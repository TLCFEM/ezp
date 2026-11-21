/*******************************************************************************
 * Copyright (C) 2025-2026 Theodore Chang
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 ******************************************************************************/
/**
 * @class pardiso
 * @brief Solver for general sparse matrices.
 *
 * It solves the system of linear equations `A * X = B` with a general sparse matrix `A`.
 * The RHS matrix `B` is a dense matrix.
 *
 * The matrix `A` should be stored in the Compressed Sparse Row (CSR) format with one-based indexing.
 * Use the call operator to set the parameters `iparm` for the solver.
 *
 * The optional template parameter `DC` selects between a centralized matrix (`distribution_category::centralized`, the default) and a matrix distributed row-wise across the MPI processes (`distribution_category::distributed_matrix`).
 *
 * The example usage can be seen as follows.
 *
 * @include example.pardiso.cpp
 * @include example.pardiso.distributed.cpp
 *
 * @author tlc
 * @date 23/03/2025
 * @version 1.0.0
 * @file pardiso.hpp
 * @{
 */

#ifndef PARDISO_HPP
#define PARDISO_HPP

#ifdef EZP_MKL

#include "abstract/sparse_solver.hpp"

#include <mpl/mpl.hpp>

namespace ezp {
    enum matrix_type : std::int8_t {
        real_and_symmetric_positive_definite = 2,
        real_and_symmetric_indefinite = -2,
        real_and_structurally_symmetric = 1,
        real_and_nonsymmetric = 11,
        complex_and_hermitian_positive_definite = 4,
        complex_and_hermitian_indefinite = -4,
        complex_and_structurally_symmetric = 3,
        complex_and_symmetric = 6,
        complex_and_nonsymmetric = 13
    };

    enum message_level : std::int8_t {
        no_output = 0,
        print_statistical_information = 1
    };

    /**
     * @enum distribution_category
     * @brief Distribution of the matrix.
     *
     * The enumerators map directly to the supported values of `iparm[39]` of the Intel MKL Cluster Sparse Solver.
     * The right-hand side and the solution are always stored on the master process.
     */
    enum class distribution_category : std::int8_t {
        /** The master process stores the whole matrix. */
        centralized = 0,
        /** Each process stores a contiguous row block of the matrix. */
        distributed_matrix = 1
    };

    template<data_t DT, index_t IT, distribution_category DC = distribution_category::centralized> class pardiso final {
        static constexpr IT one{1}, negone{-1}, PARDISO_ANA_FACT{12}, PARDISO_SOLVE{33}, PARDISO_RELEASE{-1};

        std::int64_t pt[64]{};

        IT iparm[64]{};

        sparse_csr_mat<DT, IT> a_mat{};

        const mpl::communicator& comm_world{mpl::environment::comm_world()};

        const int comm = MPI_Comm_c2f(comm_world.native_handle());

        const IT mtype, msglvl;

        bool is_allocated{false};

        bool domain_explicit{false};

        IT local_n{0}, row_begin{0}, row_end{0};

        static constexpr auto is_distributed() { return distribution_category::centralized != DC; }

        auto is_zero_based() const { return 0 != iparm[34]; }

        /**
         * @brief Fill the non-zero MKL defaults before enabling the distributed input format.
         *
         * If `iparm[0]` is zero, MKL overwrites `iparm[1]` to `iparm[63]` with defaults during the analysis phase, including `iparm[39]`.
         * The distributed category would be lost.
         * The wrapper therefore switches to user-supplied parameters and reproduces the defaults that matter for the matrix type.
         */
        auto initialize_defaults() {
            if(0 != iparm[0]) return;

            iparm[0] = 1;  // use the values supplied below
            iparm[1] = 2;  // METIS nested dissection
            iparm[5] = 0;  // write the solution to x
            iparm[7] = 0;  // MKL default iterative refinement
            iparm[20] = 1; // Bunch-Kaufman pivoting for symmetric indefinite matrices

            const auto nonsymmetric = 11 == mtype || 13 == mtype;
            const auto symmetric_indefinite = -2 == mtype || -4 == mtype || 6 == mtype;

            iparm[9] = symmetric_indefinite ? 8 : 13;
            iparm[10] = nonsymmetric ? 1 : 0;
            iparm[12] = nonsymmetric ? 1 : 0;
        }

        /**
         * @brief Determine the local row range and write the MKL domain parameters.
         *
         * The range is taken from `iparm_input_domain` when it has been set.
         * Otherwise, the local row count is taken from `sparse_csr_mat::n_local` and the local row blocks are expected to be in rank order, so the global offsets are computed with a prefix scan.
         */
        auto prepare_distribution(const sparse_csr_mat<DT, IT>& A) -> IT {
            if constexpr(!is_distributed()) {
                local_n = A.n;
                row_begin = 0;
                row_end = A.n - 1;
                return A.n_local <= 0 || A.n_local == A.n ? 0 : -1;
            }

            IT explicit_flag = domain_explicit ? 1 : 0;
            IT flag_min = explicit_flag, flag_max = explicit_flag;
            comm_world.allreduce(mpl::min<IT>(), flag_min);
            comm_world.allreduce(mpl::max<IT>(), flag_max);
            if(flag_min != flag_max) return -1; // some processes did not provide a domain

            const IT shift = is_zero_based() ? 0 : 1;

            if(explicit_flag) {
                const auto begin = iparm[40] - shift;
                const auto end = iparm[41] - shift;

                if(begin < 0 || end < begin || end >= A.n) return -1;

                local_n = end - begin + 1;
                row_begin = begin;
                row_end = end;
            }
            else {
                const auto count = A.n_local;

                IT prefix = 0;
                comm_world.scan(mpl::plus<IT>(), count, prefix);

                if(count <= 0) return -1;

                local_n = count;
                row_begin = prefix - count;
                row_end = prefix - 1;

                iparm[40] = row_begin + shift;
                iparm[41] = row_end + shift;
            }

            IT total_local = local_n;
            comm_world.allreduce(mpl::plus<IT>(), total_local);

            return total_local == A.n ? 0 : -1;
        }

        auto alloc(sparse_csr_mat<DT, IT>&& A) {
            dealloc();
            is_allocated = true;
            a_mat = std::move(A);
            IT error{-1};
            if constexpr(sizeof(IT) == 4) cluster_sparse_solver(pt, &one, &one, &mtype, &PARDISO_ANA_FACT, &a_mat.n, a_mat.data, a_mat.row_ptr, a_mat.col_idx, nullptr, &negone, iparm, &msglvl, nullptr, nullptr, &comm, &error);
            else if constexpr(sizeof(IT) == 8) cluster_sparse_solver_64(pt, &one, &one, &mtype, &PARDISO_ANA_FACT, &a_mat.n, a_mat.data, a_mat.row_ptr, a_mat.col_idx, nullptr, &negone, iparm, &msglvl, nullptr, nullptr, &comm, &error);
            return sync_error(error);
        }

        auto dealloc() {
            if(!is_allocated) return;
            is_allocated = false;
            IT error;
            if constexpr(sizeof(IT) == 4) cluster_sparse_solver(pt, &one, &one, &mtype, &PARDISO_RELEASE, &negone, nullptr, nullptr, nullptr, nullptr, &negone, iparm, &msglvl, nullptr, nullptr, &comm, &error);
            else if constexpr(sizeof(IT) == 8) cluster_sparse_solver_64(pt, &one, &one, &mtype, &PARDISO_RELEASE, &negone, nullptr, nullptr, nullptr, nullptr, &negone, iparm, &msglvl, nullptr, nullptr, &comm, &error);
            for(auto& i : pt) i = 0;
        }

        auto sync_error(IT error) {
            comm_world.allreduce(mpl::min<IT>(), error);
            return error;
        }

    public:
        explicit pardiso(const IT mtype, const IT msglvl = 0)
            : pardiso(matrix_type{static_cast<std::int8_t>(mtype)}, message_level{static_cast<std::int8_t>(msglvl)}) {}

        explicit pardiso(const matrix_type mtype, const message_level msglvl = no_output)
            : mtype(mtype)
            , msglvl(msglvl) {
            if constexpr(floating_t<DT>) {
                if(mtype != real_and_symmetric_positive_definite && mtype != real_and_symmetric_indefinite && mtype != real_and_structurally_symmetric && mtype != real_and_nonsymmetric) throw std::invalid_argument("matrix type does not match the data type");
            }
            else {
                if(mtype != complex_and_hermitian_positive_definite && mtype != complex_and_hermitian_indefinite && mtype != complex_and_structurally_symmetric && mtype != complex_and_symmetric && mtype != complex_and_nonsymmetric) throw std::invalid_argument("matrix type does not match the data type");
            }

            iparm[39] = static_cast<IT>(DC);
        }

        pardiso(const pardiso& other)
            : a_mat(other.a_mat)
            , mtype(other.mtype)
            , msglvl(other.msglvl)
            , domain_explicit(other.domain_explicit)
            , local_n(other.local_n)
            , row_begin(other.row_begin)
            , row_end(other.row_end) { std::copy_n(other.iparm, 64, iparm); }
        pardiso(pardiso&&) = delete;
        pardiso& operator=(const pardiso&) = delete;
        pardiso& operator=(pardiso&&) = delete;

        ~pardiso() { dealloc(); }

        auto& operator()(const IT index) { return iparm[index]; }

        auto& iparm_default_value(const auto config) {
            iparm[0] = config;
            return *this;
        };
        auto& iparm_reducing_ordering(const auto config) {
            iparm[1] = config;
            return *this;
        };
        auto& iparm_user_permutation(const auto config) {
            iparm[4] = config;
            return *this;
        };
        auto& iparm_iterative_refinement(const auto config) {
            iparm[7] = config;
            return *this;
        };
        auto& iparm_pivoting_perturbation(const auto config) {
            iparm[9] = config;
            return *this;
        };
        auto& iparm_scaling(const auto config) {
            iparm[10] = config;
            return *this;
        };
        auto& iparm_transpose_matrix(const auto config) {
            iparm[11] = config;
            return *this;
        };
        auto& iparm_weighted_matching(const auto config) {
            iparm[12] = config;
            return *this;
        };
        auto& iparm_nnz_factor(const auto config) {
            iparm[17] = config;
            return *this;
        };
        auto& iparm_pivoting_type(const auto config) {
            iparm[20] = config;
            return *this;
        };
        auto& iparm_matrix_checker(const auto config) {
            iparm[26] = config;
            return *this;
        };
        auto& iparm_precision(const auto config) {
            iparm[27] = config;
            return *this;
        };
        auto& iparm_partial_solve(const auto config) {
            iparm[30] = config;
            return *this;
        };
        auto& iparm_zero_based_indexing(const auto config) {
            iparm[34] = config;
            return *this;
        };
        auto& iparm_input_domain(const IT begin, const IT end) {
            iparm[40] = begin;
            iparm[41] = end;
            domain_explicit = true;
            return *this;
        };
        auto& iparm_schur_complement(const auto config) {
            iparm[35] = config;
            return *this;
        };
        auto& iparm_out_of_core(const auto config) {
            iparm[59] = config;
            return *this;
        };

        auto validate_rhs(const full_mat<DT, IT>& B, const IT n, IT& nrhs) -> bool {
            const auto on_master = 0 == comm_world.rank();
            IT valid = 1;

            if(on_master) {
                nrhs = B.n_cols;
                if(B.n_rows != n || nrhs <= 0 || nullptr == B.data) valid = 0;
            }
            comm_world.bcast(0, nrhs);

            comm_world.allreduce(mpl::min<IT>(), valid);

            return 0 != valid && nrhs > 0;
        }

        auto solve_impl(full_mat<DT, IT>&& B, const IT nrhs) -> IT {
            const auto on_master = 0 == comm_world.rank();

            std::vector<DT> b_ref, x_ref;

            if constexpr(!is_distributed()) {
                const auto size = on_master ? a_mat.n * nrhs : IT{0};
                b_ref.resize(size);
                x_ref.resize(size);

                if(on_master) std::copy_n(B.data, b_ref.size(), b_ref.data());
            }
            else {
                // The right-hand side and the solution are stored centrally on the master process.
                b_ref.resize(local_n * nrhs);
                if(on_master) x_ref.resize(a_mat.n * nrhs);

                // Repack the right-hand side into local row blocks because MKL expects the local rows to be contiguous with leading dimension equal to the local number of rows.
                const auto size = static_cast<std::size_t>(comm_world.size());
                std::vector<IT> counts(size), begins(size), offsets(size);

                comm_world.allgather(local_n, counts.data());
                comm_world.allgather(row_begin, begins.data());

                std::vector<DT> packed;
                if(on_master) {
                    IT total = 0;
                    for(std::size_t I = 0; I < size; ++I) {
                        offsets[I] = total;
                        total += counts[I];
                    }

                    packed.resize(static_cast<std::size_t>(total * nrhs));
                    for(std::size_t I = 0; I < size; ++I)
                        for(IT K = 0; K < nrhs; ++K)
                            std::copy_n(B.data + begins[I] + K * a_mat.n, counts[I], packed.data() + (offsets[I] * nrhs + K * counts[I]));
                }

                mpl::contiguous_layouts<DT> send_layouts;
                mpl::displacements send_displs;
                for(std::size_t I = 0; I < size; ++I) {
                    send_layouts.push_back(mpl::contiguous_layout<DT>(static_cast<std::size_t>(counts[I] * nrhs)));
                    send_displs.push_back(static_cast<MPI_Aint>(offsets[I] * nrhs) * static_cast<MPI_Aint>(sizeof(DT)));
                }

                comm_world.scatterv(0, on_master ? packed.data() : nullptr, send_layouts, send_displs, b_ref.data(), mpl::contiguous_layout<DT>(static_cast<std::size_t>(local_n * nrhs)));
            }

            void* b_ptr = b_ref.empty() ? nullptr : static_cast<void*>(b_ref.data());
            void* x_ptr = x_ref.empty() ? nullptr : static_cast<void*>(x_ref.data());

            // Let MKL write the solution into x.
            // This keeps the input and output buffers separate.
            const auto write_to_b = iparm[5];
            iparm[5] = 0;

            IT error{-1};
            if constexpr(sizeof(IT) == 4) cluster_sparse_solver(pt, &one, &one, &mtype, &PARDISO_SOLVE, &a_mat.n, a_mat.data, a_mat.row_ptr, a_mat.col_idx, nullptr, &nrhs, iparm, &msglvl, b_ptr, x_ptr, &comm, &error);
            else if constexpr(sizeof(IT) == 8) cluster_sparse_solver_64(pt, &one, &one, &mtype, &PARDISO_SOLVE, &a_mat.n, a_mat.data, a_mat.row_ptr, a_mat.col_idx, nullptr, &nrhs, iparm, &msglvl, b_ptr, x_ptr, &comm, &error);

            iparm[5] = write_to_b;

            error = sync_error(error);
            if(error < 0) return error;

            if(!x_ref.empty()) std::copy_n(x_ref.data(), x_ref.size(), B.data);

            return error;
        }

        IT solve(sparse_csr_mat<DT, IT>&& A, full_mat<DT, IT>&& B) {
            if(A.n <= 0) return -1;

            iparm[39] = static_cast<IT>(DC);

            if constexpr(is_distributed()) initialize_defaults();

            if constexpr(std::is_same_v<DT, double> || std::is_same_v<DT, complex16>) iparm[27] = 0;
            else if constexpr(std::is_same_v<DT, float> || std::is_same_v<DT, complex8>) iparm[27] = 1;

            auto error = sync_error(prepare_distribution(A));
            if(error < 0) return error;

            IT nrhs = 0;
            if(!validate_rhs(B, A.n, nrhs)) return -1;

            error = sync_error(alloc(std::move(A)));
            if(error < 0) return error;

            return solve_impl(std::move(B), nrhs);
        }

        IT solve(full_mat<DT, IT>&& B) {
            if(a_mat.n <= 0) return -1;
            if constexpr(is_distributed())
                if(local_n <= 0) return -1;

            iparm[39] = static_cast<IT>(DC);

            IT nrhs = 0;
            if(!validate_rhs(B, a_mat.n, nrhs)) return -1;

            return solve_impl(std::move(B), nrhs);
        }
    };
} // namespace ezp

#endif

#endif

//! @}
