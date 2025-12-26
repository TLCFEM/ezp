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
 * @class mumps
 * @brief Solver for general sparse matrices.
 *
 * It solves the system of linear equations `A * X = B` with a general sparse matrix `A`.
 * The RHS matrix `B` is a dense matrix.
 *
 * The matrix `A` should be stored in the Compressed Sparse Row (CSR) format with one-based indexing.
 *
 * To set control parameters, use the overloaded function call operator, which allows access to the `icntl` array.
 *
 * @code
    auto solver = mumps<double, int_t>();
    solver(3) = 0; // suppress output
    solver.icntl_printing_level(0); // equivalent to the above
 * @endcode
 *
 * The example usage can be seen as follows.
 *
 * @include example.mumps.cpp
 *
 * @author tlc
 * @date 23/03/2025
 * @version 1.0.0
 * @file mumps.hpp
 * @{
 */

#ifndef MUMPS_HPP
#define MUMPS_HPP

#include "../external/mumps/cmumps_c.h"
#include "../external/mumps/dmumps_c.h"
#include "../external/mumps/smumps_c.h"
#include "../external/mumps/zmumps_c.h"
#include "abstract/sparse_solver.hpp"

#include <mpl/mpl.hpp>

namespace ezp {
    namespace detail {
        template<typename> struct mumps_struc {};
        template<> struct mumps_struc<double> {
            using struct_type = DMUMPS_STRUC_C;
            using entry_type = double;
            static auto mumps_c(DMUMPS_STRUC_C* ptr) { return dmumps_c(ptr); }
        };
        template<> struct mumps_struc<float> {
            using struct_type = SMUMPS_STRUC_C;
            using entry_type = float;
            static auto mumps_c(SMUMPS_STRUC_C* ptr) { return smumps_c(ptr); }
        };
        template<> struct mumps_struc<complex16> {
            using struct_type = ZMUMPS_STRUC_C;
            using entry_type = mumps_double_complex;
            static auto mumps_c(ZMUMPS_STRUC_C* ptr) { return zmumps_c(ptr); }
        };
        template<> struct mumps_struc<complex8> {
            using struct_type = CMUMPS_STRUC_C;
            using entry_type = mumps_complex;
            static auto mumps_c(CMUMPS_STRUC_C* ptr) { return cmumps_c(ptr); }
        };
    } // namespace detail

    enum symmetric_pattern : std::int8_t {
        unsymmetric = 0,
        symmetric_positive_definite = 1,
        symmetric_indefinite = 2
    };

    enum parallel_mode : std::int8_t {
        no_host = 0,
        host_involved = 1
    };

    template<data_t DT, index_t IT> class mumps final {
        using struct_t = typename detail::mumps_struc<DT>::struct_type;
        using entry_t = typename detail::mumps_struc<DT>::entry_type;

        struct_t id{};

        const mpl::communicator& comm_world{mpl::environment::comm_world()};

        auto sync_error() {
            IT error = id.infog[0] < 0 ? -1 : 0;
            comm_world.allreduce(mpl::min<IT>(), error);
            return error;
        }

        auto perform_job(const IT job) {
            id.job = job;
            detail::mumps_struc<DT>::mumps_c(&id);
        }

    public:
        explicit mumps(const symmetric_pattern sym = unsymmetric, const parallel_mode par = no_host) {
            id.comm_fortran = MPI_Comm_c2f(comm_world.native_handle());
            id.sym = sym;
            // force par=1 if there is only one process
            id.par = comm_world.size() == 1 ? 1 : par;
            perform_job(-1);
        }

        mumps(const mumps& other)
            : mumps(symmetric_pattern{static_cast<std::int8_t>(other.id.sym)}, parallel_mode{static_cast<std::int8_t>(other.id.par)}) {}
        mumps(mumps&&) = delete;
        mumps& operator=(const mumps&) = delete;
        mumps& operator=(mumps&&) = delete;

        ~mumps() { perform_job(-2); }

        /**
         * @brief Overloaded function call operator to access elements of the `icntl` array.
         *
         * @param index The index of the element to access.
         * @return A reference to the element at the specified index in the `icntl` array.
         */
        auto& operator()(const IT index) { return id.icntl[index]; }

        auto& icntl_output_error_message(const auto config) {
            id.icntl[0] = config;
            return *this;
        }
        auto& icntl_output_diagnostic_statistics_warning(const auto config) {
            id.icntl[1] = config;
            return *this;
        }
        auto& icntl_output_global_information(const auto config) {
            id.icntl[2] = config;
            return *this;
        }
        auto& icntl_printing_level(const auto config) {
            id.icntl[3] = config;
            return *this;
        }
        // auto& icntl_matrix_input_format(const auto config) {
        //     id.icntl[4] = config;
        //     return *this;
        // }
        auto& icntl_permutation_and_scaling(const auto config) {
            id.icntl[5] = config;
            return *this;
        }
        auto& icntl_symmetric_permutation(const auto config) {
            id.icntl[6] = config;
            return *this;
        }
        auto& icntl_scaling_strategy(const auto config) {
            id.icntl[7] = config;
            return *this;
        }
        auto& icntl_transpose_matrix(const auto config) {
            id.icntl[8] = config;
            return *this;
        }
        auto& icntl_iterative_refinement(const auto config) {
            id.icntl[9] = config;
            return *this;
        }
        auto& icntl_error_analysis(const auto config) {
            id.icntl[10] = config;
            return *this;
        }
        auto& icntl_ordering_strategy(const auto config) {
            id.icntl[11] = config;
            return *this;
        }
        auto& icntl_root_parallelism(const auto config) {
            id.icntl[12] = config;
            return *this;
        }
        auto& icntl_working_space_percentage_increase(const auto config) {
            id.icntl[13] = config;
            return *this;
        }
        auto& icntl_compression_block_format(const auto config) {
            id.icntl[14] = config;
            return *this;
        }
        auto& icntl_openmp_threads(const auto config) {
            id.icntl[15] = config;
            return *this;
        }
        auto& icntl_distribution_strategy_input(const auto config) {
            id.icntl[17] = config;
            return *this;
        }
        auto& icntl_schur_complement(const auto config) {
            id.icntl[18] = config;
            return *this;
        }
        // auto& icntl_rhs_format(const auto config) {
        //     id.icntl[19] = config;
        //     return *this;
        // }
        auto& icntl_distribution_strategy_solution(const auto config) {
            id.icntl[20] = config;
            return *this;
        }
        auto& icntl_out_of_core(const auto config) {
            id.icntl[21] = config;
            return *this;
        }
        auto& icntl_maximum_working_memory(const auto config) {
            id.icntl[22] = config;
            return *this;
        }
        auto& icntl_null_pivot_row_detection(const auto config) {
            id.icntl[23] = config;
            return *this;
        }
        auto& icntl_deficient_and_null_space_basis(const auto config) {
            id.icntl[24] = config;
            return *this;
        }
        auto& icntl_schur_complement_solution(const auto config) {
            id.icntl[25] = config;
            return *this;
        }
        auto& icntl_rhs_block_size(const auto config) {
            id.icntl[26] = config;
            return *this;
        }
        auto& icntl_ordering_computation(const auto config) {
            id.icntl[27] = config;
            return *this;
        }
        // auto& icntl_parallel_ordering_tool(const auto config) {
        //     id.icntl[28] = config;
        //     return *this;
        // }
        auto& icntl_inverse_computation(const auto config) {
            id.icntl[29] = config;
            return *this;
        }
        // auto& icntl_discard_factorization(const auto config) {
        //     id.icntl[30] = config;
        //     return *this;
        // }
        auto& icntl_forward_elimination(const auto config) {
            id.icntl[31] = config;
            return *this;
        }
        auto& icntl_determinant_computation(const auto config) {
            id.icntl[32] = config;
            return *this;
        }
        auto& icntl_out_of_core_file(const auto config) {
            id.icntl[33] = config;
            return *this;
        }
        auto& icntl_blr(const auto config) {
            id.icntl[34] = config;
            return *this;
        }
        auto& icntl_blr_variant(const auto config) {
            id.icntl[35] = config;
            return *this;
        }
        auto& icntl_blr_compression(const auto config) {
            id.icntl[36] = config;
            return *this;
        }
        auto& icntl_lu_compression_rate(const auto config) {
            id.icntl[37] = config;
            return *this;
        }
        auto& icntl_block_compression_rate(const auto config) {
            id.icntl[38] = config;
            return *this;
        }
        auto& icntl_tree_parallelism(const auto config) {
            id.icntl[47] = config;
            return *this;
        }
        auto& icntl_compact_working_space(const auto config) {
            id.icntl[48] = config;
            return *this;
        }
        auto& icntl_rank_revealing_factorization(const auto config) {
            id.icntl[55] = config;
            return *this;
        }
        auto& icntl_symbolic_factorization(const auto config) {
            id.icntl[57] = config;
            return *this;
        }

        auto& info() { return id.info; }
        auto& rinfo() { return id.rinfo; }
        auto& infog() { return id.infog; }
        auto& rinfog() { return id.rinfog; }

        IT solve(sparse_coo_mat<DT, IT>&& A, full_mat<DT, IT>&& B) {
            if(A.n != B.n_rows) return IT{-1};

            // ReSharper disable CppCStyleCast
            id.n = A.n;
            id.nnz = A.nnz;
            id.irn = A.row;
            id.jcn = A.col;
            id.a = (entry_t*)A.data;

            id.lrhs = B.n_rows;
            id.nrhs = B.n_cols;
            id.rhs = (entry_t*)B.data;
            // ReSharper restore CppCStyleCast

            perform_job(6);

            return sync_error();
        }

        IT solve(full_mat<DT, IT>&& B) {
            if(id.n != B.n_rows) return IT{-1};

            // ReSharper disable CppCStyleCast
            id.lrhs = B.n_rows;
            id.nrhs = B.n_cols;
            id.rhs = (entry_t*)B.data;
            // ReSharper restore CppCStyleCast

            perform_job(3);

            return sync_error();
        }

        /**
         * @brief Computes the determinant of a matrix using MUMPS library data.
         *
         * This function calculates the determinant based on the values stored in the
         * MUMPS library's internal data structures. It uses the `rinfog` and `infog`
         * arrays to retrieve necessary information for the computation.
         *
         * @note The result is meaningful only if the solver is configured to compute
         * the determinant (`icntl_determinant_computation` or directly `id.icntl[32]`).
         *
         * @tparam DT The data type of the determinant (e.g., float, double).
         * @return A `std::complex<DT>` representing the determinant of the matrix.
         *
         * @note The computation involves:
         *       - `rinfog[11]` as the real part of the determinant.
         *       - `rinfog[12]` as the imaginary part, if applicable.
         *       - `infog[33]` to determine the power of 2 scaling factor.
         *       - If `DT` is a floating-point type, the imaginary part is set to zero.
         */
        auto det() const {
            using WT = work_t<DT>;

            const auto a = id.rinfog[11];
            const auto b = floating_t<DT> ? DT{0} : id.rinfog[12];
            const auto c = std::pow(WT{2}, id.infog[33]);

            return std::complex<WT>{a, b} * c;
        }

        auto sign_det() const
            requires floating_t<DT>
        { return id.rinfog[11] > work_t<DT>{0} ? 1 : -1; }
    };
} // namespace ezp

#endif

//! @}
