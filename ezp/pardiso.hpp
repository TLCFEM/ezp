/*******************************************************************************
 * Copyright (C) 2025 Theodore Chang
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
 * The example usage can be seen as follows.
 *
 * @include example.pardiso.cpp
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

    template<data_t DT, index_t IT> class pardiso final {
        static constexpr IT one{1}, negone{-1}, PARDISO_ANA_FACT{12}, PARDISO_SOLVE{33}, PARDISO_RELEASE{-1};

        std::int64_t pt[64]{};

        IT iparm[64]{};

        sparse_csr_mat<DT, IT> a_mat{};

        const mpl::communicator& comm_world{mpl::environment::comm_world()};

        const int comm = MPI_Comm_c2f(comm_world.native_handle());

        const IT mtype, msglvl;

        bool is_allocated{false};

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
            : pardiso(matrix_type{static_cast<std::int8_t>(mtype)}, message_level{static_cast<std::int8_t>(msglvl)}) {};

        explicit pardiso(const matrix_type mtype, const message_level msglvl = no_output)
            : mtype(mtype)
            , msglvl(msglvl) {};

        pardiso(const pardiso& other)
            : a_mat(other.a_mat)
            , mtype(other.mtype)
            , msglvl(other.msglvl) { std::copy_n(other.iparm, 64, iparm); }
        pardiso(pardiso&&) = delete;
        pardiso& operator=(const pardiso&) = delete;
        pardiso& operator=(pardiso&&) = delete;

        ~pardiso() { dealloc(); }

        auto& operator()(const IT index) { return iparm[index]; }

        auto& iparm_default_value(const auto config) { return iparm[0] = config; };
        auto& iparm_reducing_ordering(const auto config) { return iparm[1] = config; };
        auto& iparm_user_permutation(const auto config) { return iparm[4] = config; };
        auto& iparm_iterative_refinement(const auto config) { return iparm[7] = config; };
        auto& iparm_pivoting_perturbation(const auto config) { return iparm[9] = config; };
        auto& iparm_scaling(const auto config) { return iparm[10] = config; };
        auto& iparm_transpose_matrix(const auto config) { return iparm[11] = config; };
        auto& iparm_weighted_matching(const auto config) { return iparm[12] = config; };
        auto& iparm_nnz_factor(const auto config) { return iparm[17] = config; };
        auto& iparm_pivoting_type(const auto config) { return iparm[20] = config; };
        auto& iparm_matrix_checker(const auto config) { return iparm[26] = config; };
        // auto& iparm_precision(const auto config) { return iparm[27] = config; };
        auto& iparm_partial_solve(const auto config) { return iparm[30] = config; };
        auto& iparm_zero_based_indexing(const auto config) { return iparm[34] = config; };
        auto& iparm_schur_complement(const auto config) { return iparm[35] = config; };
        auto& iparm_out_of_core(const auto config) { return iparm[59] = config; };

        IT solve(sparse_csr_mat<DT, IT>&& A, full_mat<DT, IT>&& B) {
            if(A.n != B.n_rows) return -1;

            auto error = sync_error(alloc(std::move(A)));
            if(error < 0) return error;

            iparm[39] = 0; // force centralised input/output

            if constexpr(std::is_same_v<DT, double> || std::is_same_v<DT, complex16>) iparm[27] = 0;
            else if constexpr(std::is_same_v<DT, float> || std::is_same_v<DT, complex8>) iparm[27] = 1;

            return solve(std::move(B));
        }

        IT solve(full_mat<DT, IT>&& B) {
            if(a_mat.n != B.n_rows) return -1;

            std::vector<DT> b_ref;
            if(0 == comm_world.rank()) b_ref.resize(B.n_rows * B.n_cols);

            void *b_ptr, *x_ptr;

            if(0 == iparm[0] || 0 == iparm[5]) {
                // b unchanged, x has solution
                if(0 == comm_world.rank()) std::copy_n(B.data, b_ref.size(), b_ref.data());
                b_ptr = b_ref.data();
                x_ptr = B.data;
            }
            else {
                // b has solution, x is still used
                b_ptr = B.data;
                x_ptr = b_ref.data();
            }

            IT error{-1};
            if constexpr(sizeof(IT) == 4) cluster_sparse_solver(pt, &one, &one, &mtype, &PARDISO_SOLVE, &a_mat.n, a_mat.data, a_mat.row_ptr, a_mat.col_idx, nullptr, &B.n_cols, iparm, &msglvl, b_ptr, x_ptr, &comm, &error);
            else if constexpr(sizeof(IT) == 8) cluster_sparse_solver_64(pt, &one, &one, &mtype, &PARDISO_SOLVE, &a_mat.n, a_mat.data, a_mat.row_ptr, a_mat.col_idx, nullptr, &B.n_cols, iparm, &msglvl, b_ptr, x_ptr, &comm, &error);

            return sync_error(error);
        }
    };
} // namespace ezp

#endif

#endif

//! @}
