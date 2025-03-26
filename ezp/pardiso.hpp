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

#include <ezp/abstract/traits.hpp>
#include <mpl/mpl.hpp>

namespace ezp {
    enum matrix_type : int {
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

    enum message_level : int {
        no_output = 0,
        print_statistical_information = 1
    };

    template<data_t DT, index_t IT> class pardiso final {
        const mpl::communicator& comm_world{mpl::environment::comm_world()};

        const int comm = MPI_Comm_c2f(comm_world.native_handle());

        static constexpr IT one{1};

        const IT mtype, msglvl;

        std::int64_t pt[64]{};

        IT iparm[64]{};

        auto sync_error(IT error) {
            comm_world.allreduce(mpl::min<IT>(), error);
            return error;
        }

    public:
        pardiso(const IT mtype, const IT msglvl = 0)
            : pardiso(matrix_type{static_cast<int>(mtype)}, message_level{static_cast<int>(msglvl)}) {};

        pardiso(const matrix_type mtype, const message_level msglvl = message_level::no_output)
            : mtype(mtype)
            , msglvl(msglvl) {};

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
        auto& iparm_precision(const auto config) { return iparm[27] = config; };
        auto& iparm_partial_solve(const auto config) { return iparm[30] = config; };
        auto& iparm_zero_based_indexing(const auto config) { return iparm[34] = config; };
        auto& iparm_schur_complement(const auto config) { return iparm[35] = config; };
        auto& iparm_out_of_core(const auto config) { return iparm[59] = config; };

        IT solve(sparse_csr_mat<DT, IT>&& A, full_mat<DT, IT>&& B) {
            IT error = 0;
            if(0 == comm_world.rank() && A.row_ptr[A.n] != A.nnz + (0 == iparm[0] || 0 == iparm[34])) error = -1;

            error = sync_error(error);
            if(error < 0) return error;

            if(A.n != B.n_rows) return -1;

            iparm[39] = 0; // force centralised input/output

            std::vector<DT> b_ref;
            if(0 == comm_world.rank()) b_ref.resize(B.n_rows * B.n_cols);

            void *b_ptr, *x_ptr;

            if(0 == iparm[0] || 0 == iparm[5]) {
                // b unchanged, x has solution
                if(0 == comm_world.rank()) std::copy(B.data, B.data + b_ref.size(), b_ref.data());
                b_ptr = b_ref.data();
                x_ptr = B.data;
            }
            else {
                // b has solution, x is still used
                b_ptr = B.data;
                x_ptr = b_ref.data();
            }

            IT phase = 13;
            if constexpr(sizeof(IT) == 4) {
                using E = std::int32_t;
                cluster_sparse_solver(pt, (E*)&one, (E*)&one, (E*)&mtype, (E*)&phase, (E*)&A.n, A.data, (E*)A.row_ptr, (E*)A.col_idx, nullptr, (E*)&B.n_cols, (E*)iparm, (E*)&msglvl, b_ptr, x_ptr, &comm, (E*)&error);
            }
            else if constexpr(sizeof(IT) == 8) {
                using E = std::int64_t;
                cluster_sparse_solver_64(pt, (E*)&one, (E*)&one, (E*)&mtype, (E*)&phase, (E*)&A.n, A.data, (E*)A.row_ptr, (E*)A.col_idx, nullptr, (E*)&B.n_cols, (E*)iparm, (E*)&msglvl, b_ptr, x_ptr, &comm, (E*)&error);
            }

            const auto info = sync_error(error);

            phase = -1;
            if constexpr(sizeof(IT) == 4) {
                using E = std::int32_t;
                cluster_sparse_solver(pt, (E*)&one, (E*)&one, (E*)&mtype, (E*)&phase, (E*)&A.n, nullptr, (E*)A.row_ptr, (E*)A.col_idx, nullptr, (E*)&B.n_cols, (E*)iparm, (E*)&msglvl, nullptr, nullptr, &comm, (E*)&error);
            }
            else if constexpr(sizeof(IT) == 8) {
                using E = std::int64_t;
                cluster_sparse_solver_64(pt, (E*)&one, (E*)&one, (E*)&mtype, (E*)&phase, (E*)&A.n, nullptr, (E*)A.row_ptr, (E*)A.col_idx, nullptr, (E*)&B.n_cols, (E*)iparm, (E*)&msglvl, nullptr, nullptr, &comm, (E*)&error);
            }

            return info;
        }
    };
} // namespace ezp

#endif

#endif

//! @}
