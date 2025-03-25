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
 * @class lis
 * @brief Iterative solver for general sparse matrices.
 *
 * It solves the system of linear equations `A * X = B` with a general sparse matrix `A`.
 * The RHS matrix `B` is a dense matrix.
 *
 * The example usage can be seen as follows.
 *
 * @include example.lis.cpp
 *
 * @author tlc
 * @date 23/03/2025
 * @version 1.0.0
 * @file lis.hpp
 * @{
 */

#ifndef LIS_HPP
#define LIS_HPP

#include <ezp.lis.h>
#include <ezp/abstract/traits.hpp>
#include <mpl/mpl.hpp>

namespace ezp {
    struct lis_env {
        const mpl::communicator& comm_world{mpl::environment::comm_world()};

        lis_env() {
            lis_do_not_handle_mpi();
            lis_initialize(nullptr, nullptr);
        }
        ~lis_env() { lis_finalize(); }

        auto& comm() const { return comm_world; }
        auto rank() const { return comm_world.rank(); }
    };

    auto& get_lis_env() {
        static const lis_env env;
        return env;
    }

    template<index_t IT> class lis {
        const lis_env& env = get_lis_env();
        const MPI_Comm comm = env.comm().native_handle();

        LIS_SOLVER solver;
        LIS_MATRIX a_loc;

        bool is_set = false;

        auto sync_error(IT error) {
            env.comm().allreduce(mpl::min<IT>(), error);
            return error;
        }

        auto deregister_matrix() {
            if(is_set) lis_matrix_unset(a_loc);
            is_set = false;
        }

        auto register_matrix(const sparse_csr_mat<LIS_SCALAR, IT>& A) {
            deregister_matrix();
            lis_matrix_set_size(a_loc, 0 == env.rank() ? A.n : 0, 0);
            lis_matrix_set_csr(0 == env.rank() ? A.nnz : 0, A.irn, A.jcn, A.data, a_loc);
            lis_matrix_assemble(a_loc);
            is_set = true;
        }

        LIS_VECTOR create_vector(const IT n) {
            LIS_VECTOR v;
            lis_vector_create(comm, &v);
            lis_vector_set_size(v, 0 == env.rank() ? n : 0, 0);
            return v;
        }

    public:
        lis(const char* setting) {
            lis_solver_create(&solver);
            lis_solver_set_option(setting, solver);
            lis_matrix_create(comm, &a_loc);
        }

        ~lis() {
            deregister_matrix();
            lis_matrix_destroy(a_loc);
            lis_solver_destroy(solver);
        }

        IT solve(sparse_csr_mat<LIS_SCALAR, IT>&& A, full_mat<LIS_SCALAR, IT>&& B) {
            IT error = 0;
            if(0 == env.rank() && A.irn[A.n] != A.nnz) error = -1;

            error = sync_error(error);
            if(error < 0) return error;

            if(A.n != B.n_rows) return -1;

            register_matrix(A);

            std::vector<LIS_SCALAR> b_ref;
            if(0 == env.rank()) {
                b_ref.resize(B.n_rows * B.n_cols);
                std::copy(B.data, B.data + b_ref.size(), b_ref.data());
            }

            auto b_loc = create_vector(B.n_rows);
            auto x_loc = create_vector(B.n_rows);

            for(auto I = 0, J = 0; I < B.n_cols; ++I, J += B.n_rows) {
                lis_vector_set(b_loc, 0 == env.rank() ? b_ref.data() + J : nullptr);
                lis_vector_set(x_loc, 0 == env.rank() ? B.data + J : nullptr);

                lis_solve(a_loc, b_loc, x_loc, solver);

                lis_vector_unset(b_loc);
                lis_vector_unset(x_loc);
            }

            lis_vector_destroy(b_loc);
            lis_vector_destroy(x_loc);

            return 0;
        }
    };
} // namespace ezp

#endif

//! @}
