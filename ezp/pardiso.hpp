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
    template<data_t DT, index_t IT> class pardiso final {
        const mpl::communicator& comm_world{mpl::environment::comm_world()};

        const int comm = MPI_Comm_c2f(comm_world.native_handle());

        const IT mtype, maxfct, mnum, msglvl;

        std::int64_t pt[64]{};

        IT iparm[64]{};

    public:
        pardiso(const IT mtype, const IT maxfct = 1, const IT mnum = 1, const IT msglvl = 0)
            : mtype(mtype)
            , maxfct(maxfct)
            , mnum(mnum)
            , msglvl(msglvl) {};

        auto& operator()(const IT index) { return iparm[index]; }

        IT solve(sparse_csr_mat<DT, IT>&& A, full_mat<DT, IT>&& B) {
            if(A.n != B.n_rows) return -1;

            std::vector<DT> b_ref;
            if(0 == comm_world.rank()) {
                b_ref.resize(B.n_rows * B.n_cols);
                std::copy(B.data, B.data + b_ref.size(), b_ref.data());
            }

            iparm[5] = 0; // write solution into x

            IT error = -1;

            IT phase = 13;
            if constexpr(sizeof(IT) == 4) {
                using E = std::int32_t;
                cluster_sparse_solver(pt, (E*)&maxfct, (E*)&mnum, (E*)&mtype, (E*)&phase, (E*)&A.n, A.data, (E*)A.irn, (E*)A.jcn, nullptr, (E*)&B.n_cols, (E*)iparm, (E*)&msglvl, b_ref.data(), B.data, &comm, (E*)&error);
            }
            else if constexpr(sizeof(IT) == 8) {
                using E = std::int64_t;
                cluster_sparse_solver_64(pt, (E*)&maxfct, (E*)&mnum, (E*)&mtype, (E*)&phase, (E*)&A.n, A.data, (E*)A.irn, (E*)A.jcn, nullptr, (E*)&B.n_cols, (E*)iparm, (E*)&msglvl, b_ref.data(), B.data, &comm, (E*)&error);
            }

            phase = -1;
            if constexpr(sizeof(IT) == 4) {
                using E = std::int32_t;
                cluster_sparse_solver(pt, (E*)&maxfct, (E*)&mnum, (E*)&mtype, (E*)&phase, (E*)&A.n, nullptr, (E*)A.irn, (E*)A.jcn, nullptr, (E*)&B.n_cols, (E*)iparm, (E*)&msglvl, nullptr, nullptr, &comm, (E*)&error);
            }
            else if constexpr(sizeof(IT) == 8) {
                using E = std::int64_t;
                cluster_sparse_solver_64(pt, (E*)&maxfct, (E*)&mnum, (E*)&mtype, (E*)&phase, (E*)&A.n, nullptr, (E*)A.irn, (E*)A.jcn, nullptr, (E*)&B.n_cols, (E*)iparm, (E*)&msglvl, nullptr, nullptr, &comm, (E*)&error);
            }

            return error;
        }
    };
} // namespace ezp

#endif

#endif

//! @}
