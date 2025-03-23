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
 * @class mumps
 * @brief Solver for general sparse matrices.
 *
 * It solves the system of linear equations `A*X=B` with a general sparse matrix `A`.
 * The RHS matrix `B` is a dense matrix.
 *
 * The solver automatically detects whether `A` is distributed on many processes or
 * centralized on the root process.
 * The matrix `A` should be stored in the Compressed Sparse Row (CSR) format with one-based indexing.
 *
 * To set control parameters, use the overloaded function call operator, which allows access to the `icntl` array.
 *
 * @code
   auto solver = mumps<double, int_t>();
   solver(3) = 0; // suppress output
 * @endcode
 *
 * The example usage can be seen as follows.
 *
 * @include ../examples/example.mumps.cpp
 *
 * @author tlc
 * @date 23/03/2025
 * @version 1.0.0
 * @file mumps.hpp
 * @{
 */

#ifndef MUMPS_HPP
#define MUMPS_HPP

#include <ezp/abstract/traits.hpp>
#include <mpl/mpl.hpp>
#include <mumps/cmumps_c.h>
#include <mumps/dmumps_c.h>
#include <mumps/smumps_c.h>
#include <mumps/zmumps_c.h>

namespace ezp {
    namespace detail {
        template<typename DT> struct mumps_struc {};
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

    template<data_t DT, index_t IT> class mumps final {
        using struct_t = typename detail::mumps_struc<DT>::struct_type;
        using entry_t = typename detail::mumps_struc<DT>::entry_type;

        struct_t id;

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
        explicit mumps(const int sym = 0) {
            id.comm_fortran = MPI_Comm_c2f(comm_world.native_handle());
            id.sym = sym;
            perform_job(-1);
        };

        ~mumps() { perform_job(-2); };

        /**
         * @brief Overloaded function call operator to access elements of the `icntl` array.
         *
         * @param index The index of the element to access.
         * @return A reference to the element at the specified index in the `icntl` array.
         */
        auto& operator()(const IT index) { return id.icntl[index]; }

        IT solve(sparse_csr_mat<DT, IT>&& A, full_mat<DT, IT>&& B) {
            if(A.n != B.n_rows) return -1;

            int has_data = A.is_valid() ? 1 : 0;
            comm_world.allreduce(mpl::plus<int>(), has_data);
            if(has_data > 1) {
                id.icntl[17] = 3;

                id.nnz_loc = A.nnz;
                id.irn_loc = A.irn;
                id.jcn_loc = A.jcn;
                id.a_loc = (entry_t*)A.data;
            }
            else {
                id.nnz = A.nnz;
                id.irn = A.irn;
                id.jcn = A.jcn;
                id.a = (entry_t*)A.data;
            }

            id.n = A.n;
            id.nrhs = B.n_cols;
            id.rhs = (entry_t*)B.data;

            perform_job(6);

            return sync_error();
        }

        IT solve(full_mat<DT, IT>&& B) {
            if(id.n != B.n_rows) return -1;

            id.nrhs = B.n_cols;
            id.rhs = (entry_t*)B.data;

            perform_job(3);

            return sync_error();
        }
    };
} // namespace ezp

#endif

//! @}
