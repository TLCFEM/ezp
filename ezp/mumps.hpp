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

#ifndef MUMPS_HPP
#define MUMPS_HPP

#include <ezp/abstract/traits.hpp>
#include <mumps/cmumps_c.h>
#include <mumps/dmumps_c.h>
#include <mumps/smumps_c.h>
#include <mumps/zmumps_c.h>

namespace ezp {
    namespace detail {
        template<typename DT> struct mumps_struc {};
        template<> struct mumps_struc<double> {
            using type = DMUMPS_STRUC_C;
            using data_type = double;
            static auto mumps_c(DMUMPS_STRUC_C* ptr) { return dmumps_c(ptr); }
        };
        template<> struct mumps_struc<float> {
            using type = SMUMPS_STRUC_C;
            using data_type = float;
            static auto mumps_c(SMUMPS_STRUC_C* ptr) { return smumps_c(ptr); }
        };
        template<> struct mumps_struc<complex16> {
            using type = ZMUMPS_STRUC_C;
            using data_type = mumps_double_complex;
            static auto mumps_c(ZMUMPS_STRUC_C* ptr) { return zmumps_c(ptr); }
        };
        template<> struct mumps_struc<complex8> {
            using type = CMUMPS_STRUC_C;
            using data_type = mumps_complex;
            static auto mumps_c(CMUMPS_STRUC_C* ptr) { return cmumps_c(ptr); }
        };
    } // namespace detail

    template<data_t DT, index_t IT> class mumps final {
        using struct_t = typename detail::mumps_struc<DT>::type;
        using element_t = typename detail::mumps_struc<DT>::data_type;

        struct_t id;

        const auto& comm_world{mpl::environment::comm_world()};

    public:
        explicit mumps(const int sym) {
            id.comm_fortran = MPI_Comm_c2f(comm_world.native_handle());
            id.sym = sym;
            id.job = -1;
            mumps_struc<DT>::mumps_c(&id);
        };

        ~mumps() {
            id.job = -2;
            mumps_struc<DT>::mumps_c(&id);
        };

        auto& operator()(const IT index) { return id.icntl[index]; }

        IT solve(sparse_csr_mat<DT, IT>&& A, full_mat<DT, IT>&& B) {
            id.n = A.n;
            id.nnz = A.nnz;
            id.irn = A.ia;
            id.jcn = A.ja;
            id.a = (element_t*)A.a;
            id.rhs = (element_t*)B.data;

            id.job = 6;
            mumps_struc<DT>::mumps_c(&id);

            IT error = id.infog[0] < 0 ? -1 : 0;
            comm_world.allreduce(mpl::min<IT>(), error);

            return error;
        }
    };
} // namespace ezp

#endif