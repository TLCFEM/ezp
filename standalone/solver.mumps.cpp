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

#include <ezp/abstract/ezp.h>
#include <mpl/mpl.hpp>
#include <mumps/dmumps_c.h>
#include <mumps/smumps_c.h>

const auto& comm_world{mpl::environment::comm_world()};
const auto& parent = mpl::inter_communicator::parent();

template<typename DT> struct mumps_struc {};
template<> struct mumps_struc<double> {
    using type = DMUMPS_STRUC_C;
    static auto mumps_c(DMUMPS_STRUC_C* ptr) { return dmumps_c(ptr); }
};
template<> struct mumps_struc<float> {
    using type = SMUMPS_STRUC_C;
    static auto mumps_c(SMUMPS_STRUC_C* ptr) { return smumps_c(ptr); }
};

template<typename DT, typename IT> int run(const IT (&config)[5]) {
    const auto sym = config[0];
    const auto nrhs = config[1];
    const auto n = config[2];
    const auto nnz = config[3];
    const auto msglvl = config[4];

    using struct_t = mumps_struc<DT>::type;

    struct_t id;
    id.comm_fortran = MPI_Comm_c2f(comm_world.native_handle());
    id.sym = sym;

    id.job = -1;
    mumps_struc<DT>::mumps_c(&id);

    id.icntl[3] = msglvl;

    std::vector<IT> ia, ja;
    std::vector<DT> a, b;

    if(0 == comm_world.rank()) {
        ia.resize(n + 1);
        ja.resize(nnz);
        a.resize(nnz);
        b.resize(n * nrhs);

        mpl::irequest_pool requests;

        requests.push(parent.irecv(ia, 0, mpl::tag_t{0}));
        requests.push(parent.irecv(ja, 0, mpl::tag_t{1}));
        requests.push(parent.irecv(a, 0, mpl::tag_t{2}));
        requests.push(parent.irecv(b, 0, mpl::tag_t{3}));

        requests.waitall();

        id.n = n;
        id.nnz = nnz;
        id.irn = ia.data();
        id.jcn = ja.data();
        id.a = a.data();
        id.rhs = b.data();
    }

    id.job = 6;
    mumps_struc<DT>::mumps_c(&id);

    IT error = id.infog[0] < 0 ? -1 : 0;
    comm_world.allreduce(mpl::min<IT>(), error);

    if(0 == comm_world.rank()) {
        parent.send(error, 0);
        if(0 == error) parent.send(b, 0);
    }

    id.job = -2;
    mumps_struc<DT>::mumps_c(&id);

    return 0;
}

template<typename IT> auto prepare() {
    const auto all = mpl::communicator(parent, mpl::communicator::order_high);

    IT config[5]{};

    all.bcast(0, config);

    return run<double>(config);
}

int main(int argc, char** argv) {
    if(!parent.is_valid()) {
        printf("This program must be invoked by the host application.\n");
        return 0;
    }

    return prepare<int_t>();
}
