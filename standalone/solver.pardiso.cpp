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

#ifdef EZP_MKL

#include <ezp/abstract/traits.hpp>
#include <mpl/mpl.hpp>

const auto& comm_world{mpl::environment::comm_world()};
const auto& parent = mpl::inter_communicator::parent();

template<typename DT, typename IT> int run(const IT (&config)[7], IT (&iparm)[64]) {
    const auto mtype = config;
    const auto nrhs = config + 1;
    const auto maxfct = config + 2;
    const auto mnum = config + 3;
    const auto msglvl = config + 4;
    const auto n = config + 5;
    const auto nnz = config + 6;

    const auto nb = *n * *nrhs;

    std::vector<IT> ia(*n + 1), ja(*nnz);
    std::vector<DT> a(*nnz), b(nb), x(nb);

    if(0 == comm_world.rank()) {
        mpl::irequest_pool requests;

        requests.push(parent.irecv(ia, 0, mpl::tag_t{0}));
        requests.push(parent.irecv(ja, 0, mpl::tag_t{1}));
        requests.push(parent.irecv(a, 0, mpl::tag_t{2}));
        requests.push(parent.irecv(b, 0, mpl::tag_t{3}));

        requests.waitall();
    }

    std::int64_t pt[64]{};

    const auto comm = MPI_Comm_c2f(comm_world.native_handle());

    IT error = 0;

    IT phase = 13;
    if constexpr(sizeof(IT) == 4) {
        using E = int;
        cluster_sparse_solver(pt, (E*)maxfct, (E*)mnum, (E*)mtype, (E*)&phase, (E*)n, a.data(), (E*)ia.data(), (E*)ja.data(), nullptr, (E*)nrhs, (E*)iparm, (E*)msglvl, b.data(), x.data(), &comm, (E*)&error);
    }
    else {
        using E = long long int;
        cluster_sparse_solver_64(pt, (E*)maxfct, (E*)mnum, (E*)mtype, (E*)&phase, (E*)n, a.data(), (E*)ia.data(), (E*)ja.data(), nullptr, (E*)nrhs, (E*)iparm, (E*)msglvl, b.data(), x.data(), &comm, (E*)&error);
    }

    if(0 == comm_world.rank()) {
        parent.send(error, 0);
        if(0 == error) parent.send(x, 0);
    }

    phase = -1;
    if constexpr(sizeof(IT) == 4) {
        using E = int;
        cluster_sparse_solver(pt, (E*)maxfct, (E*)mnum, (E*)mtype, (E*)&phase, (E*)n, nullptr, (E*)ia.data(), (E*)ja.data(), nullptr, (E*)nrhs, (E*)iparm, (E*)msglvl, nullptr, nullptr, &comm, (E*)&error);
    }
    else {
        using E = long long int;
        cluster_sparse_solver_64(pt, (E*)maxfct, (E*)mnum, (E*)mtype, (E*)&phase, (E*)n, nullptr, (E*)ia.data(), (E*)ja.data(), nullptr, (E*)nrhs, (E*)iparm, (E*)msglvl, nullptr, nullptr, &comm, (E*)&error);
    }

    return 0;
}

int main(int argc, char** argv) {
    if(!parent.is_valid()) {
        printf("This program must be invoked by the host application.\n");
        return 0;
    }

    const auto all = mpl::communicator(parent, mpl::communicator::order_high);

    int_t config[7]{};
    int_t iparm[64]{};

    all.bcast(0, config);
    all.bcast(0, iparm);

    if(config[0] == 1 || config[0] == 1 || config[0] == 2 || config[0] == -2 || config[0] == 11) {
        if(0 == iparm[27]) return run<double>(config, iparm);
        return run<float>(config, iparm);
    }

    if(0 == iparm[27]) return run<complex16>(config, iparm);
    return run<complex8>(config, iparm);
}

#endif
