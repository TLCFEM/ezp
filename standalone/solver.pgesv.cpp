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

#include <ezp/pgesv.hpp>
#include <mpl/mpl.hpp>

template<ezp::data_t DT> int run(const int N, const int NRHS) {
    const auto& comm_world{mpl::environment::comm_world()};
    const auto& parent = mpl::inter_communicator::parent();

    std::vector<DT> A, B;

    if(0 == comm_world.rank()) {
        A.resize(N * N);
        B.resize(N * NRHS);

        mpl::irequest_pool requests;

        requests.push(parent.irecv(A, 0, mpl::tag_t{0}));
        requests.push(parent.irecv(B, 0, mpl::tag_t{1}));

        requests.waitall();
    }

    const auto error = ezp::pgesv<DT, int>().solve({N, N, A.data()}, {N, NRHS, B.data()});

    if(0 == comm_world.rank()) {
        parent.send(error, 0);
        if(0 == error) parent.send(B, 0);
    }

    return 0;
}

int main(int argc, char** argv) {
    ezp::blacs_env<int>::do_not_manage_mpi();

    const auto& comm_world{mpl::environment::comm_world()};
    const auto& parent = mpl::inter_communicator::parent();

    if(!parent.is_valid()) {
        printf("This program must be invoked by the host application.\n");
        return 0;
    }

    const auto all = mpl::communicator(parent, mpl::communicator::order_high);

    int config[3]{};

    all.bcast(0, config);

    const auto N = config[0];
    const auto NRHS = config[1];
    const auto FLOAT = config[2];

    return FLOAT > 0 ? run<double>(N, NRHS) : run<float>(N, NRHS);
}
