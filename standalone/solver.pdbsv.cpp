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
 * @brief Standalone `pdbsv` solver.
 *
 * This program is a standalone application that solves a system of linear equations
 * using the `pdbsv` solver.
 *
 * The caller spawns this program as a worker process.
 *
 * The caller must send three buffers to the worker process:
 * - an integer array of size 5 containing the matrix size (`N`), number of sub-diagonals (`KL`),
 *   number of super-diagonals (`KU`), number of right-hand sides (`NRHS`),
 *   and the data type (> 0 for `double`, < 0 for `float`),
 * - a buffer containing the matrix `A`, size `N x (KL + KU + 1)`,
 * - a buffer containing the right-hand side `B`, size `N x NRHS`.
 *
 * The error code (0 for success) will be sent back to the root process of the caller.
 * If error code is 0, the solution will be sent back as well.
 *
 * The example caller logic can be seen as follows.
 *
 * @include standalone/runner.pdbsv.cpp
 *
 * @author tlc
 * @date 07/03/2025
 * @version 1.0.0
 * @file solver.pdbsv.cpp
 * @{
 */

#include <ezp/pdbsv.hpp>
#include <mpl/mpl.hpp>

const auto& comm_world{mpl::environment::comm_world()};
const auto& parent = mpl::inter_communicator::parent();

template<ezp::data_t DT> int run(const int N, const int KL, const int KU, const int NRHS) {
    std::vector<DT> A, B;

    if(0 == comm_world.rank()) {
        A.resize(N * (KL + KU + 1));
        B.resize(N * NRHS);

        mpl::irequest_pool requests;

        requests.push(parent.irecv(A, 0, mpl::tag_t{0}));
        requests.push(parent.irecv(B, 0, mpl::tag_t{1}));

        requests.waitall();
    }

    const auto error = ezp::pdbsv<DT, int>().solve({N, N, KL, KU, A.data()}, {N, NRHS, B.data()});

    if(0 == comm_world.rank()) {
        parent.send(error, 0);
        if(0 == error) parent.send(B, 0);
    }

    return 0;
}

int main(int argc, char** argv) {
    ezp::blacs_env<int>::do_not_manage_mpi();

    if(!parent.is_valid()) {
        printf("This program must be invoked by the host application.\n");
        return 0;
    }

    const auto all = mpl::communicator(parent, mpl::communicator::order_high);

    int config[5]{};

    all.bcast(0, config);

    const auto N = config[0];
    const auto KL = config[1];
    const auto KU = config[2];
    const auto NRHS = config[3];
    const auto FLOAT = config[4];

    return FLOAT > 0 ? run<double>(N, KL, KU, NRHS) : run<float>(N, KL, KU, NRHS);
}

//! @}
