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
 * @brief Standalone `pgbsv` solver.
 *
 * This program is a standalone application that solves a system of linear equations
 * using the `pgbsv` solver.
 *
 * The caller spawns this program as a worker process.
 *
 * The caller must send three buffers to the worker process:
 * - an integer array of size 5 containing the matrix size (`N`), number of sub-diagonals (`KL`),
 *   number of super-diagonals (`KU`), number of right-hand sides (`NRHS`),
 *   and the data type,
 * - a buffer containing the matrix `A`, size `N x (KL + KU + 1)`,
 * - a buffer containing the right-hand side `B`, size `N x NRHS`.
 *
 * The data type has the following meaning:
 * - 2-digit positive: complex16,
 * - 1-digit positive: double,
 * - 1-digit negative: float,
 * - 2-digit negative: complex8.
 *
 * @note The `pgbsv` solver requires the lead dimension of `A` to be `2 * (KL + KU) + 1`.
 * The top `KL + KU` rows do not contain any data. The bottom `KL + KU + 1` rows contain
 * the matrix `A`.
 * @note The caller is expected to send the bottom `KL + KU + 1` rows of `A` only.
 * This is to avoid sending unnecessary data and optimize the communication.
 *
 * The error code (0 for success) will be sent back to the root process of the caller.
 * If error code is 0, the solution will be sent back as well.
 *
 * The example caller logic can be seen as follows.
 *
 * @include runner.cpp
 *
 * @author tlc
 * @date 07/03/2025
 * @version 1.0.0
 * @file solver.pgbsv.cpp
 * @{
 */

#include <ezp/pgbsv.hpp>
#include <mpl/mpl.hpp>

const auto& comm_world{mpl::environment::comm_world()};
const auto& parent = mpl::inter_communicator::parent();

template<ezp::data_t DT> int run(const int N, const int KL, const int KU, const int NRHS) {
    std::vector<DT> A, B;

    if(0 == comm_world.rank()) {
        const auto KLU = KL + KU;
        const auto LDA = 2 * KLU + 1;

        A.resize(N * LDA);
        B.resize(N * NRHS);

        mpl::irequest_pool requests;

        // ! Expect the caller to send bottom KL + KU + 1 rows of A only.
        // ! However, the layout of A is still N x LDA.
        // ! Thus, a strided layout is used to receive the data.
        requests.push(parent.irecv(A.data() + KLU, mpl::strided_vector_layout<DT>(N, LDA - KLU, LDA), 0, mpl::tag_t{0}));
        requests.push(parent.irecv(B, 0, mpl::tag_t{1}));

        requests.waitall();
    }

    const auto error = ezp::pgbsv<DT, int>().solve({N, N, KL, KU, A.data()}, {N, NRHS, B.data()});

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

    if(FLOAT >= 10) return run<complex16>(N, KL, KU, NRHS);
    if(FLOAT >= 0) return run<double>(N, KL, KU, NRHS);
    if(FLOAT > -10) return run<float>(N, KL, KU, NRHS);

    return run<complex8>(N, KL, KU, NRHS);
}

//! @}
