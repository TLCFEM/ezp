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
 * @brief Example caller to the standalone `ppbsv` solver.
 *
 * @author tlc
 * @date 07/03/2025
 * @version 1.0.0
 * @file runner.ppbsv.cpp
 * @{
 */

#include <mpl/mpl.hpp>

int main(int argc, char* argv[]) {
    constexpr int NUM_NODE = 2;

    const auto& comm_world{mpl::environment::comm_world()};
    const auto worker = comm_world.spawn(0, NUM_NODE, {"solver.ppbsv"});
    const auto all = mpl::communicator(worker, mpl::communicator::order_low);

    constexpr auto N = 6, KLU = 1, NRHS = 1;

    int config[]{N, KLU, NRHS, 1};

    std::vector<double> A(N * (KLU + 1), 0.), B(N * NRHS, 1.);

    for(auto I = 0; I < N; ++I) A[I + I * KLU] = I + 1.;

    all.bcast(0, config);

    mpl::irequest_pool requests;

    requests.push(worker.isend(A, 0, mpl::tag_t{0}));
    requests.push(worker.isend(B, 0, mpl::tag_t{1}));

    requests.waitall();

    int error = -1;
    worker.recv(error, 0);
    if(0 == error) worker.recv(B, 0);

    for(auto I = 0; I < N; ++I) printf("x[%d] = %+.6f\n", I, B[I]);

    return 0;
}

//! @}
