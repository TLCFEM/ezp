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
 * @brief Example caller to the standalone solvers.
 *
 * @author tlc
 * @date 07/03/2025
 * @version 1.0.0
 * @file runner.cpp
 * @{
 */

#include <iostream>
#include <mpl/mpl.hpp>

int main(int argc, char** argv) {
    if(argc < 2) {
        std::cout << "Usage: runner ge|po|gb|db|pb [n]\n";
        std::cout << "Example: runner ge 3\n";
        return 0;
    }
    constexpr auto N = 6, NRHS = 1;

    std::vector<double> A, B(N * NRHS, 1.);
    std::vector<int> config;

    const auto type = std::string(argv[1]);
    std::string solver;

    if("ge" == type) {
        solver = "solver.pgesv";

        config = {N, NRHS, 1};

        A.resize(N * N, 0.);
        for(auto I = 0; I < N; ++I) A[I * N + I] = I + 1.;
    }
    else if("po" == type) {
        solver = "solver.pposv";

        config = {N, NRHS, 1};

        A.resize(N * N, 0.);
        for(auto I = 0; I < N; ++I) A[I * N + I] = I + 1.;
    }
    else if("gb" == type) {
        solver = "solver.pgbsv";

        constexpr auto KL = 1, KU = 1;

        config = {N, KL, KU, NRHS, 1};

        A.resize(N * (KL + KU + 1), 0.);
        for(auto I = 0; I < N; ++I) A[KU + I * (KL + KU + 1)] = I + 1.;
    }
    else if("db" == type) {
        solver = "solver.pdbsv";

        constexpr auto KL = 1, KU = 1;

        config = {N, KL, KU, NRHS, 1};

        A.resize(N * (KL + KU + 1), 0.);
        for(auto I = 0; I < N; ++I) A[KU + I * (KL + KU + 1)] = I + 1.;
    }
    else if("pb" == type) {
        solver = "solver.ppbsv";

        constexpr auto KLU = 1;

        config = {N, KLU, NRHS, 1};

        A.resize(N * (KLU + 1), 0.);
        for(auto I = 0; I < N; ++I) A[I + I * KLU] = I + 1.;
    }
    else {
        std::cout << "Usage: runner ge|po|gb|db|pb [n]\n";
        return 0;
    }

    const auto& comm_world{mpl::environment::comm_world()};
    const auto worker = comm_world.spawn(0, argc < 3 ? 1 : std::abs(std::stoi(argv[2])), {solver});
    const auto all = mpl::communicator(worker, mpl::communicator::order_low);

    all.bcast(0, config.data(), mpl::contiguous_layout<int>(config.size()));

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
