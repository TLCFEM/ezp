/*******************************************************************************
 * Copyright (C) 2025-2026 Theodore Chang
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
 * @brief Standalone `pardiso` solver.
 *
 * This program is a standalone application that solves a system of linear
 * equations using the `pardiso` solver.
 *
 * The caller spawns this program as a worker process.
 *
 * The matrix `A` is stored in the compressed sparse row (CSR) format.
 * The caller must send five buffers to the worker process:
 *
 * - an integer array of size 5
 * - a buffer containing the row index of matrix `A`, size `N + 1`,
 * - a buffer containing the column index of matrix `A`, size `NNZ`,
 * - a buffer containing the value of matrix `A`, size `NNZ`,
 * - a buffer containing the right-hand side `B`, size `N x NRHS`.
 *
 * The error code (0 for success) will be sent back to the root process of the caller.
 * If error code is 0, the solution will be sent back as well.
 *
 * The example caller logic can be seen as follows.
 *
 * @include runner.pardiso.cpp
 *
 * @author tlc
 * @date 26/03/2025
 * @version 1.0.0
 * @file solver.pardiso.cpp
 * @{
 */

#ifdef EZP_MKL
#include <ezp/pardiso.parser.hpp>
#include <mpl/mpl.hpp>

const auto& comm_world{mpl::environment::comm_world()};
const auto& parent = mpl::inter_communicator::parent();

template<typename DT, typename IT> int run(const IT (&config)[5], IT (&iparm)[64], const std::string_view command) {
    const auto mtype = config[0];
    const auto msglvl = config[1];
    const auto n = config[2];
    const auto nnz = config[3];
    const auto nrhs = config[4];

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
    }

    ezp::pardiso<DT, IT> solver(mtype, msglvl);
    for(auto i = 0; i < 64; i++) solver(i) = iparm[i];
    pardiso_set(std::string(command), solver);

    const auto error = solver.solve({n, nnz, ia.data(), ja.data(), a.data()}, {n, nrhs, b.data()});

    if(0 == comm_world.rank()) {
        parent.send(error, 0);
        if(0 == error) parent.send(b, 0);
    }

    return 0;
}

template<typename IT> auto prepare(const std::string_view command) {
    const auto all = mpl::communicator(parent, mpl::communicator::order_high);

    IT config[5]{};
    IT iparm[64]{};

    all.bcast(0, config);
    all.bcast(0, iparm);

    if(config[0] == 1 || config[0] == 1 || config[0] == 2 || config[0] == -2 || config[0] == 11) {
        if(0 == iparm[27]) return run<double>(config, iparm, command);
        return run<float>(config, iparm, command);
    }

    if(0 == iparm[27]) return run<complex16>(config, iparm, command);
    return run<complex8>(config, iparm, command);
}

int main(int argc, char** argv) {
    if(!parent.is_valid()) {
        printf("This program must be invoked by the host application.\n");
        return 0;
    }

    std::string command{};
    for(int i = 1; i < argc; i++) {
        command += ' ';
        command += argv[i];
    }

    if(argc > 1) return prepare<std::int64_t>(command);

    return prepare<std::int32_t>(command);
}
#else
#include <iostream>

int main(int, char**) {
    std::cout << "This program must be compiled with the EZP_MKL macro defined.\n";
    return 0;
}
#endif

//! @}
