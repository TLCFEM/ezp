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
 * @brief Standalone `mumps` solver.
 *
 * This program is a standalone application that solves a system of linear equations
 * using the `mumps` solver.
 *
 * The caller spawns this program as a worker process.
 *
 * The matrix `A` is stored in the coordinate (COO) format.
 * The caller must send five buffers to the worker process:
 * - an integer array of size 5
 *   - the symmetry type of matrix `A`,
 *   - the number of right-hand sides,
 *   - the number of rows of matrix `A`,
 *   - the number of non-zero elements of matrix `A`,
 *   - the data type of matrix `A` (see below),
 * - a buffer containing the row index of matrix `A`, size `NNZ`,
 * - a buffer containing the column index of matrix `A`, size `NNZ`,
 * - a buffer containing the value of matrix `A`, size `NNZ`,
 * - a buffer containing the right-hand side `B`, size `N x NRHS`.
 *
 * The data type has the following meaning:
 * - 2-digit positive: complex16,
 * - 1-digit positive: double,
 * - 1-digit negative: float,
 * - 2-digit negative: complex8.
 *
 * The error code (0 for success) will be sent back to the root process of the caller.
 * If error code is 0, the solution will be sent back as well.
 *
 * The example caller logic can be seen as follows.
 *
 * @include runner.mumps.cpp
 *
 * @author tlc
 * @date 23/03/2025
 * @version 1.0.0
 * @file solver.mumps.cpp
 * @{
 */

#include <ezp/mumps.parser.hpp>
#include <mpl/mpl.hpp>

const auto& comm_world{mpl::environment::comm_world()};
const auto& parent = mpl::inter_communicator::parent();

static constexpr auto config_size = 5;

template<typename DT, typename IT> int run(const IT (&config)[config_size], const std::string_view command) {
    const auto sym = config[0];
    const auto nrhs = config[1];
    const auto n = config[2];
    const auto nnz = config[3];

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnarrowing"
    auto solver = ezp::mumps<DT, IT>(ezp::symmetric_pattern{sym});
#pragma GCC diagnostic pop

    mumps_set(std::string(command), solver);

    std::vector<IT> ia, ja;
    std::vector<DT> a, b;

    if(0 == comm_world.rank()) {
        ia.resize(nnz);
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

    const auto error = solver.solve({n, nnz, ia.data(), ja.data(), a.data()}, {n, nrhs, b.data()});

    if(0 == comm_world.rank()) {
        parent.send(error, 0);
        if(0 == error) parent.send(b, 0);
    }

    return 0;
}

template<typename IT> auto prepare(const std::string_view command) {
    const auto all = mpl::communicator(parent, mpl::communicator::order_high);

    IT config[config_size]{};

    all.bcast(0, config);

    const auto FLOAT = config[config_size - 1];

    if(FLOAT >= 10) return run<complex16>(config, command);
    if(FLOAT >= 0) return run<double>(config, command);
    if(FLOAT > -10) return run<float>(config, command);

    return run<complex8>(config, command);
}

int main(const int argc, char** argv) {
    if(!parent.is_valid()) {
        printf("This program must be invoked by the host application.\n");
        return 0;
    }

    std::string command{};
    for(int i = 1; i < argc; i++) {
        command += ' ';
        command += argv[i];
    }

    return prepare<int>(command);
}

//! @}
