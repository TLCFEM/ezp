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
 * @brief Standalone `lis` solver.
 *
 * This program is a standalone application that solves a system of linear equations
 * using the `lis` solver.
 *
 * The caller spawns this program as a worker process.
 *
 * The matrix `A` is stored in the compressed sparse row (CSR) format.
 * The matrix `A` uses the zero-based indexing.
 * The caller must send six buffers to the worker process:
 * - an integer array of length 4, containing:
 *   - the length of the option string,
 *   - the number of rows of matrix `A`,
 *   - the number of non-zero elements of matrix `A`,
 *   - the number of right-hand sides,
 * - an option string,
 * - a buffer containing the row index of matrix `A`, size `NNZ`,
 * - a buffer containing the column index of matrix `A`, size `NNZ`,
 * - a buffer containing the value of matrix `A`, size `NNZ`,
 * - a buffer containing the right-hand side `B`, size `N x NRHS`.
 *
 * The error code (0 for success) will be sent back to the root process of the caller.
 * If error code is 0, the solution will be sent back as well.
 *
 * The example caller logic can be seen as follows.
 *
 * @include runner.lis.cpp
 *
 * @author tlc
 * @date 26/03/2025
 * @version 1.0.0
 * @file solver.lis.cpp
 * @{
 */

#include <ezp/lis.hpp>
#include <mpl/mpl.hpp>

int main(int, char**) {
    const auto& comm_world{mpl::environment::comm_world()};
    const auto& parent = mpl::inter_communicator::parent();

    if(!parent.is_valid()) {
        printf("This program must be invoked by the host application.\n");
        return 0;
    }

    const auto all = mpl::communicator(parent, mpl::communicator::order_high);

    LIS_INT config[4]{};
    all.bcast(0, config);

    std::string opt(config[0], 'x');
    all.bcast(0, opt.data(), mpl::contiguous_layout<char>(config[0]));

    const auto n = config[1];
    const auto nnz = config[2];
    const auto nrhs = config[3];

    auto solver = ezp::lis(opt.c_str());

    std::vector<LIS_INT> ia, ja;
    std::vector<LIS_SCALAR> a, b;

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

    const auto error = solver.solve({n, nnz, ia.data(), ja.data(), a.data()}, {n, nrhs, b.data()});

    if(0 == comm_world.rank()) {
        parent.send(error, 0);
        if(0 == error) parent.send(b, 0);
    }

    return 0;
}

//! @}
