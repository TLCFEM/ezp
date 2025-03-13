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
 * @brief Standalone `pgesvx` solver.
 *
 * This program is a standalone application that solves a system of linear equations
 * using the `pgesvx` solver.
 * The solver always tries to equilibrate the system.
 *
 * The caller spawns this program as a worker process.
 *
 * The caller must send three buffers to the worker process:
 * - an integer array of size 3 containing the matrix size (`N`),
 *   number of right-hand sides (`NRHS`), and the data type,
 * - a buffer containing the matrix `A`, size `N x N`,
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
 * @include runner.cpp
 *
 * @author tlc
 * @date 12/03/2025
 * @version 1.0.0
 * @file solver.pgesvx.cpp
 * @{
 */

#include "solver.full.hpp"

int main(int, char**) {
    ezp::blacs_env<int>::do_not_manage_mpi();

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

    if(FLOAT >= 10) return run<complex16, int_t, ezp::pgesvx<complex16, int_t>>(N, NRHS);
    if(FLOAT >= 0) return run<double, int_t, ezp::pgesvx<double, int_t>>(N, NRHS);
    if(FLOAT > -10) return run<float, int_t, ezp::pgesvx<float, int_t>>(N, NRHS);

    return run<complex8, int_t, ezp::pgesvx<complex8, int_t>>(N, NRHS);
}

//! @}
