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
 * @brief Example caller to the `mumps` solver with a distributed matrix.
 *
 * Each process stores the triplets of a contiguous block of rows of the matrix.
 * The pattern and the values are provided in local triplets, which corresponds to `ICNTL(18)=3`.
 * The right-hand side and the solution are stored on the host process.
 *
 * Run it with, for example:
 *
 * @code
 * mpirun -n 2 ./example.mumps.distributed
 * @endcode
 *
 * @author tlc
 * @date 21/11/2026
 * @version 1.0.0
 * @file example.mumps.distributed.cpp
 * @{
 */

#include <ezp/mumps.hpp>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

int main() {
    const auto& comm_world{mpl::environment::comm_world()};

    constexpr int N = 12, NRHS = 2;

    const auto rank = static_cast<int>(comm_world.rank());
    const auto size = static_cast<int>(comm_world.size());

    if(size > N) {
        if(0 == rank) std::cout << "Use at most " << N << " processes for this example.\n";
        return 0;
    }

    const auto block = N / size;
    const auto remainder = N % size;

    const auto local_n = block + (rank < remainder ? 1 : 0);
    const auto begin = rank * block + std::min(rank, remainder);

    // Each process owns the rows [begin, begin + local_n) of the global matrix.
    // The global matrix is tridiagonal: -1 on the off-diagonals and 4 on the diagonal.
    std::vector<int> irn, jcn;
    std::vector<double> a;

    const auto add = [&](const int row, const int col, const double value) {
        irn.push_back(row + 1); // one-based row index
        jcn.push_back(col + 1); // one-based column index
        a.push_back(value);
    };

    for(auto i = 0; i < local_n; ++i) {
        const auto row = begin + i;

        if(row > 0) add(row, row - 1, -1.);
        add(row, row, 4.);
        if(row + 1 < N) add(row, row + 1, -1.);
    }

    // The exact solution is a vector of ones.
    const auto rhs = [](const int row) {
        auto value = 4.;
        if(row > 0) value -= 1.;
        if(row + 1 < N) value -= 1.;
        return value;
    };

    // Only the host process stores the right-hand side and receives the solution.
    std::vector<double> b;
    if(0 == rank) {
        b.resize(N * NRHS);
        for(auto K = 0; K < NRHS; ++K)
            for(auto I = 0; I < N; ++I) b[I + K * N] = rhs(I);
    }

    // The distributed category uses ICNTL(18)=3 and defaults to a working host.
    auto solver = ezp::mumps<double, int, ezp::distribution_category::distributed_matrix>();
    solver.icntl_printing_level(0);

    // The number of entries is local to this process; row and column indices are global and one-based.
    const auto info = solver.solve({N, static_cast<int>(irn.size()), irn.data(), jcn.data(), a.data()}, {N, NRHS, b.data()});

    if(0 == rank) {
        auto error = 0.;
        for(const auto value : b) error = std::max(error, std::abs(value - 1.));

        std::cout << "info = " << info << ", max error = " << std::scientific << error << '\n';
    }

    return info;
}

//! @}
