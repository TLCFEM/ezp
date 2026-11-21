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
 * @brief Example caller to the `pardiso` solver with a distributed matrix.
 *
 * Each process stores a contiguous block of rows of the matrix. The right-hand
 * side and the solution are stored on the master process; the solver scatters
 * the right-hand side internally, which is required by MKL for distributed
 * matrix input (`iparm[39] = 1`).
 *
 * Run it with, for example:
 *
 * @code
 * mpirun -n 2 ./example.pardiso.distributed
 * @endcode
 *
 * @author tlc
 * @date 21/11/2026
 * @version 1.0.0
 * @file example.pardiso.distributed.cpp
 * @{
 */

#include <ezp/pardiso.hpp>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

int main() {
#ifdef EZP_MKL
    const auto& comm_world{mpl::environment::comm_world()};

    constexpr int_t N = 12, NRHS = 2;

    const auto rank = static_cast<int_t>(comm_world.rank());
    const auto size = static_cast<int_t>(comm_world.size());

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
    std::vector<int_t> ia(local_n + 1);
    std::vector<int_t> ja;
    std::vector<double> a;

    ja.reserve(3 * local_n);
    a.reserve(3 * local_n);

    for(int_t i = 0; i < local_n; ++i) {
        const auto row = begin + i;

        ia[i] = static_cast<int_t>(ja.size()) + 1;

        if(row > 0) {
            ja.push_back(row); // one-based column index of row - 1
            a.push_back(-1.);
        }

        ja.push_back(row + 1);
        a.push_back(4.);

        if(row + 1 < N) {
            ja.push_back(row + 2);
            a.push_back(-1.);
        }
    }
    ia[local_n] = static_cast<int_t>(ja.size()) + 1;

    // The exact solution is a vector of ones.
    const auto rhs = [](const int_t row) {
        auto value = 4.;
        if(row > 0) value -= 1.;
        if(row + 1 < N) value -= 1.;
        return value;
    };

    // Only the master process stores the right-hand side and receives the solution.
    std::vector<double> b;
    if(0 == rank) {
        b.resize(N * NRHS);
        for(int_t K = 0; K < NRHS; ++K)
            for(int_t I = 0; I < N; ++I) b[I + K * N] = rhs(I);
    }

    auto solver = ezp::pardiso<double, int_t, ezp::distribution_category::distributed_matrix>(ezp::matrix_type::real_and_nonsymmetric, ezp::message_level::no_output);

    // The local blocks are supplied in rank order. Alternatively, call
    // solver.iparm_input_domain(begin + 1, begin + local_n) to set the bounds explicitly.
    const auto info = solver.solve({N, local_n, static_cast<int_t>(ja.size()), ia.data(), ja.data(), a.data()}, {N, NRHS, b.data()});

    if(0 == rank) {
        auto error = 0.;
        for(const auto value : b) error = std::max(error, std::abs(value - 1.));

        std::cout << "info = " << info << ", max error = " << std::scientific << error << '\n';
    }

    return 0;
#else
    std::cerr << "MKL not enabled.\n";
    return 0;
#endif
}

//! @}
