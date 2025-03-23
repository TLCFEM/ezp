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
 * @brief Example caller to the `mumps` solver.
 *
 * @author tlc
 * @date 23/03/2025
 * @version 1.0.0
 * @file example.mumps.cpp
 * @{
 */

#include <ezp/mumps.hpp>
#include <iomanip>
#include <iostream>
#include <mpl/mpl.hpp>

using namespace ezp;

int main() {
    const auto& comm_world{mpl::environment::comm_world()};

    constexpr int N = 10, NRHS = 1;

    std::vector<int_t> ia, ja;
    std::vector<double> a, b;

    if(0 == comm_world.rank()) {
        ia.resize(N + 1);
        ja.resize(N);
        a.resize(N);
        b.resize(N * NRHS);

        for(auto i = 0; i < N; i++) ia[i] = ja[i] = a[i] = i + 1;
        ia[N] = N + 1;

        std::fill(b.begin(), b.end(), 1.);
    }

    auto solver = mumps<double, int_t>();

    // need to wrap the data in sparse_csr_mat objects
    const auto info = solver.solve({N, N + 1, ia.data(), ja.data(), a.data()}, {N, NRHS, b.data()});

    if(0 == comm_world.rank()) {
        std::cout << std::setprecision(10) << "Info: " << info << '\n';
        std::cout << "Solution:\n";
        for(const double i : b) std::cout << i << '\n';
    }

    return info;
}

//! @}
