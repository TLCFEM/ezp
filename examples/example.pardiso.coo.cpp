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
 * @brief Example caller to the `pardiso` solver.
 *
 * The `pardiso` solver expects input matrices in the compressed sparse row (CSR) format.
 * This example demonstrates how to convert a matrix in the coordinate list (COO) format to the CSR format using the `sparse_csr_mat` constructor, and then solve a linear system using the `pardiso` solver.
 *
 * @author tlc
 * @date 07/5/2026
 * @version 1.0.0
 * @file example.pardiso.coo.cpp
 * @{
 */

#include <ezp/pardiso.hpp>
#include <iomanip>
#include <iostream>

int main() {
#ifdef EZP_MKL
    const auto& comm_world{mpl::environment::comm_world()};

    auto solver = ezp::pardiso<double, int_t>(ezp::matrix_type::real_and_nonsymmetric, ezp::message_level::no_output);

    int_t N = 10, NRHS = 1;

    std::vector<int_t> ia, ja;
    std::vector<double> a, b;

    const auto populate = [&] {
        if(0 != comm_world.rank()) return;

        // initialise one-based COO matrix on the root process
        ia.resize(N);
        ja.resize(N);
        a.resize(N);
        b.resize(N * NRHS);

        for(auto i = 0; i < N; i++) ia[i] = ja[i] = static_cast<int_t>(a[i] = i + 1);

        std::ranges::fill(b, 1.);
    };

    populate();

    const ezp::sparse_coo_mat coo_system{N, N, ia.data(), ja.data(), a.data()};

    // need to wrap the data in sparse_csr_mat objects
    auto info = solver.solve(ezp::sparse_csr_mat<double, int_t>{coo_system, true}, {N, NRHS, b.data()});

    const auto print = [&] {
        if(0 != comm_world.rank()) return;

        std::cout << std::fixed << std::setprecision(10) << "Info: " << info << '\n';
        std::cout << "Solution:\n";
        for(const auto i : b) std::cout << i << '\n';
    };

    print();

    N = 20;

    populate();
    info = solver.solve(ezp::sparse_csr_mat<double, int_t>{ezp::sparse_coo_mat{N, N, ia.data(), ja.data(), a.data()}, true}, {N, NRHS, b.data()});
    print();

    return info;
#else
    std::cerr << "MKL not enabled.\n";
    return 0;
#endif
}

//! @}
