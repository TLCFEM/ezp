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
 * @brief Example caller to the `pgbsv` solver.
 *
 * @author tlc
 * @date 07/03/2025
 * @version 1.0.0
 * @file example.pgbsv.cpp
 * @{
 */

#include <ezp/pgbsv.hpp>
#include <iomanip>
#include <iostream>

using namespace ezp;

int main() {
    // get the current blacs environment
    const auto& env = get_env<int_t>();

    constexpr auto N = 10, NRHS = 1, KL = 2, KU = 2;
    constexpr auto LDA = 2 * (KL + KU) + 1;

    // storage for the matrices A and B
    std::vector<double> A, B;

    // helper function to convert 2D indices to 1D indices
    // the band symmetric matrix used for gbsv subroutine requires the matrix to be stored with a leading dimension of (2 * (KL + KU) + 1)
    // see Fig. 4.10 https://netlib.org/scalapack/slug/node84.html
    const auto IDX = par_dgbsv<int_t>::indexer{N, KL, KU};

    if(0 == env.rank()) {
        // the matrices are only initialized on the root process
        A.resize(N * LDA, 0.);
        B.resize(N * NRHS, 1.);

        for(auto I = 0; I < N; ++I) A[IDX(I, I)] = I + 1;
    }

    // create a parallel solver
    // it uses a one-dimensional process grid
    // it takes the number of processes as arguments
    auto solver = par_dgbsv(env.size());

    // need to wrap the data in full_mat objects
    // it requires the number of rows and columns of the matrix, and a pointer to the data
    // on non-root processes, the data pointer is nullptr as the vector is empty
    // solver.solve(band_mat{N, N, KL, KU, A.data()}, full_mat{N, NRHS, B.data()});
    const auto info = solver.solve({N, N, KL, KU, A.data()}, {N, NRHS, B.data()});

    if(0 == env.rank() && 0 == info) {
        std::cout << std::setprecision(6) << std::fixed << "Info: " << info << '\n';
        std::cout << "Solution:\n";
        for(auto i = 0u; i < B.size(); ++i) std::cout << B[i] << '\n';
    }

    return info;
}

//! @}
