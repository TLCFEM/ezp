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
 * @brief Example caller to the `pgesv` solver.
 *
 * @author tlc
 * @date 07/03/2025
 * @version 1.0.0
 * @file example.pgesv.cpp
 * @{
 */

#include <ezp/pgesv.hpp>
#include <iomanip>
#include <iostream>

using namespace ezp;

int main() {
    // get the current blacs environment
    const auto& env = get_env<int_t>();

    constexpr auto N = 6, NRHS = 2;

    // storage for the matrices A and B
    std::vector<double> A, B;

    // helper function to convert 2D indices to 1D indices
    const auto IDX = par_dgesv<int_t>::indexer{N};

    if(0 == env.rank()) {
        // the matrices are only initialized on the root process
        A.resize(N * N, 0.);
        B.resize(N * NRHS);

        static constexpr auto M = 5.10156648;

        for(auto I = 0; I < N; ++I) {
            B[I] = A[IDX(I, I)] = I + 1;
            B[I + N] = (I + 1) * M;
        }
    }

    // create a parallel solver
    // it takes the number of rows and columns of the process grid as arguments
    auto solver = par_dgesv<int_t>();

    // need to wrap the data in full_mat objects
    // it requires the number of rows and columns of the matrix, and a pointer to the data
    // on non-root processes, the data pointer is nullptr as the vector is empty
    // solver.solve(full_mat{N, N, A.data()}, full_mat{N, NRHS, B.data()});
    const auto info = solver.solve({N, N, A.data()}, {N, NRHS, B.data()});

    if(0 == env.rank() && 0 == info) {
        std::cout << std::setprecision(10) << "Info: " << info << '\n';
        std::cout << "Solution:\n";
        for(const double i : B) std::cout << i << '\n';
    }

    return info;
}

//! @}
