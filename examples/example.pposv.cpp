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
 * @brief Example caller to the `pposv` solver.
 *
 * @author tlc
 * @date 11/03/2025
 * @version 1.0.0
 * @file example.pposv.cpp
 * @{
 */

#include <ezp/pposv.hpp>
#include <iomanip>
#include <iostream>

using namespace ezp;

class mat {
public:
    int_t n_rows, n_cols;

private:
    std::vector<double> storage;

public:
    mat(const int_t rows, const int cols)
        : n_rows(rows)
        , n_cols(cols) {}

    auto init() { storage.resize(n_rows * n_cols); }

    auto& operator()(const int_t i, const int_t j) { return storage[i + j * n_cols]; }

    auto& operator[](const int_t i) { return storage[i]; }

    auto begin() { return storage.begin(); }

    auto end() { return storage.end(); }
};

int main() {
    // get the current blacs environment
    const auto& env = get_env<>();

    constexpr auto N = 6, NRHS = 2;

    // storage for the matrices A and B using the custom class
    // acceptable objects shall have members `.n_rows` and `.n_cols`
    // and have any of the following methods `.mem()`, `.memptr()`, `.data()`
    // or contiguous iterators
    // all above methods shall return a pointer to the first element
    mat A(N, N), B(N, NRHS);

    if(0 == env.rank()) {
        // the matrices are only initialized on the root process
        A.init();
        B.init();

        for(auto I = 0; I < N; ++I) {
            B[I] = A(I, I) = I + 1;
            B[I + N] = I + 4;
        }
    }

    // create a parallel solver
    // and send custom matrix objects to the solver
    const auto info = par_dposv<int_t>().solve(A, B);

    if(0 == env.rank() && 0 == info) {
        std::cout << std::setprecision(10) << "Info: " << info << '\n';
        std::cout << "Solution:\n";
        for(const auto i : B) std::cout << i << '\n';
    }

    return info;
}

//! @}
