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
 * @brief Example caller to the `ppbsv` solver.
 *
 * @author tlc
 * @date 11/03/2025
 * @version 1.0.0
 * @file example.ppbsv.cpp
 * @{
 */

#include <ezp/ppbsv.hpp>
#include <iomanip>
#include <iostream>

// get the current blacs environment
const auto& env = ezp::get_env<>();

class general_mat {
public:
    int_t n_rows{}, n_cols{};

private:
    std::vector<double> storage;

public:
    auto init(const int_t rows, const int cols) {
        n_rows = rows;
        n_cols = cols;
        if(0 == env.rank()) storage.resize(rows * cols);
    }

    explicit general_mat(const int_t rows = 0, const int cols = 0) { init(rows, cols); }

    auto& operator[](const int_t i) { return storage[i]; }

    auto begin() { return storage.begin(); }

    auto end() { return storage.end(); }
};

class bandsymm_mat {
public:
    int_t n_rows{}, n_cols{}, klu{};

private:
    std::vector<double> storage;

    ezp::ppbsv<double, int_t>::indexer indexer{0, 0};

public:
    auto init(const int_t n, const int bandwidth) {
        n_rows = n;
        n_cols = n;
        klu = bandwidth;
        if(0 == env.rank()) storage.resize(n * (klu + 1));
        indexer = {n, klu};
    }

    explicit bandsymm_mat(const int_t n = 0, const int bandwidth = 0) { init(n, bandwidth); }

    auto& operator()(const int_t i, const int_t j) { return storage[indexer(i, j)]; }

    auto data() { return storage.data(); }
};

int main() {
    // storage for the matrices A and B
    bandsymm_mat A;
    general_mat B;

    constexpr auto N = 6, NRHS = 2, KLU = 1;
    // the matrices are only initialized on the root process
    A.init(N, KLU);
    B.init(N, NRHS);

    if(0 == env.rank()) {
        static constexpr auto M = 5.10156648;

        for(auto I = 0; I < N; ++I) {
            B[I] = A(I, I) = I + 1;
            B[N + I] = (I + 1) * M;
        }
    }

    // create a parallel solver
    // it uses a one-dimensional process grid
    // it takes the number of processes as arguments
    auto solver = ezp::par_dpbsv(env.size());

    // use custom matrix objects
    const auto info = solver.solve(A, B);

    if(0 == env.rank() && 0 == info) {
        std::cout << std::setprecision(10) << "Info: " << info << '\n';
        std::cout << "Solution:\n";
        for(const double i : B) std::cout << i << '\n';
    }

    return info;
}

//! @}
