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

#include <chrono>
#include <ezp/pdbsv.hpp>
#include <random>
#include <thread>

using namespace ezp;
using namespace std::chrono;

#ifdef EZP_ENABLE_TEST
#include <catch2/catch_amalgamated.hpp>
TEST_CASE("Random PDBSV", "[Simple Solver]") {
#else
#define REQUIRE(...)

void random_pdbsv() {
#endif
    const auto& env = get_env<int>();

    std::mt19937 gen(duration_cast<days>(system_clock::now().time_since_epoch()).count() - 1);

    const auto KL = std::uniform_int_distribution(0, 20)(gen);
    const auto KU = std::uniform_int_distribution(0 == KL ? 1 : 0, 20)(gen);
    const auto NRHS = std::uniform_int_distribution(1, 20)(gen);
    const auto N = std::uniform_int_distribution(std::max(KL, KU) + 2, 100)(gen);

    const auto LDA = KL + KU + 1;
    const auto OFFSET = KU;

    const auto IDX = [&](const int r, const int c) {
        if(r - c > KL || c - r > KU) return -1;
        return OFFSET + r - c + c * LDA;
    };

    std::vector<double> A, B;

    if(0 == env.rank()) {
        A.resize(N * LDA, 0.);
        B.resize(N * NRHS, 1.);

        std::uniform_real_distribution dist_v(0., 1.);

        for(auto I = 0; I < N; ++I) A[IDX(I, I)] = 10. * dist_v(gen) + 10.;

        // std::uniform_int_distribution dist_idx(0, N - 1);

        // for(auto I = 0; I < N * N; ++I) {
        //     const auto position = IDX(dist_idx(gen), dist_idx(gen));
        //     if(position >= 0) A[position] += 10. * dist_v(gen);
        // }
    }

    // create a parallel solver
    // it uses a one-dimensional process grid
    // it takes the number of processes as arguments
    auto solver = par_ddbsv(env.size());

    // need to wrap the data in full_mat objects
    // it requires the number of rows and columns of the matrix, and a pointer to the data
    // on non-root processes, the data pointer is nullptr as the vector is empty
    // solver.solve(band_mat{N, N, KL, KU, A.data()}, full_mat{N, NRHS, B.data()});
    const auto info = solver.solve({N, N, KL, KU, A.data()}, {N, NRHS, B.data()});

    if(0 == env.rank()) REQUIRE(info == 0);
}

#ifdef EZP_ENABLE_TEST
#else
int main(const int argc, const char*[]) {
    volatile int i = 0;
    if(argc <= 1)
        while(0 == i) std::this_thread::sleep_for(seconds(10));

    random_pdbsv();

    return 0;
}
#endif