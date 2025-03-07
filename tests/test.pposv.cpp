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
#include <cmath>
#include <ezp/pposv.hpp>
#include <random>
#include <thread>

using namespace ezp;
using namespace std::chrono;

#ifdef EZP_ENABLE_TEST
#include <catch2/catchy.hpp>
TEST_CASE("Random PPOSV", "[Simple Solver]") {
#else
#define REQUIRE(...)

void random_pposv() {
#endif
    const auto& env = get_env<int>();

    const auto rows = std::max(1, static_cast<int>(std::sqrt(env.size())));
    const auto cols = env.size() / rows;

    for(auto K = 0; K < 1000; ++K) {
        const auto seed = blacs_context<int>().amx(static_cast<int>(duration_cast<seconds>(system_clock::now().time_since_epoch()).count()));
        std::mt19937 gen(seed);

        const auto NRHS = std::uniform_int_distribution(1, 20)(gen);
        const auto N = std::uniform_int_distribution(100, 400)(gen);

        const auto IDX = [=](const int r, const int c) { return r + c * N; };

        std::vector<double> A, B;

        if(0 == env.rank()) {
            A.resize(N * N, 0.);
            B.resize(N * NRHS, 1.);

            std::uniform_real_distribution dist_v(0., 1.);

            for(auto I = 0; I < N; ++I) {
                A[IDX(I, I)] = 10. + 10. * dist_v(gen);
                for(auto J = I + 1; J < N; ++J) A[IDX(I, J)] = A[IDX(J, I)] = dist_v(gen);
            }
        }

        const auto info = par_dposv(rows, cols).solve({N, N, A.data()}, {N, NRHS, B.data()});

        if(0 == env.rank()) REQUIRE(info == 0);
    }
}

#ifdef EZP_ENABLE_TEST
#else
int main(const int argc, const char*[]) {
    volatile int i = 0;
    if(argc <= 1)
        while(0 == i) std::this_thread::sleep_for(seconds(10));

    random_pposv();

    return 0;
}
#endif