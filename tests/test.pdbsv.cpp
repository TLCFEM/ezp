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
#include <catch2/catchy.hpp>
#else
#define REQUIRE(...)
#endif

#ifdef EZP_ENABLE_TEST
TEST_CASE("Random PDDBSV", "[Simple Solver]") {
#else
void random_pddbsv() {
#endif
    const auto& env = get_env<int>();

    const auto context = blacs_context<int>();

    for(auto K = 0; K < 100; ++K) {
        const auto seed = context.amx(static_cast<int>(duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count()));
        std::mt19937 gen(seed);

        auto band = std::uniform_int_distribution(1, 20);
        const auto KL = band(gen);
        const auto KU = band(gen);
        const auto NRHS = band(gen);
        const auto N = std::uniform_int_distribution(std::max(KL, KU) + 1, 400)(gen);

        const auto LDA = KL + KU + 1;

        const auto IDX = par_ddbsv<int>::indexer{N, KL, KU};

        std::vector<double> A, B;

        if(0 == env.rank()) {
            A.resize(N * LDA, 0.);
            B.resize(N * NRHS, 1.);

            std::uniform_real_distribution dist_v(0., 1.);

            for(auto I = 0; I < N; ++I) A[IDX(I, I)] = 10. * dist_v(gen) + 10.;

            std::uniform_int_distribution dist_idx(0, N - 1);

            for(auto I = 0; I < N * N; ++I)
                if(const auto position = IDX(dist_idx(gen), dist_idx(gen)); position >= 0) A[position] += dist_v(gen);
        }

        const auto info = par_ddbsv(env.size()).solve({N, N, KL, KU, A.data()}, {N, NRHS, B.data()});

        if(0 == env.rank()) REQUIRE(info == 0);
    }
}

#ifdef EZP_ENABLE_TEST
TEST_CASE("Random PSDBSV", "[Simple Solver]") {
#else
void random_psdbsv() {
#endif
    const auto& env = get_env<int>();

    const auto context = blacs_context<int>();

    for(auto K = 0; K < 100; ++K) {
        const auto seed = context.amx(static_cast<int>(duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count()));
        std::mt19937 gen(seed);

        auto band = std::uniform_int_distribution(1, 20);
        const auto KL = band(gen);
        const auto KU = band(gen);
        const auto NRHS = band(gen);
        const auto N = std::uniform_int_distribution(std::max(KL, KU) + 1, 400)(gen);

        const auto LDA = KL + KU + 1;

        const auto IDX = par_sdbsv<int>::indexer{N, KL, KU};

        std::vector<float> A, B;

        if(0 == env.rank()) {
            A.resize(N * LDA, 0.f);
            B.resize(N * NRHS, 1.f);

            std::uniform_real_distribution dist_v(0.f, 1.f);

            for(auto I = 0; I < N; ++I) A[IDX(I, I)] = 10.f * dist_v(gen) + 10.f;

            std::uniform_int_distribution dist_idx(0, N - 1);

            for(auto I = 0; I < N * N; ++I)
                if(const auto position = IDX(dist_idx(gen), dist_idx(gen)); position >= 0) A[position] += dist_v(gen);
        }

        const auto info = par_sdbsv(env.size()).solve({N, N, KL, KU, A.data()}, {N, NRHS, B.data()});

        if(0 == env.rank()) REQUIRE(info == 0);
    }
}

#ifdef EZP_ENABLE_TEST
#else
int main(const int argc, const char*[]) {
    volatile int i = 0;
    if(argc <= 1)
        while(0 == i) std::this_thread::sleep_for(seconds(10));

    random_pddbsv();
    random_psdbsv();

    return 0;
}
#endif