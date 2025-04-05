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
#include <ezp/pgbsv.hpp>
#include <random>
#include <thread>

using namespace ezp;
using namespace std::chrono;

#ifdef EZP_ENABLE_TEST
#include <catch2/catchy.hpp>
#else
#define REQUIRE(...)
#endif

static auto REPEAT = 10;

template<data_t DT> auto random_pgbsv() {
    using solver_t = pgbsv<DT, int_t>;

    const auto& env = get_env<int_t>();

    const auto context = blacs_context<int_t>();

    auto band = std::uniform_int_distribution(1, 10);

    for(auto K = 0; K < REPEAT; ++K) {
        const auto seed = context.amn(static_cast<int_t>(duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count()));
        std::mt19937 gen(seed);

        const auto KL = band(gen);
        const auto KU = band(gen);
        const auto NRHS = band(gen);
        const auto N = std::uniform_int_distribution(std::max(KL, KU) + 1, 100)(gen);

        if(0 == env.rank()) printf("Seed: %d, N: %d, KL: %d, KU: %d, NRHS: %d\n", seed, N, KL, KU, NRHS);

        const auto LDA = 2 * (KL + KU) + 1;

        const auto IDX = typename solver_t::indexer{N, KL, KU};

        std::vector<DT> A, B;

        if(0 == env.rank()) {
            A.resize(N * LDA, DT{0.});
            B.resize(N * NRHS, DT{1.});

            std::uniform_real_distribution dist_v(0.f, 1.f);

            for(auto I = 0; I < N; ++I) A[IDX(I, I)] = 10.f * dist_v(gen) + 10.f;
        }

        const auto info = solver_t().solve({N, N, KL, KU, A.data()}, {N, NRHS, B.data()});

        if(0 == env.rank()) REQUIRE(info == 0);
    }
};

#ifdef EZP_ENABLE_TEST
TEST_CASE("Random PDGBSV", "[Simple Solver]") {
#else
void random_pdgbsv() {
#endif
    random_pgbsv<double>();
}

#ifdef EZP_ENABLE_TEST
TEST_CASE("Random PSGBSV", "[Simple Solver]") {
#else
void random_psgbsv() {
#endif
    random_pgbsv<float>();
}

#ifdef EZP_ENABLE_TEST
TEST_CASE("Random PZGBSV", "[Simple Solver]") {
#else
void random_pzgbsv() {
#endif
    random_pgbsv<complex16>();
}

#ifdef EZP_ENABLE_TEST
TEST_CASE("Random PCGBSV", "[Simple Solver]") {
#else
void random_pcgbsv() {
#endif
    random_pgbsv<complex8>();
}

#ifndef EZP_ENABLE_TEST
int main(const int argc, const char* argv[]) {
    if(argc <= 1) {
        volatile int i = 0;
        while(0 == i) std::this_thread::sleep_for(seconds(10));
    }
    else REPEAT = std::atoi(argv[1]);

    random_pdgbsv();
    random_psgbsv();
    random_pzgbsv();
    random_pcgbsv();

    return 0;
}
#endif
