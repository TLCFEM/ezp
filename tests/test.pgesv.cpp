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
#include <ezp/pgesv.hpp>
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

template<data_t DT, char ODER = 'R'> auto random_pgesv() {
    using solver_t = pgesv<DT, int_t, ODER>;

    const auto& env = get_env<int_t>();

    const auto context = blacs_context<int_t>();

    for(auto K = 0; K < REPEAT; ++K) {
        const auto seed = context.amx(static_cast<int_t>(duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count()));
        std::mt19937 gen(seed);

        const auto NRHS = std::uniform_int_distribution(1, 20)(gen);
        const auto N = std::uniform_int_distribution(100, 500)(gen);

        printf("Seed: %d, N: %d, NRHS: %d\n", seed, N, NRHS);

        const auto IDX = typename solver_t::indexer{N};

        std::vector<DT> A, B;

        if(0 == env.rank()) {
            A.resize(N * N, DT{0.});
            B.resize(N * NRHS, DT{1.});

            std::uniform_real_distribution dist_v(0.f, 1.f);

            for(auto I = 0; I < N; ++I)
                for(auto J = I; J < std::min(N, I + 2); ++J) A[IDX(I, J)] = dist_v(gen);
        }

        const auto info = solver_t().solve({N, N, A.data()}, {N, NRHS, B.data()});

        if(0 == env.rank()) REQUIRE(info == 0);
    }
}

#ifdef EZP_ENABLE_TEST
TEST_CASE("Random PDGESV", "[Simple Solver]") {
#else
void random_pdgesv() {
#endif
    random_pgesv<double>();
}

#ifdef EZP_ENABLE_TEST
TEST_CASE("Random PDGESVC", "[Simple Solver]") {
#else
void random_pdgesv_c() {
#endif
    random_pgesv<double, 'C'>();
}

#ifdef EZP_ENABLE_TEST
TEST_CASE("Random PSGESV", "[Simple Solver]") {
#else
void random_psgesv() {
#endif
    random_pgesv<float>();
}

#ifdef EZP_ENABLE_TEST
TEST_CASE("Random PSGESVC", "[Simple Solver]") {
#else
void random_psgesv_c() {
#endif
    random_pgesv<double, 'C'>();
}

#ifdef EZP_ENABLE_TEST
TEST_CASE("Random PZGESV", "[Simple Solver]") {
#else
void random_pzgesv() {
#endif
    random_pgesv<complex16>();
}

#ifdef EZP_ENABLE_TEST
TEST_CASE("Random PZGESVC", "[Simple Solver]") {
#else
void random_pzgesv_c() {
#endif
    random_pgesv<complex16, 'C'>();
}

#ifdef EZP_ENABLE_TEST
TEST_CASE("Random PCGESV", "[Simple Solver]") {
#else
void random_pcgesv() {
#endif
    random_pgesv<complex8>();
}

#ifdef EZP_ENABLE_TEST
TEST_CASE("Random PCGESVC", "[Simple Solver]") {
#else
void random_pcgesv_c() {
#endif
    random_pgesv<complex8, 'C'>();
}

#ifndef EZP_ENABLE_TEST
int main(const int argc, const char* argv[]) {
    volatile int i = 0;
    if(argc <= 1)
        while(0 == i) std::this_thread::sleep_for(seconds(10));
    else REPEAT = std::atoi(argv[1]);

    random_pdgesv();
    random_pdgesv_c();
    random_psgesv();
    random_psgesv_c();
    random_pzgesv();
    random_pzgesv_c();
    random_pcgesv();
    random_pcgesv_c();

    return 0;
}
#endif