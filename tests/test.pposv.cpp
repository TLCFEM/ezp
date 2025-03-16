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
#include <ezp/pposv.hpp>
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

template<data_t DT, char UL = 'L', char ODER = 'R'> auto random_pposv() {
    using solver_t = pposv<DT, int_t, UL, ODER>;

    const auto& env = get_env<int_t>();

    const auto context = blacs_context<int_t>();

    for(auto K = 0; K < REPEAT; ++K) {
        const auto seed = context.amx(static_cast<int_t>(duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count()));
        std::mt19937 gen(seed);

        const auto NRHS = std::uniform_int_distribution(1, 10)(gen);
        const auto N = std::uniform_int_distribution(NRHS + 1, 100)(gen);

        if(0 == env.rank()) printf("Seed: %d, N: %d, NRHS: %d\n", seed, N, NRHS);

        const auto IDX = typename solver_t::indexer{N};

        std::vector<DT> A, B;

        if(0 == env.rank()) {
            A.resize(N * N, DT{0.});
            B.resize(N * NRHS, DT{0.});

            std::uniform_real_distribution dist_v(0.f, 1.f);

            for(auto I = 0; I < N; ++I) {
                A[IDX(I, I)] = DT{10.f + 10.f * dist_v(gen)};
                for(auto J = I + 1; J < std::min(N, I + 5); ++J) A[IDX(I, J)] = A[IDX(J, I)] = dist_v(gen);
            }
        }

        const auto info = solver_t().solve({N, N, A.data()}, {N, NRHS, B.data()});

        if(0 == env.rank()) REQUIRE(info == 0);
    }
}

#ifdef EZP_ENABLE_TEST
TEST_CASE("Random PDPOSV", "[Simple Solver]") {
#else
void random_pdposv() {
#endif
    random_pposv<double>();
}

#ifdef EZP_ENABLE_TEST
TEST_CASE("Random PDPOSVC", "[Simple Solver]") {
#else
void random_pdposv_c() {
#endif
    random_pposv<double, 'L', 'C'>();
}

#ifdef EZP_ENABLE_TEST
TEST_CASE("Random PSPOSV", "[Simple Solver]") {
#else
void random_psposv() {
#endif
    random_pposv<float>();
}

#ifdef EZP_ENABLE_TEST
TEST_CASE("Random PSPOSVC", "[Simple Solver]") {
#else
void random_psposv_c() {
#endif
    random_pposv<float, 'L', 'C'>();
}

#ifdef EZP_ENABLE_TEST
TEST_CASE("Random PZPOSV", "[Simple Solver]") {
#else
void random_pzposv() {
#endif
    random_pposv<complex16>();
}

#ifdef EZP_ENABLE_TEST
TEST_CASE("Random PZPOSVC", "[Simple Solver]") {
#else
void random_pzposv_c() {
#endif
    random_pposv<complex16, 'L', 'C'>();
}

#ifdef EZP_ENABLE_TEST
TEST_CASE("Random PCPOSV", "[Simple Solver]") {
#else
void random_pcposv() {
#endif
    random_pposv<complex8>();
}

#ifdef EZP_ENABLE_TEST
TEST_CASE("Random PCPOSVC", "[Simple Solver]") {
#else
void random_pcposv_c() {
#endif
    random_pposv<complex8, 'L', 'C'>();
}

#ifdef EZP_ENABLE_TEST
#else
int main(const int argc, const char* argv[]) {
    volatile int i = 0;
    if(argc <= 1)
        while(0 == i) std::this_thread::sleep_for(seconds(10));
    else REPEAT = std::atoi(argv[1]);

    random_pdposv();
    random_pdposv_c();
    random_psposv();
    random_psposv_c();
    random_pzposv();
    random_pzposv_c();
    random_pcposv();
    random_pcposv_c();

    return 0;
}
#endif