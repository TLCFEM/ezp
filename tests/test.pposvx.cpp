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
#include <ezp/pposvx.hpp>
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

template<data_t DT, char ODER = 'R'> auto random_pposvx() {
    using solver_t = pposvx<DT, int_t, 'L', ODER>;

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

            std::uniform_real_distribution dist_v(1.f, 2.f);

            for(auto I = 0; I < N; ++I) {
                A[IDX(I, I)] = 10.f * dist_v(gen);
                for(auto J = I + 1; J < std::min(N, I + 2); ++J) A[IDX(I, J)] = A[IDX(J, I)] = dist_v(gen);
            }
        }

        const auto info = solver_t().solve({N, N, A.data()}, {N, NRHS, B.data()});

        if(0 == env.rank()) REQUIRE(info == 0);
    }
}

#ifdef EZP_ENABLE_TEST
TEST_CASE("Random PDPOSVX", "[Expert Solver]") {
#else
void random_pdposvx() {
#endif
    random_pposvx<double>();
}

#ifdef EZP_ENABLE_TEST
TEST_CASE("Random PDPOSVXC", "[Expert Solver]") {
#else
void random_pdposvx_c() {
#endif
    random_pposvx<double, 'C'>();
}

#ifdef EZP_ENABLE_TEST
TEST_CASE("Random PSPOSVX", "[Expert Solver]") {
#else
void random_psposvx() {
#endif
    random_pposvx<float>();
}

#ifdef EZP_ENABLE_TEST
TEST_CASE("Random PSPOSVXC", "[Expert Solver]") {
#else
void random_psposvx_c() {
#endif
    random_pposvx<double, 'C'>();
}

#ifdef EZP_ENABLE_TEST
TEST_CASE("Random PZPOSVX", "[Expert Solver]") {
#else
void random_pzposvx() {
#endif
    random_pposvx<complex16>();
}

#ifdef EZP_ENABLE_TEST
TEST_CASE("Random PZPOSVXC", "[Expert Solver]") {
#else
void random_pzposvx_c() {
#endif
    random_pposvx<complex16, 'C'>();
}

#ifdef EZP_ENABLE_TEST
TEST_CASE("Random PCPOSVX", "[Expert Solver]") {
#else
void random_pcposvx() {
#endif
    random_pposvx<complex8>();
}

#ifdef EZP_ENABLE_TEST
TEST_CASE("Random PCPOSVXC", "[Expert Solver]") {
#else
void random_pcposvx_c() {
#endif
    random_pposvx<complex8, 'C'>();
}

#ifndef EZP_ENABLE_TEST
int main(const int argc, const char* argv[]) {
    volatile int i = 0;
    if(argc <= 1)
        while(0 == i) std::this_thread::sleep_for(seconds(10));
    else REPEAT = std::atoi(argv[1]);

    random_pdposvx();
    random_pdposvx_c();
    random_psposvx();
    random_psposvx_c();
    random_pzposvx();
    random_pzposvx_c();
    random_pcposvx();
    random_pcposvx_c();

    return 0;
}
#endif