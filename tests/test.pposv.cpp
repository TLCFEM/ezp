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

#ifdef EZP_ENABLE_TEST
TEST_CASE("Random PDPOSV", "[Simple Solver]") {
#else
void random_pdposv() {
#endif
    const auto& env = get_env<int>();

    const auto context = blacs_context<int>();

    for(auto K = 0; K < 100; ++K) {
        const auto seed = context.amx(static_cast<int>(duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count()));
        std::mt19937 gen(seed);

        const auto NRHS = std::uniform_int_distribution(1, 20)(gen);
        const auto N = std::uniform_int_distribution(100, 400)(gen);

        const auto IDX = par_dposv<int>::indexer{N};

        std::vector<double> A, B;

        if(0 == env.rank()) {
            A.resize(N * N, 0.);
            B.resize(N * NRHS, 1.);

            std::uniform_real_distribution dist_v(0., 1.);

            for(auto I = 0; I < N; ++I) {
                A[IDX(I, I)] = 10. + 10. * dist_v(gen);
                for(auto J = I + 1; J < std::min(N, I + 5); ++J) A[IDX(I, J)] = A[IDX(J, I)] = dist_v(gen);
            }
        }

        const auto info = par_dposv<int>().solve({N, N, A.data()}, {N, NRHS, B.data()});

        if(0 == env.rank()) REQUIRE(info == 0);
    }
}

#ifdef EZP_ENABLE_TEST
TEST_CASE("Random PDPOSVC", "[Simple Solver]") {
#else
void random_pdposv_c() {
#endif
    const auto& env = get_env<int>();

    const auto context = blacs_context<int>();

    for(auto K = 0; K < 100; ++K) {
        const auto seed = context.amx(static_cast<int>(duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count()));
        std::mt19937 gen(seed);

        const auto NRHS = std::uniform_int_distribution(1, 20)(gen);
        const auto N = std::uniform_int_distribution(100, 400)(gen);

        const auto IDX = par_dposv_c<int>::indexer{N};

        std::vector<double> A, B;

        if(0 == env.rank()) {
            A.resize(N * N, 0.);
            B.resize(N * NRHS, 1.);

            std::uniform_real_distribution dist_v(0., 1.);

            for(auto I = 0; I < N; ++I) {
                A[IDX(I, I)] = 10. + 10. * dist_v(gen);
                for(auto J = I + 1; J < std::min(N, I + 5); ++J) A[IDX(I, J)] = A[IDX(J, I)] = dist_v(gen);
            }
        }

        const auto info = par_dposv_c<int>().solve({N, N, A.data()}, {N, NRHS, B.data()});

        if(0 == env.rank()) REQUIRE(info == 0);
    }
}

#ifdef EZP_ENABLE_TEST
TEST_CASE("Random PSPOSV", "[Simple Solver]") {
#else
void random_psposv() {
#endif
    const auto& env = get_env<int>();

    const auto context = blacs_context<int>();

    for(auto K = 0; K < 100; ++K) {
        const auto seed = context.amx(static_cast<int>(duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count()));
        std::mt19937 gen(seed);

        const auto NRHS = std::uniform_int_distribution(1, 20)(gen);
        const auto N = std::uniform_int_distribution(100, 400)(gen);

        const auto IDX = par_sposv<int>::indexer{N};

        std::vector<float> A, B;

        if(0 == env.rank()) {
            A.resize(N * N, 0.f);
            B.resize(N * NRHS, 1.f);

            std::uniform_real_distribution dist_v(0.f, 1.f);

            for(auto I = 0; I < N; ++I) {
                A[IDX(I, I)] = 10.f + 10.f * dist_v(gen);
                for(auto J = I + 1; J < std::min(N, I + 5); ++J) A[IDX(I, J)] = A[IDX(J, I)] = dist_v(gen);
            }
        }

        const auto info = par_sposv<int>().solve({N, N, A.data()}, {N, NRHS, B.data()});

        if(0 == env.rank()) REQUIRE(info == 0);
    }
}

#ifdef EZP_ENABLE_TEST
TEST_CASE("Random PSPOSVC", "[Simple Solver]") {
#else
void random_psposv_c() {
#endif
    const auto& env = get_env<int>();

    const auto context = blacs_context<int>();

    for(auto K = 0; K < 100; ++K) {
        const auto seed = context.amx(static_cast<int>(duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count()));
        std::mt19937 gen(seed);

        const auto NRHS = std::uniform_int_distribution(1, 20)(gen);
        const auto N = std::uniform_int_distribution(100, 400)(gen);

        const auto IDX = par_sposv_c<int>::indexer{N};

        std::vector<float> A, B;

        if(0 == env.rank()) {
            A.resize(N * N, 0.f);
            B.resize(N * NRHS, 1.f);

            std::uniform_real_distribution dist_v(0.f, 1.f);

            for(auto I = 0; I < N; ++I) {
                A[IDX(I, I)] = 10.f + 10.f * dist_v(gen);
                for(auto J = I + 1; J < std::min(N, I + 5); ++J) A[IDX(I, J)] = A[IDX(J, I)] = dist_v(gen);
            }
        }

        const auto info = par_sposv_c<int>().solve({N, N, A.data()}, {N, NRHS, B.data()});

        if(0 == env.rank()) REQUIRE(info == 0);
    }
}

#ifdef EZP_ENABLE_TEST
#else
int main(const int argc, const char*[]) {
    volatile int i = 0;
    if(argc <= 1)
        while(0 == i) std::this_thread::sleep_for(seconds(10));

    random_pdposv();
    random_pdposv_c();
    random_psposv();
    random_psposv_c();

    return 0;
}
#endif