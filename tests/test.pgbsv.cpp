/*******************************************************************************
 * Copyright (C) 2025-2026 Theodore Chang
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
#include <ezp/pgbsv.hpp>
#include <random>
#include <thread>

#include "check.hpp"

using namespace ezp;
using namespace ezp::test;
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

        std::vector<DT> A, B, RHS, XEXACT;

        if(0 == env.rank()) {
            A.resize(N * LDA, DT{0.});
            B.resize(N * NRHS, DT{0.});
            RHS.resize(N * NRHS);
            XEXACT.resize(N * NRHS);

            std::uniform_real_distribution dist_v(0.f, 1.f);
            std::uniform_real_distribution dist_x(-1.f, 1.f);

            for(auto I = 0; I < N; ++I) {
                A[IDX(I, I)] = 10.f * dist_v(gen) + 10.f;
                for(auto J = std::max(0, I - KU); J < std::min(N, I + KL + 1); ++J)
                    if(I != J && IDX(I, J) >= 0) A[IDX(I, J)] = dist_v(gen);
            }

            // Pick a known exact solution and form `B = A * XEXACT` from band storage.
            for(auto& x : XEXACT) x = dist_x(gen);
            for(auto C = 0; C < NRHS; ++C)
                for(auto I = 0; I < N; ++I) {
                    DT sum{0};
                    for(auto J = 0; J < N; ++J)
                        if(const auto position = IDX(I, J); position >= 0) sum += A[position] * XEXACT[J + C * N];
                    B[I + C * N] = RHS[I + C * N] = sum;
                }
        }

        const auto info = solver_t().solve({N, N, KL, KU, A.data()}, {N, NRHS, B.data()});

        if(0 == env.rank()) {
            REQUIRE(info == 0);
            // After the solve, `B` holds the computed solution `X`.
            const auto A_FULL = rebuild_full_band(A, IDX, N);
            const auto res = full_residual(A_FULL, B, RHS, N, NRHS);
            REQUIRE(res < residual_tol<DT>());
            printf("Relative residual: %e\n", static_cast<double>(res));
        }
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
