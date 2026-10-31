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
#include <ezp/pgesv.hpp>
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

template<data_t DT, char ODER = 'R'> auto random_pgesv() {
    using solver_t = pgesv<DT, int_t, ODER>;

    const auto& env = get_env<int_t>();

    const auto context = blacs_context<int_t>();

    for(auto K = 0; K < REPEAT; ++K) {
        const auto seed = context.amx(static_cast<int_t>(duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count()));
        std::mt19937 gen(seed);

        const auto NRHS = std::uniform_int_distribution(1, 10)(gen);
        const auto N = std::uniform_int_distribution(NRHS + 1, 100)(gen);

        if(0 == env.rank()) printf("Seed: %d, N: %d, NRHS: %d\n", seed, N, NRHS);

        const auto IDX = typename solver_t::indexer{N};

        std::vector<DT> A, B, RHS, XEXACT;

        if(0 == env.rank()) {
            A.resize(N * N, DT{0.});
            B.resize(N * NRHS, DT{0.});
            RHS.resize(N * NRHS);
            XEXACT.resize(N * NRHS);

            std::uniform_real_distribution dist_v(0.f, 1.f);
            std::uniform_real_distribution dist_x(-1.f, 1.f);

            for(auto I = 0; I < N; ++I)
                for(auto J = I; J < std::min(N, I + 2); ++J) A[IDX(I, J)] = dist_v(gen) + .5f;

            // Pick a known exact solution and form `B = A * XEXACT`.
            for(auto& x : XEXACT) x = dist_x(gen);
            for(auto C = 0; C < NRHS; ++C)
                for(auto I = 0; I < N; ++I) {
                    DT sum{0};
                    for(auto J = 0; J < N; ++J) sum += A[IDX(I, J)] * XEXACT[J + C * N];
                    B[I + C * N] = RHS[I + C * N] = sum;
                }
        }

        auto solver = solver_t();

        const auto info = solver.solve({N, N, A.data()}, {N, NRHS, B.data()});

        if(0 == env.rank()) {
            REQUIRE(info == 0);
            // After the solve, `B` holds the computed solution `X`. Check the residual
            // against the unmodified original matrix `A` (before `det` overwrites it).
            const auto res = full_residual(A, B, RHS, N, NRHS);
            REQUIRE(res < residual_tol<DT>());
            printf("Relative residual: %e\n", static_cast<double>(res));
        }

        const auto det = solver.det({N, N, A.data()});

        if(0 == env.rank() && std::is_same_v<DT, double>) printf("Determinant: %e\n", det);
    }
}

/**
 * @brief Solves a small system with an analytically known solution and determinant.
 *
 * The matrix `A = [[2,1,0],[0,3,1],[1,0,4]]` has `det(A) = 25` and, for the
 * exact solution `X = [1,2,3]`, the right-hand side is `B = [4,9,13]`.
 * This provides an exact-reference check of both the solution and the determinant.
 */
template<data_t DT, char ODER = 'R'> auto known_pgesv() {
    using solver_t = pgesv<DT, int_t, ODER>;

    const auto& env = get_env<int_t>();

    constexpr int_t N{3};
    constexpr int_t NRHS{1};

    std::vector<DT> A, B;

    if(0 == env.rank()) {
        A.resize(N * N, DT{0.});
        B.resize(N * NRHS);

        const auto IDX = typename solver_t::indexer{N};
        A[IDX(0, 0)] = DT{2};
        A[IDX(0, 1)] = DT{1};
        A[IDX(1, 2)] = DT{1};
        A[IDX(2, 0)] = DT{1};
        A[IDX(2, 2)] = DT{4};
        A[IDX(1, 1)] = DT{3};

        // B = A * X with X = [1, 2, 3]
        B[0] = DT{4};
        B[1] = DT{9};
        B[2] = DT{13};
    }

    const auto RHS = B;

    auto solver = solver_t();
    const auto info = solver.solve({N, N, A.data()}, {N, NRHS, B.data()});

    if(0 == env.rank()) {
        REQUIRE(info == 0);
        // Check the residual against the unmodified matrix `A` before `det` overwrites it.
        const auto res = full_residual(A, B, RHS, N, NRHS);
        REQUIRE(res < residual_tol<DT>());
        printf("Known PDGESV: solution residual=%e\n", static_cast<double>(res));
    }

    const auto det = solver.det({N, N, A.data()});

    if(0 == env.rank()) {
        REQUIRE(std::abs(det - DT{25}) < residual_tol<DT>());
        printf("Known PDGESV: det=%e (expected 25)\n", static_cast<double>(std::abs(det)));
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
TEST_CASE("Known PDGESV", "[Simple Solver][Known]") {
#else
void known_pdgesv() {
#endif
    known_pgesv<double>();
}

#ifdef EZP_ENABLE_TEST
TEST_CASE("Random PDGESVC", "[Simple Solver]") {
#else
void random_pdgesv_c() {
#endif
    random_pgesv<double, 'C'>();
}

#ifdef EZP_ENABLE_TEST
TEST_CASE("Known PDGESVC", "[Simple Solver][Known]") {
#else
void known_pdgesv_c() {
#endif
    known_pgesv<double, 'C'>();
}

#ifdef EZP_ENABLE_TEST
TEST_CASE("Random PSGESV", "[Simple Solver]") {
#else
void random_psgesv() {
#endif
    random_pgesv<float>();
}

#ifdef EZP_ENABLE_TEST
TEST_CASE("Known PSGESV", "[Simple Solver][Known]") {
#else
void known_psgesv() {
#endif
    known_pgesv<float>();
}

#ifdef EZP_ENABLE_TEST
TEST_CASE("Random PSGESVC", "[Simple Solver]") {
#else
void random_psgesv_c() {
#endif
    random_pgesv<float, 'C'>();
}

#ifdef EZP_ENABLE_TEST
TEST_CASE("Known PSGESVC", "[Simple Solver][Known]") {
#else
void known_psgesv_c() {
#endif
    known_pgesv<float, 'C'>();
}

#ifdef EZP_ENABLE_TEST
TEST_CASE("Random PZGESV", "[Simple Solver]") {
#else
void random_pzgesv() {
#endif
    random_pgesv<complex16>();
}

#ifdef EZP_ENABLE_TEST
TEST_CASE("Known PZGESV", "[Simple Solver][Known]") {
#else
void known_pzgesv() {
#endif
    known_pgesv<complex16>();
}

#ifdef EZP_ENABLE_TEST
TEST_CASE("Random PZGESVC", "[Simple Solver]") {
#else
void random_pzgesv_c() {
#endif
    random_pgesv<complex16, 'C'>();
}

#ifdef EZP_ENABLE_TEST
TEST_CASE("Known PZGESVC", "[Simple Solver][Known]") {
#else
void known_pzgesv_c() {
#endif
    known_pgesv<complex16, 'C'>();
}

#ifdef EZP_ENABLE_TEST
TEST_CASE("Random PCGESV", "[Simple Solver]") {
#else
void random_pcgesv() {
#endif
    random_pgesv<complex8>();
}

#ifdef EZP_ENABLE_TEST
TEST_CASE("Known PCGESV", "[Simple Solver][Known]") {
#else
void known_pcgesv() {
#endif
    known_pgesv<complex8>();
}

#ifdef EZP_ENABLE_TEST
TEST_CASE("Random PCGESVC", "[Simple Solver]") {
#else
void random_pcgesv_c() {
#endif
    random_pgesv<complex8, 'C'>();
}

#ifdef EZP_ENABLE_TEST
TEST_CASE("Known PCGESVC", "[Simple Solver][Known]") {
#else
void known_pcgesv_c() {
#endif
    known_pgesv<complex8, 'C'>();
}

#ifndef EZP_ENABLE_TEST
int main(const int argc, const char* argv[]) {
    if(argc <= 1) {
        volatile int i = 0;
        while(0 == i) std::this_thread::sleep_for(seconds(10));
    }
    else REPEAT = std::atoi(argv[1]);

    random_pdgesv();
    known_pdgesv();
    random_pdgesv_c();
    known_pdgesv_c();
    random_psgesv();
    known_psgesv();
    random_psgesv_c();
    known_psgesv_c();
    random_pzgesv();
    known_pzgesv();
    random_pzgesv_c();
    known_pzgesv_c();
    random_pcgesv();
    known_pcgesv();
    random_pcgesv_c();
    known_pcgesv_c();

    return 0;
}
#endif
