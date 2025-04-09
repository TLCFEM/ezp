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
#include <ezp/pardiso.hpp>
#include <random>
#include <thread>

#ifdef EZP_MKL

using namespace ezp;
using namespace std::chrono;

#ifdef EZP_ENABLE_TEST
#include <catch2/catchy.hpp>
#else
#define REQUIRE(...)
#endif

static auto REPEAT = 10;

template<data_t DT> auto random_pardiso() {
    const auto& comm_world{mpl::environment::comm_world()};

    blacs_env<>::do_not_manage_mpi();

    using solver_t = pardiso<DT, int_t>;

    for(auto K = 0; K < REPEAT; ++K) {
        auto seed = static_cast<int_t>(duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count());
        comm_world.allreduce(mpl::max<int_t>(), seed);
        std::mt19937 gen(seed);

        const auto NRHS = std::uniform_int_distribution(1, 10)(gen);
        const auto N = std::uniform_int_distribution(1, 100)(gen);

        if(0 == comm_world.rank()) printf("Seed: %d, N: %d, NRHS: %d\n", seed, N, NRHS);

        std::vector<int_t> ia, ja;
        std::vector<DT> a, b;

        if(0 == comm_world.rank()) {
            ia.resize(N + 1);
            ja.resize(N);
            a.resize(N);
            b.resize(N * NRHS);

            for(auto i = 0; i < N; i++) {
                ia[i] = ja[i] = i + 1;
                a[i] = i + 1;
            }
            ia[N] = N + 1;

            std::fill(b.begin(), b.end(), 1.);
        }

        int_t mtype;
        if constexpr(std::is_same_v<DT, double>) mtype = 11;
        else if constexpr(std::is_same_v<DT, float>) mtype = 11;
        else if constexpr(std::is_same_v<DT, complex16>) mtype = 13;
        else if constexpr(std::is_same_v<DT, complex8>) mtype = 13;

        auto solver = solver_t(mtype);

        solver(0) = 1;  // solver default parameters overriden with provided by iparm
        solver(1) = 3;  // use METIS for fill-in reordering
        solver(7) = 2;  // max number of iterative refinement steps
        solver(9) = 13; // perturb the pivot elements with 1E-13
        solver(10) = 1; // use nonsymmetric permutation and scaling MPS
        solver(12) = 1; // switch on Maximum Weighted Matching algorithm (default for non-symmetric)

        const auto info = solver.solve({N, N, ia.data(), ja.data(), a.data()}, {N, NRHS, b.data()});

        if(0 == comm_world.rank()) REQUIRE(info == 0);
    }
}

#ifdef EZP_ENABLE_TEST
TEST_CASE("Random PARDISO", "[Sparse Solver]") {
#else
void random_dpardiso() {
#endif
    random_pardiso<double>();
}

#ifdef EZP_ENABLE_TEST
TEST_CASE("Random SPARDISO", "[Sparse Solver]") {
#else
void random_spardiso() {
#endif
    random_pardiso<float>();
}

#ifdef EZP_ENABLE_TEST
TEST_CASE("Random ZPARDISO", "[Sparse Solver]") {
#else
void random_zpardiso() {
#endif
    random_pardiso<complex16>();
}

#ifdef EZP_ENABLE_TEST
TEST_CASE("Random CPARDISO", "[Sparse Solver]") {
#else
void random_cpardiso() {
#endif
    random_pardiso<complex8>();
}

#ifndef EZP_ENABLE_TEST
int main(const int argc, const char* argv[]) {
    if(argc <= 1) {
        volatile int i = 0;
        while(0 == i) std::this_thread::sleep_for(seconds(10));
    }
    else REPEAT = std::atoi(argv[1]);

    random_dpardiso();
    random_spardiso();
    random_zpardiso();
    random_cpardiso();

    return 0;
}
#endif
#elif !defined(EZP_ENABLE_TEST)
int main() { return 0; }
#endif
