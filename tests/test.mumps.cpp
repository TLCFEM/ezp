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
#include <ezp/mumps.parser.hpp>
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

template<data_t DT> auto random_mumps() {
    const auto& comm_world{mpl::environment::comm_world()};

    blacs_env<>::do_not_manage_mpi();

    auto solver = mumps<DT, int_t>();
    mumps_set("--printing-level 0", solver);

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
            ia.resize(N);
            ja.resize(N);
            a.resize(N);
            b.resize(N * NRHS);

            for(auto i = 0; i < N; i++) {
                ia[i] = ja[i] = i + 1;
                a[i] = i + 1;
            }

            std::fill(b.begin(), b.end(), 1.);
        }

        const auto info = solver.solve({N, N, ia.data(), ja.data(), a.data()}, {N, NRHS, b.data()});

        if(0 == comm_world.rank()) REQUIRE(info == 0);
    }
}

#ifdef EZP_ENABLE_TEST
TEST_CASE("Random DMUMPS", "[Sparse Solver]") {
#else
void random_dmumps() {
#endif
    random_mumps<double>();
}

#ifdef EZP_ENABLE_TEST
TEST_CASE("Random SMUMPS", "[Sparse Solver]") {
#else
void random_smumps() {
#endif
    random_mumps<float>();
}

#ifdef EZP_ENABLE_TEST
TEST_CASE("Random ZMUMPS", "[Sparse Solver]") {
#else
void random_zmumps() {
#endif
    random_mumps<complex16>();
}

#ifdef EZP_ENABLE_TEST
TEST_CASE("Random CMUMPS", "[Sparse Solver]") {
#else
void random_cmumps() {
#endif
    random_mumps<complex8>();
}

#ifndef EZP_ENABLE_TEST
int main(const int argc, const char* argv[]) {
    volatile int i = 0;
    if(argc <= 1)
        while(0 == i) std::this_thread::sleep_for(seconds(10));
    else REPEAT = std::atoi(argv[1]);

    random_dmumps();
    random_smumps();
    random_zmumps();
    random_cmumps();

    return 0;
}
#endif
