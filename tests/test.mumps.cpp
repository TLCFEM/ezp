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

    auto solver = mumps<DT, int>();
    mumps_set("--printing-level 0", solver);

    for(auto K = 0; K < REPEAT; ++K) {
        auto seed = static_cast<int>(duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count());
        comm_world.allreduce(mpl::max<int>(), seed);
        std::mt19937 gen(seed);

        const auto NRHS = std::uniform_int_distribution(1, 10)(gen);
        const auto N = std::uniform_int_distribution(1, 100)(gen);

        if(0 == comm_world.rank()) printf("Seed: %d, N: %d, NRHS: %d\n", seed, N, NRHS);

        std::vector<int> ia, ja;
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

        [[maybe_unused]] auto info = solver.solve({N, N, ia.data(), ja.data(), a.data()}, {N, NRHS, b.data()});

        if(0 == comm_world.rank()) REQUIRE(info == 0);

        info = solver.solve({N, NRHS, b.data()});

        if(0 == comm_world.rank()) REQUIRE(info == 0);
    }

    solver.icntl_output_error_message(0);
    solver.icntl_output_diagnostic_statistics_warning(0);
    solver.icntl_output_global_information(0);
    solver.icntl_printing_level(0);
    solver.icntl_permutation_and_scaling(0);
    solver.icntl_symmetric_permutation(0);
    solver.icntl_scaling_strategy(0);
    solver.icntl_transpose_matrix(0);
    solver.icntl_iterative_refinement(0);
    solver.icntl_error_analysis(0);
    solver.icntl_ordering_strategy(0);
    solver.icntl_root_parallelism(0);
    solver.icntl_working_space_percentage_increase(0);
    solver.icntl_compression_block_format(0);
    solver.icntl_openmp_threads(0);
    solver.icntl_distribution_strategy_input(0);
    solver.icntl_schur_complement(0);
    solver.icntl_distribution_strategy_solution(0);
    solver.icntl_out_of_core(0);
    solver.icntl_maximum_working_memory(0);
    solver.icntl_null_pivot_row_detection(0);
    solver.icntl_deficient_and_null_space_basis(0);
    solver.icntl_schur_complement_solution(0);
    solver.icntl_rhs_block_size(0);
    solver.icntl_ordering_computation(0);
    solver.icntl_inverse_computation(0);
    solver.icntl_forward_elimination(0);
    solver.icntl_determinant_computation(0);
    solver.icntl_out_of_core_file(0);
    solver.icntl_blr(0);
    solver.icntl_blr_variant(0);
    solver.icntl_blr_compression(0);
    solver.icntl_lu_compression_rate(0);
    solver.icntl_block_compression_rate(0);
    solver.icntl_tree_parallelism(0);
    solver.icntl_compact_working_space(0);
    solver.icntl_rank_revealing_factorization(0);
    solver.icntl_symbolic_factorization(0);
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
    if(argc <= 1) {
        volatile int i = 0;
        while(0 == i) std::this_thread::sleep_for(seconds(10));
    }
    else REPEAT = std::atoi(argv[1]);

    random_dmumps();
    random_smumps();
    random_zmumps();
    random_cmumps();

    return 0;
}
#endif
