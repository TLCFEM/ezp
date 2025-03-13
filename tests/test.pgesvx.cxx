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
#include <ezp/pgesvx.hpp>
#include <random>
#include <thread>

using namespace ezp;
using namespace std::chrono;

static auto N = 10;

template<data_t DT, char ODER = 'R'> auto random_pgesvx() {
    using solver_t = pgesvx<DT, int_t, ODER>;

    const auto& env = get_env<int_t>();

    const auto context = blacs_context<int_t>();

    for(auto K = 0; K < N; ++K) {
        auto seed = context.amx(static_cast<int_t>(duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count()));
        seed = 1019740621;
        std::mt19937 gen(seed);

        const auto NRHS = std::uniform_int_distribution(1, 2)(gen);
        const auto N = std::uniform_int_distribution(5, 10)(gen);

        printf("Seed: %d, N: %d, NRHS: %d\n", seed, N, NRHS);

        const auto IDX = typename solver_t::indexer{N};

        std::vector<DT> A, B;

        if(0 == env.rank()) {
            A.resize(N * N, DT{0.});
            B.resize(N * NRHS, DT{1.});

            std::uniform_real_distribution dist_v(0.f, 1.f);

            for(auto I = 0; I < N; ++I) {
                A[IDX(I, I)] = 10.f * dist_v(gen);
                for(auto J = I + 1; J < std::min(N, I + 2); ++J) A[IDX(I, J)] = dist_v(gen);
            }
        }

        [[maybe_unused]] const auto info = solver_t().solve({N, N, A.data()}, {N, NRHS, B.data()});
    }
}

int main(const int argc, const char* argv[]) {
    volatile int i = 0;
    if(argc <= 1)
        while(0 == i) std::this_thread::sleep_for(seconds(10));
    else N = std::atoi(argv[1]);

    random_pgesvx<double>();

    return 0;
}