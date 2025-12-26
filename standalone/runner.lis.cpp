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

#include <mpl/mpl.hpp>

int main(int argc, char** argv) {
    constexpr int NUM_NODE = 2;

    const auto& comm_world{mpl::environment::comm_world()};
    const auto worker = comm_world.spawn(0, NUM_NODE, {"solver.lis"});
    const auto all = mpl::communicator(worker, mpl::communicator::order_low);

    int config[4]{};

    std::string option; // "-print all -p ilu -ilu_fill 1 -i fgmres" option string
    for(auto i = 1; i < argc; i++) {
        option += argv[i];
        if(i < argc - 1) option += ' ';
    }

    printf("option: %s\n", option.c_str());

    constexpr int N = 40, NRHS = 1;

    config[0] = option.size(); // length of the option string
    config[1] = N;             // n
    config[2] = N;             // nnz
    config[3] = NRHS;          // nrhs

    std::vector<int> ia(N + 1), ja(N);
    std::vector<double> a(N), b(N * NRHS, 1.);

    for(auto i = 0; i < N; i++) {
        ia[i] = ja[i] = i;
        a[i] = i + 1;
    }
    ia[N] = N;

    all.bcast(0, config);
    all.bcast(0, option.data(), mpl::contiguous_layout<char>(config[0]));

    mpl::irequest_pool requests;

    requests.push(worker.isend(ia, 0, mpl::tag_t{0}));
    requests.push(worker.isend(ja, 0, mpl::tag_t{1}));
    requests.push(worker.isend(a, 0, mpl::tag_t{2}));
    requests.push(worker.isend(b, 0, mpl::tag_t{3}));

    requests.waitall();

    int error = -1;
    worker.recv(error, 0);
    if(0 == error) worker.recv(b, 0);

    for(decltype(b.size()) i = 0; i < b.size(); i++) printf("x[%lu] = %+.8f\n", i, b[i]);

    return 0;
}
