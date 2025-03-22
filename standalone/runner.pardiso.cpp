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

#include <mpl/mpl.hpp>

int main(int, char**) {
    constexpr int NUM_NODE = 2;

    const auto& comm_world{mpl::environment::comm_world()};
    const auto worker = comm_world.spawn(0, NUM_NODE, {"solver.pardiso"});
    const auto all = mpl::communicator(worker, mpl::communicator::order_low);

    int config[7]{};
    int iparm[64]{};

    iparm[0] = 1;   // solver default parameters overriden with provided by iparm
    iparm[1] = 3;   // use METIS for fill-in reordering
    iparm[5] = 0;   // write solution into x
    iparm[7] = 2;   // max number of iterative refinement steps
    iparm[9] = 13;  // perturb the pivot elements with 1E-13
    iparm[10] = 1;  // use nonsymmetric permutation and scaling MPS
    iparm[12] = 1;  // switch on Maximum Weighted Matching algorithm (default for non-symmetric)
    // iparm[17] = -1; // output: Number of nonzeros in the factor LU
    // iparm[18] = -1; // output: Mflops for LU factorization
    iparm[26] = 0;  // check input data for correctness
    iparm[39] = 0;  // input: matrix/rhs/solution stored on master

    constexpr int N = 100, NRHS = 1;

    config[0] = 11;   // mtype
    config[1] = NRHS; // nrhs
    config[2] = 1;    // maxfct
    config[3] = 1;    // mnum
    config[4] = 0;    // msglvl
    config[5] = N;    // n
    config[6] = N;    // nnz

    std::vector<int> ia(N + 1), ja(N);
    std::vector<double> a(N), b(N * NRHS, 1.);

    for(auto i = 0; i < N; i++) ia[i] = ja[i] = a[i] = i + 1;
    ia[N] = N + 1;

    all.bcast(0, config);
    all.bcast(0, iparm);

    mpl::irequest_pool requests;

    requests.push(worker.isend(ia, 0, mpl::tag_t{0}));
    requests.push(worker.isend(ja, 0, mpl::tag_t{1}));
    requests.push(worker.isend(a, 0, mpl::tag_t{2}));
    requests.push(worker.isend(b, 0, mpl::tag_t{3}));

    requests.waitall();

    int error = -1;
    worker.recv(error, 0);
    if(0 == error) worker.recv(b, 0);

    for(auto i = 0; i < b.size(); i++) printf("x[%ld] = %+.8f\n", i, b[i]);

    return 0;
}
