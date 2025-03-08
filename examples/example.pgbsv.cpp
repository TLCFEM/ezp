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
/**
 * @brief Example caller to the `pgbsv` solver.
 *
 * @author tlc
 * @date 07/03/2025
 * @version 1.0.0
 * @file example.pgbsv.cpp
 * @{
 */

#include <ezp/pgbsv.hpp>
#include <iomanip>
#include <iostream>

using namespace ezp;

int main() {
    // get the current blacs environment
    const auto& env = get_env<int>();

    constexpr auto N = 10, NRHS = 1, KL = 2, KU = 2;
    constexpr auto LDA = 2 * (KL + KU) + 1;
    constexpr auto OFFSET = 2 * KU + KL;

    // storage for the matrices A and B
    std::vector<double> A, B;

    // helper function to convert 2D indices to 1D indices
    // the band symmetric matrix used for gbsv subroutine requires the matrix to be stored with a leading dimension of (2 * (KL + KU) + 1)
    // see Fig. 4.10 https://netlib.org/scalapack/slug/node84.html
    const auto IDX = par_dgbsv<int>::indexer{N, KL, KU};

    if(0 == env.rank()) {
        // the matrices are only initialized on the root process
        A.resize(N * LDA, 0.);
        B.resize(N * NRHS);

        A[IDX(0, 0)] = 1.741102;
        A[IDX(0, 1)] = 0.597832;
        A[IDX(0, 2)] = 0.859107;
        A[IDX(1, 0)] = 0.654704;
        A[IDX(1, 1)] = 1.839290;
        A[IDX(1, 2)] = 0.284697;
        A[IDX(1, 3)] = 0.093657;
        A[IDX(2, 0)] = 0.064187;
        A[IDX(2, 1)] = 0.851994;
        A[IDX(2, 2)] = 2.753108;
        A[IDX(2, 3)] = 0.504381;
        A[IDX(2, 4)] = 0.018659;
        A[IDX(3, 1)] = 0.858028;
        A[IDX(3, 2)] = 0.255440;
        A[IDX(3, 3)] = 3.308421;
        A[IDX(3, 4)] = 0.682957;
        A[IDX(3, 5)] = 0.777682;
        A[IDX(4, 2)] = 0.341441;
        A[IDX(4, 3)] = 0.237288;
        A[IDX(4, 4)] = 1.849747;
        A[IDX(4, 5)] = 0.376083;
        A[IDX(4, 6)] = 0.406906;
        A[IDX(5, 3)] = 0.452318;
        A[IDX(5, 4)] = 0.429588;
        A[IDX(5, 5)] = 2.486945;
        A[IDX(5, 6)] = 0.680845;
        A[IDX(5, 7)] = 0.887935;
        A[IDX(6, 4)] = 0.823589;
        A[IDX(6, 5)] = 0.420532;
        A[IDX(6, 6)] = 2.626641;
        A[IDX(6, 7)] = 0.269718;
        A[IDX(6, 8)] = 0.002824;
        A[IDX(7, 5)] = 0.133123;
        A[IDX(7, 6)] = 0.333038;
        A[IDX(7, 7)] = 1.966010;
        A[IDX(7, 8)] = 0.764672;
        A[IDX(7, 9)] = 0.462412;
        A[IDX(8, 6)] = 0.119309;
        A[IDX(8, 7)] = 0.886263;
        A[IDX(8, 8)] = 2.895127;
        A[IDX(8, 9)] = 0.445419;
        A[IDX(9, 7)] = 0.455375;
        A[IDX(9, 8)] = 0.024524;
        A[IDX(9, 9)] = 1.200660;

        B[0] = 1.261587;
        B[1] = 0.988482;
        B[2] = 3.130108;
        B[3] = 1.785095;
        B[4] = 1.095000;
        B[5] = 1.316414;
        B[6] = 0.804297;
        B[7] = 1.723804;
        B[8] = 1.628384;
        B[9] = 1.369979;
    }

    // create a parallel solver
    // it uses a one-dimensional process grid
    // it takes the number of processes as arguments
    auto solver = par_dgbsv(env.size());

    // need to wrap the data in full_mat objects
    // it requires the number of rows and columns of the matrix, and a pointer to the data
    // on non-root processes, the data pointer is nullptr as the vector is empty
    // solver.solve(band_mat{N, N, KL, KU, A.data()}, full_mat{N, NRHS, B.data()});
    const auto info = solver.solve({N, N, KL, KU, A.data()}, {N, NRHS, B.data()});

    if(0 == env.rank()) {
        std::cout << std::setprecision(6) << std::fixed << "Info: " << info << '\n';
        std::cout << "Solution:\n";
        std::vector<double> X(N * NRHS);
        X[0] = 0.128021;
        X[1] = 0.326264;
        X[2] = 0.981995;
        X[3] = 0.266945;
        X[4] = 0.307322;
        X[5] = 0.207357;
        X[6] = 0.122664;
        X[7] = 0.523065;
        X[8] = 0.253049;
        X[9] = 0.937470;
        for(auto i = 0; i < B.size(); ++i) std::cout << B[i] << ' ' << X[i] << " abs(diff): " << std::abs(B[i] - X[i]) << '\n';
    }

    return 0;
}

//! @}
