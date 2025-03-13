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

#ifndef SOLVER_FULL_HPP
#define SOLVER_FULL_HPP

#include <ezp/ezp>
#include <mpl/mpl.hpp>

inline const auto& comm_world{mpl::environment::comm_world()};
inline const auto& parent = mpl::inter_communicator::parent();

template<ezp::data_t DT, ezp::index_t IT, typename solver_t> int run(const int N, const int NRHS) {
    std::vector<DT> A, B;

    if(0 == comm_world.rank()) {
        A.resize(N * N);
        B.resize(N * NRHS);

        mpl::irequest_pool requests;

        requests.push(parent.irecv(A, 0, mpl::tag_t{0}));
        requests.push(parent.irecv(B, 0, mpl::tag_t{1}));

        requests.waitall();
    }

    const auto error = solver_t().solve({N, N, A.data()}, {N, NRHS, B.data()});

    if(0 == comm_world.rank()) {
        parent.send(error, 0);
        if(0 == error) parent.send(B, 0);
    }

    return 0;
}

#endif // SOLVER_FULL_HPP