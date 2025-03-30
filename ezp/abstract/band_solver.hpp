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

#ifndef BAND_SOLVER_HPP
#define BAND_SOLVER_HPP

#include "abstract_solver.hpp"

namespace ezp::detail {
    template<data_t DT, index_t IT, wrapper_t WT> class band_solver : public abstract_solver<DT, IT, WT> {
    protected:
        blacs_context<IT> ctx, trans_ctx;

    public:
        explicit band_solver(const IT rows)
            : abstract_solver<DT, IT, WT>()
            , ctx(rows, 1, 'R')
            , trans_ctx(1, rows, 'R') {}
    };
} // namespace ezp::detail

#endif // BAND_SOLVER_HPP
