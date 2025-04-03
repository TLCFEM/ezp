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

#ifndef ABSTRACT_SOLVER_HPP
#define ABSTRACT_SOLVER_HPP

#include "traits.hpp"

namespace ezp::detail {
    template<data_t DT, index_t IT, wrapper_t WT> class abstract_solver {
        using wrapper_type = WT;

    protected:
        static constexpr IT ZERO{0}, ONE{1};

        template<full_container_t CT> auto to_full(CT&& custom) {
            if constexpr(has_mem<CT>) return full_mat<DT, IT>{custom.n_rows, custom.n_cols, custom.mem()};
            else if constexpr(has_memptr<CT>) return full_mat<DT, IT>{custom.n_rows, custom.n_cols, custom.memptr()};
            else if constexpr(has_data_method<CT>) return full_mat<DT, IT>{custom.n_rows, custom.n_cols, custom.data()};
            else if constexpr(has_iterator<CT>) return full_mat<DT, IT>{custom.n_rows, custom.n_cols, &(*custom.begin())};
            else static_assert(always_false_v<CT>, "invalid container type");
        }

    public:
        abstract_solver() = default;

        virtual ~abstract_solver() = default;

        template<full_container_t CT> IT solve(CT&& B) { return solve(to_full(B)); }

        virtual IT solve(WT&&, full_mat<DT, IT>&&) = 0;

        virtual IT solve(full_mat<DT, IT>&&) = 0;
    };
} // namespace ezp::detail

#endif // ABSTRACT_SOLVER_HPP
