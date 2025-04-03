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

        template<band_container_t CT> auto to_band(CT&& custom) {
            if constexpr(has_mem<CT>) return band_mat<DT, IT>{custom.n_rows, custom.n_cols, custom.kl, custom.ku, custom.mem()};
            else if constexpr(has_memptr<CT>) return band_mat<DT, IT>{custom.n_rows, custom.n_cols, custom.kl, custom.ku, custom.memptr()};
            else if constexpr(has_data_method<CT>) return band_mat<DT, IT>{custom.n_rows, custom.n_cols, custom.kl, custom.ku, custom.data()};
            else if constexpr(has_iterator<CT>) return band_mat<DT, IT>{custom.n_rows, custom.n_cols, custom.kl, custom.ku, &(*custom.begin())};
            else static_assert(always_false_v<CT>, "invalid container type");
        }

        template<band_symm_container_t CT> auto to_band_symm(CT&& custom) {
            if constexpr(has_mem<CT>) return band_symm_mat<DT, IT>{custom.n_rows, custom.n_cols, custom.klu, custom.mem()};
            else if constexpr(has_memptr<CT>) return band_symm_mat<DT, IT>{custom.n_rows, custom.n_cols, custom.klu, custom.memptr()};
            else if constexpr(has_data_method<CT>) return band_symm_mat<DT, IT>{custom.n_rows, custom.n_cols, custom.klu, custom.data()};
            else if constexpr(has_iterator<CT>) return band_symm_mat<DT, IT>{custom.n_rows, custom.n_cols, custom.klu, &(*custom.begin())};
            else static_assert(always_false_v<CT>, "invalid container type");
        }

    public:
        explicit band_solver(const IT rows)
            : abstract_solver<DT, IT, WT>()
            , ctx(rows, 1, 'R')
            , trans_ctx(1, rows, 'R') {}
    };
} // namespace ezp::detail

#endif // BAND_SOLVER_HPP
