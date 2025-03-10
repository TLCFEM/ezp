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

#ifndef FULL_SOLVER_HPP
#define FULL_SOLVER_HPP

#include "abstract_solver.hpp"

namespace ezp::detail {
    template<data_t DT, index_t IT, char ODER = 'R'> class full_solver : public abstract_solver<IT> {
        struct full_system {
            IT n{-1}, block{-1}, rows{-1};
            desc<IT> desc_a;
            std::vector<DT> a, b;
            std::vector<IT> ipiv;
        };

    protected:
        full_system loc;

        blacs_context<IT> ctx;

        auto init_storage(const IT n) {
            loc.n = n;
            loc.block = std::max(IT{1}, static_cast<IT>(std::sqrt(ctx.row_block(loc.n) * ctx.col_block(loc.n))));
            loc.rows = ctx.rows(loc.n, loc.block);
            loc.desc_a = ctx.desc_l(loc.n, loc.n, loc.block, loc.rows);

            loc.a.resize(loc.rows * ctx.cols(loc.n, loc.block));
            loc.ipiv.resize(loc.rows + loc.block);
        }

    public:
        full_solver()
            : abstract_solver<IT>()
            , ctx(ODER) {}

        full_solver(const IT rows, const IT cols)
            : abstract_solver<IT>()
            , ctx(rows, cols, ODER) {}

        class indexer {
            const IT n;

        public:
            explicit indexer(const full_mat<DT, IT>& A)
                : n(A.n) {}

            explicit indexer(const IT N)
                : n(N) {}

            auto operator()(const IT i, const IT j) const {
                if(i < 0 || i >= n || j < 0 || j >= n) return IT{-1};
                return i + j * n;
            }
        };

        template<container_t CT> IT solve(full_mat<CT, IT>&& B) {
            if constexpr(has_mem<CT>) return solve({B.n_rows, B.n_cols, B.mem()});
            if constexpr(has_memptr<CT>) return solve({B.n_rows, B.n_cols, B.memptr()});
            if constexpr(has_data<CT>) return solve({B.n_rows, B.n_cols, B.data()});
            if constexpr(has_iterator<CT>) return solve({B.n_rows, B.n_cols, B.begin()});

            // should never reach here
            static_assert(always_false_v<CT>, "invalid container type");
        }

        virtual IT solve(full_mat<DT, IT>&&, full_mat<DT, IT>&&) = 0;
        virtual IT solve(full_mat<DT, IT>&&) = 0;
    };
} // namespace ezp::detail

#endif // FULL_SOLVER_HPP