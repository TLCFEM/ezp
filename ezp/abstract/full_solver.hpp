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

#ifndef FULL_SOLVER_HPP
#define FULL_SOLVER_HPP

#include "abstract_solver.hpp"

namespace ezp::detail {
    template<data_t DT, index_t IT, char ODER = 'R'> class full_solver : public abstract_solver<DT, IT, full_mat<DT, IT>> {
        using base_t = abstract_solver<DT, IT, full_mat<DT, IT>>;

        struct full_system {
            IT n{-1}, block{-1}, rows{-1}, cols{-1};
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
            loc.cols = ctx.cols(loc.n, loc.block);
            loc.desc_a = ctx.desc_l(loc.n, loc.n, loc.block, loc.rows);

            loc.a.resize(loc.rows * loc.cols);
            loc.ipiv.resize(loc.rows + loc.block);
        }

        auto gather_pivot() {
            const auto ipiv_l = ctx.desc_l(loc.n, 1, loc.block, loc.rows);
            const auto ipiv_g = ctx.desc_g(loc.n, 1);

            std::vector<IT> ipiv;
            if(0 == ctx.rank) ipiv.resize(loc.n);

            ctx.copy_to(loc.ipiv.data(), ipiv_l.data(), ipiv.data(), ipiv_g.data());

            return ipiv;
        }

        using base_t::to_full;

    public:
        full_solver()
            : base_t()
            , ctx(ODER) {}

        full_solver(const IT rows, const IT cols)
            : base_t()
            , ctx(rows, cols, ODER) {}

        class indexer {
            IT n, m;

        public:
            explicit indexer(const full_mat<DT, IT>& A)
                : n(A.n_rows)
                , m(A.n_cols) {}

            explicit indexer(const IT N)
                : indexer(N, N) {}

            indexer(const IT N, const IT M)
                : n(N)
                , m(M) {}

            auto operator()(const IT i, const IT j) const {
                if(i < 0 || i >= n || j < 0 || j >= n) return IT{-1};
                return i + j * n;
            }
        };

        using base_t::solve;

        template<full_container_t AT, full_container_t BT> IT solve(AT&& A, BT&& B) { return solve(to_full(std::forward<AT>(A)), to_full(std::forward<BT>(B))); }
        template<full_container_t AT> IT solve(AT&& A, full_mat<DT, IT>&& B) { return solve(to_full(std::forward<AT>(A)), std::move(B)); }
        template<full_container_t BT> IT solve(full_mat<DT, IT>&& A, BT&& B) { return solve(std::move(A), to_full(std::forward<BT>(B))); }
    };
} // namespace ezp::detail

#endif // FULL_SOLVER_HPP
