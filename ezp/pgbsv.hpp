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
 * @class pgbsv
 * @brief Solver for general band matrices.
 *
 * @note Although the `pgbsv` solver supports `KL=0` and/or `KU=0`, a zero (half) bandwidth
 * would lead to unwanted warning message from ScaLAPACK.
 * @note See: https://github.com/Reference-ScaLAPACK/scalapack/issues/116
 *
 * It solves the system of linear equations `A*X=B` with a general band matrix `A`.
 * The band matrix `A` has `KL` sub-diagonals and `KU` super-diagonals.
 * It shall be stored in the following format.
 * The band storage scheme is illustrated by the following example, when
 * `M=N=6`, `KL=2`, `KU=1`.
 *
 * ```
 *     .    .    .    .    .    .
 *     .    .    .    .    .    .
 *     .    .    .    .    .    .
 *     .   a12  a23  a34  a45  a56
 *    a11  a22  a33  a44  a55  a66
 *    a21  a32  a43  a54  a65   .
 *    a31  a42  a53  a64   .    .
 * ```
 *
 * The lead dimension should be `2*(KL+KU)+1`.
 *
 * With zero based indexing, for a general band matrix `A`, the element at row `i` and
 * column `j` is stored at `A[IDX(i, j)]`.
 *
 * @code
    const auto IDX = [&](const int i, const int j) {
        if(i - j > KL || j - i > KU) return -1;
        return 2 * KU + KL + i + 2 * j * (KL + KU);
    };
 * @endcode
 *
 * The example usage can be seen as follows.
 *
 * @include ../examples/example.pgbsv.cpp
 *
 * @author tlc
 * @date 07/03/2025
 * @version 1.0.0
 * @file pgbsv.hpp
 * @{
 */

#ifndef PGBSV_HPP
#define PGBSV_HPP

#include "abstract/band_solver.hpp"

namespace ezp {
    template<data_t DT, index_t IT> class pgbsv final : public detail::band_solver<DT, IT, band_mat<DT, IT>> {
        using base_t = detail::band_solver<DT, IT, band_mat<DT, IT>>;

        struct band_system {
            IT n{-1}, kl{-1}, ku{-1}, lead{-1}, block{-1}, lines{-1};
            desc<IT> desc1d_a;
            std::vector<DT> a, b, work;
            std::vector<IT> ipiv;
        };

        band_system loc;

        auto init_storage(const IT n, const IT kl, const IT ku) {
            loc.n = n;
            loc.kl = kl;
            loc.ku = ku;
            loc.lead = 2 * (loc.kl + loc.ku) + 1;
            loc.block = std::max(loc.n / std::max(IT{1}, this->ctx.n_rows - 1) + 1, std::max(loc.kl + loc.ku + 1, this->ctx.row_block(loc.n)));
            loc.block = std::min(loc.block, loc.n);
            loc.lines = this->ctx.rows(loc.n, loc.block);
            loc.desc1d_a = {501, this->trans_ctx.context, loc.n, loc.block, 0, loc.lead, 0, 0, 0};

            // see: https://github.com/Reference-ScaLAPACK/scalapack/issues/117
            loc.a.resize(loc.lead * loc.lines + loc.ku);
            loc.ipiv.resize(std::min(loc.n, loc.lines + loc.kl + loc.ku), -987654);
        }

        using base_t::to_band;
        using base_t::to_full;

    public:
        explicit pgbsv(const IT rows = get_env<IT>().size())
            : base_t(rows) {}

        class indexer {
            IT n, kl, ku;

        public:
            explicit indexer(const band_mat<DT, IT>& A)
                : n(A.n)
                , kl(A.kl)
                , ku(A.ku) {}

            indexer(const IT N, const IT KL, const IT KU)
                : n(N)
                , kl(KL)
                , ku(KU) {}

            auto operator()(const IT i, const IT j) const {
                if(i < 0 || i >= n || j < 0 || j >= n) return IT{-1};
                if(i - j > kl || j - i > ku) return IT{-1};
                return 2 * ku + kl + i + 2 * j * (kl + ku);
            }
        };

        template<band_container_t AT, full_container_t BT> IT solve(AT&& A, BT&& B) { return solve(to_band(A), to_full(B)); }

        IT solve(band_mat<DT, IT>&& A, full_mat<DT, IT>&& B) {
            if(!this->ctx.is_valid() || !this->trans_ctx.is_valid()) return 0;

            if(A.n_rows != A.n_cols || A.n_cols != B.n_rows) return -1;

            init_storage(A.n_cols, A.kl, A.ku);

            // pretend that A is a full matrix of size (2*(kl+ku)+1) x n
            // redistribute A to the process grid
            this->trans_ctx.scatter(
                full_mat<DT, IT>(loc.lead, loc.n, A.data, A.distributed),
                this->trans_ctx.desc_g(loc.lead, loc.n),
                loc.a,
                this->trans_ctx.desc_l(loc.lead, loc.n, loc.lead, loc.block, loc.lead)
            );

            const IT laf = (loc.block + 6 * loc.kl + 13 * loc.ku) * (loc.kl + loc.ku);
            const IT lwork = std::max(B.n_cols * (loc.block + 2 * loc.kl + 4 * loc.ku), IT{1});
            loc.work.resize(laf + lwork);

            IT info{-1};
            // ReSharper disable CppCStyleCast
            if constexpr(std::is_same_v<DT, double>) {
                using E = double;
                pdgbtrf(&loc.n, &loc.kl, &loc.ku, (E*)loc.a.data(), &this->ONE, loc.desc1d_a.data(), loc.ipiv.data(), (E*)loc.work.data(), &laf, (E*)(loc.work.data() + laf), &lwork, &info);
            }
            else if constexpr(std::is_same_v<DT, float>) {
                using E = float;
                psgbtrf(&loc.n, &loc.kl, &loc.ku, (E*)loc.a.data(), &this->ONE, loc.desc1d_a.data(), loc.ipiv.data(), (E*)loc.work.data(), &laf, (E*)(loc.work.data() + laf), &lwork, &info);
            }
            else if constexpr(std::is_same_v<DT, complex16>) {
                using E = complex16;
                pzgbtrf(&loc.n, &loc.kl, &loc.ku, (E*)loc.a.data(), &this->ONE, loc.desc1d_a.data(), loc.ipiv.data(), (E*)loc.work.data(), &laf, (E*)(loc.work.data() + laf), &lwork, &info);
            }
            else if constexpr(std::is_same_v<DT, complex8>) {
                using E = complex8;
                pcgbtrf(&loc.n, &loc.kl, &loc.ku, (E*)loc.a.data(), &this->ONE, loc.desc1d_a.data(), loc.ipiv.data(), (E*)loc.work.data(), &laf, (E*)(loc.work.data() + laf), &lwork, &info);
            }
            // ReSharper restore CppCStyleCast

            if((info = this->trans_ctx.amx(info)) != 0) return info;

            return solve(std::move(B));
        }

        IT solve(full_mat<DT, IT>&& B) {
            static constexpr char TRANS = 'N';

            if(B.n_rows != loc.n) return -1;

            if(!this->ctx.is_valid() || !this->trans_ctx.is_valid()) return 0;

            const auto lead_b = std::max(loc.block, loc.lines);

            loc.b.resize(lead_b * B.n_cols);

            const auto full_desc_b = this->ctx.desc_g(B.n_rows, B.n_cols);
            const auto local_desc_b = this->ctx.desc_l(B.n_rows, B.n_cols, loc.block, B.n_cols, lead_b);

            this->ctx.scatter(B, full_desc_b, loc.b, local_desc_b);

            const IT laf = (loc.block + 6 * loc.kl + 13 * loc.ku) * (loc.kl + loc.ku);
            const IT lwork = std::max(B.n_cols * (loc.block + 2 * loc.kl + 4 * loc.ku), IT{1});
            loc.work.resize(laf + lwork);

            desc<IT> desc1d_b{502, this->trans_ctx.context, loc.n, loc.block, 0, lead_b, 0, 0, 0};

            IT info{-1};
            // ReSharper disable CppCStyleCast
            if constexpr(std::is_same_v<DT, double>) {
                using E = double;
                pdgbtrs(&TRANS, &loc.n, &loc.kl, &loc.ku, &B.n_cols, (E*)loc.a.data(), &this->ONE, loc.desc1d_a.data(), loc.ipiv.data(), (E*)loc.b.data(), &this->ONE, desc1d_b.data(), (E*)loc.work.data(), &laf, (E*)(loc.work.data() + laf), &lwork, &info);
            }
            else if constexpr(std::is_same_v<DT, float>) {
                using E = float;
                psgbtrs(&TRANS, &loc.n, &loc.kl, &loc.ku, &B.n_cols, (E*)loc.a.data(), &this->ONE, loc.desc1d_a.data(), loc.ipiv.data(), (E*)loc.b.data(), &this->ONE, desc1d_b.data(), (E*)loc.work.data(), &laf, (E*)(loc.work.data() + laf), &lwork, &info);
            }
            else if constexpr(std::is_same_v<DT, complex16>) {
                using E = complex16;
                pzgbtrs(&TRANS, &loc.n, &loc.kl, &loc.ku, &B.n_cols, (E*)loc.a.data(), &this->ONE, loc.desc1d_a.data(), loc.ipiv.data(), (E*)loc.b.data(), &this->ONE, desc1d_b.data(), (E*)loc.work.data(), &laf, (E*)(loc.work.data() + laf), &lwork, &info);
            }
            else if constexpr(std::is_same_v<DT, complex8>) {
                using E = complex8;
                pcgbtrs(&TRANS, &loc.n, &loc.kl, &loc.ku, &B.n_cols, (E*)loc.a.data(), &this->ONE, loc.desc1d_a.data(), loc.ipiv.data(), (E*)loc.b.data(), &this->ONE, desc1d_b.data(), (E*)loc.work.data(), &laf, (E*)(loc.work.data() + laf), &lwork, &info);
            }
            // ReSharper restore CppCStyleCast

            if((info = this->trans_ctx.amx(info)) == 0) this->ctx.gather(loc.b, local_desc_b, B, full_desc_b);

            return info;
        }
    };

    template<index_t IT> using par_dgbsv = pgbsv<double, IT>;
    template<index_t IT> using par_sgbsv = pgbsv<float, IT>;
    template<index_t IT> using par_zgbsv = pgbsv<complex16, IT>;
    template<index_t IT> using par_cgbsv = pgbsv<complex8, IT>;
} // namespace ezp

#endif // PGBSV_HPP

//! @}
