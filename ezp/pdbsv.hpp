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
 * @class pdbsv
 * @brief Solver for general band matrices.
 *
 * @note Although the `pdbsv` solver supports `KL=0` and/or `KU=0`, a zero (half) bandwidth
 * would lead to unwanted warning message from ScaLAPACK.
 * @note See: https://github.com/Reference-ScaLAPACK/scalapack/issues/116
 *
 * It solves the system of linear equations `A * X = B` with a general band matrix `A`.
 * The band matrix `A` has `KL` sub-diagonals and `KU` super-diagonals.
 * It shall be stored in the following format.
 * The band storage scheme is illustrated by the following example, when
 * `M=N=6`, `KL=2`, `KU=1`.
 *
 * ```
 *     .   a12  a23  a34  a45  a56
 *    a11  a22  a33  a44  a55  a66
 *    a21  a32  a43  a54  a65   .
 *    a31  a42  a53  a64   .    .
 * ```
 *
 * The lead dimension should be `(KL+KU+1)`.
 *
 * With zero based indexing, for a general band matrix `A`, the element at row `i` and
 * column `j` is stored at `A[IDX(i, j)]`.
 *
 * @code
   const auto IDX = [&](const int i, const int j) {
       if(i - j > KL || j - i > KU) return -1;
       return KU + i - j + j * (KL + KU + 1);
   };
 * @endcode
 *
 * The example usage can be seen as follows.
 *
 * @include example.pdbsv.cpp
 *
 * @author tlc
 * @date 07/03/2025
 * @version 1.0.0
 * @file pdbsv.hpp
 * @{
 */

#ifndef PDBSV_HPP
#define PDBSV_HPP

#include "abstract/band_solver.hpp"

namespace ezp {
    template<data_t DT, index_t IT> class pdbsv final : public detail::band_solver<DT, IT, band_mat<DT, IT>> {
        using base_t = detail::band_solver<DT, IT, band_mat<DT, IT>>;

        struct band_system {
            IT n{-1}, kl{-1}, ku{-1}, max_klu{-1}, lead{-1}, block{-1}, lines{-1};
            desc<IT> desc1d_a;
            std::vector<DT> a, b, work;
        };

        band_system loc;

        /**
         * @brief Initialize the storage for a matrix with given dimensions and bandwidth.
         *
         * @tparam IT Integer type for matrix dimensions and bandwidth.
         * @param n Number of rows/columns in the matrix.
         * @param kl Number of sub-diagonals (lower bandwidth).
         * @param ku Number of super-diagonals (upper bandwidth).
         *
         * This function sets up the storage parameters for a band matrix, including the leading dimension,
         * block size, and the number of lines. It also resizes the storage vector to accommodate the matrix data.
         */
        auto init_storage(const IT n, const IT kl, const IT ku) {
            loc.n = n;
            loc.kl = kl;
            loc.ku = ku;
            loc.max_klu = std::max(loc.kl, loc.ku);
            loc.lead = loc.kl + loc.ku + 1;
            loc.block = std::max(loc.n / std::max(IT{1}, this->ctx.n_rows - 1) + 1, std::max(2 * loc.max_klu, this->ctx.row_block(loc.n)));
            loc.block = std::min(loc.block, loc.n);
            loc.lines = this->ctx.rows(loc.n, loc.block);
            loc.desc1d_a = {501, this->trans_ctx.context, loc.n, loc.block, 0, loc.lead, 0, 0, 0};

            loc.a.resize(loc.lead * loc.lines);
        }

        using base_t::to_band;
        using base_t::to_full;

    public:
        explicit pdbsv(const IT rows = get_env<IT>().size())
            : base_t(rows) {}

        class indexer {
            IT n, kl, ku;

        public:
            explicit indexer(const band_mat<DT, IT>& A)
                : n(A.n_rows)
                , kl(A.kl)
                , ku(A.ku) {}

            indexer(const IT N, const IT KL, const IT KU)
                : n(N)
                , kl(KL)
                , ku(KU) {}

            auto operator()(const IT i, const IT j) const {
                if(i < 0 || i >= n || j < 0 || j >= n) return IT{-1};
                if(i - j > kl || j - i > ku) return IT{-1};
                return ku + i + j * (kl + ku);
            }
        };

        template<band_container_t AT, full_container_t BT> IT solve(AT&& A, BT&& B) { return solve(to_band(std::forward<AT>(A)), to_full(std::forward<BT>(B))); }
        template<band_container_t AT> IT solve(AT&& A, full_mat<DT, IT>&& B) { return solve(to_band(std::forward<AT>(A)), to_full(std::forward<full_mat<DT, IT>>(B))); }

        IT solve(band_mat<DT, IT>&& A, full_mat<DT, IT>&& B) override {
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

            const IT laf = loc.block * (loc.kl + loc.ku) + 6 * loc.max_klu * loc.max_klu;
            const IT lwork = loc.max_klu * std::max(2 * B.n_cols - 1, loc.max_klu);
            loc.work.resize(laf + lwork);

            IT info{-1};
            // ReSharper disable CppCStyleCast
            if constexpr(std::is_same_v<DT, double>) {
                using E = double;
                pddbtrf(&loc.n, &loc.kl, &loc.ku, (E*)loc.a.data(), &this->ONE, loc.desc1d_a.data(), (E*)loc.work.data(), &laf, (E*)(loc.work.data() + laf), &lwork, &info);
            }
            else if constexpr(std::is_same_v<DT, float>) {
                using E = float;
                psdbtrf(&loc.n, &loc.kl, &loc.ku, (E*)loc.a.data(), &this->ONE, loc.desc1d_a.data(), (E*)loc.work.data(), &laf, (E*)(loc.work.data() + laf), &lwork, &info);
            }
            else if constexpr(std::is_same_v<DT, complex16>) {
                using E = complex16;
                pzdbtrf(&loc.n, &loc.kl, &loc.ku, (E*)loc.a.data(), &this->ONE, loc.desc1d_a.data(), (E*)loc.work.data(), &laf, (E*)(loc.work.data() + laf), &lwork, &info);
            }
            else if constexpr(std::is_same_v<DT, complex8>) {
                using E = complex8;
                pcdbtrf(&loc.n, &loc.kl, &loc.ku, (E*)loc.a.data(), &this->ONE, loc.desc1d_a.data(), (E*)loc.work.data(), &laf, (E*)(loc.work.data() + laf), &lwork, &info);
            }
            // ReSharper restore CppCStyleCast

            if((info = this->trans_ctx.amx(info)) != 0) return info;

            return solve(std::move(B));
        }

        IT solve(full_mat<DT, IT>&& B) override {
            static constexpr char TRANS = 'N';

            if(B.n_rows != loc.n) return -1;

            if(!this->ctx.is_valid() || !this->trans_ctx.is_valid()) return 0;

            const auto lead_b = std::max(loc.block, loc.lines);

            loc.b.resize(lead_b * B.n_cols);

            const auto full_desc_b = this->ctx.desc_g(B.n_rows, B.n_cols);
            const auto local_desc_b = this->ctx.desc_l(B.n_rows, B.n_cols, loc.block, B.n_cols, lead_b);

            this->ctx.scatter(B, full_desc_b, loc.b, local_desc_b);

            const IT laf = loc.block * (loc.kl + loc.ku) + 6 * loc.max_klu * loc.max_klu;
            const IT lwork = loc.max_klu * std::max(2 * B.n_cols - 1, loc.max_klu);
            loc.work.resize(laf + lwork);

            desc<IT> desc1d_b{502, this->trans_ctx.context, loc.n, loc.block, 0, lead_b, 0, 0, 0};

            IT info{-1};
            // ReSharper disable CppCStyleCast
            if constexpr(std::is_same_v<DT, double>) {
                using E = double;
                pddbtrs(&TRANS, &loc.n, &loc.kl, &loc.ku, &B.n_cols, (E*)loc.a.data(), &this->ONE, loc.desc1d_a.data(), (E*)loc.b.data(), &this->ONE, desc1d_b.data(), (E*)loc.work.data(), &laf, (E*)(loc.work.data() + laf), &lwork, &info);
            }
            else if constexpr(std::is_same_v<DT, float>) {
                using E = float;
                psdbtrs(&TRANS, &loc.n, &loc.kl, &loc.ku, &B.n_cols, (E*)loc.a.data(), &this->ONE, loc.desc1d_a.data(), (E*)loc.b.data(), &this->ONE, desc1d_b.data(), (E*)loc.work.data(), &laf, (E*)(loc.work.data() + laf), &lwork, &info);
            }
            else if constexpr(std::is_same_v<DT, complex16>) {
                using E = complex16;
                pzdbtrs(&TRANS, &loc.n, &loc.kl, &loc.ku, &B.n_cols, (E*)loc.a.data(), &this->ONE, loc.desc1d_a.data(), (E*)loc.b.data(), &this->ONE, desc1d_b.data(), (E*)loc.work.data(), &laf, (E*)(loc.work.data() + laf), &lwork, &info);
            }
            else if constexpr(std::is_same_v<DT, complex8>) {
                using E = complex8;
                pcdbtrs(&TRANS, &loc.n, &loc.kl, &loc.ku, &B.n_cols, (E*)loc.a.data(), &this->ONE, loc.desc1d_a.data(), (E*)loc.b.data(), &this->ONE, desc1d_b.data(), (E*)loc.work.data(), &laf, (E*)(loc.work.data() + laf), &lwork, &info);
            }
            // ReSharper restore CppCStyleCast

            if((info = this->trans_ctx.amx(info)) == 0) this->ctx.gather(loc.b, local_desc_b, B, full_desc_b);

            return info;
        }
    };

    template<index_t IT> using par_ddbsv = pdbsv<double, IT>;
    template<index_t IT> using par_sdbsv = pdbsv<float, IT>;
    template<index_t IT> using par_zdbsv = pdbsv<complex16, IT>;
    template<index_t IT> using par_cdbsv = pdbsv<complex8, IT>;
} // namespace ezp

#endif // PDBSV_HPP

//! @}
