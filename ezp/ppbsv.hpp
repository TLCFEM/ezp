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
 * @class ppbsv
 * @brief Solver for symmetric band positive definite matrices.
 *
 * @note Although the `ppbsv` solver supports KLU, a zero (half) bandwidth
 * would lead to unwanted warning message from ScaLAPACK.
 * @note See: https://github.com/Reference-ScaLAPACK/scalapack/issues/116
 *
 * It solves the system of linear equations `A*X=B` with a symmetric band positive definite matrix `A`.
 * The band matrix `A` has `KLU` sub-diagonals.
 * It shall be stored in the following format.
 * The band storage scheme is illustrated by the following example, when
 * `M=N=6`, `KLU=2`.
 *
 * For `UPLO='L'`, the lower half is stored.
 *
 * ```
 *    a11  a22  a33  a44  a55  a66
 *    a21  a32  a43  a54  a65   .
 *    a31  a42  a53  a64   .    .
 * ```
 *
 * The lead dimension should be `KLU+1`.
 *
 * With zero based indexing, for a general band matrix `A`, the element at row `i` and
 * column `j` is stored at `A[IDX(i, j)]`.
 *
 * @code
    const auto IDX = [&](int i, int j) {
        if(i > j) std::swap(i, j);
        if(j - i > KLU) return -1;
        return 2 * j - i + (j + 1) * KLU;
    };
 * @endcode
 *
 * For `UPLO='U'`, the upper half is stored.
 *
 * ```
 *     .    .   a13  a24  a35  a46
 *     .   a12  a23  a34  a45  a56
 *    a11  a22  a33  a44  a55  a66
 * ```
 *
 * The lead dimension should be `KLU+1`.
 *
 * With zero based indexing, for a general band matrix `A`, the element at row `i` and
 * column `j` is stored at `A[IDX(i, j)]`.
 *
 * @code
    const auto IDX = [&](int i, int j) {
        if(i < j) std::swap(i, j);
        if(i - j > KLU) return -1;
        return i + j * KLU;
    };
 * @endcode
 *
 * @author tlc
 * @date 07/03/2025
 * @version 1.0.0
 * @file ppbsv.hpp
 * @{
 */

#ifndef PPBSV_HPP
#define PPBSV_HPP

#include "abstract/band_solver.hpp"

namespace ezp {
    template<data_t DT, index_t IT, char UL = 'L'> class ppbsv final : public detail::band_solver<IT> {
        static constexpr char UPLO = UL;

        struct band_system {
            IT n{-1}, klu{-1}, lead{-1}, block{-1}, lines{-1};
            desc<IT> desc1d_a;
            std::vector<DT> a, b, work;
        };

        band_system loc;

        auto init_storage(const IT n, const IT klu) {
            loc.n = n;
            loc.klu = klu;
            loc.lead = loc.klu + 1;
            loc.block = std::max(loc.n / std::max(1, this->ctx.n_rows - 1) + 1, std::max(2 * loc.klu, this->ctx.row_block(loc.n)));
            loc.block = std::min(loc.block, loc.n);
            loc.lines = this->ctx.rows(loc.n, loc.block);
            loc.desc1d_a = {501, this->trans_ctx.context, loc.n, loc.block, 0, loc.lead, 0, 0, 0};

            loc.a.resize(loc.lead * loc.lines);
        }

    public:
        explicit ppbsv(const IT rows)
            : detail::band_solver<IT>(rows) {}

        class indexer {
            const IT n, klu;

        public:
            indexer(const band_mat<DT, IT>& A)
                : n(A.n)
                , klu(A.klu) {}

            indexer(const IT N, const IT KLU)
                : n(N)
                , klu(KLU) {}

            auto operator()(IT i, IT j) const {
                if(i < 0 || i >= n || j < 0 || j >= n) return -1;
                if('L' == UL) {
                    if(i < j) std::swap(i, j);
                    if(i - j > klu) return -1;
                    return i + j * klu;
                }
                else {
                    if(i > j) std::swap(i, j);
                    if(j - i > klu) return -1;
                    return 2 * j - i + (j + 1) * klu;
                }
            }
        };

        IT solve(band_symm_mat<DT, IT>&& A, full_mat<DT, IT>&& B) {
            if(!this->ctx.is_valid() || !this->trans_ctx.is_valid()) return 0;

            init_storage(A.n_cols, A.klu);

            // pretend that A is a full matrix of size (2*(kl+ku)+1) x n
            // redistribute A to the process grid
            this->trans_ctx.scatter(
                full_mat<DT, IT>(loc.lead, loc.n, A.data, A.distributed),
                this->trans_ctx.desc_g(loc.lead, loc.n),
                loc.a,
                this->trans_ctx.desc_l(loc.lead, loc.n, loc.lead, loc.block, loc.lead)
            );

            const IT laf = (loc.block + 2 * loc.klu) * loc.klu;
            const IT lwork = loc.klu * std::max(B.n_cols, loc.klu);
            loc.work.resize(laf + lwork);

            IT info;
            // ReSharper disable CppCStyleCast
            if(std::is_same_v<DT, double>) {
                using E = double;
                pdpbtrf(&UPLO, &loc.n, &loc.klu, (E*)loc.a.data(), &this->ONE, loc.desc1d_a.data(), (E*)loc.work.data(), &laf, (E*)(loc.work.data() + laf), &lwork, &info);
            }
            else {
                using E = float;
                pspbtrf(&UPLO, &loc.n, &loc.klu, (E*)loc.a.data(), &this->ONE, loc.desc1d_a.data(), (E*)loc.work.data(), &laf, (E*)(loc.work.data() + laf), &lwork, &info);
            }
            // ReSharper restore CppCStyleCast

            if((info = this->trans_ctx.amx(info)) != 0) return info;

            return solve(std::move(B));
        }

        IT solve(full_mat<DT, IT>&& B) {
            if(B.n_rows != loc.n) return -1;

            if(!this->ctx.is_valid() || !this->trans_ctx.is_valid()) return 0;

            const auto lead_b = std::max(loc.block, loc.lines);

            loc.b.resize(lead_b * B.n_cols);

            const auto full_desc_b = this->ctx.desc_g(B.n_rows, B.n_cols);
            const auto local_desc_b = this->ctx.desc_l(B.n_rows, B.n_cols, loc.block, B.n_cols, lead_b);

            this->ctx.scatter(B, full_desc_b, loc.b, local_desc_b);

            const IT laf = (loc.block + 2 * loc.klu) * loc.klu;
            const IT lwork = loc.klu * std::max(B.n_cols, loc.klu);
            loc.work.resize(laf + lwork);

            desc<IT> desc1d_b{502, this->trans_ctx.context, loc.n, loc.block, 0, lead_b, 0, 0, 0};

            IT info;
            // ReSharper disable CppCStyleCast
            if(std::is_same_v<DT, double>) {
                using E = double;
                pdpbtrs(&UPLO, &loc.n, &loc.klu, &B.n_cols, (E*)loc.a.data(), &this->ONE, loc.desc1d_a.data(), (E*)loc.b.data(), &this->ONE, desc1d_b.data(), (E*)loc.work.data(), &laf, (E*)(loc.work.data() + laf), &lwork, &info);
            }
            else {
                using E = float;
                pspbtrs(&UPLO, &loc.n, &loc.klu, &B.n_cols, (E*)loc.a.data(), &this->ONE, loc.desc1d_a.data(), (E*)loc.b.data(), &this->ONE, desc1d_b.data(), (E*)loc.work.data(), &laf, (E*)(loc.work.data() + laf), &lwork, &info);
            }
            // ReSharper restore CppCStyleCast

            info = this->trans_ctx.amx(info);

            if(0 == info) this->ctx.gather(loc.b, local_desc_b, B, full_desc_b);

            return info;
        }
    };

    template<index_t IT, char UL = 'L'> using par_dpbsv = ppbsv<double, IT, UL>;
    template<index_t IT, char UL = 'L'> using par_spbsv = ppbsv<float, IT, UL>;
    template<index_t IT> using par_dpbsv_u = ppbsv<double, IT, 'U'>;
    template<index_t IT> using par_spbsv_u = ppbsv<float, IT, 'U'>;
} // namespace ezp

#endif // PPBSV_HPP

//! @}
