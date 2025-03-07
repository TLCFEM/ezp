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
 * @brief General band matrix.
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
    template<data_t DT, index_t IT, char ODER = 'R'> class pgbsv final : public band_solver<IT, ODER> {
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
            loc.block = std::max(loc.n / std::max(1, this->ctx.n_rows - 1) + 1, std::max(loc.kl + loc.ku + 1, this->ctx.row_block(loc.n)));
            loc.block = std::min(loc.block, loc.n);
            loc.lines = this->ctx.rows(loc.n, loc.block);
            loc.desc1d_a = {501, this->trans_ctx.context, loc.n, loc.block, 0, loc.lead, 0, 0, 0};

            // see: https://github.com/Reference-ScaLAPACK/scalapack/issues/117
            loc.a.resize(loc.lead * loc.lines + loc.ku);
            loc.ipiv.resize(std::min(loc.n, loc.lines + loc.kl + loc.ku), -987654);
        }

    public:
        explicit pgbsv(const IT rows)
            : band_solver<IT, ODER>(rows) {}

        IT solve(band_mat<DT, IT>&& A, full_mat<DT, IT>&& B) {
            if(!this->ctx.is_valid() || !this->trans_ctx.is_valid()) return 0;

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
            const IT lwork = std::max(B.n_cols * (loc.block + 2 * loc.kl + 4 * loc.ku), 1);
            loc.work.resize(laf + lwork);

            IT info;
            // ReSharper disable CppCStyleCast
            if(std::is_same_v<DT, double>) {
                using E = double;
                pdgbtrf(&loc.n, &loc.kl, &loc.ku, (E*)loc.a.data(), &this->ONE, loc.desc1d_a.data(), loc.ipiv.data(), (E*)loc.work.data(), &laf, (E*)(loc.work.data() + laf), &lwork, &info);
            }
            else {
                using E = float;
                psgbtrf(&loc.n, &loc.kl, &loc.ku, (E*)loc.a.data(), &this->ONE, loc.desc1d_a.data(), loc.ipiv.data(), (E*)loc.work.data(), &laf, (E*)(loc.work.data() + laf), &lwork, &info);
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
            const IT lwork = std::max(B.n_cols * (loc.block + 2 * loc.kl + 4 * loc.ku), 1);
            loc.work.resize(laf + lwork);

            desc<IT> desc1d_b{502, this->trans_ctx.context, loc.n, loc.block, 0, lead_b, 0, 0, 0};

            IT info;
            // ReSharper disable CppCStyleCast
            if(std::is_same_v<DT, double>) {
                using E = double;
                pdgbtrs(&TRANS, &loc.n, &loc.kl, &loc.ku, &B.n_cols, (E*)loc.a.data(), &this->ONE, loc.desc1d_a.data(), loc.ipiv.data(), (E*)loc.b.data(), &this->ONE, desc1d_b.data(), (E*)loc.work.data(), &laf, (E*)(loc.work.data() + laf), &lwork, &info);
            }
            else {
                using E = float;
                psgbtrs(&TRANS, &loc.n, &loc.kl, &loc.ku, &B.n_cols, (E*)loc.a.data(), &this->ONE, loc.desc1d_a.data(), loc.ipiv.data(), (E*)loc.b.data(), &this->ONE, desc1d_b.data(), (E*)loc.work.data(), &laf, (E*)(loc.work.data() + laf), &lwork, &info);
            }
            // ReSharper restore CppCStyleCast

            info = this->trans_ctx.amx(info);

            if(0 == info) this->ctx.gather(loc.b, local_desc_b, B, full_desc_b);

            return info;
        }
    };

    template<index_t IT, char ODER = 'R'> using par_dgbsv = pgbsv<double, IT, ODER>;
    template<index_t IT, char ODER = 'R'> using par_sgbsv = pgbsv<float, IT, ODER>;
    template<index_t IT> using par_dgbsv_c = pgbsv<double, IT, 'C'>;
    template<index_t IT> using par_sgbsv_c = pgbsv<float, IT, 'C'>;
} // namespace ezp

#endif // PGBSV_HPP

//! @}
