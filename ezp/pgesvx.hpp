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
 * @class pgesvx
 * @brief Solver for general full matrices (expert driver).
 * @author tlc
 * @date 12/03/2025
 * @version 1.0.0
 * @file pgesvx.hpp
 * @{
 */

#ifndef PGESVX_HPP
#define PGESVX_HPP

#include "abstract/full_solver.hpp"

namespace ezp {
    template<data_t DT, index_t IT, char FCT = 'E', char EQD = 'B', char ODER = 'R'> class pgesvx final : public detail::full_solver<DT, IT, ODER> {
        static constexpr char FACT = FCT;
        static constexpr char EQUED = EQD;
        static constexpr char TRANS = 'N';

        using base_t = detail::full_solver<DT, IT, ODER>;

    public:
        pgesvx()
            : base_t() {}

        pgesvx(const IT rows, const IT cols)
            : base_t(rows, cols) {}

        using base_t::solve;

        IT solve(full_mat<DT, IT>&& A, full_mat<DT, IT>&& B) override {
            if(!this->ctx.is_valid()) return 0;

            if(A.n_rows != A.n_cols || A.n_cols != B.n_rows) return -1;

            this->init_storage(A.n_rows);

            std::vector<DT> af(this->loc.a.size());

            this->ctx.scatter(A, this->ctx.desc_g(A.n_rows, A.n_cols), af, this->loc.desc_a);

            this->loc.b.resize(this->loc.rows * this->ctx.cols(B.n_cols, this->loc.block));

            const auto full_desc_b = this->ctx.desc_g(B.n_rows, B.n_cols);
            const auto loc_desc_b = this->ctx.desc_l(B.n_rows, B.n_cols, this->loc.block, this->loc.rows);

            this->ctx.scatter(B, full_desc_b, this->loc.b, loc_desc_b);

            auto equed = EQUED;

            std::vector<DT> r(this->loc.rows), c(this->loc.cols);
            std::vector<DT> x(this->loc.b.size());

            const auto ceil = [](const IT a, const IT b) { return (a + b - 1) / b; };
            const auto ceil_a = std::max(IT{1}, ceil(this->ctx.n_rows - 1, this->ctx.n_cols));
            const auto ceil_b = std::max(IT{1}, ceil(this->ctx.n_cols - 1, this->ctx.n_rows));

            const auto pgecon_lwork = 2 * (this->loc.rows + this->loc.cols) + std::max(IT{2}, std::max(this->loc.block * ceil_a, this->loc.cols + this->loc.block * ceil_b));
            const auto pgerfs_lwork = 3 * this->loc.rows;
            const auto lwork = this->loc.cols + std::max(pgecon_lwork, pgerfs_lwork);
            std::vector<DT> work(lwork);

            auto liwork = this->loc.cols;
            std::vector<IT> iwork(liwork);

            DT rcond, ferr, berr;

            IT info{-1};
            // ReSharper disable CppCStyleCast
            if constexpr(std::is_same_v<DT, double>) {
                using E = double;
                pdgesvx(&FACT, &TRANS, &this->loc.n, &B.n_cols, (E*)af.data(), &this->ONE, &this->ONE, this->loc.desc_a.data(), (E*)this->loc.a.data(), &this->ONE, &this->ONE, this->loc.desc_a.data(), this->loc.ipiv.data(), &equed, (E*)r.data(), (E*)c.data(), (E*)B.data, &this->ONE, &this->ONE, loc_desc_b.data(), (E*)x.data(), &this->ONE, &this->ONE, loc_desc_b.data(), (E*)&rcond, (E*)&ferr, (E*)&berr, (E*)work.data(), &lwork, iwork.data(), &liwork, &info);
            }
            else if constexpr(std::is_same_v<DT, float>) {
                using E = float;
                psgesvx(&FACT, &TRANS, &this->loc.n, &B.n_cols, (E*)af.data(), &this->ONE, &this->ONE, this->loc.desc_a.data(), (E*)this->loc.a.data(), &this->ONE, &this->ONE, this->loc.desc_a.data(), this->loc.ipiv.data(), &equed, (E*)r.data(), (E*)c.data(), (E*)B.data, &this->ONE, &this->ONE, loc_desc_b.data(), (E*)x.data(), &this->ONE, &this->ONE, loc_desc_b.data(), (E*)&rcond, (E*)&ferr, (E*)&berr, (E*)work.data(), &lwork, iwork.data(), &liwork, &info);
            }
            else if constexpr(std::is_same_v<DT, complex16>) {
                using E = complex16;
                pzgesvx(&FACT, &TRANS, &this->loc.n, &B.n_cols, (E*)af.data(), &this->ONE, &this->ONE, this->loc.desc_a.data(), (E*)this->loc.a.data(), &this->ONE, &this->ONE, this->loc.desc_a.data(), this->loc.ipiv.data(), &equed, (E*)r.data(), (E*)c.data(), (E*)B.data, &this->ONE, &this->ONE, loc_desc_b.data(), (E*)x.data(), &this->ONE, &this->ONE, loc_desc_b.data(), (E*)&rcond, (E*)&ferr, (E*)&berr, (E*)work.data(), &lwork, iwork.data(), &liwork, &info);
            }
            else if constexpr(std::is_same_v<DT, complex8>) {
                using E = complex8;
                pcgesvx(&FACT, &TRANS, &this->loc.n, &B.n_cols, (E*)af.data(), &this->ONE, &this->ONE, this->loc.desc_a.data(), (E*)this->loc.a.data(), &this->ONE, &this->ONE, this->loc.desc_a.data(), this->loc.ipiv.data(), &equed, (E*)r.data(), (E*)c.data(), (E*)B.data, &this->ONE, &this->ONE, loc_desc_b.data(), (E*)x.data(), &this->ONE, &this->ONE, loc_desc_b.data(), (E*)&rcond, (E*)&ferr, (E*)&berr, (E*)work.data(), &lwork, iwork.data(), &liwork, &info);
            }
            // ReSharper restore CppCStyleCast

            if((info = this->ctx.amx(info)) != 0) return info;

            return solve(std::move(B));
        }

        IT solve(full_mat<DT, IT>&& B) override {
            IT info{-1};
            return info;
        }
    };

    template<index_t IT, char FCT = 'E', char EQD = 'B', char ODER = 'R'> using par_dgesvx = pgesvx<double, IT, FCT, EQD, ODER>;
    template<index_t IT, char FCT = 'E', char EQD = 'B', char ODER = 'R'> using par_sgesvx = pgesvx<float, IT, FCT, EQD, ODER>;
    template<index_t IT, char FCT = 'E', char EQD = 'B', char ODER = 'R'> using par_zgesvx = pgesvx<complex16, IT, FCT, EQD, ODER>;
    template<index_t IT, char FCT = 'E', char EQD = 'B', char ODER = 'R'> using par_cgesvx = pgesvx<complex8, IT, FCT, EQD, ODER>;
} // namespace ezp

#endif // PGESVX_HPP

//! @}
