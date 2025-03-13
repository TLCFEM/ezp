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
 * @class pposvx
 * @brief Solver for general full matrices (expert driver).
 * @author tlc
 * @date 12/03/2025
 * @version 1.0.0
 * @file pposvx.hpp
 * @{
 */

#ifndef PPOSVX_HPP
#define PPOSVX_HPP

#include "abstract/full_solver.hpp"

#include <numeric>

namespace ezp {
    template<data_t DT, index_t IT, char UL = 'L', char ODER = 'R'> class pposvx final : public detail::full_solver<DT, IT, ODER> {
        static constexpr char FACT = 'E';
        static constexpr char UPLO = UL;

        using base_t = detail::full_solver<DT, IT, ODER>;

        static auto ceil(const IT a, const IT b) { return (a + b - 1) / b; }

        auto compute_lwork() {
            const auto ceil_a = std::max(IT{1}, ceil(this->ctx.n_rows - 1, this->ctx.n_cols));
            const auto ceil_b = std::max(IT{1}, ceil(this->ctx.n_cols - 1, this->ctx.n_rows));
            const auto ppocon_lwork = 2 * (this->loc.rows + this->loc.cols) + std::max(IT{2}, std::max(this->loc.block * ceil_a, this->loc.cols + this->loc.block * ceil_b));

            const auto pporfs_lwork = 3 * this->loc.rows;

            return std::max(ppocon_lwork, pporfs_lwork);
        }

        auto compute_lrwork() {
            const auto lcmp = std::lcm(this->ctx.n_rows, this->ctx.n_cols) / this->ctx.n_rows;

            return this->loc.rows + 2 * this->loc.cols + this->loc.block * ceil(ceil(this->loc.rows, this->loc.block), lcmp);
        }

        struct expert_system {
            IT lwork;
            std::vector<DT> af, work;
            std::vector<work_t<DT>> sr, sc;
        };

        expert_system exp;

        auto init_expert_storage() {
            exp.lwork = compute_lwork();
            exp.af.resize(this->loc.a.size());
            exp.work.resize(exp.lwork);
            exp.sr.resize(this->loc.rows);
            exp.sc.resize(this->loc.cols);
        }

    public:
        pposvx()
            : base_t() {}

        pposvx(const IT rows, const IT cols)
            : base_t(rows, cols) {}

        using base_t::solve;

        IT solve(full_mat<DT, IT>&& A, full_mat<DT, IT>&& B) override {
            if(!this->ctx.is_valid()) return 0;

            if(A.n_rows != A.n_cols || A.n_cols != B.n_rows) return -1;

            this->init_storage(A.n_rows);
            init_expert_storage();

            this->ctx.scatter(A, this->ctx.desc_g(A.n_rows, A.n_cols), exp.af, this->loc.desc_a);

            const auto loc_cols_b = this->ctx.cols(B.n_cols, this->loc.block);
            this->loc.b.resize(this->loc.rows * loc_cols_b);

            const auto full_desc_b = this->ctx.desc_g(B.n_rows, B.n_cols);
            const auto loc_desc_b = this->ctx.desc_l(B.n_rows, B.n_cols, this->loc.block, this->loc.rows);

            this->ctx.scatter(B, full_desc_b, this->loc.b, loc_desc_b);

            std::vector<work_t<DT>> ferr(loc_cols_b), berr(loc_cols_b);
            work_t<DT> rcond;

            std::vector<DT> x(this->loc.b.size());

            auto equed = 'X';

            IT info{-1};
            // ReSharper disable CppCStyleCast
            if constexpr(std::is_same_v<DT, double>) {
                using E = double;

                const auto liwork = this->loc.rows;
                std::vector<IT> iwork(liwork);

                pdposvx(&FACT, &UPLO, &this->loc.n, &B.n_cols, (E*)exp.af.data(), &this->ONE, &this->ONE, this->loc.desc_a.data(), (E*)this->loc.a.data(), &this->ONE, &this->ONE, this->loc.desc_a.data(), &equed, (E*)exp.sr.data(), (E*)exp.sc.data(), (E*)this->loc.b.data(), &this->ONE, &this->ONE, loc_desc_b.data(), (E*)x.data(), &this->ONE, &this->ONE, loc_desc_b.data(), (E*)&rcond, (E*)ferr.data(), (E*)berr.data(), (E*)exp.work.data(), &exp.lwork, iwork.data(), &liwork, &info);
            }
            else if constexpr(std::is_same_v<DT, float>) {
                using E = float;

                const auto liwork = this->loc.rows;
                std::vector<IT> iwork(liwork);

                psposvx(&FACT, &UPLO, &this->loc.n, &B.n_cols, (E*)exp.af.data(), &this->ONE, &this->ONE, this->loc.desc_a.data(), (E*)this->loc.a.data(), &this->ONE, &this->ONE, this->loc.desc_a.data(), &equed, (E*)exp.sr.data(), (E*)exp.sc.data(), (E*)this->loc.b.data(), &this->ONE, &this->ONE, loc_desc_b.data(), (E*)x.data(), &this->ONE, &this->ONE, loc_desc_b.data(), (E*)&rcond, (E*)ferr.data(), (E*)berr.data(), (E*)exp.work.data(), &exp.lwork, iwork.data(), &liwork, &info);
            }
            else if constexpr(std::is_same_v<DT, complex16>) {
                using E = complex16;
                using BE = work_t<complex16>;

                const auto lrwork = compute_lrwork();
                std::vector<BE> rwork(lrwork);

                pzposvx(&FACT, &UPLO, &this->loc.n, &B.n_cols, (E*)exp.af.data(), &this->ONE, &this->ONE, this->loc.desc_a.data(), (E*)this->loc.a.data(), &this->ONE, &this->ONE, this->loc.desc_a.data(), &equed, (BE*)exp.sr.data(), (BE*)exp.sc.data(), (E*)this->loc.b.data(), &this->ONE, &this->ONE, loc_desc_b.data(), (E*)x.data(), &this->ONE, &this->ONE, loc_desc_b.data(), (BE*)&rcond, (BE*)ferr.data(), (BE*)berr.data(), (E*)exp.work.data(), &exp.lwork, rwork.data(), &lrwork, &info);
            }
            else if constexpr(std::is_same_v<DT, complex8>) {
                using E = complex8;
                using BE = work_t<complex8>;

                const auto lrwork = compute_lrwork();
                std::vector<BE> rwork(lrwork);

                pcposvx(&FACT, &UPLO, &this->loc.n, &B.n_cols, (E*)exp.af.data(), &this->ONE, &this->ONE, this->loc.desc_a.data(), (E*)this->loc.a.data(), &this->ONE, &this->ONE, this->loc.desc_a.data(), &equed, (BE*)exp.sr.data(), (BE*)exp.sc.data(), (E*)this->loc.b.data(), &this->ONE, &this->ONE, loc_desc_b.data(), (E*)x.data(), &this->ONE, &this->ONE, loc_desc_b.data(), (BE*)&rcond, (BE*)ferr.data(), (BE*)berr.data(), (E*)exp.work.data(), &exp.lwork, rwork.data(), &lrwork, &info);
            }
            // ReSharper restore CppCStyleCast

            if((info = this->ctx.amx(info)) != 0) return info;

            if(equed == 'C' || equed == 'B')
                for(auto i = 0; i < loc_cols_b; ++i)
                    for(auto j = 0; j < this->loc.rows; ++j) x[j * loc_cols_b + i] /= exp.sc[j];

            this->ctx.gather(x, loc_desc_b, B, full_desc_b);

            return info;
        }

        IT solve(full_mat<DT, IT>&& B) override {
            IT info{-1};
            return info;
        }
    };

    template<index_t IT, char UL = 'L', char ODER = 'R'> using par_dposvx = pposvx<double, IT, UL, ODER>;
    template<index_t IT, char UL = 'L', char ODER = 'R'> using par_sposvx = pposvx<float, IT, UL, ODER>;
    template<index_t IT, char UL = 'L', char ODER = 'R'> using par_zposvx = pposvx<complex16, IT, UL, ODER>;
    template<index_t IT, char UL = 'L', char ODER = 'R'> using par_cposvx = pposvx<complex8, IT, UL, ODER>;
} // namespace ezp

#endif // PPOSVX_HPP

//! @}
