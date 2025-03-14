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
 *
 * It solves the system of linear equations `A*X=B` with a full general matrix `A`.
 * The matrix `A` is stored in a `NxN` block.
 * The matrix `B` is stored in a `NxNRHS` block.
 *
 * @note There is a known bug in the `pgesvx` solver that causes the program to hang.
 * See details: https://github.com/Reference-ScaLAPACK/scalapack/issues/119
 * @note Enabling equilibration may cause the program to hang.
 * Please check the status of the bug before setting `FACT` to 'E'.
 *
 * The example usage can be seen as follows.
 *
 * @include ../examples/example.pgesvx.cpp
 *
 * @author tlc
 * @date 12/03/2025
 * @version 1.0.0
 * @file pgesvx.hpp
 * @{
 */

#ifndef PGESVX_HPP
#define PGESVX_HPP

#include "abstract/full_solver.hpp"

#include <numeric>

namespace ezp {
    template<data_t DT, index_t IT, char FCT = 'E', char ODER = 'R'> class pgesvx final : public detail::full_solver<DT, IT, ODER> {
        static constexpr char TRANS = 'N';
        static constexpr char FACT = FCT;

        using base_t = detail::full_solver<DT, IT, ODER>;

        static auto ceil(const IT a, const IT b) { return (a + b - 1) / b; }

        auto compute_lwork() {
            const auto ceil_a = std::max(IT{1}, ceil(this->ctx.n_rows - 1, this->ctx.n_cols));
            const auto ceil_b = std::max(IT{1}, ceil(this->ctx.n_cols - 1, this->ctx.n_rows));
            const auto pgecon_lwork = 2 * (this->loc.rows + this->loc.cols) + std::max(IT{2}, std::max(this->loc.block * ceil_a, this->loc.cols + this->loc.block * ceil_b));

            const auto lcmq = std::lcm(this->ctx.n_rows, this->ctx.n_cols) / this->ctx.n_cols;
            const auto nqb = ceil(this->loc.n, this->loc.block * this->ctx.n_cols);
            const auto pgerfs_lwork = 4 * this->loc.rows + this->loc.cols + this->loc.block * ceil(nqb, lcmq);

            return this->loc.rows + std::max(pgecon_lwork, pgerfs_lwork);
        }

        struct expert_system {
            IT lwork;
            std::vector<DT> af, work;
            std::vector<work_t<DT>> r, c;
        };

        expert_system exp;

        auto init_expert_storage() {
            exp.lwork = compute_lwork();
            exp.af.resize(this->loc.a.size());
            exp.work.resize(exp.lwork);
            exp.r.resize(this->loc.rows);
            exp.c.resize(this->loc.cols);
        }

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

                pdgesvx(&FACT, &TRANS, &this->loc.n, &B.n_cols, (E*)exp.af.data(), &this->ONE, &this->ONE, this->loc.desc_a.data(), (E*)this->loc.a.data(), &this->ONE, &this->ONE, this->loc.desc_a.data(), this->loc.ipiv.data(), &equed, (E*)exp.r.data(), (E*)exp.c.data(), (E*)this->loc.b.data(), &this->ONE, &this->ONE, loc_desc_b.data(), (E*)x.data(), &this->ONE, &this->ONE, loc_desc_b.data(), (E*)&rcond, (E*)ferr.data(), (E*)berr.data(), (E*)exp.work.data(), &exp.lwork, iwork.data(), &liwork, &info);
            }
            else if constexpr(std::is_same_v<DT, float>) {
                using E = float;

                const auto liwork = this->loc.rows;
                std::vector<IT> iwork(liwork);

                psgesvx(&FACT, &TRANS, &this->loc.n, &B.n_cols, (E*)exp.af.data(), &this->ONE, &this->ONE, this->loc.desc_a.data(), (E*)this->loc.a.data(), &this->ONE, &this->ONE, this->loc.desc_a.data(), this->loc.ipiv.data(), &equed, (E*)exp.r.data(), (E*)exp.c.data(), (E*)this->loc.b.data(), &this->ONE, &this->ONE, loc_desc_b.data(), (E*)x.data(), &this->ONE, &this->ONE, loc_desc_b.data(), (E*)&rcond, (E*)ferr.data(), (E*)berr.data(), (E*)exp.work.data(), &exp.lwork, iwork.data(), &liwork, &info);
            }
            else if constexpr(std::is_same_v<DT, complex16>) {
                using E = complex16;
                using BE = work_t<complex16>;

                const auto lrwork = std::max(this->loc.rows, 2 * this->loc.cols);
                std::vector<BE> rwork(lrwork);

                pzgesvx(&FACT, &TRANS, &this->loc.n, &B.n_cols, (E*)exp.af.data(), &this->ONE, &this->ONE, this->loc.desc_a.data(), (E*)this->loc.a.data(), &this->ONE, &this->ONE, this->loc.desc_a.data(), this->loc.ipiv.data(), &equed, (BE*)exp.r.data(), (BE*)exp.c.data(), (E*)this->loc.b.data(), &this->ONE, &this->ONE, loc_desc_b.data(), (E*)x.data(), &this->ONE, &this->ONE, loc_desc_b.data(), (BE*)&rcond, (BE*)ferr.data(), (BE*)berr.data(), (E*)exp.work.data(), &exp.lwork, rwork.data(), &lrwork, &info);
            }
            else if constexpr(std::is_same_v<DT, complex8>) {
                using E = complex8;
                using BE = work_t<complex8>;

                const auto lrwork = std::max(this->loc.rows, 2 * this->loc.cols);
                std::vector<BE> rwork(lrwork);

                pcgesvx(&FACT, &TRANS, &this->loc.n, &B.n_cols, (E*)exp.af.data(), &this->ONE, &this->ONE, this->loc.desc_a.data(), (E*)this->loc.a.data(), &this->ONE, &this->ONE, this->loc.desc_a.data(), this->loc.ipiv.data(), &equed, (BE*)exp.r.data(), (BE*)exp.c.data(), (E*)this->loc.b.data(), &this->ONE, &this->ONE, loc_desc_b.data(), (E*)x.data(), &this->ONE, &this->ONE, loc_desc_b.data(), (BE*)&rcond, (BE*)ferr.data(), (BE*)berr.data(), (E*)exp.work.data(), &exp.lwork, rwork.data(), &lrwork, &info);
            }
            // ReSharper restore CppCStyleCast

            if((info = this->ctx.amx(info)) != 0) return info;

            this->ctx.gather(x, loc_desc_b, B, full_desc_b);

            return info;
        }

        IT solve(full_mat<DT, IT>&& B) override { return IT{-1}; }
    };

    template<index_t IT, char FCT = 'E', char ODER = 'R'> using par_dgesvx = pgesvx<double, IT, FCT, ODER>;
    template<index_t IT, char FCT = 'E', char ODER = 'R'> using par_sgesvx = pgesvx<float, IT, FCT, ODER>;
    template<index_t IT, char FCT = 'E', char ODER = 'R'> using par_zgesvx = pgesvx<complex16, IT, FCT, ODER>;
    template<index_t IT, char FCT = 'E', char ODER = 'R'> using par_cgesvx = pgesvx<complex8, IT, FCT, ODER>;
} // namespace ezp

#endif // PGESVX_HPP

//! @}
