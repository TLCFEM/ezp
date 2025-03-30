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
 * @class pgesv
 * @brief Solver for general full matrices.
 *
 * It solves the system of linear equations `A * X = B` with a full general matrix `A`.
 * The matrix `A` is stored in a `N x N` block.
 * The matrix `B` is stored in a `N x NRHS` block.
 *
 * The example usage can be seen as follows.
 *
 * @include example.pgesv.cpp
 *
 * @author tlc
 * @date 07/03/2025
 * @version 1.0.0
 * @file pgesv.hpp
 * @{
 */

#ifndef PGESV_HPP
#define PGESV_HPP

#include "abstract/full_solver.hpp"

namespace ezp {
    template<data_t DT, index_t IT, char ODER = 'R'> class pgesv final : public detail::full_solver<DT, IT, ODER> {
        using base_t = detail::full_solver<DT, IT, ODER>;

    public:
        pgesv()
            : base_t() {}

        pgesv(const IT rows, const IT cols)
            : base_t(rows, cols) {}

        using base_t::solve;

        IT solve(full_mat<DT, IT>&& A, full_mat<DT, IT>&& B) override {
            if(!this->ctx.is_valid()) return 0;

            if(A.n_rows != A.n_cols || A.n_cols != B.n_rows) return -1;

            this->init_storage(A.n_rows);

            this->ctx.scatter(A, this->ctx.desc_g(A.n_rows, A.n_cols), this->loc.a, this->loc.desc_a);

            IT info{-1};
            // ReSharper disable CppCStyleCast
            if constexpr(std::is_same_v<DT, double>) {
                using E = double;
                pdgetrf(&this->loc.n, &this->loc.n, (E*)this->loc.a.data(), &this->ONE, &this->ONE, this->loc.desc_a.data(), this->loc.ipiv.data(), &info);
            }
            else if constexpr(std::is_same_v<DT, float>) {
                using E = float;
                psgetrf(&this->loc.n, &this->loc.n, (E*)this->loc.a.data(), &this->ONE, &this->ONE, this->loc.desc_a.data(), this->loc.ipiv.data(), &info);
            }
            else if constexpr(std::is_same_v<DT, complex16>) {
                using E = complex16;
                pzgetrf(&this->loc.n, &this->loc.n, (E*)this->loc.a.data(), &this->ONE, &this->ONE, this->loc.desc_a.data(), this->loc.ipiv.data(), &info);
            }
            else if constexpr(std::is_same_v<DT, complex8>) {
                using E = complex8;
                pcgetrf(&this->loc.n, &this->loc.n, (E*)this->loc.a.data(), &this->ONE, &this->ONE, this->loc.desc_a.data(), this->loc.ipiv.data(), &info);
            }
            // ReSharper restore CppCStyleCast

            if((info = this->ctx.amx(info)) != 0) return info;

            return solve(std::move(B));
        }

        IT solve(full_mat<DT, IT>&& B) override {
            static constexpr char TRANS = 'N';

            if(B.n_rows != this->loc.n) return -1;

            if(!this->ctx.is_valid()) return 0;

            this->loc.b.resize(this->loc.rows * this->ctx.cols(B.n_cols, this->loc.block));

            const auto full_desc_b = this->ctx.desc_g(B.n_rows, B.n_cols);
            const auto loc_desc_b = this->ctx.desc_l(B.n_rows, B.n_cols, this->loc.block, this->loc.rows);

            this->ctx.scatter(B, full_desc_b, this->loc.b, loc_desc_b);

            IT info{-1};
            // ReSharper disable CppCStyleCast
            if constexpr(std::is_same_v<DT, double>) {
                using E = double;
                pdgetrs(&TRANS, &this->loc.n, &B.n_cols, (E*)this->loc.a.data(), &this->ONE, &this->ONE, this->loc.desc_a.data(), this->loc.ipiv.data(), (E*)this->loc.b.data(), &this->ONE, &this->ONE, loc_desc_b.data(), &info);
            }
            else if constexpr(std::is_same_v<DT, float>) {
                using E = float;
                psgetrs(&TRANS, &this->loc.n, &B.n_cols, (E*)this->loc.a.data(), &this->ONE, &this->ONE, this->loc.desc_a.data(), this->loc.ipiv.data(), (E*)this->loc.b.data(), &this->ONE, &this->ONE, loc_desc_b.data(), &info);
            }
            else if constexpr(std::is_same_v<DT, complex16>) {
                using E = complex16;
                pzgetrs(&TRANS, &this->loc.n, &B.n_cols, (E*)this->loc.a.data(), &this->ONE, &this->ONE, this->loc.desc_a.data(), this->loc.ipiv.data(), (E*)this->loc.b.data(), &this->ONE, &this->ONE, loc_desc_b.data(), &info);
            }
            else if constexpr(std::is_same_v<DT, complex8>) {
                using E = complex8;
                pcgetrs(&TRANS, &this->loc.n, &B.n_cols, (E*)this->loc.a.data(), &this->ONE, &this->ONE, this->loc.desc_a.data(), this->loc.ipiv.data(), (E*)this->loc.b.data(), &this->ONE, &this->ONE, loc_desc_b.data(), &info);
            }
            // ReSharper restore CppCStyleCast

            if((info = this->ctx.amx(info)) == 0) this->ctx.gather(this->loc.b, loc_desc_b, B, full_desc_b);

            return info;
        }
    };

    template<index_t IT, char ODER = 'R'> using par_dgesv = pgesv<double, IT, ODER>;
    template<index_t IT, char ODER = 'R'> using par_sgesv = pgesv<float, IT, ODER>;
    template<index_t IT, char ODER = 'R'> using par_zgesv = pgesv<complex16, IT, ODER>;
    template<index_t IT, char ODER = 'R'> using par_cgesv = pgesv<complex8, IT, ODER>;
    template<index_t IT> using par_dgesv_c = pgesv<double, IT, 'C'>;
    template<index_t IT> using par_sgesv_c = pgesv<float, IT, 'C'>;
    template<index_t IT> using par_zgesv_c = pgesv<complex16, IT, 'C'>;
    template<index_t IT> using par_cgesv_c = pgesv<complex8, IT, 'C'>;
} // namespace ezp

#endif // PGESV_HPP

//! @}
