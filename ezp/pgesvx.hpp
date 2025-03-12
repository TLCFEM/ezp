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
    template<data_t DT, index_t IT, char ODER = 'R'> class pgesvx final : public detail::full_solver<DT, IT, ODER> {
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
            IT info{-1};
            return info;
        }
    };

    template<index_t IT, char ODER = 'R'> using par_dgesvx = pgesvx<double, IT, ODER>;
    template<index_t IT, char ODER = 'R'> using par_sgesvx = pgesvx<float, IT, ODER>;
    template<index_t IT, char ODER = 'R'> using par_zgesvx = pgesvx<complex16, IT, ODER>;
    template<index_t IT, char ODER = 'R'> using par_cgesvx = pgesvx<complex8, IT, ODER>;
    template<index_t IT> using par_dgesvx_c = pgesvx<double, IT, 'C'>;
    template<index_t IT> using par_sgesvx_c = pgesvx<float, IT, 'C'>;
    template<index_t IT> using par_zgesvx_c = pgesvx<complex16, IT, 'C'>;
    template<index_t IT> using par_cgesvx_c = pgesvx<complex8, IT, 'C'>;
} // namespace ezp

#endif // PGESVX_HPP

//! @}
