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
/**
 * @file check.hpp
 * @brief Shared helpers for numerically verifying the correctness of solvers.
 *
 * The checks compute the componentwise relative residual `||A*X - B|| / (||A||*||X||)`
 * of a computed solution `X` against a known right-hand side `B`. The matrix `A` may be
 * given either directly (full column-major storage) or reconstructed from band storage.
 *
 * These helpers are meant to be used from the distributed test programs. All computations
 * are performed locally on the process that holds the gathered solution `X` and `B` (i.e.
 * the root process).
 */

#ifndef EZP_TEST_CHECK_HPP
#define EZP_TEST_CHECK_HPP

#include <ezp/abstract/traits.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace ezp::test {
    /**
     * @brief Returns a copy of a value with the correct phase for mirroring a
     *        Hermitian off-diagonal entry (identity for real data).
     */
    template<typename T> auto conjugate(const T& v) {
        if constexpr(complex_t<T>) return std::conj(v);
        else return v;
    }

    /**
     * @brief Computes the infinity norm of a column-major `n x n` matrix.
     */
    template<data_t DT> auto mat_inf_norm(const std::vector<DT>& a, const int_t n) {
        using R = work_t<DT>;
        auto nrm = R{0};
        for(auto r = int_t{0}; r < n; ++r) {
            auto row_sum = R{0};
            for(auto c = int_t{0}; c < n; ++c) row_sum += std::abs(a[r + c * n]);
            nrm = std::max(nrm, row_sum);
        }
        return nrm;
    }

    /**
     * @brief Computes the componentwise relative residual `||A*X - B|| / (||A||*||X||)`.
     *
     * @param a Column-major `n x n` matrix.
     * @param x Column-major `n x nrhs` computed solution (output of the solver).
     * @param b Column-major `n x nrhs` right-hand side used to solve the system.
     * @param n Number of rows/columns of `a`.
     * @param nrhs Number of right-hand sides.
     * @return The relative residual, or `1` if the normalization vanishes.
     */
    template<data_t DT> auto full_residual(const std::vector<DT>& a, const std::vector<DT>& x, const std::vector<DT>& b, const int_t n, const int_t nrhs) {
        using R = work_t<DT>;
        auto res = R{0}, nrm_x = R{0};
        for(auto c = int_t{0}; c < nrhs; ++c)
            for(auto r = int_t{0}; r < n; ++r) {
                auto err = DT{0};
                for(auto k = int_t{0}; k < n; ++k) err += a[r + k * n] * x[k + c * n];
                res = std::max(res, std::abs(err - b[r + c * n]));
                nrm_x = std::max(nrm_x, std::abs(x[r + c * n]));
            }
        const auto nrm_a = mat_inf_norm(a, n);
        return res / std::max(R{1}, nrm_a * nrm_x);
    }

    /**
     * @brief Returns a numerical-correctness tolerance for a given data type.
     *
     * The tolerance is chosen to comfortably accommodate round-off while still
     * catching genuine correctness regressions.
     */
    template<data_t DT> constexpr auto residual_tol() {
        if constexpr(std::is_same_v<work_t<DT>, double>) return 1e-12;
        else return 1e-5;
    }

    /**
     * @brief Reconstructs a full column-major `n x n` matrix from general band storage.
     *
     * @param a Band-storage buffer.
     * @param idx The solver's band indexer, mapping `(row, col)` to a storage position
     *            or `-1` when the entry lies outside the band.
     * @param n Dimension of the full matrix.
     * @return The reconstructed full matrix.
     */
    template<data_t DT, typename IDX> auto rebuild_full_band(const std::vector<DT>& a, const IDX& idx, const int_t n) {
        std::vector<DT> a_full(n * n, DT{0});
        for(auto c = int_t{0}; c < n; ++c)
            for(auto r = int_t{0}; r < n; ++r)
                if(const auto p = idx(r, c); p >= 0) a_full[r + c * n] = a[p];
        return a_full;
    }

    /**
     * @brief Reconstructs a full symmetric/Hermitian `n x n` matrix from band storage.
     *
     * Only one triangle is stored; the other is rebuilt by mirroring (with conjugation
     * for Hermitian complex data).
     *
     * @param a Band-storage buffer (stored triangle only).
     * @param idx The solver's band indexer.
     * @param n Dimension of the full matrix.
     * @param uplo `'L'` or `'U'` indicating the stored triangle.
     * @return The reconstructed full symmetric matrix.
     */
    template<data_t DT, char UL, typename IDX> auto rebuild_full_symm_band(const std::vector<DT>& a, const IDX& idx, const int_t n) {
        std::vector<DT> a_full(n * n, DT{0});
        for(auto i = int_t{0}; i < n; ++i)
            for(auto j = i; j < n; ++j) {
                int_t p;
                if constexpr('L' == UL) p = idx(j, i); // lower triangle
                else p = idx(i, j);                     // upper triangle
                if(p < 0) continue;
                const auto val = a[p];
                a_full[i + j * n] = conjugate(val);
                a_full[j + i * n] = val;
            }
        return a_full;
    }
} // namespace ezp::test

#endif // EZP_TEST_CHECK_HPP
