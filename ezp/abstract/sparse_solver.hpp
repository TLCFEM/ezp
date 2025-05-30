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

#ifndef SPARSE_SOLVER_HPP
#define SPARSE_SOLVER_HPP

#include "traits.hpp"

#include <algorithm>
#include <numeric>
#ifdef EZP_TBB
#include <tbb/parallel_sort.h>
#define ezp_sort tbb::parallel_sort
#else
#define ezp_sort std::sort
#endif

namespace ezp {
    template<data_t DT, index_t IT> struct sparse_coo_mat {
        IT n, nnz;
        IT *row, *col;
        DT* data;

        sparse_coo_mat(const IT n, const IT nnz, IT* const row, IT* const col, DT* const data)
            : n(n)
            , nnz(nnz)
            , row(row)
            , col(col)
            , data(data) {}

        auto is_valid() const { return row && col && data; }
    };

    namespace detail {
        template<index_t IT> class csr_comparator {
            const IT* const row_idx;
            const IT* const col_idx;

        public:
            csr_comparator(const IT* const in_row_idx, const IT* const in_col_idx)
                : row_idx(in_row_idx)
                , col_idx(in_col_idx) {}

            bool operator()(const IT idx_a, const IT idx_b) const {
                if(row_idx[idx_a] == row_idx[idx_b]) return col_idx[idx_a] < col_idx[idx_b];
                return row_idx[idx_a] < row_idx[idx_b];
            }
        };

        template<typename T> bool approx_equal(T x, T y, int ulp = 2)
            requires(!std::numeric_limits<T>::is_integer)
        { return std::fabs(x - y) <= std::numeric_limits<T>::epsilon() * std::fabs(x + y) * ulp || std::fabs(x - y) < std::numeric_limits<T>::min(); }
    } // namespace detail

    template<data_t DT, index_t IT> struct sparse_csr_mat final {
        IT n, nnz;
        IT *row_ptr, *col_idx;
        DT* data;

        std::vector<IT> row_storage, col_storage;
        std::vector<DT> data_storage;

        sparse_csr_mat() = default;
        sparse_csr_mat(const sparse_csr_mat& other)
            : n(other.n)
            , row_ptr(other.row_ptr)
            , col_idx(other.col_idx)
            , data(other.data)
            , row_storage(other.row_storage)
            , col_storage(other.col_storage)
            , data_storage(other.data_storage) {
            if(!row_storage.empty()) row_ptr = row_storage.data();
            if(!col_storage.empty()) col_idx = col_storage.data();
            if(!data_storage.empty()) data = data_storage.data();
        }
        sparse_csr_mat(sparse_csr_mat&&) = delete;
        sparse_csr_mat& operator=(const sparse_csr_mat&) = delete;
        sparse_csr_mat& operator=(sparse_csr_mat&&) = default;
        ~sparse_csr_mat() = default;

        sparse_csr_mat(const IT n, const IT nnz, IT* const row_ptr, IT* const col_idx, DT* const data)
            : n(n)
            , nnz(nnz)
            , row_ptr(row_ptr)
            , col_idx(col_idx)
            , data(data) {}

        template<data_t DT2, index_t IT2> explicit sparse_csr_mat(const sparse_coo_mat<DT2, IT2>& coo, const bool one_based = false, const bool full = false)
            : n(IT{coo.n})
            , nnz(IT{coo.nnz}) {
            if(!coo.is_valid()) return;

            std::vector<IT2> index(nnz);
            std::iota(index.begin(), index.end(), IT2(0));
            ezp_sort(index.begin(), index.end(), detail::csr_comparator(coo.row, coo.col));

            row_storage.resize(nnz);
            col_storage.resize(nnz);
            data_storage.resize(nnz);

            for(auto I = IT{0}; I < nnz; ++I) {
                row_storage[I] = coo.row[index[I]];
                col_storage[I] = coo.col[index[I]];
                data_storage[I] = coo.data[index[I]];
            }

            condense(one_based, full);

            row_ptr = row_storage.data();
            col_idx = col_storage.data();
            data = data_storage.data();
        }

        auto condense(const bool one_based, const bool full) {
            auto last_row = row_storage[0], last_col = col_storage[0];

            auto current_pos = IT{0};
            auto last_sum = DT{0};

            auto populate = [&] {
                if(detail::approx_equal(last_sum, DT(0)) && (!full || last_row != last_col)) return;
                row_storage[current_pos] = last_row;
                col_storage[current_pos] = last_col;
                data_storage[current_pos] = last_sum;
                ++current_pos;
                last_sum = DT{0};
            };

            for(auto I = IT{0}; I < nnz; ++I) {
                if(last_row != row_storage[I] || last_col != col_storage[I]) {
                    populate();
                    last_row = row_storage[I];
                    last_col = col_storage[I];
                }
                last_sum += data_storage[I];
            }

            populate();

            nnz = current_pos;

            auto current_row = current_pos = IT{0};
            auto shift = one_based ? IT{1} : IT{0};

            while(current_pos < nnz)
                if(row_storage[current_pos] < current_row + shift) ++current_pos;
                else row_storage[current_row++] = current_pos + shift;

            row_storage[0] = IT{0} + shift;
            row_storage[n] = nnz + shift;

            row_storage.resize(n + 1);
            col_storage.resize(nnz);
            data_storage.resize(nnz);
        }

        auto is_valid() { return row_ptr && col_idx && data; }
    };
} // namespace ezp

#endif // SPARSE_SOLVER_HPP
