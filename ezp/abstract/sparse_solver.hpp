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

#ifndef SPARSE_SOLVER_HPP
#define SPARSE_SOLVER_HPP

#include "traits.hpp"

#include <numeric>
#ifdef EZP_TBB
#include <tbb/parallel_sort.h>
#define ezp_sort tbb::parallel_sort
#else
#include <algorithm>
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

        template<data_t T> bool approx_equal(const T x, const T y, const int ulp = 2) {
            using R = decltype(std::abs(x));
            return std::abs(x - y) <= std::numeric_limits<R>::epsilon() * std::abs(x + y) * ulp || std::abs(x - y) < std::numeric_limits<R>::min();
        }
    } // namespace detail

    template<data_t DT, index_t IT> struct sparse_csr_mat final {
        IT n{}, nnz{};
        IT *row_ptr{nullptr}, *col_idx{nullptr};
        DT* data{nullptr};

        std::vector<IT> row_storage, col_storage;
        std::vector<DT> data_storage;

        sparse_csr_mat() = default;
        sparse_csr_mat(const sparse_csr_mat& other)
            : n(other.n)
            , nnz(other.nnz)
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

        /**
         * Construct a CSR matrix from a COO matrix.
         * The input matrix will be sorted, duplicated entries will be summed.
         * @param coo The input matrix in the COO format.
         * @param one_based Indicate if the input matrix uses one based indexing (if true) or zero based indexing (if false). This flag will be used to generate correct row indexing.
         * @param full Some solvers require diagonal entries to be explicitly present even if they are trivial, if true, all diagonal entries will be explicitly set regardless of the actual values.
         */
        template<data_t DT2, index_t IT2> explicit sparse_csr_mat(const sparse_coo_mat<DT2, IT2>& coo, const bool one_based = false, const bool full = false)
            : n(IT{coo.n})
            , nnz(IT{coo.nnz}) {
            if(!coo.is_valid()) return;

            std::vector<IT2> index(nnz);
            std::iota(index.begin(), index.end(), IT2(0));
            ezp_sort(index.begin(), index.end(), detail::csr_comparator(coo.row, coo.col));

            row_storage.resize(n + 1);
            col_storage.resize(nnz);
            data_storage.resize(nnz);

            condense(coo, index, one_based, full);

            row_ptr = row_storage.data();
            col_idx = col_storage.data();
            data = data_storage.data();
        }

        /**
         * Sum duplicated entries, drop trivial entries and build the row pointer in a single pass.
         * @param coo The input matrix in the COO format.
         * @param index The permutation that sorts the COO entries by row then by column.
         * @param one_based Indicate if the input matrix uses one based indexing.
         * @param full If true, explicitly present diagonal entries are kept even if they are trivial.
         */
        template<data_t DT2, index_t IT2> auto condense(const sparse_coo_mat<DT2, IT2>& coo, const std::vector<IT2>& index, const bool one_based, const bool full) {
            const auto shift = one_based ? IT{1} : IT{0};

            auto current_pos = IT{0}, current_row = IT{0};
            auto last_row = IT{0}, last_col = IT{0};
            auto last_sum = DT{0};

            auto populate = [&] {
                if(!detail::approx_equal(last_sum, DT{0}) || (full && last_row == last_col)) {
                    while(current_row + shift <= last_row) row_storage[current_row++] = current_pos + shift;
                    col_storage[current_pos] = last_col;
                    data_storage[current_pos++] = last_sum;
                }
                last_sum = DT{0};
            };

            if(nnz > 0) {
                last_row = static_cast<IT>(coo.row[index[0]]);
                last_col = static_cast<IT>(coo.col[index[0]]);
            }

            for(auto I = IT{0}; I < nnz; ++I) {
                const auto idx = index[I];
                const auto row = static_cast<IT>(coo.row[idx]), col = static_cast<IT>(coo.col[idx]);
                if(last_row != row || last_col != col) {
                    populate();
                    last_row = row;
                    last_col = col;
                }
                last_sum += coo.data[idx];
            }

            if(nnz > 0) populate();

            nnz = current_pos;

            // remaining (trailing empty) rows and the end pointer
            while(current_row <= n) row_storage[current_row++] = nnz + shift;

            col_storage.resize(nnz);
            data_storage.resize(nnz);

            if(full) insert_missing_diagonals(shift);
        }

        /**
         * Ensure every diagonal entry is explicitly present.
         * The main pass above can only keep a diagonal entry that already appears
         * (with any value, including a trivial one) in the input COO matrix.
         * If a diagonal entry is entirely absent from the input, it must be
         * synthesized here with a value of zero, as documented for the `full` flag.
         * @param shift The indexing shift (0 or 1) used for row/col pointers.
         */
        auto insert_missing_diagonals(const IT shift) {
            IT missing{0};
            for(auto row = IT{0}; row < n; ++row) {
                auto found = false;
                for(auto k = row_storage[row] - shift; k < row_storage[row + 1] - shift; ++k)
                    if(col_storage[k] == row + shift) {
                        found = true;
                        break;
                    }
                if(!found) ++missing;
            }

            if(0 == missing) return;

            std::vector<IT> new_row(n + 1);
            std::vector<IT> new_col(nnz + missing);
            std::vector<DT> new_data(nnz + missing);

            auto pos = IT{0};
            for(auto row = IT{0}; row < n; ++row) {
                new_row[row] = pos + shift;
                auto inserted = false;
                for(auto k = row_storage[row] - shift; k < row_storage[row + 1] - shift; ++k) {
                    const auto col = col_storage[k];
                    if(!inserted && col > row + shift) {
                        new_col[pos] = row + shift;
                        new_data[pos++] = DT{0};
                        inserted = true;
                    }
                    new_col[pos] = col;
                    new_data[pos++] = data_storage[k];
                    if(col == row + shift) inserted = true;
                }
                if(!inserted) {
                    new_col[pos] = row + shift;
                    new_data[pos++] = DT{0};
                }
            }
            new_row[n] = pos + shift;

            nnz = pos;
            row_storage = std::move(new_row);
            col_storage = std::move(new_col);
            data_storage = std::move(new_data);
        }

        auto is_valid() { return row_ptr && col_idx && data; }
    };
} // namespace ezp

#endif // SPARSE_SOLVER_HPP
