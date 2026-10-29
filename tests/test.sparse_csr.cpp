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

// Regression tests for ezp::sparse_csr_mat construction and lifetime.
//
// These tests exercise only the pure C++ COO -> CSR conversion logic in
// ezp/abstract/sparse_solver.hpp and therefore do not require MPI, ScaLAPACK
// or MKL. They are deliberately kept independent of the distributed solver
// drivers so they can be built and run on any toolchain.

#include <ezp/abstract/sparse_solver.hpp>

#include <vector>

using namespace ezp;

#ifdef EZP_ENABLE_TEST
#include <catch2/catchy.hpp>
#else
#include <cassert>
#define REQUIRE(...) assert((__VA_ARGS__))
#endif

// A valid COO/CSR round trip must put the values at the expected positions
// and report the matrix as valid, so that the empty-input fix below does not
// regress ordinary (already working) construction.
static void valid_zero_based() {
    std::vector<double> data{1., 2., 3.};
    std::vector<int_t> row{0, 0, 2}, col{0, 1, 2};

    sparse_coo_mat<double, int_t> coo{3, 3, row.data(), col.data(), data.data()};
    sparse_csr_mat<double, int_t> csr(coo, false, false);

    REQUIRE(csr.is_valid());
    REQUIRE(csr.n == 3);
    REQUIRE(csr.nnz == 3);
    REQUIRE(csr.row_ptr[0] == 0);
    REQUIRE(csr.row_ptr[1] == 2); // row 0 has two entries
    REQUIRE(csr.row_ptr[2] == 2); // row 1 is empty
    REQUIRE(csr.row_ptr[3] == 3);
    REQUIRE(csr.col_idx[0] == 0);
    REQUIRE(csr.col_idx[1] == 1);
    REQUIRE(csr.col_idx[2] == 2);
    REQUIRE(csr.data[0] == 1.);
    REQUIRE(csr.data[1] == 2.);
    REQUIRE(csr.data[2] == 3.);
}

// One-based indexing must keep row pointers offset by one.
static void valid_one_based() {
    std::vector<double> data{5., 7.};
    std::vector<int_t> row{1, 3}, col{1, 3};

    sparse_coo_mat<double, int_t> coo{3, 2, row.data(), col.data(), data.data()};
    sparse_csr_mat<double, int_t> csr(coo, true, false);

    REQUIRE(csr.is_valid());
    REQUIRE(csr.nnz == 2);
    REQUIRE(csr.row_ptr[0] == 1);
    REQUIRE(csr.row_ptr[1] == 2);
    REQUIRE(csr.row_ptr[2] == 2); // row 2 is empty
    REQUIRE(csr.row_ptr[3] == 3);
    REQUIRE(csr.col_idx[0] == 1);
    REQUIRE(csr.col_idx[1] == 3);
    REQUIRE(csr.data[0] == 5.);
    REQUIRE(csr.data[1] == 7.);
}

// The `full` flag must force every diagonal entry to be explicitly present,
// even when a diagonal entry is entirely absent from the input COO (not just
// when it is present with a trivial value). Regression: the single pass in
// condense() could only keep a diagonal entry that already appeared in the
// input; it never synthesized a missing one, so `full` silently failed to
// honour its documented contract whenever a row had no diagonal entry at all.
static void full_forces_absent_diagonal() {
    // 3x3 matrix with entries only at (0,1) and (2,0); rows 0, 1 and 2 all
    // lack an explicit diagonal entry.
    std::vector<double> data{4., 6.};
    std::vector<int_t> row{0, 2}, col{1, 0};

    sparse_coo_mat<double, int_t> coo{3, 2, row.data(), col.data(), data.data()};
    sparse_csr_mat<double, int_t> csr(coo, false, true);

    REQUIRE(csr.is_valid());
    REQUIRE(csr.nnz == 5); // 2 original entries + 3 synthesized zero diagonals

    for(int_t r = 0; r < 3; ++r) REQUIRE(std::find(csr.col_idx + csr.row_ptr[r], csr.col_idx + csr.row_ptr[r + 1], r) != csr.col_idx + csr.row_ptr[r + 1]);
}

// Constructing from an invalid (or empty, nnz == 0) COO must yield a
// well-defined object. Regression: the raw pointers used to be left
// uninitialized on the early-return path, so is_valid() could report true on
// dangling stack pointers and reading the matrix was undefined behaviour.
static void invalid_input_is_not_valid() {
    sparse_coo_mat<double, int_t> coo{3, 0, nullptr, nullptr, nullptr};
    sparse_csr_mat<double, int_t> csr(coo, false, false);

    REQUIRE(!csr.is_valid());
    REQUIRE(csr.row_ptr == nullptr);
    REQUIRE(csr.col_idx == nullptr);
    REQUIRE(csr.data == nullptr);
}

// A default-constructed matrix must also be well-defined (null pointers).
static void default_constructed_is_not_valid() {
    sparse_csr_mat<double, int_t> csr;

    REQUIRE(!csr.is_valid());
    REQUIRE(csr.row_ptr == nullptr);
    REQUIRE(csr.col_idx == nullptr);
    REQUIRE(csr.data == nullptr);
}

#ifdef EZP_ENABLE_TEST
TEST_CASE("Valid zero-based sparse CSR", "[Sparse CSR]") {
    valid_zero_based();
}

TEST_CASE("Valid one-based sparse CSR", "[Sparse CSR]") {
    valid_one_based();
}

TEST_CASE("Full flag forces absent diagonal entries", "[Sparse CSR]") {
    full_forces_absent_diagonal();
}

TEST_CASE("Invalid COO input is not valid CSR", "[Sparse CSR]") {
    invalid_input_is_not_valid();
}

TEST_CASE("Default-constructed CSR is not valid", "[Sparse CSR]") {
    default_constructed_is_not_valid();
}
#else
int main() {
    valid_zero_based();
    valid_one_based();
    full_forces_absent_diagonal();
    invalid_input_is_not_valid();
    default_constructed_is_not_valid();
    return 0;
}
#endif
