# pardiso

The `pardiso` solver supports the following input types.

* data type: DSZC
* index type: `std::int32_t`, `std::int64_t`

`pardiso` wraps the [Intel MKL Cluster Sparse Solver](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2025-1/cluster-sparse-solver.html).
It requires the macro `EZP_MKL` to be defined, which is set automatically when MKL is detected.

## Constructor

```cpp
auto solver = ezp::pardiso<double, int_t>(
    ezp::matrix_type::real_and_nonsymmetric,
    ezp::message_level::no_output
);
```

The `matrix_type` enum specifies the structure of the matrix.
Available values:

| Enum | Description |
|------|-------------|
| `real_and_structurally_symmetric` | Structurally symmetric |
| `real_and_symmetric_positive_definite` | Symmetric positive definite |
| `real_and_symmetric_indefinite` | Symmetric indefinite |
| `real_and_nonsymmetric` | General non-symmetric |
| `complex_and_hermitian_positive_definite` | Hermitian positive definite |
| `complex_and_hermitian_indefinite` | Hermitian indefinite |
| `complex_and_structurally_symmetric` | Structurally symmetric complex |
| `complex_and_symmetric` | Symmetric complex |
| `complex_and_nonsymmetric` | General complex non-symmetric |

The `message_level` enum controls solver output: `no_output` or `print_statistical_information`.

## Input Format

The matrix $A$ must be provided as a `sparse_csr_mat` with **one-based** indexing.
Fields required:

* `n` — number of rows (and columns),
* `nnz` — number of non-zero entries,
* `row_ptr` — array of length `n+1` with one-based row pointers,
* `col_idx` — array of length `nnz` with one-based column indices,
* `data` — array of length `nnz`, values.

The right-hand side $B$ is a dense `full_mat` with `n_rows = N` and `n_cols = NRHS`.

## COO-to-CSR Conversion

If the matrix is available in COO format, it can be converted to CSR automatically.

```cpp
const ezp::sparse_coo_mat coo{N, N, ia.data(), ja.data(), a.data()};
// pass true for one-based indexing
solver.solve(ezp::sparse_csr_mat<double, int_t>{coo, true}, {N, NRHS, b.data()});
```

## Usage Example

```cpp
#include <ezp/pardiso.hpp>

auto solver = ezp::pardiso<double, int_t>(
    ezp::matrix_type::real_and_nonsymmetric,
    ezp::message_level::no_output
);

int_t N = 10, NRHS = 1;
std::vector<int_t> ia, ja;
std::vector<double> a, b;

// populate one-based CSR diagonal matrix on root process
if(0 == mpl::environment::comm_world().rank()) {
    ia.resize(N + 1);  ja.resize(N);  a.resize(N);  b.resize(N * NRHS);
    for(auto i = 0; i < N; i++) { ia[i] = ja[i] = static_cast<int_t>(a[i] = i + 1); }
    ia[N] = N + 1;
    std::ranges::fill(b, 1.);
}

const auto info = solver.solve({N, N, ia.data(), ja.data(), a.data()}, {N, NRHS, b.data()});
```

The factorization is retained internally and reused when `solve` is called again with a new right-hand side.

## Solver Options

There are two ways to configure the solver.

### Member Methods

The `operator()` operator provides access to `iparm` array.
There are explicitly named member methods, which give the same functionality.

```cpp
solver(26) = 1;
solver.iparm_matrix_checker(1);
```

When using the explicit methods, it is possible to chain multiple methods together.

```cpp
solver.iparm_matrix_checker(1).iparm_iterative_refinement(2);
```

### Verbose String Arguments

The `pardiso.parser.hpp` provides a parsing function.
Setting parameters can be done via the following.

```cpp
#include <ezp/pardiso.parser.hpp>

pardiso_set("--matrix-checker 1 --iterative-refinement 2", solver);
```

All available options are listed as follows.

```text
Usage: [--default-value INT] [--reducing-ordering INT] [--user-permutation INT] [--iterative-refinement INT] [--pivoting-perturbation INT] [--scaling INT] [--transpose-matrix INT] [--weighted-matching INT] [--nnz-factor INT] [--pivoting-type INT] [--matrix-checker INT] [--partial-solve INT] [--zero-based-indexing INT] [--schur-complement INT] [--out-of-core INT]

Optional arguments:
  --default-value INT          [0] Use default values. [default: 0]
  --reducing-ordering INT      [1] Fill-in reducing ordering for the input matrix. [default: 2]
  --user-permutation INT       [4] User permutation. [default: 0]
  --iterative-refinement INT   [7] Iterative refinement step. [default: 0]
  --pivoting-perturbation INT  [9] Pivoting perturbation.
  --scaling INT                [10] Scaling vectors. [default: 1]
  --transpose-matrix INT       [11] Solve with transposed or conjugate transposed matrix A. [default: 0]
  --weighted-matching INT      [12] Improved accuracy using (non-)symmetric weighted matching. [default: 1]
  --nnz-factor INT             [17] Report the number of non-zero elements in the factors. [default: -1]
  --pivoting-type INT          [20] Pivoting for symmetric indefinite matrices. [default: 1]
  --matrix-checker INT         [26] Matrix checker. [default: 0]
  --partial-solve INT          [30] Partial solve and computing selected components of the solution vectors. [default: 0]
  --zero-based-indexing INT    [34] One- or zero-based indexing of columns and rows. [default: 0]
  --schur-complement INT       [35] Schur complement matrix computation control. [default: 0]
  --out-of-core INT            [59] Solver mode. [default: 0]
```
