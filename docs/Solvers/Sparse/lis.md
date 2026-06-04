# lis

The `lis` solver supports the following input types.

* data type: D (`double` only)
* index type: `std::int32_t`, `std::int64_t`

`lis` is an iterative solver from the [Lis library](http://www.ssisc.org/lis/).
It requires a matrix in **zero-based CSR** (Compressed Sparse Row) format.

## Constructor

The constructor accepts an optional string of solver options (see [Solver Options](#solver-options) below).

```cpp
auto solver = lis("-i fgmres -p ilu");
```

## Input Format

The matrix $A$ must be provided as a `sparse_csr_mat` with **zero-based** indexing.
Fields required:

* `n` — number of rows (and columns) of the square matrix,
* `nnz` — number of non-zero entries,
* `row_ptr` — array of length `n+1`, where `row_ptr[i]` is the index of the first non-zero in row `i`,
* `col_idx` — array of length `nnz`, column indices of the non-zero entries,
* `data` — array of length `nnz`, values of the non-zero entries.

The right-hand side $B$ is a dense `full_mat` with `n_rows = N` and `n_cols = NRHS`.

## Usage Example

```cpp
#include <ezp/lis.hpp>

using namespace ezp;

int N = 10, NRHS = 1;
std::vector<int_t> ia, ja;
std::vector<double> a, b;

// populate zero-based CSR diagonal matrix on root process
if(0 == mpl::environment::comm_world().rank()) {
    ia.resize(N + 1);  ja.resize(N);  a.resize(N);  b.resize(N * NRHS);
    for(auto i = 0; i < N; i++) { ia[i] = ja[i] = i; a[i] = i + 1; }
    ia[N] = N;
    std::ranges::fill(b, 1.);
}

auto solver = lis("-print all -p ilu -ilu_fill 1 -i fgmres");
const auto info = solver.solve({N, N, ia.data(), ja.data(), a.data()}, {N, NRHS, b.data()});
```

## Solver Options

Call the `set_option` method to change solver options after construction.

```cpp
solver.set_option("-i fgmres -p ilu -print 2");
```

All available options are documented in the [official Lis documentation](https://www.ssisc.org/lis/).
Commonly used options include:

| Option | Description |
|--------|-------------|
| `-i <method>` | Iterative solver, e.g., `cg`, `bicgstab`, `gmres`, `fgmres` |
| `-p <precond>` | Preconditioner, e.g., `none`, `ilu`, `ssor` |
| `-ilu_fill <n>` | ILU fill-in level |
| `-maxiter <n>` | Maximum number of iterations |
| `-tol <val>` | Convergence tolerance |
| `-print <level>` | Verbosity level (0=none, 1=summary, 2=iterations) |
