# pgesv

The `pgesv` solver supports the following input types.

- data type: DSZC
- index type: `std::int32_t`, `std::int64_t`

`pgesv` solves a square system `A * X = B` using LU decomposition with partial pivoting.
The matrix `A` of size $N$ is stored in a memory block of size $N^2$.

## Type Aliases

For convenience, the following type aliases are provided.

```cpp
par_dgesv<IT>    // double precision
par_sgesv<IT>    // single precision
par_zgesv<IT>    // double-precision complex
par_cgesv<IT>    // single-precision complex
par_dgesv_c<IT>  // double precision, column-major process grid
// etc.
```

For example, `par_dgesv<int>` is equivalent to `pgesv<double, int, 'R'>`.

## Constructor

The solver takes three template arguments.

1. data type, e.g., `double`, `float`, `std::complex<double>`, `std::complex<float>`;
2. index type, e.g., `std::int32_t`, `std::int64_t`;
3. process grid order, `R` or `C`.

The solver uses a 2D process grid.
The grid order can be either `R`ow major or `C`olumn major.
With `np_row` and `np_col` denoting the numbers of rows and columns of the process grid, a solver is constructed as follows.

```cpp
auto solver = pgesv<double, int, 'R'>(np_row, np_col);
```

The library can also determine the grid dimensions automatically.

```cpp
auto solver = pgesv<double, int, 'R'>();
```

## Indexer

A nested `indexer` helper converts 2D row-major indices into the 1D column-major offsets required by `ScaLAPACK`'s Fortran-order storage.

```cpp
const auto IDX = par_dgesv<int>::indexer{N};
// A[IDX(i, j)] accesses row i, column j of an N×N column-major matrix
```

## Solving

Prepare matrices $A$ and $B$ on the root process in whatever form is convenient.
The two matrices must be wrapped into `template<data_t DT, index_t IT> struct full_mat` objects, which require

1. the number of rows of the matrix, `n_rows`;
2. the number of columns of the matrix, `n_cols`;
3. a pointer to the data, `data`.

Then call the solver.

```cpp
constexpr auto N = 6, NRHS = 2;
// storage for the matrices A and B
std::vector<double> A, B;
// ...
// ... storage is populated by some utility on the root process
// ...
const auto info = solver.solve({N, N, A.data()}, {N, NRHS, B.data()});
```

The factorization is stored internally as long as the solver object stays alive.
A second solve reuses the stored factorization.

```cpp
const auto info = solver.solve({N, NRHS, B.data()});
```

Calling `solve(full_mat<DT, IT>&& A, full_mat<DT, IT>&& B)` again clears the previous factorization automatically.
On return, `B` is replaced by the solution.
In the example above, `std::vector<double> B` then contains the solution.

## Determinant

`pgesv` also provides a `det()` method that computes the determinant from the stored LU factorization.
The factorization must be performed before `det()` is called.

```cpp
const auto info = solver.solve({N, N, A.data()}, {N, NRHS, B.data()});
// reuse the stored factorization for determinant computation
const auto determinant = solver.det({N, N, A.data()});
```

The determinant is computed on the root process and returned to all processes.
