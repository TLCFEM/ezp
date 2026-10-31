# pposvx

The `pposvx` solver supports the following input types.

- data type: DSZC
- index type: `std::int32_t`, `std::int64_t`

`pposvx` solves a symmetric/Hermitian positive definite system `A * X = B` using Cholesky decomposition.
The matrix `A` of size $N$ is stored in a memory block of size $N^2$.
It is the expert driver, and it can apply equilibration and iterative refinement.

## Constructor

The solver takes four template arguments.

1. data type, e.g., `double`, `float`, `std::complex<double>`, `std::complex<float>`;
2. index type, e.g., `std::int32_t`, `std::int64_t`;
3. symmetry flag, `U` or `L`;
4. process grid order, `R` or `C`.

The solver uses a 2D process grid.
The grid order can be either `R`ow major or `C`olumn major.
The symmetry flag indicates whether the `U`pper or `L`ower triangle of the matrix is stored.
If only half of the matrix is populated, this flag must be set properly.
If the whole symmetric matrix is populated, either value works.
With `np_row` and `np_col` denoting the numbers of rows and columns of the process grid, a solver is constructed as follows.

```cpp
auto solver = pposvx<double, int, 'L', 'R'>(np_row, np_col);
```

The library can also determine the grid dimensions automatically.

```cpp
auto solver = pposvx<double, int, 'L', 'R'>();
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

The factorization **cannot** be reused.
Every call to `solve(full_mat<DT, IT>&& A, full_mat<DT, IT>&& B)` invalidates the previous factorization and refactorizes even if $A$ does not change.
This is a costly solver.
If `pposv` suffices, use `pposvx` only in advanced cases.
On return, `B` is replaced by the solution.
In the example above, `std::vector<double> B` then contains the solution.
