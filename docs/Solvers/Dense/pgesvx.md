# pgesvx

The `pgesvx` solver supports the following input types.

- data type: DSZC
- index type: `std::int32_t`, `std::int64_t`

`pgesvx` solves a square system `A * X = B` using LU decomposition with partial pivoting.
The matrix `A` of size $N$ is stored in a memory block of size $N^2$.
It is the expert driver, and it can apply equilibration and iterative refinement.

!!! warning
    The reference `ScaLAPACK` `pgesvx` has a known bug that may cause the program to hang.
    See [Reference-ScaLAPACK/scalapack#119](https://github.com/Reference-ScaLAPACK/scalapack/issues/119).
    In particular, enabling equilibration may hang the program.
    Check the status of the bug before setting the equilibration flag to `'E'`.

## Constructor

The solver takes four template arguments.

1. data type, e.g., `double`, `float`, `std::complex<double>`, `std::complex<float>`;
2. index type, e.g., `std::int32_t`, `std::int64_t`;
3. equilibration flag, `F`, `N`, or `E`;
4. process grid order, `R` or `C`.

The solver uses a 2D process grid.
The grid order can be either `R`ow major or `C`olumn major.
With `np_row` and `np_col` denoting the numbers of rows and columns of the process grid, a solver is constructed as follows.

```cpp
auto solver = pgesvx<double, int, 'E', 'R'>(np_row, np_col);
```

The library can also determine the grid dimensions automatically.

```cpp
auto solver = pgesvx<double, int, 'E', 'R'>();
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
If `pgesv` suffices, use `pgesvx` only in advanced cases.
On return, `B` is replaced by the solution.
In the example above, `std::vector<double> B` then contains the solution.
