# pposvx

The `pposvx` solves a square matrix using LU decomposition with partial pivoting.
The matrix of size $N$ is stored in a memory block of size $N^2$.
The solver can apply equilibration and iterative refinement.
This is an expert solver.

## Constructor

There are four template arguments.

1. data type, e.g., `double`, `float`, `std::complex<double>`, `std::complex<float>`.
2. index type, e.g., `std::int32_t`, `std::int64_t`.
3. symmetry flag, e.g., `U`, `L`.
4. proccess grid order, `R` or `C`.

This solver uses a 2D process grid, so one can choose from `R`ow major or `C`olumn major ordering.

The symmetry flag can be either `U`pper or `L`ower.
If only half of the matrix is populated, this flag needs to be set properly.
If the whole symmetric matrix is populated, either one will work.

With `np_row` and `np_col` denoting the numbers of rows and columns of the process grid, a solver can be constructed using the following.

```cpp
auto solver = pposvx<double, int, 'L', 'R'>(np_row, np_col);
```

It is possible to let the library automatically determine the numbers of rows and columns, then the solver can be constructed as

```cpp
auto solver = pposvx<double, int, 'L', 'R'>();
```

## Solving

Prepare matrices $A$ and $B$ on the root process in whatever form of choice.
Those two matrices need to be wrapped into `template<data_t DT, index_t IT> struct full_mat` objects, which require

1. the number of rows of the matrix, `n_rows`,
2. the number of columns of the matrix, `n_cols`,
3. a pointer pointing to the data, `data`.

Then one can call the solver via

```cpp
constexpr auto N = 6, NRHS = 2;
// storage for the matrices A and B
std::vector<double> A, B;
// ...
// ... storage is populated by some utility on the root process
// ...
const auto info = solver.solve({N, N, A.data()}, {N, NRHS, B.data()});
```

The factorization **cannnot** be reused.
Thus, every call to `solve(full_mat<DT, IT>&& A, full_mat<DT, IT>&& B)` will invalidate the previous factorization and perform the factorization again even if the matrix $A$ does not change.
This is a costly solver, if `pposv` suffices, `pposvx` shall only be used in advanced cases.

On return, `B` will be replaced by the solution.
In the above example, `std::vector<double> B` will contain the solution.
