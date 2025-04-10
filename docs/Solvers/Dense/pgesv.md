# pgesv

The `pgesv` solver supports the following input types.

* data type: DSZC
* index type: `std::int32_t`, `std::int64_t`

The `pgesv` solves a square matrix using LU decomposition with partial pivoting.
The matrix of size $N$ is stored in a memory block of size $N^2$.

## Constructor

There are three template arguments.

1. data type, e.g., `double`, `float`, `std::complex<double>`, `std::complex<float>`.
2. index type, e.g., `std::int32_t`, `std::int64_t`.
3. proccess grid order, `R` or `C`.

This solver uses a 2D process grid, so one can choose from `R`ow major or `C`olumn major ordering.

With `np_row` and `np_col` denoting the numbers of rows and columns of the process grid, a solver can be constructed using the following.

```cpp
auto solver = pgesv<double, int, 'R'>(np_row, np_col);
```

It is possible to let the library automatically determine the numbers of rows and columns, then the solver can be constructed as

```cpp
auto solver = pgesv<double, int, 'R'>();
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

The factorization will be stored internally as long as the solver object stays alive.
Thus to reuse the factorization, a second solve can call the following.

```cpp
const auto info = solver.solve({N, NRHS, B.data()});
```

A second call to `solve(full_mat<DT, IT>&& A, full_mat<DT, IT>&& B)` will clear previous factorizations automatically.

On return, `B` will be replaced by the solution.
In the above example, `std::vector<double> B` will contain the solution.
