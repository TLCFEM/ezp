# ppbsv

The `ppbsv` solver supports the following input types.

* data type: DSZC
* index type: `std::int32_t`, `std::int64_t`

The `ppbsv` solves a square real banded symmetric positive definite distributed matrix with bandwidth $KLU$ using Cholesky factorization.
The matrix of size $N$ is stored in a memory block of size $N*(KLU+1)$.

## Constructor

There are two template arguments.

1. data type, e.g., `double`, `float`, `std::complex<double>`, `std::complex<float>`.
2. index type, e.g., `std::int32_t`, `std::int64_t`.
3. flag to indicate which half is stored, `U` or `L`.

This solver uses a 1D process grid.
It takes a single argument that represents the number of rows in the grid.

```cpp
auto solver = ppbsv<double, int, 'U'>(np_row);
```

Since the matrix is symmetric, `ScaLAPCK` expect only half of the matrix.
Thus the caller must provide the matrix exactly stored in contiguous memory block $N*(KLU+1)$.
For different `UPLO` flags, the internal storage varies.

## Solving

```cpp
constexpr auto N = 6, NRHS = 2, KLU = 2;
// storage for the matrices A and B
std::vector<double> A, B;
// ...
// ... storage is populated by some utility on the root process
// ...
const auto info = solver.solve({N, N, KLU, A.data()}, {N, NRHS, B.data()});
```
