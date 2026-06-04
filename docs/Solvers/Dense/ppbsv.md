# ppbsv

The `ppbsv` solver supports the following input types.

* data type: DSZC
* index type: `std::int32_t`, `std::int64_t`

The `ppbsv` solves a square real banded symmetric positive definite distributed matrix with bandwidth $KLU$ using Cholesky factorization.
The matrix of size $N$ is stored in a memory block of size $N \times (KLU+1)$.

## Constructor

There are three template arguments.

1. data type, e.g., `double`, `float`, `std::complex<double>`, `std::complex<float>`.
2. index type, e.g., `std::int32_t`, `std::int64_t`.
3. `UPLO` flag indicating which triangular half is stored, `U` (upper) or `L` (lower).

This solver uses a 1D process grid.
It takes a single argument that represents the number of rows in the grid.

```cpp
auto solver = ppbsv<double, int, 'U'>(np_row);
```

Since the matrix is symmetric, `ScaLAPACK` expects only half of the matrix.
Thus the caller must provide the matrix exactly stored in a contiguous memory block of size $N \times (KLU+1)$.
For different `UPLO` flags, the internal storage layout varies.

## Indexer

A nested `indexer` converts logical 2D indices $(i, j)$ into the correct 1D offset for the symmetric band storage layout, respecting the `UPLO` flag.

```cpp
const auto IDX = par_dpbsv<int>::indexer{N, KLU};
A.resize(N * (KLU + 1), 0.);
for(auto I = 0; I < N; ++I) A[IDX(I, I)] = I + 1;
```

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
