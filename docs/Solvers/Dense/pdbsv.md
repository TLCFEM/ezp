# pdbsv

The `pdbsv` solver supports the following input types.

* data type: DSZC
* index type: `std::int32_t`, `std::int64_t`

The `pdbsv` solves a square real banded diagonally dominant-like distributed matrix with bandwidth $KL$, $KU$ using LU decomposition with **no pivoting**.
The matrix of size $N$ is stored in a memory block of size $N \times (KL + KU + 1)$.

!!! note
    `pdbsv` does not perform partial pivoting.
    It is faster than `pgbsv` but requires the matrix to be diagonally dominant or positive definite to be numerically stable.
    Use `pgbsv` for general banded matrices.

## Constructor

There are two template arguments.

1. data type, e.g., `double`, `float`, `std::complex<double>`, `std::complex<float>`.
2. index type, e.g., `std::int32_t`, `std::int64_t`.

This solver uses a 1D process grid.
It takes a single argument that represents the number of rows in the grid.

```cpp
auto solver = pdbsv<double, int>(np_row);
```

## Indexer

A nested `indexer` converts logical 2D indices $(i, j)$ into the correct 1D offset for the `ScaLAPACK` band storage layout.
See [Fig. 4.9](https://netlib.org/scalapack/slug/node84.html) for the exact storage format.
The leading dimension of the stored matrix is `KL + KU + 1`.

```cpp
const auto IDX = par_ddbsv<int>::indexer{N, KL, KU};
A.resize(N * (KL + KU + 1), 0.);
for(auto I = 0; I < N; ++I) A[IDX(I, I)] = I + 1;
```

## Solving

```cpp
constexpr auto N = 6, NRHS = 2, KL = 1, KU = 2;
// storage for the matrices A and B
std::vector<double> A, B;
// ...
// ... storage is populated by some utility on the root process
// ...
const auto info = solver.solve({N, N, KL, KU, A.data()}, {N, NRHS, B.data()});
```
