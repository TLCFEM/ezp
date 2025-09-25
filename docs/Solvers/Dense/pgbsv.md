# pgbsv

The `pgbsv` solver supports the following input types.

* data type: DSZC
* index type: `std::int32_t`, `std::int64_t`

The `pgbsv` solves a square real banded distributed matrix with bandwidth $KL$, $KU$ using LU decomposition with partial pivoting.
The matrix of size $N$ is stored in a memory block of size $N*(KL+KU+1)$.

## Constructor

There are two template arguments.

1. data type, e.g., `double`, `float`, `std::complex<double>`, `std::complex<float>`.
2. index type, e.g., `std::int32_t`, `std::int64_t`.

This solver uses a 1D process grid.
It takes a single argument that represents the number of rows in the grid.

```cpp
auto solver = pgbsv<double, int>(np_row);
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
