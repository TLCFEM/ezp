# mumps

The `mumps` solver supports the following input types.

- data type: DSZC
- index type: `std::int32_t`

[`MUMPS`](https://mumps-solver.org/) supports both 32-bit and 64-bit integer indexing.
There are, however, some caveats.

1. If 64-bit integer is enabled via `MUMPS_INTSIZE64`, all integers become 64-bit, so MKL and MPI must also use 64-bit indexing.
   This may cause compatibility problems, because an MPI implementation with 64-bit integers is not available everywhere, and other libraries may only link against the conventional 32-bit version.
2. Even without 64-bit integer enabled, `MUMPS` uses a 64-bit integer for `nnz`.
   As long as the matrix size does not exceed two billion, the 32-bit integer version is sufficient.

`MUMPS_INTSIZE64` is **not** enabled, even when `EZP_USE_64BIT_INT` is switched on, to maximise compatibility.
The underlying `mumps` library therefore links against 32-bit `ScaLAPACK` and `MPI`.

## Constructor

```cpp
auto solver = mumps<double, int>();
// optionally specify the symmetry pattern:
//   mumps<double, int>(ezp::symmetric_positive_definite)
//   mumps<double, int>(ezp::symmetric_indefinite)
//   mumps<double, int>(ezp::unsymmetric)  // default
// and the parallel mode (whether the host process participates in the factorization):
//   mumps<double, int>(ezp::unsymmetric, ezp::host_involved)
```

The `symmetric_pattern` argument defaults to `unsymmetric`.
The `parallel_mode` argument defaults to `no_host`.
When a single process is used, the host is always forced to participate.

## Input Format

The matrix $A$ must be provided as a `sparse_coo_mat` with **one-based** indexing.
The required fields are:

- `n` — the number of rows (and columns);
- `nnz` — the number of non-zero entries;
- `row` — an array of length `nnz` with one-based row indices;
- `col` — an array of length `nnz` with one-based column indices;
- `data` — an array of length `nnz` holding the values.

The right-hand side $B$ is a dense `full_mat` with `n_rows = N` and `n_cols = NRHS`.

## Usage Example

```cpp
#include <ezp/mumps.hpp>

using namespace ezp;

int N = 10, NRHS = 1;
std::vector<int> ia, ja;
std::vector<double> a, b;

// populate one-based COO diagonal matrix on root process
if(0 == mpl::environment::comm_world().rank()) {
    ia.resize(N);  ja.resize(N);  a.resize(N);  b.resize(N * NRHS);
    for(auto i = 0; i < N; i++) ia[i] = ja[i] = a[i] = i + 1;
    std::ranges::fill(b, 1.);
}

auto solver = mumps<double, int>();
solver.icntl_printing_level(0);              // suppress output
solver.icntl_determinant_computation(1);     // enable determinant

const auto info = solver.solve({N, N, ia.data(), ja.data(), a.data()}, {N, NRHS, b.data()});

if(0 == mpl::environment::comm_world().rank() && info == 0) {
    std::cout << "sign(det): " << solver.sign_det() << '\n';
    for(const auto x : b) std::cout << x << '\n';
}
```

The factorization is retained internally.
After the first `solve`, a subsequent call to `solve` with a different right-hand side reuses the stored factorization.

## Determinant

When `icntl_determinant_computation(1)` is set, the determinant is available after solving.
`sign_det()` returns the sign of the determinant for real data types.
`det()` returns the full determinant as a `std::complex` value.

## Solver Options

There are two ways to configure the solver.

### Member Methods

The `operator()` operator provides access to the `icntl` array.
Explicitly named member methods give the same functionality.

```cpp
solver(3) = 1;
solver.icntl_printing_level(1);
```

The explicit methods can be chained.

```cpp
solver.icntl_printing_level(1).icntl_iterative_refinement(2);
```

### Verbose String Arguments

The `mumps.parser.hpp` header provides a parsing function.
Parameters can be set through a string.

```cpp
#include <ezp/mumps.parser.hpp>

mumps_set("--printing-level 1 --iterative-refinement 2", solver);
```

All available options are listed below.

```text
Usage: mumps [--output-error-message INT] [--output-diagnostic-statistics-warning INT] [--output-global-information INT] [--printing-level INT] [--permutation-and-scaling INT] [--symmetric-permutation INT] [--scaling-strategy INT] [--transpose-matrix INT] [--iterative-refinement INT] [--error-analysis INT] [--ordering-strategy INT] [--root-parallelism INT] [--working-space-percentage-increase INT] [--compression-block-format INT] [--openmp-threads INT] [--distribution-strategy-input INT] [--schur-complement INT] [--distribution-strategy-solution INT] [--out-of-core INT] [--maximum-working-memory INT] [--null-pivot-row-detection INT] [--deficient-and-null-space-basis INT] [--schur-complement-solution INT] [--rhs-block-size INT] [--ordering-computation INT] [--inverse-computation INT] [--forward-elimination INT] [--determinant-computation INT] [--out-of-core-file INT] [--blr INT] [--blr-variant INT] [--blr-compression INT] [--lu-compression-rate INT] [--block-compression-rate INT] [--tree-parallelism INT] [--compact-working-space INT] [--rank-revealing-factorization INT] [--symbolic-factorization INT]

Optional arguments:
  --output-error-message INT                  [1] output stream for error messages. [default: 6]
  --output-diagnostic-statistics-warning INT  [2] output stream for diagnostic printing and statistics local to each MPI process. [default: 0]
  --output-global-information INT             [3] output stream for global information, collected on the host. [default: 6]
  --printing-level INT                        [4] level of printing for error, warning, and diagnostic messages. [default: 2]
  --permutation-and-scaling INT               [6] permutes the matrix to a zero-free diagonal and/or scale the matrix. [default: 7]
  --symmetric-permutation INT                 [7] computes a symmetric permutation (ordering) to determine the pivot order to be used for the factorization in case of sequential analysis. [default: 7]
  --scaling-strategy INT                      [8] describes the scaling strategy. [default: 77]
  --transpose-matrix INT                      [9] computes the solution using A or transpose of A. [default: 1]
  --iterative-refinement INT                  [10] applies the iterative refinement to the computed solution. [default: 0]
  --error-analysis INT                        [11] computes statistics related to an error analysis of the linear system solved. [default: 0]
  --ordering-strategy INT                     [12] defines an ordering strategy for symmetric matrices and is used, in conjunction with ICNTL(6), to add constraints to the ordering algorithm. [default: 0]
  --root-parallelism INT                      [13] controls the parallelism of the root node (enabling or not the use of ScaLAPACK) and also its splitting. [default: 0]
  --working-space-percentage-increase INT     [14] controls the percentage increase in the estimated working space. [default: 30]
  --compression-block-format INT              [15] exploits compression of the input matrix resulting from a block format. [default: 0]
  --openmp-threads INT                        [16] controls the setting of the number of OpenMP threads by MUMPS when the setting of multithreading is not possible outside MUMPS. [default: 0]
  --distribution-strategy-input INT           [18] defines the strategy for the distributed input matrix. [default: 0]
  --schur-complement INT                      [19] computes the Schur complement matrix. [default: 0]
  --distribution-strategy-solution INT        [21] determines the distribution (centralized or distributed) of the solution vectors. [default: 0]
  --out-of-core INT                           [22] controls the in-core/out-of-core (OOC) factorization and solve. [default: 0]
  --maximum-working-memory INT                [23] corresponds to the maximum size of the working memory in MB that MUMPS can allocate per working process. [default: 0]
  --null-pivot-row-detection INT              [24] controls the detection of "null pivot rows". [default: 0]
  --deficient-and-null-space-basis INT        [25] allows the computation of a solution of a deficient matrix and also of a null space basis. [default: 0]
  --schur-complement-solution INT             [26] drives the solution phase if a Schur complement matrix has been computed. [default: 0]
  --rhs-block-size INT                        [27] controls the blocking size for multiple right-hand sides. [default: -32]
  --ordering-computation INT                  [28] determines whether a sequential or parallel computation of the ordering is performed. [default: 0]
  --inverse-computation INT                   [30] computes a user-specified set of entries in the inverse of the original matrix. [default: 0]
  --forward-elimination INT                   [32] performs the forward elimination of the right-hand sides during the factorization. [default: 0]
  --determinant-computation INT               [33] computes the determinant of the input matrix. [default: 0]
  --out-of-core-file INT                      [34] controls the conservation of the OOC files. [default: 0]
  --blr INT                                   [35] controls the activation of the BLR feature. [default: 0]
  --blr-variant INT                           [36] controls the choice of BLR factorization variant. [default: 0]
  --blr-compression INT                       [37] controls the BLR compression of the contribution blocks. [default: 0]
  --lu-compression-rate INT                   [38] estimates compression rate of LU factors. [default: 600]
  --block-compression-rate INT                [39] estimates compression rate of contribution blocks. [default: 500]
  --tree-parallelism INT                      [48] controls multithreading with tree parallelism. [default: 1]
  --compact-working-space INT                 [49] compacts workarray at the end of factorization phase. [default: 0]
  --rank-revealing-factorization INT          [56] detects pseudo-singularities during factorization and factorizes the root node with a rank-revealing method. [default: 0]
  --symbolic-factorization INT                [58] defines options for symbolic factorization. [default: 2]
```
