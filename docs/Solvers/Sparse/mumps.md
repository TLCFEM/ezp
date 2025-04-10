# mumps

The `mumps` solver supports the following input types.

* data type: DSZC
* index type: `std::int32_t` (32-bit, typically `int`)

[`MUMPS`](https://mumps-solver.org/) supports both 32-bit and 64-bit integer indexing.
There are some caveats.

1. If 64-bit integer is enabled via `MUMPS_INTSIZE64`, all integers are 64-bit, thus MKL and MPI need to use 64-bit integer for indexing as well.
   This may cause compatibility problems as 1) MPI with 64-bit integer may not be available on various platforms, 2) other libraries may only need linkage to conventional 32-bit version.
2. Even if 64-bit integer is **not** enabled, `MUMPS` uses 64-bit integer for `nnz`, thus as long as the size of the matrix does not exceed 2 billion, the 32-bit integer version is sufficient.

To allow maximum compatibility, `MUMPS_INTSIZE64` is **not** enabled, even if `EZP_USE_64BIT_INT` is switched on.
Thus the underlying `mumps` library needs to be linked to 32-bit ScaLAPACK and MPI.

## Solver Options

There are two ways to configure the solver.

### Member Methods

The `operator()` operator provides access to `icntl` array.
There are explicitly named memeber methods, which give the same functionality.

```cpp
solver(3) = 1;
solver.icntl_printing_level(1);
```

When using the explicit methods, it is possible to chain multiple methods together.

```cpp
solver.icntl_printing_level(1).icntl_iterative_refinement(2);
```

### Verbose String Arguments

The `mumps.parser.hpp` provides a parsing function.
Setting parameters can be done via the following.

```cpp
#include <ezp/mumps.parser.hpp>

mumps_set("--printing-level 1 --iterative-refinement 2", solver);
```

All available options are listed as follows.

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
