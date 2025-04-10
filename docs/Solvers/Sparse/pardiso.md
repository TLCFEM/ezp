# pardiso

The `pardiso` solver supports the following input types.

* data type: DSZC
* index type: `std::int32_t`, `std::int64_t`

## Solver Options

There are two ways to configure the solver.

### Member Methods

The `operator()` operator provides access to `iparm` array.
There are explicitly named memeber methods, which give the same functionality.

```cpp
solver(26) = 1;
solver.iparm_matrix_checker(1);
```

When using the explicit methods, it is possible to chain multiple methods together.

```cpp
solver.iparm_matrix_checker(1).iparm_iterative_refinement(2);
```

### Verbose String Arguments

The `pardiso.parser.hpp` provides a parsing function.
Setting parameters can be done via the following.

```cpp
#include <ezp/pardiso.parser.hpp>

pardiso_set("--matrix-checker 1 --iterative-refinement 2", solver);
```

All available options are listed as follows.

```text
Usage: [--default-value INT] [--reducing-ordering INT] [--user-permutation INT] [--iterative-refinement INT] [--pivoting-perturbation INT] [--scaling INT] [--transpose-matrix INT] [--weighted-matching INT] [--nnz-factor INT] [--pivoting-type INT] [--matrix-checker INT] [--partial-solve INT] [--zero-based-indexing INT] [--schur-complement INT] [--out-of-core INT]

Optional arguments:
  --default-value INT          [0] Use default values. [default: 0]
  --reducing-ordering INT      [1] Fill-in reducing ordering for the input matrix. [default: 2]
  --user-permutation INT       [4] User permutation. [default: 0]
  --iterative-refinement INT   [7] Iterative refinement step. [default: 0]
  --pivoting-perturbation INT  [9] Pivoting perturbation.
  --scaling INT                [10] Scaling vectors. [default: 1]
  --transpose-matrix INT       [11] Solve with transposed or conjugate transposed matrix A. [default: 0]
  --weighted-matching INT      [12] Improved accuracy using (non-)symmetric weighted matching. [default: 1]
  --nnz-factor INT             [17] Report the number of non-zero elements in the factors. [default: -1]
  --pivoting-type INT          [20] Pivoting for symmetric indefinite matrices. [default: 1]
  --matrix-checker INT         [26] Matrix checker. [default: 0]
  --partial-solve INT          [30] Partial solve and computing selected components of the solution vectors. [default: 0]
  --zero-based-indexing INT    [34] One- or zero-based indexing of columns and rows. [default: 0]
  --schur-complement INT       [35] Schur complement matrix computation control. [default: 0]
  --out-of-core INT            [59] Solver mode. [default: 0]
```
