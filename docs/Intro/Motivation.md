# Motivation

As of this writing, it is necessary to directly call `ScaLAPACK` subroutines via the `C` interface if one wants to solve linear systems using `ScaLAPACK`.
Since both `MPI` and `LAPACK`/`BLAS` have fabulous `C++` wrappers (see [`mpl`](https://github.com/rabauke/mpl) and [`Armadillo`](https://arma.sourceforge.net/download.html)), it would be nice if `ScaLAPACK` can be used in a similar OO manner.

Such a wrapper is advantageous and beneficial in the context of `C++` applications due to the following reasons.

1. It avoids direct interactions with the `C` API, which requires manual memory management.
   The caller has to follow pretty much the same procedural style.
2. A well-designed OO approach can significantly reduce the code size.

`ezp` is designed to provide an intuitive interface that resembles the non-MPI version of linear systems solvers.
The majority of `ScaLAPACK`, `BLACS` and `MPI` communication details shall be well hidden from the caller so that solving a system can be as simple as `solver.solve(A, B)`.
Of course, a complete, working example would be longer than a single line of code, due to the distributed-memory nature of MPI.
However, as shown in the front page, a minimum example spans merely a few lines of code.

```cpp
#include <ezp/pgesv.hpp>
#include <iostream>

int main() {
    constexpr auto N = 6, NRHS = 2;

    std::vector<double> A, B;

    if(0 == ezp::get_env<int>().rank()) {
        A.resize(N * N, 0.);
        B.resize(N * NRHS, 1.);

        const auto IDX = ezp::par_dgesv<int>::indexer{N};
        for(auto I = 0; I < N; ++I) A[IDX(I, I)] = I;
    }

    ezp::par_dgesv<int>().solve({N, N, A.data()}, {N, NRHS, B.data()});

    if(0 == ezp::get_env<int>().rank()) for(const auto i : B) std::cout << i << '\n';

    return 0;
}
```

## Available Solvers

`ezp` imeplements all precisions (DSZC) with both 32-bit and 64-bit indexing.

### Dense Matrix

`ezp` provides all solvers listed in [ScaLAPCK]{https://www.netlib.org/scalapack/slug/node44.html} page for **full** and **band** matrices.
Solvers for tridiagonal matrices are not implemented as they can be stored in band formats and solved by the corresponding solver.

### Sparse Matrix

`ezp` provides interface to `PARDISO` solver bundled in [Intel® oneAPI Math Kernel Library (oneMKL)](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2025-1/cluster-sparse-solver.html).

`ezp` provides interface to [`MUMPS`](https://mumps-solver.org/) solver.

### Why not SuperLU?

The codebase is messy and hard to maintain due to, for example, abuse of macros.
And it is not significantly faster than other direct solvers.

## References

1. [Configuration of a linear solver for linearly implicit time integration and efficient data transfer in parallel thermo-hydraulic computations](https://mediatum.ub.tum.de/doc/1486743/0996759907923.pdf)
2. [A Parallel Geometric Multifrontal Solver Using Hierarchically Semiseparable Structure](https://doi.org/10.1145/2830569)
3. [Caveats of three direct linear solvers for finite element analyses](https://doi.org/10.1002/nme.7545)
