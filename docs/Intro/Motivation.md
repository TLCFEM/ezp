# Motivation

To solve a linear system with `ScaLAPACK`, one currently has to call its subroutines directly through the `C` interface.
This is not a trivial task compared to solving the same problem sequentially with `LAPACK`.
One must figure out how to distribute the matrix over the process grid and allocate the required buffers of appropriate sizes.
This demands a substantial amount of boilerplate code that is tedious to write and error-prone.

Both `MPI` and `LAPACK`/`BLAS` already enjoy excellent `C++` wrappers, such as [`mpl`](https://github.com/rabauke/mpl) and [`Armadillo`](https://arma.sourceforge.net/download.html).
It would be desirable for `ScaLAPACK` to offer a similar object-oriented experience.
Such a wrapper benefits `C++` applications in the following ways.

1. It avoids direct interaction with the `C` API and its manual memory management.
2. A well-designed object-oriented approach significantly reduces the amount of code the caller writes.

`ezp` provides an intuitive interface that resembles the non-distributed version of linear system solvers.
Most of the `ScaLAPACK`, `BLACS`, and `MPI` communication details are hidden from the caller.
As a result, solving a system can be as simple as `solver.solve(A, B)`.
A complete working example is necessarily longer than a single line because of the distributed-memory nature of `MPI`.
However, as shown below, a minimal example spans only a few lines.

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

`ezp` implements dense and sparse solvers in all four `DSZC` precisions, with both 32-bit and 64-bit indexing.

### Dense Matrix

Dense solvers are built on top of `ScaLAPACK`.
Both **full** and **band** matrices are supported, and both row-major and column-major process grids are available.
Tridiagonal matrices are not implemented separately, because they can be stored in band format and solved by the corresponding band solver.

| Solver    | Matrix                                           | Decomposition         | Grid    |
|-----------|--------------------------------------------------|-----------------------|---------|
| `pgesv`   | General (partial pivoting)                       | LU                    | 2D      |
| `pgesvx`  | General (expert, equilibration + refinement)     | LU                    | 2D      |
| `pposv`   | Symmetric/Hermitian positive definite            | Cholesky              | 2D      |
| `pposvx`  | SPD (expert, equilibration + refinement)         | Cholesky              | 2D      |
| `pgbsv`   | General band (partial pivoting)                  | LU                    | 1D      |
| `pdbsv`   | General band (no pivoting)                       | LU                    | 1D      |
| `ppbsv`   | Symmetric/Hermitian positive definite band       | Cholesky              | 1D      |

### Sparse Matrix

Sparse solvers cover both direct and iterative methods.

- `pardiso` wraps the Intel MKL Cluster Sparse Solver and accepts CSR input in one- or zero-based indexing.
- `mumps` wraps the `MUMPS` direct solver and accepts COO input in one-based indexing.
- `lis` wraps the `Lis` iterative solver and accepts CSR input in zero-based indexing.

### Why not SuperLU?

The codebase is considered messy and hard to maintain.
It is not significantly faster than the other direct solvers, at least for the use cases `ezp` targets.

## References

1. [Configuration of a linear solver for linearly implicit time integration and efficient data transfer in parallel thermo-hydraulic computations](https://mediatum.ub.tum.de/doc/1486743/0996759907923.pdf)
2. [A Parallel Geometric Multifrontal Solver Using Hierarchically Semiseparable Structure](https://doi.org/10.1145/2830569)
3. [Caveats of three direct linear solvers for finite element analyses](https://doi.org/10.1002/nme.7545)
4. [Evaluating Accuracy and Efficiency of HPC Solvers for Sparse Linear Systems with Applications to PDEs](https://doi.org/10.48550/arXiv.2201.05413)
