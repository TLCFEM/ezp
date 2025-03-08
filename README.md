# ezp

[![codecov](https://codecov.io/gh/TLCFEM/ezp/graph/badge.svg?token=ME0M312F5M)](https://codecov.io/gh/TLCFEM/ezp)

`ezp` is a lightweight C++ wrapper for selected ScaLAPACK solvers for linear systems.

## Dependency

The `ezp` library is header only.
The following drivers are needed.

1. an implementation of `LAPACK` and `BLAS`, such as `OpenBLAS`, `MKL`, etc.
2. an implementation of `ScaLAPACK`
3. an implementation of `MPI`, such as `OpenMPI`, `MPICH`, etc.

## Example

It is assumed that the root node (rank 0) prepares the left hand side $$A$$ and right hand side $$B$$.
The solvers distrbute the matrices to available processes and solve the system, return the solution back to the master node.

The solvers are designed in such a way that all `BLACS` and `ScaLAPACK` details are hidden.
One shall prepare the matrices (**on the root node**) and call the solver.
The following is a typical example.
It highly resembles the sequential version of how one would typically solve a linear system.

```cpp
#include <ezp/pgesv.hpp>
#include <iomanip>
#include <iostream>

using namespace ezp;

int main() {
    // get the current blacs environment
    const auto& env = get_env<int>();

    constexpr auto N = 6, NRHS = 2;

    // storage for the matrices A and B
    std::vector<double> A, B;

    if(0 == env.rank()) {
        // the matrices are only initialized on the root process
        A.resize(N * N, 0.);
        B.resize(N * NRHS, 1.);

        // helper functor to convert 2D indices to 1D indices
        const auto IDX = par_dgesv<int>::indexer{N};

        for(auto I = 0; I < N; ++I) A[IDX(I, I)] = static_cast<double>(I);
    }

    // create a parallel solver
    // it takes the number of rows and columns of the process grid as arguments
    // or let the library automatically determine as follows
    auto solver = par_dgesv<int>();

    // need to wrap the data in full_mat objects
    // it requires the number of rows and columns of the matrix, and a pointer to the data
    // on non-root processes, the data pointer is nullptr as the vector is empty
    // solver.solve(full_mat{N, N, A.data()}, full_mat{N, NRHS, B.data()});
    const auto info = solver.solve({N, N, A.data()}, {N, NRHS, B.data()});

    if(0 == env.rank()) {
        std::cout << std::setprecision(10) << "Info: " << info << '\n';
        std::cout << "Solution:\n";
        for(auto i = 0; i < B.size(); ++i) std::cout << B[i] << '\n';
    }

    return 0;
}
```