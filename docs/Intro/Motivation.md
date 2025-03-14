# Motivation

As of this writing, it is necessary to directly call `ScaLAPACK` subroutines via the `C` interface if one wants to solve linear systems using `ScaLAPACK`.
Since both `MPI` and `LAPACK`/`BLAS` have fabulous `C++` wrappers (see [`mpl`](https://github.com/rabauke/mpl) and [`Armadillo`](https://arma.sourceforge.net/download.html)), it would be nice if `ScaLAPACK` can be used in a similar OO manner.

Such a wrapper is advantageous and beneficial in the context of `C++` applications due to the following reasons.

1. It avoids direct interactions with the `C` API, which requires manual memory management.
   The caller has to follow pretty much the same procedural style.
2. A well designed OO approach can significantly reduce the code size.

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
