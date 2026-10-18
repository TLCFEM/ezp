<h1 align="center">
  <br>
  ezp
  <br>
</h1>

<h3 align="center">Distributed linear solvers, distilled into one line of C++.</h3>

<p align="center">
  <em>Header-only · C++20 · MPI + ScaLAPACK, without the boilerplate.</em>
</p>

<p align="center">
  <a href="https://codecov.io/gh/TLCFEM/ezp"><img src="https://codecov.io/gh/TLCFEM/ezp/graph/badge.svg?token=ME0M312F5M" alt="codecov"></a>
  <a href="https://github.com/TLCFEM/ezp/actions/workflows/master.yml"><img src="https://github.com/TLCFEM/ezp/actions/workflows/master.yml/badge.svg?branch=master" alt="master"></a>
  <a href="https://www.gnu.org/licenses/gpl-3.0"><img src="https://img.shields.io/badge/license-GPLv3-blue.svg" alt="License: GPL v3"></a>
  <img src="https://img.shields.io/badge/C%2B%2B-20-blue.svg" alt="C++20">
  <img src="https://img.shields.io/badge/platform-Linux-informational.svg" alt="Platform: Linux">
</p>

<p align="center">
  <a href="https://tlcfem.github.io/ezp/">Documentation</a> &nbsp;·&nbsp;
  <a href="#quick-start">Quick start</a> &nbsp;·&nbsp;
  <a href="#supported-solvers">Solvers</a> &nbsp;·&nbsp;
  <a href="#standalone-solvers">Standalone solvers</a>
</p>

---

`ezp` is a lightweight, header-only C++ wrapper around a curated set of **distributed** linear-system solvers.
It hides the `BLACS`, `ScaLAPACK`, `MPI` and low-level `C`/`Fortran` minutiae behind an interface that feels just like solving a system sequentially.

```cpp
solver.solve(A, B);   // that's the whole idea
```

## Why `ezp`?

You may be interested if you...

- 🧩 want to **solve linear systems on distributed clusters** without memorising `ScaLAPACK` calling conventions.
- 🛠️ want to **avoid the hassles** of learning how to call various low-level `C`/`Fortran` solvers and managing their buffers by hand.
- 🌐 want an **object-oriented, `MPI`-aware** API that resembles the `LAPACK` experience you already know.
- 🚀 want to **separate the solver from the application** through standalone solver executables.

## Features

| | Feature |
|:--:|---|
| ✨ | **Ergonomic interface** — prepare $A$ and $B$ on the root process; everything else is automatic |
| 📦 | **Drop-in, header-only library** — no build step, no ABI to manage; just `#include` |
| 🧮 | **Full precision coverage** — all four `DSZC` precisions, with 32-bit *and* 64-bit indexing |
| 🖥️ | **Standalone solver binaries** — invoke the solver from any `MPI`-capable caller, including Python via `mpi4py` |
| 🧠 | **Build once, solve many** — factorization is retained and reused across right-hand sides |
| ✅ | **Validated with randomly generated inputs** — an extensive test suite backs every solver |

The library implements all precisions with both 32-bit and 64-bit integer indexing, and supports both row- and column-major process grids.

## Supported Solvers

### Dense (ScaLAPACK)

| availability | type of matrix                             | operation | solver  | package   |
|:------------:|--------------------------------------------|-----------|---------|-----------|
|      ✅       | general (partial pivoting)                 | simple    | PxGESV  | ScaLAPACK |
|      ✅       | general (partial pivoting)                 | expert    | PxGESVX | ScaLAPACK |
|      ✅       | symmetric/Hermitian positive definite      | simple    | PxPOSV  | ScaLAPACK |
|      ✅       | symmetric/Hermitian positive definite      | expert    | PxPOSVX | ScaLAPACK |
|      ✅       | general band (partial pivoting)            | simple    | PxGBSV  | ScaLAPACK |
|      ✅       | general band (no pivoting)                 | simple    | PxDBSV  | ScaLAPACK |
|      ✅       | symmetric/Hermitian positive definite band | simple    | PxPBSV  | ScaLAPACK |

### Sparse

| availability | type of matrix                             | operation | solver  | package   |
|:------------:|--------------------------------------------|-----------|---------|-----------|
|      ✅       | sparse (one- or zero-indexing CSR format)  | direct    | PARDISO | MKL       |
|      ✅       | sparse (one-indexing COO format)           | direct    | MUMPS   | MUMPS     |
|      ✅       | sparse (zero-indexing CSR format)          | iterative | Lis     | Lis       |

Expert drivers additionally provide equilibration, iterative refinement, condition estimation and determinant computation where applicable.

## Quick Start

`ezp` requires a C++20-compatible compiler and the following drivers:

1. an implementation of `LAPACK` / `BLAS`, such as `OpenBLAS`, `MKL`, `AOCL`, ...
2. an implementation of `ScaLAPACK`;
3. an implementation of `MPI`, such as `OpenMPI`, `MPICH`, `Intel® MPI`, ...

The easiest setup is the [Intel® oneAPI toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html), which supplies `MPI`, `oneMKL` (with a complete `ScaLAPACK`) in one shot:

```bash
git clone --recurse-submodules --depth 1 https://github.com/TLCFEM/ezp.git
mkdir ezp/build && cd ezp/build
. /opt/intel/oneapi/setvars.sh && cmake -DEZP_TEST=ON .. && cmake --build . --config Release
```

Using system libraries is also possible — see the [compilation guide](https://tlcfem.github.io/ezp/Intro/Compilation/) for `openSUSE`, `Fedora` and other distros.

## Minimal Example

It is assumed that the root node (rank 0) prepares the left-hand side $A$ and right-hand side $B$.
The solver then distributes the matrices across the available processes, solves the system, and returns the solution **back to the root node**.

All `BLACS` and `ScaLAPACK` details are hidden: prepare the matrices (on the root node), wrap them, and call `solve`.
The result highly resembles how one would solve a linear system sequentially.

```cpp
#include <ezp/pgesv.hpp>
#include <iomanip>
#include <iostream>

using namespace ezp;

int main() {
    // get the current blacs environment
    const auto env = get_env<int>();

    constexpr auto N = 6, NRHS = 2;

    // storage for the matrices A and B
    std::vector<double> A, B;

    // helper functor to convert 2D indices to 1D indices
    // it's likely the matrices are provided by some other subsystem
    const auto IDX = par_dgesv<int>::indexer{N};

    if(0 == env.rank()) {
        // the matrices are only initialized on the root process
        A.resize(N * N, 0.);
        B.resize(N * NRHS, 1.);

        for(auto I = 0; I < N; ++I) A[IDX(I, I)] = static_cast<double>(I + 1);
    }

    // create a parallel solver
    // it takes the number of rows and columns of the process grid as arguments,
    // or let the library automatically determine them
    auto solver = par_dgesv<int>();

    // wrap the data in full_mat objects:
    // number of rows, number of columns, and a pointer to the data
    // on non-root processes the pointer is nullptr, as the vectors are empty
    const auto info = solver.solve({N, N, A.data()}, {N, NRHS, B.data()});

    // reuse the stored factorization to compute the determinant
    const auto det = solver.det({N, N, A.data()});

    if(0 == env.rank() && 0 == info) {
        std::cout << std::setprecision(10) << "Determinant: " << det << '\n';
        std::cout << "Solution:\n";
        for(const auto i : B) std::cout << i << '\n';
    }

    return info;
}
```

More runnable examples live in [`examples/`](examples/), covering every solver and input format.

## Standalone Solvers

`MPI` can dynamically spawn worker processes via `MPI_Comm_spawn`.
`ezp` exploits this to ship its solvers as **standalone executables**, cleanly separating the main application from the solver.

<p align="center">
  <img src="https://raw.githubusercontent.com/TLCFEM/ezp/master/docs/Guide/communication.svg" alt="Standalone solver architecture" width="640">
</p>

This is particularly handy when the main application runs a different parallelism model, has latency constraints, or simply wants a clear boundary around the solver.
Because the boundary is a plain `MPI` interface, callers can be written in any `MPI`-compatible language — the bundled [`runner.py`](standalone/runner.py) demonstrates doing it from Python with [`mpi4py`](https://github.com/mpi4py/mpi4py).

Enable them at configure time with `-DEZP_STANDALONE=ON`; see the [standalone guide](https://tlcfem.github.io/ezp/Guide/Standalone/).

## Documentation

Full guides, per-solver references and the API docs are published at **<https://tlcfem.github.io/ezp/>**:

- [Motivation](https://tlcfem.github.io/ezp/Intro/Motivation/) and [Compilation](https://tlcfem.github.io/ezp/Intro/Compilation/)
- [Usage](https://tlcfem.github.io/ezp/Guide/Usage/), [Custom matrices](https://tlcfem.github.io/ezp/Guide/Custom/), [MPI management](https://tlcfem.github.io/ezp/Guide/MPI/), [Standalone solvers](https://tlcfem.github.io/ezp/Guide/Standalone/)
- Per-solver reference for [dense](https://tlcfem.github.io/ezp/Solvers/Dense/pgesv/) and [sparse](https://tlcfem.github.io/ezp/Solvers/Sparse/mumps/) solvers

> [!IMPORTANT]
> * Use branch [`master`](https://github.com/TLCFEM/ezp/tree/master) for development and testing.
> * Use branch [`lite-lib`](https://github.com/TLCFEM/ezp/tree/lite-lib) for integration into other projects:
>   ```bash
>   git clone --recursive --branch lite-lib https://github.com/TLCFEM/ezp.git
>   ```

## License

`ezp` is distributed under the terms of the **GNU General Public License v3.0**.
See [`LICENSE`](LICENSE) for details.

<p align="center"><sub>Copyright © 2025–2026 Theodore Chang</sub></p>
