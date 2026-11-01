# AGENTS.md — ezp

`ezp` is a header-only C++20 wrapper around distributed solvers (ScaLAPACK, MUMPS, Lis, PARDISO). The public API is the headers in `ezp/`; the root CMake project only builds examples, tests, and standalone executables. Development branch: `master` (PRs/CI target it); `lite-lib` is the integration-only branch.

## Layout

- `ezp/*.hpp` — one header per solver (`pgesv`, `pgbsv`, `pposv`, `mumps`, `lis`, `pardiso`, ...). Umbrellas: `ezp/ezp`, `ezp/ezp.dense`, `ezp/ezp.sparse`.
- `ezp/abstract/` — shared solver bases (`abstract_solver.hpp`, `full_solver.hpp`, `band_solver.hpp`, `sparse_solver.hpp`, `traits.hpp`).
- `ezp/abstract/ezp.h` — raw C/Fortran ABI: `extern "C"` declarations plus `EZP(name)` symbol-mangling macros.
- `examples/`, `tests/`, `standalone/` — compiled into flat executables in the build directory.
- `include/catch2/` — vendored Catch2 amalgamation; do not edit.
- `external/{mumps,lis,gk}` — vendored C/Fortran dependencies; do not edit.
- `libs/` — prebuilt static ScaLAPACK/BLAS/LAPACK used as fallback.
- `mpl/` — git submodule (MPI wrapper, branch `feature-easy-life`) with its own `mpl/AGENTS.md`; changes belong upstream.

## Build

```bash
# oneAPI is the simplest way to provide MPI + LAPACK/BLAS + ScaLAPACK
. /opt/intel/oneapi/setvars.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DEZP_TEST=ON -DEZP_STANDALONE=ON
cmake --build build -j"$(nproc)"
```

- Clone/build with submodules present: `git submodule update --init --recursive` (only `mpl/` is a submodule; the `external/` trees are vendored).
- Requires MPI (`C` **and** `Fortran` components), LAPACK/BLAS, and ScaLAPACK; Linux only.
- MKL is detected automatically and defines `EZP_MKL`. Otherwise CMake links the prebuilt `libs/libscalapack-{mpich,openmpi}.a`, choosing by inspecting the MPI installation — it must match your MPI. Use `-DEZP_USE_SYSTEM_LIBS=ON` for system libraries.
- All targets land directly in `build/` (`example.pgesv`, `test.pgesv`, `catch2`, `solver.pgesv`); there are no per-directory output folders.
- CMake uses `file(GLOB ...)`: rerun the configure step after adding/removing an example, test, or standalone solver.
- Examples and tests are built whenever the project is top-level, even without `EZP_TEST`; `EZP_TEST=ON` additionally builds the aggregate `catch2` binary.
- GCC builds use `-Wall -Wextra -pedantic -Werror`; warnings must be fixed.

## Tests

There is no CTest integration — `ctest` will not find these tests.

```bash
# Full assertion-based suite (this is the real test run)
mpirun -n 2 ./build/catch2

# Focused runs using Catch2 test names/tags
mpirun -n 2 ./build/catch2 "Random PDGESV"
mpirun -n 2 ./build/catch2 "[Expert Solver]"
./build/catch2 "[Sparse CSR]"          # pure C++, no MPI needed

# Standalone variant of one test: argument is the repeat count
mpirun -n 2 ./build/test.pgesv 10
```

- Tags: `[Simple Solver]`, `[Expert Solver]`, `[Sparse Solver]`, `[Sparse CSR]`, `[Known]`.
- Standalone solver-test executables **sleep forever when given no argument** (debugger hook); without `EZP_ENABLE_TEST` their `REQUIRE` is a no-op, so they only detect crashes. Use `catch2` for actual pass/fail validation (`test.sparse_csr` is the exception, using `assert`).
- MUMPS and Lis are ON by default. PARDISO exists only when MKL is found (`EZP_MKL`); `test.pardiso` is a no-op otherwise.
- Shared numerical checks and tolerances live in `tests/check.hpp`; new solver tests should use residual checks and the same dual `TEST_CASE` / standalone-`main` pattern as existing tests.
- `standalone/runner.py` spawns `./solver.pgesv` by relative path, so run it from the build directory (`mpi4py` required).

## Conventions and gotchas

- Index type is `int_t`, controlled by `EZP_INT64` / `-DEZP_USE_64BIT_INT=ON`; never assume 32-bit.
- `EZP_UNDERSCORE` is ON by default. When adding a BLACS/ScaLAPACK routine, declare it in `ezp/abstract/ezp.h` and add its `#define name EZP(name)` macro.
- Dense solver signature is `solver<data_type, index_type, grid_order='R'>`; `'C'` selects column-major process grids. `par_dgesv`, `par_zgesv`, etc. are convenience aliases.
- Only the root process needs matrix/RHS storage; non-root ranks pass empty/null data and the solver scatters/gathers internally.
- MPI/BLACS initialization is automatic via RAII (`get_env<IT>()`). Call `blacs_env<>::do_not_manage_mpi()` when another component owns `MPI_Finalize`.
- Format changed C++ files with `clang-format -i <file>` (`.clang-format` is LLVM-based, `ColumnLimit: 0`, 4-space indent).
- Docs are built from `docs/` with `zensical` plus `Doxyfile`; keep `README.md` and solver docs in sync when the public API changes.
