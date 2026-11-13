# AGENTS.md — ezp

`ezp` is a header-only C++20 wrapper around distributed solvers (ScaLAPACK, MUMPS, Lis, PARDISO). The public API is the headers in `ezp/`; the root CMake project has no `ezp` library target and builds only examples/tests/standalone binaries plus the vendored sparse dependencies. Development branch: `master` (PRs/CI target it); `lite-lib` is the integration-only branch.

## Layout

- `ezp/*.hpp` — one header per solver (`pgesv`, `pgbsv`, `pposv`, `mumps`, `lis`, `pardiso`, ...). Umbrellas: `ezp/ezp`, `ezp/ezp.dense`, `ezp/ezp.sparse`.
- New solver headers must also be added to the matching umbrella, with a corresponding `docs/Solvers/...` page registered in `zensical.toml`.
- `ezp/abstract/` — shared solver bases (`abstract_solver.hpp`, `full_solver.hpp`, `band_solver.hpp`, `sparse_solver.hpp`, `traits.hpp`).
- `ezp/abstract/ezp.h` — raw C/Fortran ABI: `extern "C"` declarations plus `EZP(name)` symbol-mangling macros.
- `examples/`, `tests/`, `standalone/` — compiled into flat executables named after the source stem (`tests/test.pgesv.cpp` → `build/test.pgesv`).
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
- All project executables land directly in `build/` (`example.pgesv`, `test.pgesv`, `catch2`, `solver.pgesv`); there are no per-directory output folders.
- With `EZP_TEST=ON`, `cmake --build build --target catch2 -j"$(nproc)"` builds the aggregate suite without the example/standalone executables.
- `EZP_STANDALONE=ON` builds the `solver.*` executables (with an install rule to `bin/`); it is independent of `EZP_TEST`.
- CMake uses `file(GLOB ...)`: rerun the configure step after adding/removing an example, test, or standalone solver.
- Examples and tests are built whenever the project is top-level, even without `EZP_TEST`; `EZP_TEST=ON` additionally builds the aggregate `catch2` binary.
- GCC builds use `-Wall -Wextra -pedantic -Werror`; warnings must be fixed.

## Tests

There is no CTest integration — `ctest` will not find these tests.

```bash
# Full assertion-based suite (configure with -DEZP_TEST=ON; this is the real test run)
mpirun -n 2 ./build/catch2

# Focused runs using Catch2 test names/tags
mpirun -n 2 ./build/catch2 "Random PDGESV"
mpirun -n 2 ./build/catch2 "[Expert Solver]"
./build/catch2 "[Sparse CSR]"          # pure C++, no MPI needed

# Standalone variant of one test: argument is the repeat count
mpirun -n 2 ./build/test.pgesv 10
```

- Tags: `[Simple Solver]`, `[Expert Solver]`, `[Sparse Solver]`, `[Sparse CSR]`, `[Known]`, `[Benchmarking]`.
- Every `tests/*.cpp` is compiled into `catch2` (with `EZP_ENABLE_TEST`) and also built as its own executable; new tests must use the existing dual `TEST_CASE` / `#ifndef EZP_ENABLE_TEST` `main` pattern, or the aggregate build links two `main`s.
- Standalone solver-test executables **sleep forever when given no argument** (debugger hook); without `EZP_ENABLE_TEST` their `REQUIRE` is a no-op, so they only detect crashes. Use `catch2` for actual pass/fail validation (`test.sparse_csr` is the exception, using `assert`).
- MUMPS and Lis are ON by default. PARDISO exists only when MKL is found (`EZP_MKL`); `test.pardiso` is a no-op otherwise.
- Shared numerical checks and tolerances live in `tests/check.hpp`; new solver tests should use its residual checks.
- CI smoke-runs every `build/example.*` directly and every `build/test.*` with repeat count `10`, then runs the aggregate suite with 2 ranks.
- The `[Benchmarking]` case reads `../.dirty/DKTS3.mtx` relative to the working directory and swallows failures, so it passes silently when the file is absent.
- `standalone/runner*` spawns the `solver.*` binaries by relative name/path, so run them from the build directory (`mpi4py` for the Python variants).

## Conventions and gotchas

- `int_t`, `complex8`, and `complex16` are global aliases from `ezp/abstract/ezp.h`, not members of `ezp`; `EZP_INT64` / `-DEZP_USE_64BIT_INT=ON` makes `int_t` 64-bit, so never assume 32-bit.
- `EZP_UNDERSCORE` is ON by default. When adding a BLACS/ScaLAPACK routine, declare it in `ezp/abstract/ezp.h` and add its `#define name EZP(name)` macro.
- Dense solver templates take `data_type, index_type` first; where present, `grid_order` (`'R'` default, `'C'` column-major grids) follows any `UL`/`FCT` parameter. `par_dgesv`, `par_zgesv`, etc. are convenience aliases and still require the index type (e.g. `par_dgesv<int_t>`).
- Sparse input formats differ: MUMPS takes 1-based COO (`solve()` takes `sparse_coo_mat`, despite the class comment saying CSR), Lis takes 0-based CSR, and PARDISO takes CSR that is 1-based unless `iparm_zero_based_indexing(true)` is set.
- Only the root process needs matrix/RHS storage; non-root ranks pass empty/null data and the solver scatters/gathers internally.
- MPI/BLACS initialization is automatic via RAII (`get_env<IT>()`). Call `blacs_env<>::do_not_manage_mpi()` when another component owns `MPI_Finalize`.
- New project source files should follow the existing GPLv3 header plus Doxygen `@brief`/`@file` block.
- Format changed C++ files with `clang-format -i <file>` (`.clang-format` is LLVM-based, `ColumnLimit: 0`, 4-space indent).
- Docs: `uv venv && uv pip install -r requirements.txt`, `doxygen`, `cp README.md docs/README.md`, `uv run zensical build --clean`; generated `docs/README.md`, `doxygen/`, and `site/` are gitignored. Keep `README.md` and solver docs in sync.
