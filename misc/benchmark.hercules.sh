#!/usr/bin/env bash
#
# Build and run an ezp dense-solver benchmark on the MPCDF HERCULES cluster.
#
# What it does:
#   1. loads Intel compiler/MPI/MKL plus a recent GCC for the C++ standard library;
#   2. clones TLCFEM/ezp (with the mpl submodule) unless a clone already exists;
#   3. writes a standalone `pgesv` benchmark application and compiles it against
#      the header-only library and MKL ScaLAPACK (ILP64);
#   4. writes a Slurm batch script for parallel.q and optionally submits it.
#
# Typical use from a HERCULES login node:
#   bash misc/benchmark.hercules.sh
#   bash misc/benchmark.hercules.sh --submit
#
# The default problem is a 100000 x 100000 dense-storage 5-point Laplacian with
# an exact solution of all ones.  Rank 0 holds ~75 GiB (the ezp API takes the
# full matrix on the root process), so the job asks for `--mem=0` and 32 nodes.
#
# Useful overrides (environment variables or flags):
#   N=150000 NRHS=4 NTASKS=1536 GRID=32x48 bash misc/benchmark.hercules.sh
#   WORK_ROOT=/scratch/$USER/ezp-benchmark bash misc/benchmark.hercules.sh
#
# HERCULES parallel.q: 32 nodes x 48 cores = 1536 cores maximum.
# 1500 ranks with a 30x50 BLACS grid is the default; use 1536/32x48 for a
# perfectly node-aligned run.
#
set -euo pipefail

REPO_URL=${REPO_URL:-https://github.com/TLCFEM/ezp.git}
BRANCH=${BRANCH:-master}
WORK_ROOT=${WORK_ROOT:-$HOME/ezp-benchmark}
REPO_DIR=${REPO_DIR:-$WORK_ROOT/ezp}
BENCH_DIR=${BENCH_DIR:-$WORK_ROOT/benchmark}

N=${N:-100000}
NRHS=${NRHS:-4}
NTASKS=${NTASKS:-1500}
GRID=${GRID:-} # e.g. 30x50; empty means derive a near-square factorisation
TIME_LIMIT=${TIME_LIMIT:-02:00:00}
PARTITION=${PARTITION:-parallel.q}
CPUS_PER_NODE=${CPUS_PER_NODE:-48}
MAX_NODES=${MAX_NODES:-32}

MODULE_INTEL=${MODULE_INTEL:-intel/2023.1.0.x}
MODULE_MPI=${MODULE_MPI:-impi/2021.9}
MODULE_MKL=${MODULE_MKL:-mkl/2023.1}
# icpx uses the GCC installation found last in PATH for libstdc++; CentOS 7's
# system GCC 4.8 has no <ranges>, so a recent GCC module is mandatory.
MODULE_GCC=${MODULE_GCC:-gcc/12}
MODULE_GIT=${MODULE_GIT:-git/2.43}

SUBMIT=0
DO_BUILD=1

usage() {
  cat <<'USAGE'
Build and run an ezp dense-solver benchmark on the MPCDF HERCULES cluster.

What it does:
  1. loads Intel compiler/MPI/MKL plus a recent GCC for the C++ standard library;
  2. clones TLCFEM/ezp (with the mpl submodule) unless a clone already exists;
  3. writes a standalone pgesv benchmark application and compiles it against
     the header-only library and MKL ScaLAPACK (ILP64);
  4. writes a Slurm batch script for parallel.q and optionally submits it.

Typical use from a HERCULES login node:
  bash misc/benchmark.hercules.sh
  bash misc/benchmark.hercules.sh --submit

The default problem is a 100000 x 100000 dense-storage 5-point Laplacian with
an exact solution of all ones.  It needs ~80 GB on rank 0 (the ezp API gathers
the full matrix on the root process), so the job asks for --mem=0.

Options:
  --submit              submit the generated Slurm script
  --no-build            reuse an existing benchmark binary
  --n N                 matrix dimension (default 100000)
  --nrhs N              number of right-hand sides (default 4)
  --tasks N             MPI ranks / cores (default 1500; max 1536)
  --grid RxC            BLACS grid (default: near-square factorisation)
  --time HH:MM:SS       Slurm time limit (default 02:00:00)
  --work-root DIR       clone/build location (default $HOME/ezp-benchmark)
  --repo URL            repository to clone
  --branch NAME         branch to clone (default master)
  -h, --help            show this help

HERCULES parallel.q: 32 nodes x 48 cores = 1536 cores maximum.
1500 ranks with a 30x50 BLACS grid is the default; use 1536/32x48 for a
perfectly node-aligned run.
USAGE
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

log() {
  printf '==> %s\n' "$*"
}

while (($# > 0)); do
  case "$1" in
  --submit) SUBMIT=1 ;;
  --no-build) DO_BUILD=0 ;;
  --n)
    N=$2
    shift
    ;;
  --nrhs)
    NRHS=$2
    shift
    ;;
  --tasks)
    NTASKS=$2
    shift
    ;;
  --grid)
    GRID=$2
    shift
    ;;
  --work-root)
    WORK_ROOT=$2
    REPO_DIR=$WORK_ROOT/ezp
    BENCH_DIR=$WORK_ROOT/benchmark
    shift
    ;;
  --repo)
    REPO_URL=$2
    shift
    ;;
  --branch)
    BRANCH=$2
    shift
    ;;
  --time)
    TIME_LIMIT=$2
    shift
    ;;
  -h | --help)
    usage
    exit 0
    ;;
  *) die "unknown argument: $1 (try --help)" ;;
  esac
  shift
done

[[ "$N" =~ ^[1-9][0-9]*$ ]] || die "N must be a positive integer"
[[ "$NRHS" =~ ^[1-9][0-9]*$ ]] || die "NRHS must be a positive integer"
[[ "$NTASKS" =~ ^[1-9][0-9]*$ ]] || die "NTASKS must be a positive integer"

# ---------------------------------------------------------------------------
# 1. Environment modules
# ---------------------------------------------------------------------------
if ! type module >/dev/null 2>&1; then
  for init in /etc/profile.d/modules.sh /usr/share/Modules/init/bash /usr/share/lmod/lmod/init/bash; do
    if [[ -r "$init" ]]; then
      # shellcheck disable=SC1090
      source "$init"
      break
    fi
  done
fi
type module >/dev/null 2>&1 || die "the environment-modules system is not available"

log "loading HERCULES modules"
module purge || true
module load "$MODULE_INTEL" || die "cannot load $MODULE_INTEL"
module load "$MODULE_MPI" 2>/dev/null || module load impi || die "cannot load Intel MPI (tried $MODULE_MPI and impi)"
module load "$MODULE_MKL" || die "cannot load $MODULE_MKL"
module load "$MODULE_GIT" 2>/dev/null || true
# MPCDF recommends loading the newer GCC last (no modules afterwards) so that
# icpx finds its libstdc++ headers/symbols instead of the CentOS 7 default.
module load "$MODULE_GCC" || die "cannot load $MODULE_GCC"
[[ -n "${GCC_HOME:-}" ]] || die "GCC_HOME is not set after loading $MODULE_GCC"

if command -v mpiicpx >/dev/null 2>&1; then
  MPICXX=mpiicpx
elif command -v mpiicpc >/dev/null 2>&1; then
  MPICXX=mpiicpc
else
  die "no Intel MPI C++ wrapper found (mpiicpx/mpiicpc)"
fi
command -v git >/dev/null 2>&1 || die "git is not available"

[[ -n "${MKLROOT:-}" ]] || die "MKLROOT is not set; load the MKL module first"
MKL_LIB_DIR=""
for candidate in "$MKLROOT/lib/intel64" "$MKLROOT/lib/intel64_lin"; do
  if [[ -d "$candidate" ]]; then
    MKL_LIB_DIR=$candidate
    break
  fi
done
[[ -n "$MKL_LIB_DIR" ]] || die "MKL library directory not found under $MKLROOT/lib"

for lib in libmkl_scalapack_ilp64 libmkl_blacs_intelmpi_ilp64 libmkl_intel_ilp64 libmkl_sequential libmkl_core; do
  compgen -G "$MKL_LIB_DIR/$lib.*" >/dev/null || die "missing MKL library: $MKL_LIB_DIR/$lib.*"
done

log "MPI wrapper: $MPICXX"
log "MKL:         $MKLROOT"

# ---------------------------------------------------------------------------
# 2. Clone the repository
# ---------------------------------------------------------------------------
if [[ -d "$REPO_DIR/.git" ]]; then
  log "using existing clone: $REPO_DIR"
  git -C "$REPO_DIR" submodule update --init --recursive
elif [[ -e "$REPO_DIR" ]]; then
  die "$REPO_DIR exists but is not a git clone"
else
  log "cloning $REPO_URL ($BRANCH) into $REPO_DIR"
  mkdir -p "$(dirname "$REPO_DIR")"
  git clone --recurse-submodules --depth 1 --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
fi

# ---------------------------------------------------------------------------
# 3. Write the benchmark application
# ---------------------------------------------------------------------------
mkdir -p "$BENCH_DIR"
BENCH_SRC="$BENCH_DIR/benchmark.pgesv.cpp"
BENCH_BIN="$BENCH_DIR/benchmark.pgesv"

log "writing $BENCH_SRC"
cat >"$BENCH_SRC" <<'CPP'
// Standalone ezp benchmark: factorise and solve a large dense-storage system.
//
// Usage: benchmark.pgesv [n] [nrhs] [grid_rows] [grid_cols]
//
// The matrix is a diagonally dominant 5-point Laplacian.  Its right-hand side
// is A * ones, so the exact solution is the all-ones vector.  This keeps setup
// cheap while still exercising the full dense LU factorisation and solve.

#include <ezp/pgesv.hpp>
#include <mpl/mpl.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
auto parse_positive(const char* text, const char* name) -> std::size_t {
    std::size_t pos = 0;
    const auto value = std::stoull(text, &pos);
    if(pos != std::char_traits<char>::length(text) || value == 0) throw std::invalid_argument(name);
    return static_cast<std::size_t>(value);
}
} // namespace

int main(int argc, char** argv) {
    try {
        std::size_t n = 100000;
        int_t nrhs = 4;
        int_t grid_rows = 0, grid_cols = 0;

        if(argc > 1) n = parse_positive(argv[1], "n");
        if(argc > 2) nrhs = static_cast<int_t>(parse_positive(argv[2], "nrhs"));
        if(argc > 3) grid_rows = static_cast<int_t>(parse_positive(argv[3], "grid_rows"));
        if(argc > 4) grid_cols = static_cast<int_t>(parse_positive(argv[4], "grid_cols"));

        if(n < 2) throw std::invalid_argument("n must be at least 2");
        if(nrhs < 1) throw std::invalid_argument("nrhs must be at least 1");
        if((grid_rows == 0) != (grid_cols == 0)) throw std::invalid_argument("specify both grid dimensions or neither");
        if(n > static_cast<std::size_t>(std::numeric_limits<int_t>::max())) throw std::overflow_error("n does not fit int_t");

        // MPL owns MPI_Init/MPI_Finalize here; BLACS must not finalise MPI again.
        const auto& comm_world = mpl::environment::comm_world();
        ezp::blacs_env<int_t>::do_not_manage_mpi();
        const auto& env = ezp::get_env<int_t>();
        const auto rank = env.rank();
        const auto size = static_cast<std::size_t>(env.size());

        if(grid_rows > 0 && static_cast<std::size_t>(grid_rows) * static_cast<std::size_t>(grid_cols) != size && rank == 0)
            std::cerr << "warning: grid " << grid_rows << 'x' << grid_cols << " does not use all " << size << " ranks\n";

        std::unique_ptr<ezp::par_dgesv<int_t>> solver;
        if(grid_rows > 0) solver = std::make_unique<ezp::par_dgesv<int_t>>(grid_rows, grid_cols);
        else solver = std::make_unique<ezp::par_dgesv<int_t>>();

        const auto n_i = static_cast<int_t>(n);
        std::vector<double> a, b, row_sum;

        const auto t0 = mpl::environment::wtime();
        if(rank == 0) {
            const auto bytes = n * n * sizeof(double);
            std::cout << "matrix: n=" << n << " nrhs=" << nrhs
                      << " storage=" << std::fixed << std::setprecision(1)
                      << static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0) << " GiB on rank 0\n";

            a.assign(n * n, 0.0);
            row_sum.assign(n, 0.0);

            const auto stride = std::max<std::size_t>(1, static_cast<std::size_t>(std::sqrt(static_cast<double>(n))));
            const auto idx = [n](std::size_t r, std::size_t c) { return r + c * n; };

            for(std::size_t c = 0; c < n; ++c) {
                a[idx(c, c)] = 6.0;
                row_sum[c] += 6.0;

                const auto add = [&](std::size_t r, std::size_t col) {
                    a[idx(r, col)] = -1.0;
                    row_sum[r] -= 1.0;
                };

                if(c > 0) add(c - 1, c);
                if(c + 1 < n) add(c + 1, c);
                if(stride > 1) {
                    if(c >= stride) add(c - stride, c);
                    if(c + stride < n) add(c + stride, c);
                }
            }

            b.assign(n * nrhs, 0.0);
            for(int_t j = 0; j < nrhs; ++j)
                for(std::size_t i = 0; i < n; ++i) b[i + static_cast<std::size_t>(j) * n] = row_sum[i];
        }
        const auto t1 = mpl::environment::wtime();

        const auto info1 = solver->solve(
            ezp::full_mat<double, int_t>{n_i, n_i, a.data()},
            ezp::full_mat<double, int_t>{n_i, nrhs, b.data()}
        );
        const auto t2 = mpl::environment::wtime();

        double max_error = 0.0;
        if(rank == 0 && info1 == 0)
            for(int_t j = 0; j < nrhs; ++j)
                for(std::size_t i = 0; i < n; ++i) max_error = std::max(max_error, std::abs(b[i + static_cast<std::size_t>(j) * n] - 1.0));

        if(rank == 0)
            std::cout << std::scientific << std::setprecision(3)
                      << "factor+solve_s=" << (t2 - t1)
                      << " setup_s=" << (t1 - t0)
                      << " info=" << info1
                      << " max_abs_error=" << max_error << '\n';

        // Restore the RHS and reuse the stored factorisation.  This isolates the
        // triangular-solve/scatter/gather cost from the LU factorisation.
        double reuse_error = max_error;
        double reuse_time = 0.0;
        if(info1 == 0) {
            if(rank == 0)
                for(int_t j = 0; j < nrhs; ++j)
                    for(std::size_t i = 0; i < n; ++i) b[i + static_cast<std::size_t>(j) * n] = row_sum[i];

            const auto t3 = mpl::environment::wtime();
            const auto info2 = solver->solve(ezp::full_mat<double, int_t>{n_i, nrhs, b.data()});
            const auto t4 = mpl::environment::wtime();
            reuse_time = t4 - t3;

            reuse_error = 0.0;
            if(info2 != 0) reuse_error = std::numeric_limits<double>::infinity();
            else if(rank == 0)
                for(int_t j = 0; j < nrhs; ++j)
                    for(std::size_t i = 0; i < n; ++i) reuse_error = std::max(reuse_error, std::abs(b[i + static_cast<std::size_t>(j) * n] - 1.0));

            if(rank == 0)
                std::cout << "reuse_solve_s=" << reuse_time
                          << " reuse_max_abs_error=" << reuse_error << '\n';
        }

        int status = (info1 == 0 && max_error < 1e-8 && reuse_error < 1e-8) ? 0 : 1;
        comm_world.allreduce(mpl::max<int>(), status);
        return status;
    }
    catch(const std::exception& e) {
        std::cerr << "benchmark failed: " << e.what() << '\n';
        int initialized = 0;
        MPI_Initialized(&initialized);
        if(initialized) MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
CPP

# ---------------------------------------------------------------------------
# 4. Compile
# ---------------------------------------------------------------------------
if ((DO_BUILD)); then
  log "compiling with $MPICXX"
  "$MPICXX" \
    -O3 -march=native -std=c++20 -DNDEBUG \
    -DEZP_INT64 -DEZP_UNDERSCORE -DMPICH_SKIP_MPICXX \
    -I"$REPO_DIR" -I"$REPO_DIR/mpl" \
    "$BENCH_SRC" -o "$BENCH_BIN" \
    -L"${GCC_HOME}/lib64" -Wl,-rpath,"${GCC_HOME}/lib64" \
    -L"$MKL_LIB_DIR" -Wl,-rpath,"$MKL_LIB_DIR" \
    -Wl,--start-group \
    -lmkl_scalapack_ilp64 \
    -lmkl_blacs_intelmpi_ilp64 \
    -lmkl_intel_ilp64 \
    -lmkl_sequential \
    -lmkl_core \
    -Wl,--end-group \
    -lpthread -lm -ldl
else
  [[ -x "$BENCH_BIN" ]] || die "--no-build requested but $BENCH_BIN does not exist"
fi

# ---------------------------------------------------------------------------
# 5. Derive the BLACS grid and write the Slurm script
# ---------------------------------------------------------------------------
if [[ -z "$GRID" ]]; then
  # Prefer the factor pair closest to a square; 1500 -> 30x50, 1536 -> 32x48.
  best_rows=1
  best_cols=$NTASKS
  candidate=$(awk -v n="$NTASKS" 'BEGIN { printf "%d", sqrt(n) }')
  while ((candidate >= 1)); do
    if ((NTASKS % candidate == 0)); then
      best_rows=$candidate
      best_cols=$((NTASKS / candidate))
      break
    fi
    candidate=$((candidate - 1))
  done
  GRID="${best_rows}x${best_cols}"
  if ((best_rows == 1)); then
    printf 'warning: NTASKS=%s is prime or has no useful factorisation; BLACS grid is %s\n' "$NTASKS" "$GRID" >&2
  fi
fi
[[ "$GRID" =~ ^[1-9][0-9]*x[1-9][0-9]*$ ]] || die "GRID must look like ROWSxCOLS, e.g. 30x50"
GRID_ROWS=${GRID%x*}
GRID_COLS=${GRID#*x}
((GRID_ROWS * GRID_COLS == NTASKS)) || die "GRID $GRID must contain exactly NTASKS=$NTASKS ranks"

NODES=$(((NTASKS + CPUS_PER_NODE - 1) / CPUS_PER_NODE))
((NODES <= MAX_NODES)) || die "NTASKS=$NTASKS needs $NODES nodes, but $PARTITION allows at most $MAX_NODES"

SLURM_SCRIPT="$BENCH_DIR/benchmark.slurm"
cat >"$SLURM_SCRIPT" <<EOF
#!/bin/bash -l
#SBATCH --job-name=ezp-benchmark
#SBATCH --partition=$PARTITION
#SBATCH --nodes=$NODES
#SBATCH --ntasks=$NTASKS
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=$TIME_LIMIT
#SBATCH --output=$BENCH_DIR/benchmark-%j.out
#SBATCH --error=$BENCH_DIR/benchmark-%j.err

set -euo pipefail

module purge || true
module load $MODULE_INTEL
module load $MODULE_MPI 2>/dev/null || module load impi
module load $MODULE_MKL
# Keep the newer GCC last so the runtime libstdc++ matches what icpx compiled against.
module load $MODULE_GCC

# One MPI rank per core: do not oversubscribe MKL/OpenMP threads.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=FALSE

cd "$BENCH_DIR"
echo "job=\$SLURM_JOB_ID nodes=\${SLURM_JOB_NUM_NODES:-?} tasks=\${SLURM_NTASKS:-?} grid=$GRID"
echo "hosts:"
scontrol show hostnames "\$SLURM_JOB_NODELIST" 2>/dev/null | sort -u || true

srun --ntasks=$NTASKS "$BENCH_BIN" "$N" "$NRHS" "$GRID_ROWS" "$GRID_COLS"
EOF
chmod +x "$SLURM_SCRIPT"

mem_gib=$(awk -v n="$N" 'BEGIN { printf "%.1f", n * n * 8 / 1024 / 1024 / 1024 }')
log "benchmark binary: $BENCH_BIN"
log "slurm script:     $SLURM_SCRIPT"
log "problem:          n=$N nrhs=$NRHS ranks=$NTASKS grid=$GRID nodes=$NODES"
log "memory note:      rank 0 needs about ${mem_gib} GiB for the dense matrix"

if ((SUBMIT)); then
  command -v sbatch >/dev/null 2>&1 || die "sbatch is not available"
  log "submitting job"
  sbatch "$SLURM_SCRIPT"
else
  printf '\nSubmit the benchmark with:\n  sbatch %q\n' "$SLURM_SCRIPT"
fi
