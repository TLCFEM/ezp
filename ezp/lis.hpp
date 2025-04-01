/*******************************************************************************
 * Copyright (C) 2025 Theodore Chang
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 ******************************************************************************/
/**
 * @class lis
 * @brief Iterative solver for general sparse matrices.
 *
 * It solves the system of linear equations `A * X = B` with a general sparse matrix `A`.
 * The RHS matrix `B` is a dense matrix.
 *
 * The example usage can be seen as follows.
 *
 * @include example.lis.cpp
 *
 * @author tlc
 * @date 23/03/2025
 * @version 1.0.0
 * @file lis.hpp
 * @{
 */

#ifndef LIS_HPP
#define LIS_HPP

#include "abstract/traits.hpp"

#include <mpl/mpl.hpp>

#ifdef EZP_INT64
using LIS_INT = std::int64_t;
#else
using LIS_INT = std::int32_t;
#endif

using LIS_SCALAR = double;
using LIS_REAL = double;
using LIS_Comm = MPI_Comm;

struct LIS_VECTOR_STRUCT {
    LIS_INT label;
    LIS_INT status;
    LIS_INT precision;
    LIS_INT gn;
    LIS_INT n;
    LIS_INT np;
    LIS_INT pad;
    LIS_INT origin;
    LIS_INT is_copy;
    LIS_INT is_destroy;
    LIS_INT is_scaled;
    LIS_INT my_rank;
    LIS_INT nprocs;
    LIS_Comm comm;
    LIS_INT is;
    LIS_INT ie;
    LIS_INT* ranges;
    LIS_SCALAR* value;
    LIS_SCALAR* value_lo;
    LIS_SCALAR* work;
    LIS_INT intvalue;
};

typedef LIS_VECTOR_STRUCT* LIS_VECTOR;

struct LIS_MATRIX_CORE_STRUCT {
    LIS_INT nnz;
    LIS_INT ndz;
    LIS_INT bnr;
    LIS_INT bnc;
    LIS_INT nr;
    LIS_INT nc;
    LIS_INT bnnz;
    LIS_INT nnd;
    LIS_INT maxnzr;
    LIS_INT* ptr;
    LIS_INT* row;
    LIS_INT* col;
    LIS_INT* index;
    LIS_INT* bptr;
    LIS_INT* bindex;
    LIS_SCALAR* value;
    LIS_SCALAR* work;
};

typedef LIS_MATRIX_CORE_STRUCT* LIS_MATRIX_CORE;

struct LIS_MATRIX_DIAG_STRUCT {
    LIS_INT label;
    LIS_INT status;
    LIS_INT precision;
    LIS_INT gn;
    LIS_INT n;
    LIS_INT np;
    LIS_INT pad;
    LIS_INT origin;
    LIS_INT is_copy;
    LIS_INT is_destroy;
    LIS_INT is_scaled;
    LIS_INT my_rank;
    LIS_INT nprocs;
    LIS_Comm comm;
    LIS_INT is;
    LIS_INT ie;
    LIS_INT* ranges;
    LIS_SCALAR* value;
    LIS_SCALAR* work;

    LIS_INT bn;
    LIS_INT nr;
    LIS_INT* bns;
    LIS_INT* ptr;
    LIS_SCALAR** v_value;
};

typedef LIS_MATRIX_DIAG_STRUCT* LIS_MATRIX_DIAG;

struct LIS_COMMTABLE_STRUCT {
    LIS_Comm comm;
    LIS_INT pad;
    LIS_INT neibpetot;
    LIS_INT imnnz;
    LIS_INT exnnz;
    LIS_INT wssize;
    LIS_INT wrsize;
    LIS_INT* neibpe;
    LIS_INT* import_ptr;
    LIS_INT* import_index;
    LIS_INT* export_ptr;
    LIS_INT* export_index;
    LIS_SCALAR* ws;
    LIS_SCALAR* wr;
    MPI_Request *req1, *req2;
    MPI_Status *sta1, *sta2;
};

typedef LIS_COMMTABLE_STRUCT* LIS_COMMTABLE;

struct LIS_MATRIX_STRUCT {
    LIS_INT label;
    LIS_INT status;
    LIS_INT precision;
    LIS_INT gn;
    LIS_INT n;
    LIS_INT np;
    LIS_INT pad;
    LIS_INT origin;
    LIS_INT is_copy;
    LIS_INT is_destroy;
    LIS_INT is_scaled;
    LIS_INT my_rank;
    LIS_INT nprocs;
    LIS_Comm comm;
    LIS_INT is;
    LIS_INT ie;
    LIS_INT* ranges;

    LIS_INT matrix_type;
    LIS_INT nnz;       /* CSR,CSC,MSR,JAD,VBR,COO */
    LIS_INT ndz;       /* MSR */
    LIS_INT bnr;       /* BSR,BSC */
    LIS_INT bnc;       /* BSR,BSC */
    LIS_INT nr;        /* BSR,BSC,VBR */
    LIS_INT nc;        /* BSR,BSC,VBR */
    LIS_INT bnnz;      /* BSR,BSC,VBR */
    LIS_INT nnd;       /* DIA */
    LIS_INT maxnzr;    /* ELL,JAD */
    LIS_INT* ptr;      /* CSR,CSC,JAD */
    LIS_INT* row;      /* JAD,VBR,COO */
    LIS_INT* col;      /* JAD,VBR,COO */
    LIS_INT* index;    /* CSR,CSC,MSR,DIA,ELL,JAD */
    LIS_INT* bptr;     /* BSR,BSC,VBR */
    LIS_INT* bindex;   /* BSR,BSC,VBR */
    LIS_SCALAR* value; /* CSR,CSC,MSR,DIA,ELL,JAD,BSR,BSC,VBR,DNS,COO */
    LIS_SCALAR* work;

    LIS_MATRIX_CORE L;
    LIS_MATRIX_CORE U;
    LIS_MATRIX_DIAG D;
    LIS_MATRIX_DIAG WD;

    LIS_INT is_block;
    LIS_INT pad_comm;
    LIS_INT is_pmat;
    LIS_INT is_sorted;
    LIS_INT is_splited;
    LIS_INT is_save;
    LIS_INT is_comm;
    LIS_INT is_fallocated;
    LIS_INT use_wd;
    LIS_INT conv_bnr;
    LIS_INT conv_bnc;
    LIS_INT* conv_row;
    LIS_INT* conv_col;
    LIS_INT options[10];

    LIS_INT w_annz;
    LIS_INT* w_nnz;
    LIS_INT* w_row;
    LIS_INT** w_index;
    LIS_SCALAR** w_value;
    LIS_SCALAR*** v_value;

    LIS_INT* l2g_map;
    LIS_COMMTABLE commtable;
};

typedef LIS_MATRIX_STRUCT* LIS_MATRIX;

struct LIS_MATRIX_ILU_STRUCT {
    LIS_INT n;
    LIS_INT bs;
    LIS_INT* nnz_ma;
    LIS_INT* nnz;
    LIS_INT* bsz;
    LIS_INT** index;
    LIS_SCALAR** value;
    LIS_SCALAR*** values;
};

typedef LIS_MATRIX_ILU_STRUCT* LIS_MATRIX_ILU;

struct LIS_PRECON_STRUCT {
    LIS_INT precon_type;
    LIS_MATRIX A; /* SSOR */
    LIS_MATRIX Ah;
    LIS_MATRIX_ILU L;                 /* ilu(k),ilut,iluc,sainv */
    LIS_MATRIX_ILU U;                 /* ilu(k),ilut,iluc,sainv */
    LIS_MATRIX_DIAG WD;               /* bilu(k),bilut,biluc,bjacobi */
    LIS_VECTOR D;                     /* ilu(k),ilut,iluc,jacobi,sainv */
    LIS_VECTOR Pb;                    /* i+s */
    LIS_VECTOR temp;                  /* saamg */
    LIS_REAL theta;                   /* saamg */
    LIS_VECTOR* work;                 /* adds */
    struct LIS_SOLVER_STRUCT* solver; /* hybrid */
    LIS_INT worklen;                  /* adds */
    LIS_INT level_num;                /* saamg */
    LIS_INT wsize;                    /* saamg */
    LIS_INT solver_comm;              /* saamg */
    LIS_INT my_rank;                  /* saamg */
    LIS_INT nprocs;                   /* saamg */
    LIS_INT is_copy;
    LIS_COMMTABLE commtable; /* saamg */
};

typedef LIS_PRECON_STRUCT* LIS_PRECON;

struct LIS_SOLVER_STRUCT {
    LIS_MATRIX A, Ah;
    LIS_VECTOR b, x, xx, d;
    LIS_MATRIX_DIAG WD;
    LIS_PRECON precon;
    LIS_VECTOR* work;
    LIS_REAL* rhistory;
    LIS_INT worklen;
    LIS_INT options[27];
    LIS_SCALAR params[15];
    LIS_INT retcode;
    LIS_INT iter;
    LIS_INT iter2;
    LIS_REAL resid;
    double time;
    double itime;
    double ptime;
    double p_c_time;
    double p_i_time;
    LIS_INT precision;
    LIS_REAL bnrm;
    LIS_REAL tol;
    LIS_REAL tol_switch;
    LIS_INT setup;
};

typedef LIS_SOLVER_STRUCT* LIS_SOLVER;

extern "C" {
LIS_INT lis_finalize(void);
LIS_INT lis_initialize(int* argc, char** argv[]);
LIS_INT lis_matrix_assemble(LIS_MATRIX A);
LIS_INT lis_matrix_create(LIS_Comm comm, LIS_MATRIX* Amat);
LIS_INT lis_matrix_destroy(LIS_MATRIX Amat);
LIS_INT lis_matrix_set_csr(LIS_INT nnz, LIS_INT* row, LIS_INT* index, LIS_SCALAR* value, LIS_MATRIX A);
LIS_INT lis_matrix_set_size(LIS_MATRIX A, LIS_INT local_n, LIS_INT global_n);
LIS_INT lis_matrix_unset(LIS_MATRIX A);
LIS_INT lis_solve(LIS_MATRIX A, LIS_VECTOR b, LIS_VECTOR x, LIS_SOLVER solver);
LIS_INT lis_solver_create(LIS_SOLVER* solver);
LIS_INT lis_solver_destroy(LIS_SOLVER solver);
LIS_INT lis_solver_set_option(const char* text, LIS_SOLVER solver);
LIS_INT lis_vector_create(LIS_Comm comm, LIS_VECTOR* vec);
LIS_INT lis_vector_destroy(LIS_VECTOR vec);
LIS_INT lis_vector_set_size(LIS_VECTOR vec, LIS_INT local_n, LIS_INT global_n);
LIS_INT lis_vector_set(LIS_VECTOR vec, LIS_SCALAR* value);
LIS_INT lis_vector_unset(LIS_VECTOR vec);
void lis_do_not_handle_mpi();
}

namespace ezp {
    namespace detail {
        struct lis_env {
            const mpl::communicator& comm_world{mpl::environment::comm_world()};

            lis_env() {
                lis_do_not_handle_mpi();
                lis_initialize(nullptr, nullptr);
            }
            ~lis_env() { lis_finalize(); }

            [[nodiscard]] auto rank() const { return comm_world.rank(); }

            [[nodiscard]] auto native_handle() const { return comm_world.native_handle(); }
        };

        inline auto& get_lis_env() {
            static const lis_env env;
            return env;
        }

        class lis_vector final {
            LIS_VECTOR v{};

            bool is_set = false;

            auto unset() {
                if(is_set) lis_vector_unset(v);
                is_set = false;
            }

        public:
            explicit lis_vector(const LIS_INT n) {
                lis_vector_create(get_lis_env().native_handle(), &v);
                lis_vector_set_size(v, 0 == get_lis_env().rank() ? n : 0, 0);
            }

            ~lis_vector() {
                unset();
                lis_vector_destroy(v);
            }

            auto set(LIS_SCALAR* value) {
                unset();
                is_set = true;
                lis_vector_set(v, 0 == get_lis_env().rank() ? value : nullptr);
                return v;
            }
        };

        class lis_matrix final {
            LIS_MATRIX a_mat{};

            bool is_set = false;

            auto unset() {
                if(is_set) lis_matrix_unset(a_mat);
                lis_matrix_destroy(a_mat);
                is_set = false;
            }

        public:
            lis_matrix() = default;

            lis_matrix(const lis_matrix&) {}
            lis_matrix(lis_matrix&&) = delete;
            lis_matrix& operator=(const lis_matrix&) = delete;
            lis_matrix& operator=(lis_matrix&&) = delete;

            ~lis_matrix() { unset(); }

            [[nodiscard]] auto get() const { return a_mat; }

            auto set(const sparse_csr_mat<LIS_SCALAR, LIS_INT>& A) {
                unset();
                const bool is_root = 0 == get_lis_env().rank();
                lis_matrix_create(get_lis_env().native_handle(), &a_mat);
                lis_matrix_set_size(a_mat, is_root ? A.n : 0, 0);
                lis_matrix_set_csr(is_root ? A.nnz : 0, A.row_ptr, A.col_idx, A.data, a_mat);
                lis_matrix_assemble(a_mat);
                is_set = true;
            }
        };

        class lis_solver final {
            LIS_SOLVER solver{};

        public:
            lis_solver() { lis_solver_create(&solver); }

            lis_solver(const lis_solver&) { lis_solver_create(&solver); };
            lis_solver(lis_solver&&) = delete;
            lis_solver& operator=(const lis_solver&) = delete;
            lis_solver& operator=(lis_solver&&) = delete;

            ~lis_solver() { lis_solver_destroy(solver); }

            explicit lis_solver(const char* setting) {
                lis_solver_create(&solver);
                set_option(setting);
            }

            void set_option(const char* setting) const { lis_solver_set_option(setting, solver); }

            auto solve(LIS_MATRIX A, LIS_VECTOR B, LIS_VECTOR X) const { return lis_solve(A, B, X, solver); }
        };
    } // namespace detail

    class lis final {
        const detail::lis_env& env = detail::get_lis_env();

        detail::lis_solver solver;
        detail::lis_matrix a_loc;

        [[nodiscard]] auto sync_error(LIS_INT error) const {
            env.comm_world.allreduce(mpl::min<LIS_INT>(), error);
            return error;
        }

    public:
        lis() = default;

        explicit lis(const char* setting)
            : solver(setting) {}

        void set_option(const char* setting) const { solver.set_option(setting); }

        LIS_INT solve(sparse_csr_mat<LIS_SCALAR, LIS_INT>&& A, full_mat<LIS_SCALAR, LIS_INT>&& B) {
            LIS_INT error = 0;
            if(0 == env.rank() && A.row_ptr[A.n] != A.nnz) error = -1;

            error = sync_error(error);
            if(error < 0) return error;

            a_loc.set(std::move(A));

            return solve(std::move(B));
        }

        LIS_INT solve(full_mat<LIS_SCALAR, LIS_INT>&& B) const {
            if(a_loc.get()->gn != B.n_rows) return -1;

            LIS_INT error = 0;

            std::vector<LIS_SCALAR> b_ref;
            if(0 == env.rank()) {
                b_ref.resize(B.n_rows * B.n_cols);
                std::copy_n(B.data, b_ref.size(), b_ref.data());
            }

            auto b_loc = detail::lis_vector(B.n_rows);
            auto x_loc = detail::lis_vector(B.n_rows);

            for(decltype(B.n_rows) I = 0; I < B.n_rows * B.n_cols; I += B.n_rows) {
                error = solver.solve(a_loc.get(), b_loc.set(b_ref.data() + I), x_loc.set(B.data + I));

                if(0 != error) break;
            }

            return error;
        }
    };
} // namespace ezp

#endif

//! @}
