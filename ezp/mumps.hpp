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
 * @class mumps
 * @brief Solver for general sparse matrices.
 *
 * It solves the system of linear equations `A * X = B` with a general sparse matrix `A`.
 * The RHS matrix `B` is a dense matrix.
 *
 * The matrix `A` should be stored in the Compressed Sparse Row (CSR) format with one-based indexing.
 *
 * To set control parameters, use the overloaded function call operator, which allows access to the `icntl` array.
 *
 * @code
    auto solver = mumps<double, int_t>();
    solver(3) = 0; // suppress output
    solver.icntl_printing_level(0); // equivalent to the above
 * @endcode
 *
 * The example usage can be seen as follows.
 *
 * @include example.mumps.cpp
 *
 * @author tlc
 * @date 23/03/2025
 * @version 1.0.0
 * @file mumps.hpp
 * @{
 */

#ifndef MUMPS_HPP
#define MUMPS_HPP

#include <external/mumps/cmumps_c.h>
#include <external/mumps/dmumps_c.h>
#include <external/mumps/smumps_c.h>
#include <external/mumps/zmumps_c.h>
#include <ezp/abstract/traits.hpp>
#include <mpl/mpl.hpp>

namespace ezp {
    namespace detail {
        template<typename DT> struct mumps_struc {};
        template<> struct mumps_struc<double> {
            using struct_type = DMUMPS_STRUC_C;
            using entry_type = double;
            static auto mumps_c(DMUMPS_STRUC_C* ptr) { return dmumps_c(ptr); }
        };
        template<> struct mumps_struc<float> {
            using struct_type = SMUMPS_STRUC_C;
            using entry_type = float;
            static auto mumps_c(SMUMPS_STRUC_C* ptr) { return smumps_c(ptr); }
        };
        template<> struct mumps_struc<complex16> {
            using struct_type = ZMUMPS_STRUC_C;
            using entry_type = mumps_double_complex;
            static auto mumps_c(ZMUMPS_STRUC_C* ptr) { return zmumps_c(ptr); }
        };
        template<> struct mumps_struc<complex8> {
            using struct_type = CMUMPS_STRUC_C;
            using entry_type = mumps_complex;
            static auto mumps_c(CMUMPS_STRUC_C* ptr) { return cmumps_c(ptr); }
        };
    } // namespace detail

    template<data_t DT, index_t IT> class mumps final {
        using struct_t = typename detail::mumps_struc<DT>::struct_type;
        using entry_t = typename detail::mumps_struc<DT>::entry_type;

        struct_t id;

        const mpl::communicator& comm_world{mpl::environment::comm_world()};

        auto sync_error() {
            IT error = id.infog[0] < 0 ? -1 : 0;
            comm_world.allreduce(mpl::min<IT>(), error);
            return error;
        }

        auto perform_job(const IT job) {
            id.job = job;
            detail::mumps_struc<DT>::mumps_c(&id);
        }

    public:
        explicit mumps(const int sym = 0, const int par = 0) {
            id.comm_fortran = MPI_Comm_c2f(comm_world.native_handle());
            id.sym = sym;
            id.par = par;
            // force par=1 if there is only one process
            if(comm_world.size() == 1) id.par = 1;
            perform_job(-1);
        };

        ~mumps() { perform_job(-2); };

        /**
         * @brief Overloaded function call operator to access elements of the `icntl` array.
         *
         * @param index The index of the element to access.
         * @return A reference to the element at the specified index in the `icntl` array.
         */
        auto& operator()(const IT index) { return id.icntl[index]; }

        auto icntl_output_error_message(const auto config) { return id.icntl[0] = config; }
        auto icntl_output_diagnostic_statistics_warning(const auto config) { return id.icntl[1] = config; }
        auto icntl_output_global_information(const auto config) { return id.icntl[2] = config; }
        auto icntl_printing_level(const auto config) { return id.icntl[3] = config; }
        // auto icntl_matrix_input_format(const auto config) { return id.icntl[4] = config; }
        auto icntl_permutation_and_scaling(const auto config) { return id.icntl[5] = config; }
        auto icntl_symmetric_permutation(const auto config) { return id.icntl[6] = config; }
        auto icntl_scaling_strategy(const auto config) { return id.icntl[7] = config; }
        auto icntl_transpose_matrix(const auto config) { return id.icntl[8] = config; }
        auto icntl_iterative_refinement(const auto config) { return id.icntl[9] = config; }
        auto icntl_error_analysis(const auto config) { return id.icntl[10] = config; }
        auto icntl_ordering_strategy(const auto config) { return id.icntl[11] = config; }
        auto icntl_root_parallelism(const auto config) { return id.icntl[12] = config; }
        auto icntl_working_space_percentage_increase(const auto config) { return id.icntl[13] = config; }
        auto icntl_compression_block_format(const auto config) { return id.icntl[14] = config; }
        auto icntl_openmp_threads(const auto config) { return id.icntl[15] = config; }
        auto icntl_distribution_strategy_input(const auto config) { return id.icntl[17] = config; }
        auto icntl_schur_complement(const auto config) { return id.icntl[18] = config; }
        // auto icntl_rhs_format(const auto config) { return id.icntl[19] = config; }
        auto icntl_distribution_strategy_solution(const auto config) { return id.icntl[20] = config; }
        auto icntl_out_of_core(const auto config) { return id.icntl[21] = config; }
        auto icntl_maximum_working_memory(const auto config) { return id.icntl[22] = config; }
        auto icntl_null_pivot_row_detection(const auto config) { return id.icntl[23] = config; }
        auto icntl_deficient_and_null_space_basis(const auto config) { return id.icntl[24] = config; }
        auto icntl_schur_complement_solution(const auto config) { return id.icntl[25] = config; }
        auto icntl_rhs_block_size(const auto config) { return id.icntl[26] = config; }
        auto icntl_ordering_computation(const auto config) { return id.icntl[27] = config; }
        // auto icntl_parallel_ordering_tool(const auto config) { return id.icntl[28] = config; }
        auto icntl_inverse_computation(const auto config) { return id.icntl[29] = config; }
        // auto icntl_discard_factorization(const auto config) { return id.icntl[30] = config; }
        auto icntl_forward_elimination(const auto config) { return id.icntl[31] = config; }
        auto icntl_determinant_computation(const auto config) { return id.icntl[32] = config; }
        auto icntl_out_of_core_file(const auto config) { return id.icntl[33] = config; }
        auto icntl_blr(const auto config) { return id.icntl[34] = config; }
        auto icntl_blr_variant(const auto config) { return id.icntl[35] = config; }
        auto icntl_blr_compression(const auto config) { return id.icntl[36] = config; }
        auto icntl_lu_compression_rate(const auto config) { return id.icntl[37] = config; }
        auto icntl_block_compression_rate(const auto config) { return id.icntl[38] = config; }
        auto icntl_tree_parallelism(const auto config) { return id.icntl[47] = config; }
        auto icntl_compact_working_space(const auto config) { return id.icntl[48] = config; }
        auto icntl_rank_revealing_factorization(const auto config) { return id.icntl[55] = config; }
        auto icntl_symbolic_factorization(const auto config) { return id.icntl[57] = config; }

        auto& info() { return id.info; }
        auto& rinfo() { return id.rinfo; }
        auto& infog() { return id.infog; }
        auto& rinfog() { return id.rinfog; }

        IT solve(sparse_csr_mat<DT, IT>&& A, full_mat<DT, IT>&& B) {
            if(0 == comm_world.rank() && A.irn[A.n] != A.nnz) return -1;

            if(A.n != B.n_rows) return -1;

            id.nnz = A.nnz;
            id.irn = A.irn;
            id.jcn = A.jcn;
            id.a = (entry_t*)A.data;

            id.n = A.n;
            id.lrhs = B.n_rows;
            id.nrhs = B.n_cols;
            id.rhs = (entry_t*)B.data;

            perform_job(6);

            return sync_error();
        }

        IT solve(full_mat<DT, IT>&& B) {
            if(id.n != B.n_rows) return -1;

            id.lrhs = B.n_rows;
            id.nrhs = B.n_cols;
            id.rhs = (entry_t*)B.data;

            perform_job(3);

            return sync_error();
        }
    };
} // namespace ezp

#endif

//! @}
