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
 * @fn mumps_set
 * @brief Parse MUMPS parameters.
 *
 * Usage: mumps [--output-error-message INT] [--output-diagnostic-statistics-warning INT] [--output-global-information INT] [--printing-level INT] [--permutation-and-scaling INT] [--symmetric-permutation INT] [--scaling-strategy INT] [--transpose-matrix INT] [--iterative-refinement INT] [--error-analysis INT] [--ordering-strategy INT] [--root-parallelism INT] [--working-space-percentage-increase INT] [--compression-block-format INT] [--openmp-threads INT] [--distribution-strategy-input INT] [--schur-complement INT] [--distribution-strategy-solution INT] [--out-of-core INT] [--maximum-working-memory INT] [--null-pivot-row-detection INT] [--deficient-and-null-space-basis INT] [--schur-complement-solution INT] [--rhs-block-size INT] [--ordering-computation INT] [--inverse-computation INT] [--forward-elimination INT] [--determinant-computation INT] [--out-of-core-file INT] [--blr INT] [--blr-variant INT] [--blr-compression INT] [--lu-compression-rate INT] [--block-compression-rate INT] [--tree-parallelism INT] [--compact-working-space INT] [--rank-revealing-factorization INT] [--symbolic-factorization INT]
 *
 * Optional arguments:
 *   --output-error-message INT                  [1] output stream for error messages. [default: 6]
 *   --output-diagnostic-statistics-warning INT  [2] output stream for diagnostic printing and statistics local to each MPI process. [default: 0]
 *   --output-global-information INT             [3] output stream for global information, collected on the host. [default: 6]
 *   --printing-level INT                        [4] level of printing for error, warning, and diagnostic messages. [default: 2]
 *   --permutation-and-scaling INT               [6] permutes the matrix to a zero-free diagonal and/or scale the matrix. [default: 7]
 *   --symmetric-permutation INT                 [7] computes a symmetric permutation (ordering) to determine the pivot order to be used for the factorization in case of sequential analysis. [default: 7]
 *   --scaling-strategy INT                      [8] describes the scaling strategy. [default: 77]
 *   --transpose-matrix INT                      [9] computes the solution using A or transpose of A. [default: 1]
 *   --iterative-refinement INT                  [10] applies the iterative refinement to the computed solution. [default: 0]
 *   --error-analysis INT                        [11] computes statistics related to an error analysis of the linear system solved. [default: 0]
 *   --ordering-strategy INT                     [12] defines an ordering strategy for symmetric matrices and is used, in conjunction with ICNTL(6), to add constraints to the ordering algorithm. [default: 0]
 *   --root-parallelism INT                      [13] controls the parallelism of the root node (enabling or not the use of ScaLAPACK) and also its splitting. [default: 0]
 *   --working-space-percentage-increase INT     [14] controls the percentage increase in the estimated working space. [default: 30]
 *   --compression-block-format INT              [15] exploits compression of the input matrix resulting from a block format. [default: 0]
 *   --openmp-threads INT                        [16] controls the setting of the number of OpenMP threads by MUMPS when the setting of multithreading is not possible outside MUMPS. [default: 0]
 *   --distribution-strategy-input INT           [18] defines the strategy for the distributed input matrix. [default: 0]
 *   --schur-complement INT                      [19] computes the Schur complement matrix. [default: 0]
 *   --distribution-strategy-solution INT        [21] determines the distribution (centralized or distributed) of the solution vectors. [default: 0]
 *   --out-of-core INT                           [22] controls the in-core/out-of-core (OOC) factorization and solve. [default: 0]
 *   --maximum-working-memory INT                [23] corresponds to the maximum size of the working memory in MB that MUMPS can allocate per working process. [default: 0]
 *   --null-pivot-row-detection INT              [24] controls the detection of "null pivot rows". [default: 0]
 *   --deficient-and-null-space-basis INT        [25] allows the computation of a solution of a deficient matrix and also of a null space basis. [default: 0]
 *   --schur-complement-solution INT             [26] drives the solution phase if a Schur complement matrix has been computed. [default: 0]
 *   --rhs-block-size INT                        [27] controls the blocking size for multiple right-hand sides. [default: -32]
 *   --ordering-computation INT                  [28] determines whether a sequential or parallel computation of the ordering is performed. [default: 0]
 *   --inverse-computation INT                   [30] computes a user-specified set of entries in the inverse of the original matrix. [default: 0]
 *   --forward-elimination INT                   [32] performs the forward elimination of the right-hand sides during the factorization. [default: 0]
 *   --determinant-computation INT               [33] computes the determinant of the input matrix. [default: 0]
 *   --out-of-core-file INT                      [34] controls the conservation of the OOC files. [default: 0]
 *   --blr INT                                   [35] controls the activation of the BLR feature. [default: 0]
 *   --blr-variant INT                           [36] controls the choice of BLR factorization variant. [default: 0]
 *   --blr-compression INT                       [37] controls the BLR compression of the contribution blocks. [default: 0]
 *   --lu-compression-rate INT                   [38] estimates compression rate of LU factors. [default: 600]
 *   --block-compression-rate INT                [39] estimates compression rate of contribution blocks. [default: 500]
 *   --tree-parallelism INT                      [48] controls multithreading with tree parallelism. [default: 1]
 *   --compact-working-space INT                 [49] compacts workarray at the end of factorization phase. [default: 0]
 *   --rank-revealing-factorization INT          [56] detects pseudo-singularities during factorization and factorizes the root node with a rank-revealing method. [default: 0]
 *   --symbolic-factorization INT                [58] defines options for symbolic factorization. [default: 2]
 *
 * @author tlc
 * @date 05/04/2025
 * @version 1.0.0
 * @file mumps.parser.hpp
 * @{
 */

#ifndef MUMPS_PARSER_HPP
#define MUMPS_PARSER_HPP

#include "mumps.hpp"

#include <argparse/argparse.hpp>

namespace ezp {
    namespace detail {
        static constexpr std::string mumps_name = "mumps";
    } // namespace detail

    template<data_t DT, index_t IT> auto mumps_set(std::string command, mumps<DT, IT>& solver) {
        argparse::ArgumentParser program(detail::mumps_name, "", argparse::default_arguments::none);
        program.add_argument("--output-error-message").help("[1] output stream for error messages.").default_value(6).scan<'i', int>().metavar("INT").nargs(1);
        program.add_argument("--output-diagnostic-statistics-warning").help("[2] output stream for diagnostic printing and statistics local to each MPI process.").default_value(0).scan<'i', int>().metavar("INT").nargs(1);
        program.add_argument("--output-global-information").help("[3] output stream for global information, collected on the host.").default_value(6).scan<'i', int>().metavar("INT").nargs(1);
        program.add_argument("--printing-level").help("[4] level of printing for error, warning, and diagnostic messages.").default_value(2).choices(0, 1, 2, 3, 4).scan<'i', int>().metavar("INT").nargs(1);
        program.add_argument("--permutation-and-scaling").help("[6] permutes the matrix to a zero-free diagonal and/or scale the matrix.").default_value(7).choices(0, 1, 2, 3, 4, 5, 6, 7).scan<'i', int>().metavar("INT").nargs(1);
        program.add_argument("--symmetric-permutation").help("[7] computes a symmetric permutation (ordering) to determine the pivot order to be used for the factorization in case of sequential analysis.").default_value(7).choices(0, 1, 2, 3, 4, 5, 6, 7).scan<'i', int>().metavar("INT").nargs(1);
        program.add_argument("--scaling-strategy").help("[8] describes the scaling strategy.").default_value(77).choices(-2, -1, 0, 1, 3, 4, 7, 8, 77).scan<'i', int>().metavar("INT").nargs(1);
        program.add_argument("--transpose-matrix").help("[9] computes the solution using A or transpose of A.").default_value(1).choices(0, 1).scan<'i', int>().metavar("INT").nargs(1);
        program.add_argument("--iterative-refinement").help("[10] applies the iterative refinement to the computed solution.").default_value(0).scan<'i', int>().metavar("INT").nargs(1);
        program.add_argument("--error-analysis").help("[11] computes statistics related to an error analysis of the linear system solved.").default_value(0).choices(0, 1, 2).scan<'i', int>().metavar("INT").nargs(1);
        program.add_argument("--ordering-strategy").help("[12] defines an ordering strategy for symmetric matrices and is used, in conjunction with ICNTL(6), to add constraints to the ordering algorithm.").default_value(0).choices(0, 1, 2, 3).scan<'i', int>().metavar("INT").nargs(1);
        program.add_argument("--root-parallelism").help("[13] controls the parallelism of the root node (enabling or not the use of ScaLAPACK) and also its splitting.").default_value(0).scan<'i', int>().metavar("INT").nargs(1);
        program.add_argument("--working-space-percentage-increase").help("[14] controls the percentage increase in the estimated working space.").default_value(30).scan<'i', int>().metavar("INT").nargs(1);
        program.add_argument("--compression-block-format").help("[15] exploits compression of the input matrix resulting from a block format.").default_value(0).scan<'i', int>().metavar("INT").nargs(1);
        program.add_argument("--openmp-threads").help("[16] controls the setting of the number of OpenMP threads by MUMPS when the setting of multithreading is not possible outside MUMPS.").default_value(0).choices(0, 1, 2, 3).scan<'i', int>().metavar("INT").nargs(1);
        program.add_argument("--distribution-strategy-input").help("[18] defines the strategy for the distributed input matrix.").default_value(0).scan<'i', int>().metavar("INT").nargs(1);
        program.add_argument("--schur-complement").help("[19] computes the Schur complement matrix.").default_value(0).choices(0, 1, 2, 3).scan<'i', int>().metavar("INT").nargs(1);
        program.add_argument("--distribution-strategy-solution").help("[21] determines the distribution (centralized or distributed) of the solution vectors.").default_value(0).choices(0, 1).scan<'i', int>().metavar("INT").nargs(1);
        program.add_argument("--out-of-core").help("[22] controls the in-core/out-of-core (OOC) factorization and solve.").default_value(0).choices(0, 1).scan<'i', int>().metavar("INT").nargs(1);
        program.add_argument("--maximum-working-memory").help("[23] corresponds to the maximum size of the working memory in MB that MUMPS can allocate per working process.").default_value(0).scan<'i', int>().metavar("INT").nargs(1);
        program.add_argument("--null-pivot-row-detection").help("[24] controls the detection of \"null pivot rows\".").default_value(0).choices(0, 1).scan<'i', int>().metavar("INT").nargs(1);
        program.add_argument("--deficient-and-null-space-basis").help("[25] allows the computation of a solution of a deficient matrix and also of a null space basis.").default_value(0).scan<'i', int>().metavar("INT").nargs(1);
        program.add_argument("--schur-complement-solution").help("[26] drives the solution phase if a Schur complement matrix has been computed.").default_value(0).choices(0, 1, 2).scan<'i', int>().metavar("INT").nargs(1);
        program.add_argument("--rhs-block-size").help("[27] controls the blocking size for multiple right-hand sides.").default_value(-32).scan<'i', int>().metavar("INT").nargs(1);
        program.add_argument("--ordering-computation").help("[28] determines whether a sequential or parallel computation of the ordering is performed.").default_value(0).choices(0, 1, 2).scan<'i', int>().metavar("INT").nargs(1);
        program.add_argument("--inverse-computation").help("[30] computes a user-specified set of entries in the inverse of the original matrix.").default_value(0).choices(0, 1).scan<'i', int>().metavar("INT").nargs(1);
        program.add_argument("--forward-elimination").help("[32] performs the forward elimination of the right-hand sides during the factorization.").default_value(0).choices(0, 1).scan<'i', int>().metavar("INT").nargs(1);
        program.add_argument("--determinant-computation").help("[33] computes the determinant of the input matrix.").default_value(0).choices(0, 1).scan<'i', int>().metavar("INT").nargs(1);
        program.add_argument("--out-of-core-file").help("[34] controls the conservation of the OOC files.").default_value(0).choices(0, 1).scan<'i', int>().metavar("INT").nargs(1);
        program.add_argument("--blr").help("[35] controls the activation of the BLR feature.").default_value(0).choices(0, 1, 2, 3).scan<'i', int>().metavar("INT").nargs(1);
        program.add_argument("--blr-variant").help("[36] controls the choice of BLR factorization variant.").default_value(0).choices(0, 1).scan<'i', int>().metavar("INT").nargs(1);
        program.add_argument("--blr-compression").help("[37] controls the BLR compression of the contribution blocks.").default_value(0).choices(0, 1).scan<'i', int>().metavar("INT").nargs(1);
        program.add_argument("--lu-compression-rate").help("[38] estimates compression rate of LU factors.").default_value(600).scan<'i', int>().metavar("INT").nargs(1);
        program.add_argument("--block-compression-rate").help("[39] estimates compression rate of contribution blocks.").default_value(500).scan<'i', int>().metavar("INT").nargs(1);
        program.add_argument("--tree-parallelism").help("[48] controls multithreading with tree parallelism.").default_value(1).choices(0, 1).scan<'i', int>().metavar("INT").nargs(1);
        program.add_argument("--compact-working-space").help("[49] compacts workarray at the end of factorization phase.").default_value(0).choices(0, 1, 2).scan<'i', int>().metavar("INT").nargs(1);
        program.add_argument("--rank-revealing-factorization").help("[56] detects pseudo-singularities during factorization and factorizes the root node with a rank-revealing method.").default_value(0).choices(0, 1).scan<'i', int>().metavar("INT").nargs(1);
        program.add_argument("--symbolic-factorization").help("[58] defines options for symbolic factorization.").default_value(2).choices(1, 2).scan<'i', int>().metavar("INT").nargs(1);

        std::vector<std::string> args{detail::mumps_name};
        command = command.substr(command.find('-'));
        std::istringstream iss(command);
        args.insert(args.end(), std::istream_iterator<std::string>(iss), std::istream_iterator<std::string>());
        program.parse_args(args);

        if(program.is_used("--output-error-message")) solver.icntl_output_error_message(program.get<int>("--output-error-message"));
        if(program.is_used("--output-diagnostic-statistics-warning")) solver.icntl_output_diagnostic_statistics_warning(program.get<int>("--output-diagnostic-statistics-warning"));
        if(program.is_used("--output-global-information")) solver.icntl_output_global_information(program.get<int>("--output-global-information"));
        if(program.is_used("--printing-level")) solver.icntl_printing_level(program.get<int>("--printing-level"));
        if(program.is_used("--permutation-and-scaling")) solver.icntl_permutation_and_scaling(program.get<int>("--permutation-and-scaling"));
        if(program.is_used("--symmetric-permutation")) solver.icntl_symmetric_permutation(program.get<int>("--symmetric-permutation"));
        if(program.is_used("--scaling-strategy")) solver.icntl_scaling_strategy(program.get<int>("--scaling-strategy"));
        if(program.is_used("--transpose-matrix")) solver.icntl_transpose_matrix(program.get<int>("--transpose-matrix"));
        if(program.is_used("--iterative-refinement")) solver.icntl_iterative_refinement(program.get<int>("--iterative-refinement"));
        if(program.is_used("--error-analysis")) solver.icntl_error_analysis(program.get<int>("--error-analysis"));
        if(program.is_used("--ordering-strategy")) solver.icntl_ordering_strategy(program.get<int>("--ordering-strategy"));
        if(program.is_used("--root-parallelism")) solver.icntl_root_parallelism(program.get<int>("--root-parallelism"));
        if(program.is_used("--working-space-percentage-increase")) solver.icntl_working_space_percentage_increase(program.get<int>("--working-space-percentage-increase"));
        if(program.is_used("--compression-block-format")) solver.icntl_compression_block_format(program.get<int>("--compression-block-format"));
        if(program.is_used("--openmp-threads")) solver.icntl_openmp_threads(program.get<int>("--openmp-threads"));
        if(program.is_used("--distribution-strategy-input")) solver.icntl_distribution_strategy_input(program.get<int>("--distribution-strategy-input"));
        if(program.is_used("--schur-complement")) solver.icntl_schur_complement(program.get<int>("--schur-complement"));
        if(program.is_used("--distribution-strategy-solution")) solver.icntl_distribution_strategy_solution(program.get<int>("--distribution-strategy-solution"));
        if(program.is_used("--out-of-core")) solver.icntl_out_of_core(program.get<int>("--out-of-core"));
        if(program.is_used("--maximum-working-memory")) solver.icntl_maximum_working_memory(program.get<int>("--maximum-working-memory"));
        if(program.is_used("--null-pivot-row-detection")) solver.icntl_null_pivot_row_detection(program.get<int>("--null-pivot-row-detection"));
        if(program.is_used("--deficient-and-null-space-basis")) solver.icntl_deficient_and_null_space_basis(program.get<int>("--deficient-and-null-space-basis"));
        if(program.is_used("--schur-complement-solution")) solver.icntl_schur_complement_solution(program.get<int>("--schur-complement-solution"));
        if(program.is_used("--rhs-block-size")) solver.icntl_rhs_block_size(program.get<int>("--rhs-block-size"));
        if(program.is_used("--ordering-computation")) solver.icntl_ordering_computation(program.get<int>("--ordering-computation"));
        if(program.is_used("--inverse-computation")) solver.icntl_inverse_computation(program.get<int>("--inverse-computation"));
        if(program.is_used("--forward-elimination")) solver.icntl_forward_elimination(program.get<int>("--forward-elimination"));
        if(program.is_used("--determinant-computation")) solver.icntl_determinant_computation(program.get<int>("--determinant-computation"));
        if(program.is_used("--out-of-core-file")) solver.icntl_out_of_core_file(program.get<int>("--out-of-core-file"));
        if(program.is_used("--blr")) solver.icntl_blr(program.get<int>("--blr"));
        if(program.is_used("--blr-variant")) solver.icntl_blr_variant(program.get<int>("--blr-variant"));
        if(program.is_used("--blr-compression")) solver.icntl_blr_compression(program.get<int>("--blr-compression"));
        if(program.is_used("--lu-compression-rate")) solver.icntl_lu_compression_rate(program.get<int>("--lu-compression-rate"));
        if(program.is_used("--block-compression-rate")) solver.icntl_block_compression_rate(program.get<int>("--block-compression-rate"));
        if(program.is_used("--tree-parallelism")) solver.icntl_tree_parallelism(program.get<int>("--tree-parallelism"));
        if(program.is_used("--compact-working-space")) solver.icntl_compact_working_space(program.get<int>("--compact-working-space"));
        if(program.is_used("--rank-revealing-factorization")) solver.icntl_rank_revealing_factorization(program.get<int>("--rank-revealing-factorization"));
        if(program.is_used("--symbolic-factorization")) solver.icntl_symbolic_factorization(program.get<int>("--symbolic-factorization"));

        return;
    }
} // namespace ezp

#endif

//! @}
