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
 * @fn pardiso_set
 * @brief Parse PARDISO parameters.
 *
 * Usage: [--default-value INT] [--reducing-ordering INT] [--user-permutation INT] [--iterative-refinement INT] [--pivoting-perturbation INT] [--scaling INT] [--transpose-matrix INT] [--weighted-matching INT] [--nnz-factor INT] [--pivoting-type INT] [--matrix-checker INT] [--partial-solve INT] [--zero-based-indexing INT] [--schur-complement INT] [--out-of-core INT]
 *
 * Optional arguments:
 *   --default-value INT          [0] Use default values. [default: 0]
 *   --reducing-ordering INT      [1] Fill-in reducing ordering for the input matrix. [default: 2]
 *   --user-permutation INT       [4] User permutation. [default: 0]
 *   --iterative-refinement INT   [7] Iterative refinement step. [default: 0]
 *   --pivoting-perturbation INT  [9] Pivoting perturbation.
 *   --scaling INT                [10] Scaling vectors. [default: 1]
 *   --transpose-matrix INT       [11] Solve with transposed or conjugate transposed matrix A. [default: 0]
 *   --weighted-matching INT      [12] Improved accuracy using (non-)symmetric weighted matching. [default: 1]
 *   --nnz-factor INT             [17] Report the number of non-zero elements in the factors. [default: -1]
 *   --pivoting-type INT          [20] Pivoting for symmetric indefinite matrices. [default: 1]
 *   --matrix-checker INT         [26] Matrix checker. [default: 0]
 *   --partial-solve INT          [30] Partial solve and computing selected components of the solution vectors. [default: 0]
 *   --zero-based-indexing INT    [34] One- or zero-based indexing of columns and rows. [default: 0]
 *   --schur-complement INT       [35] Schur complement matrix computation control. [default: 0]
 *   --out-of-core INT            [59] Solver mode. [default: 0]
 *
 * @author tlc
 * @date 06/04/2025
 * @version 1.0.0
 * @file pardiso.parser.hpp
 * @{
 */

#ifndef PARDISO_PARSER_HPP
#define PARDISO_PARSER_HPP

#include "pardiso.hpp"

#include <argparse/argparse.hpp>

namespace ezp {
    namespace detail {
        static constexpr std::string pardiso_name = "pardiso";
    } // namespace detail

    template<data_t DT, index_t IT> auto pardiso_set(std::string command, pardiso<DT, IT>& solver) {
        argparse::ArgumentParser program(detail::pardiso_name, "", argparse::default_arguments::none);
        program.add_argument("--default-value").help("[0] Use default values.").default_value(0).choices(0, 1).scan<'i', int>().metavar("INT").nargs(1);
        program.add_argument("--reducing-ordering").help("[1] Fill-in reducing ordering for the input matrix.").default_value(2).choices(2, 3, 10).scan<'i', int>().metavar("INT").nargs(1);
        program.add_argument("--user-permutation").help("[4] User permutation.").default_value(0).choices(0, 1, 2).scan<'i', int>().metavar("INT").nargs(1);
        program.add_argument("--iterative-refinement").help("[7] Iterative refinement step.").default_value(0).scan<'i', int>().metavar("INT").nargs(1);
        program.add_argument("--pivoting-perturbation").help("[9] Pivoting perturbation.").choices(13, 8).scan<'i', int>().metavar("INT").nargs(1);
        program.add_argument("--scaling").help("[10] Scaling vectors.").default_value(1).choices(0, 1).scan<'i', int>().metavar("INT").nargs(1);
        program.add_argument("--transpose-matrix").help("[11] Solve with transposed or conjugate transposed matrix A.").default_value(0).choices(0, 1, 2).scan<'i', int>().metavar("INT").nargs(1);
        program.add_argument("--weighted-matching").help("[12] Improved accuracy using (non-)symmetric weighted matching.").default_value(1).choices(0, 1).scan<'i', int>().metavar("INT").nargs(1);
        program.add_argument("--nnz-factor").help("[17] Report the number of non-zero elements in the factors.").default_value(-1).scan<'i', int>().metavar("INT").nargs(1);
        program.add_argument("--pivoting-type").help("[20] Pivoting for symmetric indefinite matrices.").default_value(1).choices(0, 1).scan<'i', int>().metavar("INT").nargs(1);
        program.add_argument("--matrix-checker").help("[26] Matrix checker.").default_value(0).choices(0, 1).scan<'i', int>().metavar("INT").nargs(1);
        program.add_argument("--partial-solve").help("[30] Partial solve and computing selected components of the solution vectors.").default_value(0).choices(0, 1, 2, 3).scan<'i', int>().metavar("INT").nargs(1);
        program.add_argument("--zero-based-indexing").help("[34] One- or zero-based indexing of columns and rows.").default_value(0).choices(0, 1).scan<'i', int>().metavar("INT").nargs(1);
        program.add_argument("--schur-complement").help("[35] Schur complement matrix computation control.").default_value(0).choices(0, 1, 2).scan<'i', int>().metavar("INT").nargs(1);
        program.add_argument("--out-of-core").help("[59] Solver mode.").default_value(0).choices(0, 1, 2).scan<'i', int>().metavar("INT").nargs(1);

        std::vector<std::string> args{detail::pardiso_name};
        command = command.substr(command.find('-'));
        std::istringstream iss(command);
        args.insert(args.end(), std::istream_iterator<std::string>(iss), std::istream_iterator<std::string>());
        program.parse_args(args);

        if(program.is_used("--default-value")) solver.iparm_default_value(program.get<int>("--default-value"));
        if(program.is_used("--reducing-ordering")) solver.iparm_reducing_ordering(program.get<int>("--reducing-ordering"));
        if(program.is_used("--user-permutation")) solver.iparm_user_permutation(program.get<int>("--user-permutation"));
        if(program.is_used("--iterative-refinement")) solver.iparm_iterative_refinement(program.get<int>("--iterative-refinement"));
        if(program.is_used("--pivoting-perturbation")) solver.iparm_pivoting_perturbation(program.get<int>("--pivoting-perturbation"));
        if(program.is_used("--scaling")) solver.iparm_scaling(program.get<int>("--scaling"));
        if(program.is_used("--transpose-matrix")) solver.iparm_transpose_matrix(program.get<int>("--transpose-matrix"));
        if(program.is_used("--weighted-matching")) solver.iparm_weighted_matching(program.get<int>("--weighted-matching"));
        if(program.is_used("--nnz-factor")) solver.iparm_nnz_factor(program.get<int>("--nnz-factor"));
        if(program.is_used("--pivoting-type")) solver.iparm_pivoting_type(program.get<int>("--pivoting-type"));
        if(program.is_used("--matrix-checker")) solver.iparm_matrix_checker(program.get<int>("--matrix-checker"));
        if(program.is_used("--partial-solve")) solver.iparm_partial_solve(program.get<int>("--partial-solve"));
        if(program.is_used("--zero-based-indexing")) solver.iparm_zero_based_indexing(program.get<int>("--zero-based-indexing"));
        if(program.is_used("--schur-complement")) solver.iparm_schur_complement(program.get<int>("--schur-complement"));
        if(program.is_used("--out-of-core")) solver.iparm_out_of_core(program.get<int>("--out-of-core"));

        return;
    }
} // namespace ezp

#endif

//! @}
