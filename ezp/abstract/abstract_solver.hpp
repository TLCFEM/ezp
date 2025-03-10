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

#ifndef ABSTRACT_SOLVER_HPP
#define ABSTRACT_SOLVER_HPP

#include "ezp.h"

#include <array>
#include <atomic>
#include <cmath>
#include <ranges>
#include <vector>

namespace ezp {
    template<typename T> constexpr auto always_false_v = false;

    template<typename T> concept floating_t = std::is_same_v<T, float> || std::is_same_v<T, double>;
    template<typename T> concept complex_t = std::is_same_v<T, std::complex<typename T::value_type>> && floating_t<typename T::value_type>;
    template<typename T> concept data_t = floating_t<T> || complex_t<T>;
    template<typename T> concept index_t = std::is_same_v<T, std::int32_t> || std::is_same_v<T, std::int64_t>;

    template<typename T> concept has_mem = requires(T t) { requires data_t<std::remove_pointer_t<decltype(t.mem())>>; };
    template<typename T> concept has_memptr = requires(T t) { requires data_t<std::remove_pointer_t<decltype(t.memptr())>>; };
    template<typename T> concept has_data = requires(T t) { requires data_t<std::remove_pointer_t<decltype(t.data())>>; };
    template<typename T> concept has_iterator = requires(T t) {
        requires std::ranges::contiguous_range<T>;
        requires data_t<std::remove_reference_t<decltype(*t.begin())>>;
    };
    template<typename T> concept container_t = requires(T t) { requires index_t<decltype(t.n_rows)> && index_t<decltype(t.n_cols)> && (has_mem<T> || has_memptr<T> || has_data<T> || has_iterator<T>); };

    template<typename T> concept mem_t = data_t<T> || container_t<T>;

    template<index_t IT> class blacs_env final {
        static constexpr IT ZERO{0}, ONE{1};

        static std::atomic_bool FINALIZE;

        IT _rank{-1}, _size{-1};

    public:
        blacs_env() { blacs_pinfo(&_rank, &_size); }
        ~blacs_env() { blacs_exit(FINALIZE ? &ZERO : &ONE); }

        /**
         * @brief Disables the management of MPI (Message Passing Interface) finalization.
         *
         * This static function sets the FINALIZE flag to false, indicating that the
         * MPI environment should not be finalized by `blacs`. This can be useful
         * in scenarios where MPI is managed externally or by another component.
         */
        static void do_not_manage_mpi() { FINALIZE = false; }

        auto rank() const { return _rank; }
        auto size() const { return _size; }
    };

    template<index_t IT> std::atomic_bool blacs_env<IT>::FINALIZE{true};

    /**
     * @brief Retrieves a constant reference to a static instance of blacs_env.
     *
     * This function returns a constant reference to a static instance of the
     * blacs_env class template, which is scoped to the template parameter IT.
     * The static instance ensures that the environment is initialized only once
     * and shared across all calls to this function with the same template parameter.
     *
     * @tparam IT The index type used to instantiate the blacs_env template.
     * @return A constant reference to the static blacs_env instance.
     */
    template<index_t IT> const auto& get_env() {
        static const blacs_env<IT> scoped_env;

        return scoped_env;
    }

    template<mem_t DT, index_t IT> struct full_mat {
        const IT n_rows, n_cols;
        DT* const data;
        const bool distributed = false;
    };

    template<mem_t DT, index_t IT> struct band_mat {
        const IT n_rows, n_cols, kl, ku;
        DT* const data;
        const bool distributed = false;
    };

    template<mem_t DT, index_t IT> struct band_symm_mat {
        const IT n_rows, n_cols, klu;
        DT* const data;
        const bool distributed = false;
    };

    template<index_t IT> using desc = std::array<IT, 9>;

    template<index_t IT> class blacs_context final {
        static constexpr IT ZERO{0}, ONE{1}, NEGONE{-1};
        static constexpr char SCOPE = 'A', TOP = ' ';

        IT info;

        auto init(const char order) {
            blacs_get(nullptr, &ZERO, &context);
            blacs_gridinit(&context, &order, &n_rows, &n_cols);
            blacs_pinfo(&rank, &size);
            blacs_gridinfo(&context, &n_rows, &n_cols, &my_row, &my_col);
        }

        auto release() {
            if(context >= 0) blacs_gridexit(&context);
        }

        /**
         * @brief Copies data from matrix A to matrix B using BLACS/ScaLAPACK functions.
         *
         * This function transfers data between distributed matrices that may have different
         * distributions in the process grid. It uses pxgemr2d_ routines from ScaLAPACK.
         *
         * @tparam DT The data type of the matrices (float or double)
         * @param A Pointer to source matrix
         * @param desc_a Array descriptor for matrix A
         * @param B Pointer to destination matrix
         * @param desc_b Array descriptor for matrix B
         *
         * @note The function automatically selects between double and float implementations
         *       based on the template parameter DT.
         * @note The matrices must be valid and properly distributed according to their descriptors.
         */
        template<data_t DT> auto copy_to(const DT* A, const IT* desc_a, DT* B, const IT* desc_b) {
            // ReSharper disable CppCStyleCast
            if(std::is_same_v<DT, double>) {
                using E = double;
                pdgemr2d(desc_a + 2, desc_a + 3, (E*)A, &ONE, &ONE, desc_a, (E*)B, &ONE, &ONE, desc_b, &context);
            }
            else if (std::is_same_v<DT, float>) {
                using E = float;
                psgemr2d(desc_a + 2, desc_a + 3, (E*)A, &ONE, &ONE, desc_a, (E*)B, &ONE, &ONE, desc_b, &context);
            }else if(std::is_same_v<DT, complex16>) {
                using E = complex16;
                pzgemr2d(desc_a + 2, desc_a + 3, (E*)A, &ONE, &ONE, desc_a, (E*)B, &ONE, &ONE, desc_b, &context);
            }
            else if (std::is_same_v<DT, complex8>) {
                using E = complex8;
                pcgemr2d(desc_a + 2, desc_a + 3, (E*)A, &ONE, &ONE, desc_a, (E*)B, &ONE, &ONE, desc_b, &context);
            }
            // ReSharper restore CppCStyleCast
        }

    public:
        IT n_rows, n_cols, context{-1}, rank{-1}, size{-1}, my_row{-1}, my_col{-1};

        blacs_context()
            : blacs_context(get_env<IT>().size(), 1) {}

        explicit blacs_context(const char order)
            : n_rows(-1)
            , n_cols(-1) {
            const auto& env = get_env<IT>();
            n_rows = std::max(1, static_cast<IT>(std::sqrt(env.size())));
            n_cols = env.size() / n_rows;
            init(order);
        };

        blacs_context(const IT rows, const IT cols, const char order = 'R')
            : n_rows(rows)
            , n_cols(cols) { init(order); };

        blacs_context(const blacs_context&) = delete;
        blacs_context(blacs_context&&) noexcept = delete;
        blacs_context& operator=(const blacs_context&) = delete;
        blacs_context& operator=(blacs_context&&) noexcept = delete;

        ~blacs_context() { release(); }

        /**
         * @brief Generates a descriptor for a global matrix.
         *
         * This function initializes and returns a descriptor for a global matrix
         * with the specified number of rows and columns.
         *
         * @tparam IT The integer type used for matrix dimensions.
         * @param num_rows The number of rows in the global matrix.
         * @param num_cols The number of columns in the global matrix.
         * @return A descriptor for the global matrix.
         */
        auto desc_g(const IT num_rows, const IT num_cols) {
            desc<IT> desc_t{};

            descinit(desc_t.data(), &num_rows, &num_cols, &num_rows, &num_cols, &ZERO, &ZERO, &context, &num_rows, &info);

            return desc_t;
        }

        /**
         * @brief Generates a descriptor for a local matrix.
         *
         * This function initializes a descriptor for a local matrix with the given dimensions
         * and block sizes.
         *
         * @tparam IT The integer type used for matrix dimensions and block sizes.
         * @param num_rows The number of rows in the global matrix.
         * @param num_cols The number of columns in the global matrix.
         * @param row_block The block size in the row dimension.
         * @param col_block The block size in the column dimension.
         * @param lead The leading dimension of the local matrix.
         * @return A descriptor for the local matrix.
         */
        auto desc_l(const IT num_rows, const IT num_cols, const IT row_block, const IT col_block, const IT lead) {
            desc<IT> desc_t{};

            const auto loc_lead = std::max(1, lead);
            descinit(desc_t.data(), &num_rows, &num_cols, &row_block, &col_block, &ZERO, &ZERO, &context, &loc_lead, &info);

            return desc_t;
        }

        auto desc_l(const IT num_rows, const IT num_cols, const IT block, const IT lead) { return desc_l(num_rows, num_cols, block, block, lead); }

        template<data_t DT> auto scatter(const full_mat<DT, IT>& A, const desc<IT>& desc_a, std::vector<DT>& B, const desc<IT>& desc_b) {
            if(!A.distributed) return copy_to(A.data, desc_a.data(), B.data(), desc_b.data());

            for(auto i = 0u; i < B.size(); ++i) B[i] = A.data[i];
        }

        template<data_t DT> auto gather(const std::vector<DT>& A, const desc<IT>& desc_a, const full_mat<DT, IT>& B, const desc<IT>& desc_b) {
            if(!B.distributed) return copy_to(A.data(), desc_a.data(), B.data, desc_b.data());

            for(auto i = 0u; i < A.size(); ++i) B.data[i] = A[i];
        }

        [[nodiscard]] bool is_valid() const { return my_row >= 0 && my_col >= 0; }

        /**
         * @brief Computes the row block size.
         */
        auto row_block(const IT n) const { return std::max(1, n / n_rows); }

        /**
         * @brief Computes the column block size.
         */
        auto col_block(const IT n) const { return std::max(1, n / n_cols); }

        /**
         * @brief Computes the number of local rows of the current process.
         *
         * This function calculates the number of rows of the current process
         * in a distributed matrix using the `numroc` function.
         *
         * @param n The total number of rows in the global matrix.
         * @param nb The block size used for the distribution.
         * @return The number of local rows of the current process.
         */
        auto rows(const IT n, const IT nb) const { return numroc(&n, &nb, &my_row, &ZERO, &n_rows); }

        /**
         * @brief Computes the number of local columns of the current process.
         *
         * This function calculates the number of columns of the current process
         * in a distributed matrix using the `numroc` function.
         *
         * @param n The total number of columns in the global matrix.
         * @param nb The block size used for the distribution.
         * @return The number of local columns of the calling process.
         */
        auto cols(const IT n, const IT nb) const { return numroc(&n, &nb, &my_col, &ZERO, &n_cols); }

        /**
         * @brief Perform the global amx operation.
         *
         * This function takes an integer number, which may be different on each process,
         * and computes the maximum value across all processes.
         * The result is broadcasted to all processes.
         *
         * For example, if the input number is 0, 1, 2, 3 on four processes, this function
         * will return 3 on all processes.
         *
         * @param number The integer number to be updated.
         * @return The updated integer number.
         */
        IT amx(IT number) const {
            igamx2d(&context, &SCOPE, &TOP, &ONE, &ONE, &number, &ONE, nullptr, nullptr, &NEGONE, &NEGONE, &NEGONE);
            return number;
        }

        /**
         * @brief Perform the global amn operation.
         *
         * This function takes an integer number, which may be different on each process,
         * and computes the minimum value across all processes.
         * The result is broadcast to all processes.
         *
         * For example, if the input number is 0, 1, 2, 3 on four processes, this function
         * will return 0 on all processes.
         *
         * @param number The integer number to be updated.
         * @return The updated integer number.
         */
        IT amn(IT number) const {
            igamn2d(&context, &SCOPE, &TOP, &ONE, &ONE, &number, &ONE, nullptr, nullptr, &NEGONE, &NEGONE, &NEGONE);
            return number;
        }
    };

    namespace detail {
        template<index_t IT> class abstract_solver {
        protected:
            static constexpr IT ZERO{0}, ONE{1};

        public:
            abstract_solver() = default;

            virtual ~abstract_solver() = default;
        };
    } // namespace detail
} // namespace ezp

#endif // ABSTRACT_SOLVER_HPP