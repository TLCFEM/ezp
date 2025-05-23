#include <ezp/mumps.hpp>
#include <ezp/pardiso.hpp>
#include <fstream>
#include <tuple>
#include <vector>

using flt_t = double;

#ifndef EZP_ENABLE_TEST
#include <iostream>

#define BENCHMARK(name)
#define TEST_CASE(name, tag) int main(int, char*[])
#define REQUIRE(condition)
#else
#include <catch2/catchy.hpp>

using namespace Catch;

class MasterConsoleReporter final : public ConsoleReporter {
    const bool is_master = ezp::get_env<>().rank() == 0;

public:
    using ConsoleReporter::ConsoleReporter;

    void testRunStarting(TestRunInfo const& _testRunInfo) override {
        if(is_master) ConsoleReporter::testRunStarting(_testRunInfo);
    }
    void testRunEnded(TestRunStats const& _testRunStats) override {
        if(is_master) ConsoleReporter::testRunEnded(_testRunStats);
    }
    void testCaseStarting(TestCaseInfo const& testInfo) override {
        if(is_master) ConsoleReporter::testCaseStarting(testInfo);
    }
    void testCaseEnded(TestCaseStats const& _testCaseStats) override {
        if(is_master) ConsoleReporter::testCaseEnded(_testCaseStats);
    }

    void testCasePartialStarting(TestCaseInfo const& testInfo, uint64_t partNumber) override {
        if(is_master) ConsoleReporter::testCasePartialStarting(testInfo, partNumber);
    }
    void testCasePartialEnded(TestCaseStats const& testCaseStats, uint64_t partNumber) override {
        if(is_master) ConsoleReporter::testCasePartialEnded(testCaseStats, partNumber);
    }

    void sectionStarting(SectionInfo const& _sectionInfo) override {
        if(is_master) ConsoleReporter::sectionStarting(_sectionInfo);
    }
    void sectionEnded(SectionStats const& _sectionStats) override {
        if(is_master) ConsoleReporter::sectionEnded(_sectionStats);
    }
    void assertionStarting(AssertionInfo const& _assertionInfo) override {
        if(is_master) ConsoleReporter::assertionStarting(_assertionInfo);
    }
    void assertionEnded(AssertionStats const& _assertionStats) override {
        if(is_master) ConsoleReporter::assertionEnded(_assertionStats);
    }

    void benchmarkPreparing(StringRef name) override {
        if(is_master) ConsoleReporter::benchmarkPreparing(name);
    }
    void benchmarkStarting(BenchmarkInfo const& info) override {
        if(is_master) ConsoleReporter::benchmarkStarting(info);
    }
    void benchmarkEnded(BenchmarkStats<> const& stats) override {
        if(is_master) ConsoleReporter::benchmarkEnded(stats);
    }
    void benchmarkFailed(StringRef error) override {
        if(is_master) ConsoleReporter::benchmarkFailed(error);
    }
};

CATCH_REGISTER_REPORTER("master", MasterConsoleReporter)
#endif

template<bool one_based> auto prepare(const std::string_view file_name) {
    auto file = std::ifstream(file_name.data());
    if(!file.is_open()) throw std::runtime_error("Failed to open file.");

    std::string line;
    std::getline(file, line);
    int n, nnz;
    std::istringstream(line) >> n >> n >> nnz;

    std::vector<int> row, col;
    std::vector<flt_t> val;

    if(ezp::get_env<>().rank() == 0) {
        constexpr auto base = one_based ? 0 : 1;

        row.reserve(nnz);
        col.reserve(nnz);
        val.reserve(nnz);
        while(std::getline(file, line)) {
            int i, j;
            flt_t v;
            std::istringstream(line) >> i >> j >> v;
            row.push_back(i + base);
            col.push_back(j + base);
            val.push_back(v);
        }
    }

    return std::make_tuple(n, nnz, std::move(row), std::move(col), std::move(val));
}

#ifdef EZP_MKL
auto benchmark_pardiso(const int n, const int nnz, std::vector<int>& row, std::vector<int>& col, std::vector<flt_t>& val) {
    std::vector b(n, flt_t{1});
    auto solver = ezp::pardiso<flt_t, int_t>(ezp::real_and_nonsymmetric, ezp::no_output);

    solver.iparm_default_value(1);
    solver.iparm_reducing_ordering(3);
    solver.iparm_iterative_refinement(2);
    solver.iparm_pivoting_perturbation(std::is_same_v<flt_t, double> ? 14 : 7);
    solver.iparm_weighted_matching(1);
    solver.iparm_scaling(1);

    const ezp::sparse_coo_mat coo_mat{n, nnz, row.data(), col.data(), val.data()};

    int_t info;
    BENCHMARK("PARDISO Full Solve") {
        info = solver.solve(ezp::sparse_csr_mat<flt_t, int_t>{coo_mat, true, false}, {n, 1, b.data()});
        REQUIRE(0 == info);
    };
    BENCHMARK("PARDISO Factored Solve") {
        info = solver.solve({n, 1, b.data()});
        REQUIRE(0 == info);
    };

#ifndef EZP_ENABLE_TEST
    if(ezp::get_env<>().rank() == 0) std::cout << "PARDISO Solve Info: " << info << '\n';
#endif
}
#endif

auto benchmark_mumps(const int n, const int nnz, std::vector<int>& row, std::vector<int>& col, std::vector<flt_t>& val) {
    std::vector b(n, flt_t{1});
    auto solver = ezp::mumps<double, int>();

    solver.icntl_printing_level(1);

    int info;
    BENCHMARK("MUMPS Full Solve") {
        info = solver.solve({n, nnz, row.data(), col.data(), val.data()}, {n, 1, b.data()});
        REQUIRE(0 == info);
    };
    BENCHMARK("MUMPS Factored Solve") {
        info = solver.solve({n, 1, b.data()});
        REQUIRE(0 == info);
    };

#ifndef EZP_ENABLE_TEST
    if(ezp::get_env<>().rank() == 0) std::cout << "MUMPS Solve Info: " << info << '\n';
#endif
}

TEST_CASE("Sparse Benchmark", "[Benchmarking]") {
    ezp::blacs_env<>::do_not_manage_mpi();

    try {
        auto [n, nnz, row, col, val] = prepare<true>("../misc/system5A0.mtx");
#ifdef EZP_MKL
        benchmark_pardiso(n, nnz, row, col, val);
#endif
        benchmark_mumps(n, nnz, row, col, val);
    }
    catch(...) {
    }

#ifndef EZP_ENABLE_TEST
    return 0;
#endif
}
