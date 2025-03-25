#include "external/lis/include/lis.h"

#include <stdio.h>
LIS_INT main(int argc, char* argv[]) {
    LIS_Comm comm;
    LIS_MATRIX A;
    LIS_VECTOR b, x, u;
    LIS_SOLVER solver;
    int nprocs, my_rank;
    LIS_INT err, i, n, gn, is, ie, iter;

    n = 12;
    lis_initialize(&argc, &argv);
    comm = LIS_COMM_WORLD;

    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &my_rank);

    lis_printf(comm, "\n");
    lis_printf(comm, "number of processes = %d\n", nprocs);

#ifdef _OPENMP
    lis_printf(comm, "max number of threads = %d\n", omp_get_num_procs());
    lis_printf(comm, "number of threads = %d\n", omp_get_max_threads());
#endif

    lis_matrix_create(comm, &A);
    err = lis_matrix_set_size(A, 0, n);
    CHKERR(err);
    lis_matrix_get_size(A, &n, &gn);
    lis_matrix_get_range(A, &is, &ie);
    for(i = is; i < ie; i++) {
        if(i > 0) lis_matrix_set_value(LIS_INS_VALUE, i, i - 1, -1.0, A);
        if(i < gn - 1) lis_matrix_set_value(LIS_INS_VALUE, i, i + 1, -1.0, A);
        lis_matrix_set_value(LIS_INS_VALUE, i, i, 2.0, A);
    }
    lis_matrix_set_type(A, LIS_MATRIX_CSR);
    lis_matrix_assemble(A);

    lis_vector_duplicate(A, &u);
    lis_vector_duplicate(A, &b);
    lis_vector_duplicate(A, &x);
    lis_vector_set_all(1.0, u);
    lis_matvec(A, u, b);
    lis_solver_create(&solver);
    lis_solver_set_option("-print mem", solver);
    err = lis_solver_set_optionC(solver);
    CHKERR(err);
    lis_solve(A, b, x, solver);
    lis_solver_get_iter(solver, &iter);
    lis_printf(comm, "number of iterations = %D\n", iter);
    lis_printf(comm, "\n");
    lis_vector_print(x);

    lis_matrix_destroy(A);
    lis_vector_destroy(b);
    lis_vector_destroy(x);
    lis_vector_destroy(u);
    lis_solver_destroy(solver);
    lis_finalize();
    return 0;
}