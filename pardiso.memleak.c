#include "mkl.h"
#include "mkl_cluster_sparse_solver.h"
#include "mpi.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    MKL_INT n = 1;
    MKL_INT ia[2] = {1, 2};
    MKL_INT ja[1] = {1};
    double a[1] = {1.0};

    MKL_INT mtype = 11;
    MKL_INT nrhs = 1;

    void* pt[64] = {0};

    MKL_INT iparm[64] = {0};
    MKL_INT maxfct = 1, mnum = 1, phase, msglvl = 1, error = 0;

    int comm = 0, rank = 0;

    MPI_Init(0, 0);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    comm = MPI_Comm_c2f(MPI_COMM_WORLD);

    iparm[0] = 1;   /* Solver default parameters overriden with provided by iparm */
    iparm[1] = 2;   /* Use METIS for fill-in reordering */
    iparm[5] = 0;   /* Write solution into x */
    iparm[7] = 2;   /* Max number of iterative refinement steps */
    iparm[9] = 13;  /* Perturb the pivot elements with 1E-13 */
    iparm[10] = 1;  /* Use nonsymmetric permutation and scaling MPS */
    iparm[12] = 1;  /* Switch on Maximum Weighted Matching algorithm (default for non-symmetric) */
    iparm[17] = -1; /* Output: Number of nonzeros in the factor LU */
    iparm[18] = -1; /* Output: Mflops for LU factorization */
    iparm[26] = 1;  /* Check input data for correctness */
    iparm[39] = 0;  /* Input: matrix/rhs/solution stored on master */

    phase = 11;
    cluster_sparse_solver(pt, &maxfct, &mnum, &mtype, &phase, &n, a, ia, ja, 0, &nrhs, iparm, &msglvl, 0, 0, &comm, &error);

    phase = -1;
    cluster_sparse_solver(pt, &maxfct, &mnum, &mtype, &phase, &n, 0, ia, ja, 0, &nrhs, iparm, &msglvl, 0, 0, &comm, &error);

    MPI_Finalize();

    return 0;
}
