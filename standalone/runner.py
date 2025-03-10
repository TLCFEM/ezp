#  Copyright (C) 2025 Theodore Chang
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

from array import array
import sys
from mpi4py import MPI
import numpy


def run(nprocs: int, N: int, NRHS: int):
    comm = MPI.COMM_WORLD

    # spawn the solver
    worker = comm.Spawn("./solver.pgesv", maxprocs=nprocs)
    all = worker.Merge()

    # broadcast the problem configuration
    all.Bcast(array("i", [N, NRHS, 1]), root=0)

    # send the matrices
    # ! numpy arrays are row-major by default
    # ! column-major order is required by (Sca)LAPACK
    # ! thus need to specify the order as "F"
    A = numpy.zeros((N, N), dtype=numpy.float64, order="F")
    for i in range(N):
        A[i, i] = i + 1
    print("A:\n", A)
    worker.Send(A, dest=0, tag=0)

    B = numpy.ones((N, NRHS), dtype=numpy.float64, order="F")
    for i in range(NRHS):
        B[:, i] = i + 1
    print("B:\n", B)
    worker.Send(B, dest=0, tag=1)

    # receive the error code and the solution if no error
    error = array("i", [-1])
    worker.Recv(error)
    if 0 == error[0]:
        X = numpy.ones((N, NRHS), dtype=numpy.float64, order="F")
        worker.Recv(X)
        print("Solution:\n", X)

    worker.Disconnect()
    all.Disconnect()


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: runner.py <nprocs> <N> <NRHS>")
        sys.exit(1)

    nprocs = int(sys.argv[1])
    N = int(sys.argv[2])
    NRHS = int(sys.argv[3])

    run(nprocs, N, NRHS)
