#  Copyright (C) 2025-2026 Theodore Chang
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

#
# This script runs a parallel MUMPS solver using MPI.
#
# Functions:
#     run(nprocs: int, N: int, NRHS: int):
#         Spawns the MUMPS solver with the specified number of processes and problem size.
#         Broadcasts the problem configuration and sends the matrices to the solver.
#         Receives the error code and solution from the solver.
#
# Usage:
#     runner.mumps.py <nprocs> <N> <NRHS>
#
# Arguments:
#     nprocs (int): Number of processes to spawn.
#     N (int): Size of the matrix.
#     NRHS (int): Number of right-hand sides.
#
# Example:
#     python runner.mumps.py 4 1000 10
#

import numpy
import sys
from array import array
from mpi4py import MPI


def run(nprocs: int, N: int, NRHS: int):
    comm = MPI.COMM_WORLD

    # spawn the solver
    worker = comm.Spawn("./solver.mumps", maxprocs=nprocs)
    all = worker.Merge()

    # broadcast the problem configuration
    all.Bcast(array("i", [0, NRHS, N, N, 1]), root=0)

    # send the matrices
    ia = numpy.array([x + 1 for x in range(N)], dtype=numpy.int32)
    ja = numpy.array([x + 1 for x in range(N)], dtype=numpy.int32)
    a = numpy.array([x + 1 for x in range(N)], dtype=numpy.float64)
    b = numpy.ones((N, NRHS), dtype=numpy.float64, order="F")
    for i in range(NRHS):
        b[:, i] = i + 1

    worker.Send(ia, dest=0, tag=0)
    worker.Send(ja, dest=0, tag=1)
    worker.Send(a, dest=0, tag=2)
    worker.Send(b, dest=0, tag=3)

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
        print("Usage: runner.mumps.py <nprocs> <N> <NRHS>")
        sys.exit(1)

    nprocs = int(sys.argv[1])
    N = int(sys.argv[2])
    NRHS = int(sys.argv[3])

    run(nprocs, N, NRHS)
