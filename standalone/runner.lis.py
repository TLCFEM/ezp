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

#
# This script runs a parallel solver using MPI to distribute the computation.
#
# Functions:
#     run(nprocs: int, N: int, NRHS: int):
#         Spawns the solver process, broadcasts the problem configuration, sends the matrices,
#         receives the error code and solution, and disconnects the worker and merged communicator.
#
# Usage:
#     runner.pardiso.py <nprocs> <N> <NRHS>
#
# Arguments:
#     nprocs (int): Number of processes to spawn.
#     N (int): Size of the matrix.
#     NRHS (int): Number of right-hand sides.
#
# Example:
#     python runner.lis.py 4 100 10 -print all -p ilu -ilu_fill 1 -i fgmres
#

import numpy
import sys
from array import array
from mpi4py import MPI


def run(nprocs: int, N: int, NRHS: int, option: str):
    comm = MPI.COMM_WORLD

    # spawn the solver
    worker = comm.Spawn("./solver.lis", maxprocs=nprocs)
    all = worker.Merge()

    print(f"option: {option}")

    option = numpy.frombuffer(option.encode(), dtype=numpy.int8)

    # broadcast the problem configuration
    all.Bcast(array("i", [len(option), N, N, NRHS]), root=0)
    all.Bcast(option, root=0)

    # send the matrices
    ia = numpy.array([x for x in range(N + 1)], dtype=numpy.int32)
    ja = numpy.array([x for x in range(N)], dtype=numpy.int32)
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
        print("Usage: runner.pardiso.py <nprocs> <N> <NRHS> [option]")
        sys.exit(1)

    nprocs = int(sys.argv[1])
    N = int(sys.argv[2])
    NRHS = int(sys.argv[3])
    option = " ".join(sys.argv[4:]) if len(sys.argv) > 4 else ""

    run(nprocs, N, NRHS, option)
