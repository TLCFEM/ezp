# Standalone Solver

MPI provides a mechanism to dynamically spawn worker processes via `MPI_Comm_spawn`.
By utilising this feature, it is possible to separate the main application from the solver.
The architecture can be illustrated as follows.

![architecture design](communication.svg)

The `esp` solvers are also available as standalone executables that follow the above design.
Such a form is particularly beneficial if the following holds.

1. The main application is not suitable for distributed-memory model due to, for example, latency constraints.
2. The main application adopts a different parallelism pattern.
3. Sometimes, one may want to have a clear boundary between the main application and the solver(s).

However, such a clear boundary may be a shortcoming as well.
For example, the root process needs to store the whole matrices in memory, which is subject to physical limitations.
And since the spawned processes are not persistent, solving the same system with different right-hand sides may incur unnecessary data communication.

If the matrices need to be read from IO in a distributed manner, one may also need a refined control of how the data is prepared, by probably directly using `BLACS`/`ScaLAPACK`/`MPI` functions.

## Backbone

Here, we use the `pgesv` solver as the example to illustrate some critical steps used in implementing the above architecture.

We are not using the raw `MPI` functions, instead, the `mpl` library is used.
It is necessary to get the default communicator and the parent inter communicator.

```cpp title="solver.full.hpp:24:25"
--8<--
./standalone/solver.full.hpp:24:25
--8<--
```

Since `mpl` handles `MPI` environment initialisation and finalisation, it is necessary to tell `ezp` to ignore finalising the `MPI` environment.
The function `ezp::blacs_env::do_not_manage_mpi()` must be called if `MPI` is managed by external tools.

```cpp title="solver.pgesv.cpp:54" hl_lines="1"
--8<--
./standalone/solver.pgesv.cpp:54:55
--8<--
```

The parameters of the linear system will be broadcast over by the main application first.

```cpp title="solver.pgesv.cpp:61:65" hl_lines="5"
--8<--
./standalone/solver.pgesv.cpp:61:65
--8<--
```

Knowing the problem sizes `N` and `NRHS`, the root process can initialise the containers and receive the contents of two matrices.

```cpp title="solver.full.hpp:28:40" hl_lines="3-5 9 10"
--8<--
./standalone/solver.full.hpp:28:40
--8<--
```

Solving the system follows the conventional approach, that is, create a solver object and call the solve method with data wrapped.
The solution is sent back to the caller.

```cpp title="solver.full.hpp:42:47" hl_lines="1 5"
--8<--
./standalone/solver.full.hpp:42:47
--8<--
```

## The Caller

From the above code snippets, one may observe that the caller needs to

1. broadcast `config`,
2. send `A` and `B`,
3. receive the error code and the solution `X` if no error occured.

To this end, one can declare the corresponding containers using `std::vector`.

```cpp title="runner.cpp:38:39" hl_lines="1 2"
--8<--
./standalone/runner.cpp:38:39
--8<--
```

The following creates a diagonal matrix for illustration.

```cpp title="runner.cpp:45:50" hl_lines="5 6"
--8<--
./standalone/runner.cpp:45:50
--8<--
```

Now since all the data is ready to be communicated, the actual communication is very concise and straightforward.

```cpp title="runner.cpp:111:126" hl_lines="2 5 9 10 15 16"
--8<--
./standalone/runner.cpp:111:126
--8<--
```

## Python Caller

Since there is a clear boundary between the main application and the solver, it is possible to use other MPI compatible to send/receive the problem.

For example, one can use [`mpi4py`](https://github.com/mpi4py/mpi4py) to call the solvers.

It has to be pointed out that, the ***column-major*** memory layout shall be kept.

```py title="runner.py" hl_lines="15 25 31 35 37 38"
--8<--
./standalone/runner.py:35
--8<--
```

Calling the above script with

```bash
./runner.py 2 6 2
```

yields the following is the solution, which is correct.

```text
A:
 [[1. 0. 0. 0. 0. 0.]
 [0. 2. 0. 0. 0. 0.]
 [0. 0. 3. 0. 0. 0.]
 [0. 0. 0. 4. 0. 0.]
 [0. 0. 0. 0. 5. 0.]
 [0. 0. 0. 0. 0. 6.]]
B:
 [[1. 2.]
 [1. 2.]
 [1. 2.]
 [1. 2.]
 [1. 2.]
 [1. 2.]]
Solution:
 [[1.         2.        ]
 [0.5        1.        ]
 [0.33333333 0.66666667]
 [0.25       0.5       ]
 [0.2        0.4       ]
 [0.16666667 0.33333333]]
```

## Full Reference Implementation

The following is a full reference implementation of a standalone solver and the corresponding caller logic.

??? note "runner.cpp"
    ```cpp title="runner.cpp"
    --8<--
    ./standalone/runner.cpp:27:131
    --8<--
    ```

??? note "solver.pgesv.cpp"
    ```cpp title="solver.pgesv.cpp"
    --8<--
    ./standalone/solver.pgesv.cpp:51:76
    --8<--
    ```