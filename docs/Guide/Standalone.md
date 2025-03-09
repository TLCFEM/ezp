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

```cpp title="solver.pgesv.cpp:46:49" hl_lines="3 4"
--8<--
./standalone/solver.pgesv.cpp:46:49
--8<--
```

Since `mpl` handles `MPI` environment initialisation and finalisation, it is necessary to tell `ezp` to ignore finalising the `MPI` environment.
The function `ezp::blacs_env::do_not_manage_mpi()` must be called if `MPI` is managed by external tools.

```cpp title="solver.pgesv.cpp:77" hl_lines="1"
--8<--
./standalone/solver.pgesv.cpp:77:77
--8<--
```

The parameters of the linear system will be broadcast over by the main application first.

```cpp title="solver.pgesv.cpp:84:92" hl_lines="5"
--8<--
./standalone/solver.pgesv.cpp:84:92
--8<--
```

Knowing the problem sizes `N` and `NRHS`, the root process can initialise the containers and receive the contents of two matrices.

```cpp title="solver.pgesv.cpp:54:64" hl_lines="1-3 7 8"
--8<--
./standalone/solver.pgesv.cpp:54:64
--8<--
```

Solving the system follows the conventional approach, that is, create a solver object and call the solve method with data wrapped.
The solution is sent back to the caller.

```cpp title="solver.pgesv.cpp:66:71" hl_lines="1 5"
--8<--
./standalone/solver.pgesv.cpp:66:71
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

```cpp title="runner.cpp:95:110" hl_lines="2 5 9 10 15 16"
--8<--
./standalone/runner.cpp:95:110
--8<--
```

## Full Reference Implementation

The following is a full reference implementation of a standalone solver and the corresponding caller logic.

```cpp title="runner.cpp"
--8<--
./standalone/runner.cpp
--8<--
```

```cpp title="solver.pgesv.cpp"
--8<--
./standalone/solver.pgesv.cpp
--8<--
```