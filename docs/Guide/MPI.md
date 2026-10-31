# MPI Environment Management

## Initialisation

It is **not** required to initialise the `MPI` environment with `MPI_Init` or the `BLACS` environment with `blacs_gridinit`.
All of these operations are performed automatically through [RAII](https://en.cppreference.com/w/cpp/language/raii).
The global environment is a singleton obtained from `get_env<IT>()`.
Use it to query the rank and size of the current `BLACS` environment.

```cpp
ezp::get_env<int>().rank();
ezp::get_env<int>().size();
```

## Finalisation

Finalisation is more involved.
When several `MPI`-based libraries are used together, each may manage its own initialisation and finalisation.
It then matters who finalises the `MPI` resources and in what order.
With RAII there is no reliable way to enforce an order that guarantees, for example, `blacs_exit` runs before `MPI_Finalize`.

It is, however, possible to tell `ezp` not to finalise `MPI`.
Call the following function anywhere in the application.

```cpp
ezp::blacs_env<int>::do_not_manage_mpi();
```

Once this is called, `ezp` remembers the setting.
On destruction it calls `blacs_exit(1)` instead of `blacs_exit(0)`.
The latter releases the `BLACS` resources and also calls `MPI_Finalize`, whereas the former only releases the resources.
You are then responsible for calling `MPI_Finalize`, or for letting another library finalise the `MPI` environment.
