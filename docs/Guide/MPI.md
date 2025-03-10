# MPI Environment Management

## Initialisation

It is ***not*** required to explicitly initialise either the `MPI` environment via `MPI_Init` or the `BLACS` environment via `blacs_gridinit`.
All those operations are automatically done via [`RAII`](https://en.cppreference.com/w/cpp/language/raii).

Use the following to get the rank and size of the current `BLACS` environment if needed.

```cpp
ezp::get_env<int>().rank();
ezp::get_env<int>().size();
```

## Finalisation

Finalisation is a bit more complex.
When several MPI based libraries are used together, each may have its own initialisation and finalisation processes.
Then who is in charge of finalising the `MPI` resources matters.
And the order of calling different finalisation functions matters.
Unfortunately, with `RAII`, there is no reliable way to explicitly assign an order such that, for example, `blacs_exit` is guaranteed to be called before `MPI_Finalize`.

However, it is possible to tell `ezp` to skip finalising `MPI`.
To do so, call the following function anywhere in your application.

```cpp
ezp::blacs_env<int>::do_not_manage_mpi();
```

Once this is called, `ezp` will memorise the setting and call `blacs_exit(1)` instead of `blacs_exit(0)` on destruction, the latter not only releases `blacs` resources but also further calls `MPI_Finalize`.

By such, you shall manually call `MPI_Finalize`, or probably let other libraries to finalise the `MPI` environment for you.
