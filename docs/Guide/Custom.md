# Custom Matrix Class

`ezp` supports customised matrix objects.
Besides the provided wrappers `full_mat`, `band_mat`, and `band_symm_mat`, one may define or reuse custom matrix classes to call the solvers.
All custom matrix classes must expose public members `.n_rows` and `.n_cols`.
A band matrix must additionally expose `.kl` and `.ku`, the numbers of sub-diagonals (lower bandwidth) and super-diagonals (upper bandwidth).
A band symmetric matrix must expose the additional member `.klu`, the bandwidth.

Every custom class must define at least one of the following public methods, which return a pointer to the first element.
The memory layout is assumed to be [contiguous](https://en.cppreference.com/w/cpp/named_req/ContiguousContainer).
`std::vector<T>`, `std::array<T>`, and `std::unique_ptr<T[]>` are therefore all valid storages.

1. `.mem()` -> `T*`
2. `.memptr()` -> `T*`
3. `.data()` -> `T*`
4. a contiguous iterator pair `.begin()` and `.end()`, so that `&(*begin())` -> `T*`

The possibilities are unlimited.
The simplest approach is to subclass `std::vector<T>` and add the required public members.
The `posv` example below demonstrates a custom matrix class.

```cpp title="example.pposv.cpp:27:54" hl_lines="9 25 27"
--8<--
./examples/example.pposv.cpp:27:54
--8<--
```

All processes must provide the same dimensional inputs.
Thus `.n_rows` and `.n_cols` must be valid on every process.
The actual storage, however, is only initialised on the root process.

```cpp title="example.pposv.cpp:56:91" hl_lines="12 16 17 27"
--8<--
./examples/example.pposv.cpp:56:91
--8<--
```
