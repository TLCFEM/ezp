# Custom Matrix Class

`ezp` supports customised matrix objects.

Apart from using the provided wrappers such as `full_mat`, `band_mat` and `band_symm_mat`, one can also define or use customised matrix classes to call the solver.

All custom matrix classes shall have the public member `.n_rows` and `.n_cols`.
The band matrix further requires the public member `.kl` and `.ku` which are the numbers of sub-diagonals (lower bandwidth) and super-diagonals (upper bandwidth).
The band symmetric matrix requires the additional public member `.klu` which is the bandwidth.

All custom classes must define at least one of the following public methods, that return a pointer to the first element.
It is assumed the memory layout is [contiguous](https://en.cppreference.com/w/cpp/named_req/ContiguousContainer), thus, one can use `std::vector<T>`, `std::array<T>`, `std::unique_ptr<T[]>` as the storage.

1. `.mem()` -> `T*`
2. `.memptr()` -> `T*`
3. `.data()` -> `T*`
4. contiguous iterator pair `.begin()` and `.end()`, and `&(*begin())` -> `T*`

The possibility is unlimited.
The simplest is to subclass `std::vector<T>` with additional public members.
In the example of `posv` solver, a custom matrix class is used.

```cpp title="example.pposv.cpp" hl_lines="9 27 29"
--8<--
./examples/example.pposv.cpp:27
--8<--
```