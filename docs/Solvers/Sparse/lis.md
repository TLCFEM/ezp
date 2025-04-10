# lis

The `lis` solver supports the following input types.

* data type: D
* index type: `std::int32_t`, `std::int64_t`

## Solver Options

Call the `set_option` method to set solver options.

```cpp
solver.set_option("-i fgmres -p ilu -print 2");
```

All available options can be seen in the [official documentation](https://www.ssisc.org/lis/).
