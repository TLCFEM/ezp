# Drop-in Usage

`ezp` is designed to be used as a drop-in library.
The `master` branch contains additional files needed for developing the project itself, which are not necessary for using it.
A lean version is available in the `lite-lib` branch.
Clone the repository with the following command to obtain only the files needed to use the library.

```bash
git clone --recursive --branch lite-lib https://github.com/TLCFEM/ezp.git
```

Because `ezp` is header-only, no build step or ABI management is required.
Include the desired solver header and call `solve`.

```cpp
#include <ezp/pgesv.hpp>

int main() {
    // wrap the matrices and solve
    solver.solve({N, N, A.data()}, {N, NRHS, B.data()});
}
```
