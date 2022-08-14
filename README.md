# pfPySpectra

pfpyspectra based on [pyspectra](https://github.com/NLESC-JCER/pyspectra.git), Python interface to the [C++ Spectra library](https://github.com/yixuan/spectra)

## Eigensolvers
**pfPySpecta** offers two general interfaces to [Spectra](https://github.com/yixuan/spectra): **eigensolver** and **eigensolverh**. For sparse
and symmetric matrices respectively.

These two functions would invoke the most suitable method based on the information provided by the user.

**Note**:

  The available selection_rules to compute a portion of the spectrum are:
  *  LargestMagn
  *  LargestReal
  *  LargestImag
  *  LargestAlge
  *  SmallestMagn
  *  SmallestReal
  *  SmallestImag
  *  SmallestAlge
  *  BothEnds

## Eigensolvers Sparse Interface
You can call directly the sparse interface. You would need
to import the following module:
```python
import numpy as np
import scipy.sparse as sp

from pfpyspectra import eigensolver, eigensolverh
```
The following functions are available in the spectra_sparse_interface:
*  ```py
   general_eigensolver(
    mat: scipy.sparse, eigenpairs: int, basis_size: int, selection_rule: str)
    -> (np.ndarray, np.ndarray)
   ```
*  ```py
   general_real_shift_eigensolver(
   mat: scipy.sparse, eigenpairs: int, basis_size: int, shift: float, selection_rule: str)
   -> (np.ndarray, np.ndarray)
   ```
*  ```py
   general_complex_shift_eigensolver(
     mat: scipy.sparse, eigenpairs: int, basis_size: int,
     shift_real: float, shift_imag: float, selection_rule: str)
     -> (np.ndarray, np.ndarray)
   ```
*  ```py
   symmetric_eigensolver(
     mat: scipy.sparse, eigenpairs: int, basis_size: int, selection_rule: str)
     -> (np.ndarray, np.ndarray)
   ```
*  ```py
   symmetric_shift_eigensolver(
     mat: scipy.sparse, eigenpairs: int, basis_size: int, shift: float, selection_rule: str)
     -> (np.ndarray, np.ndarray)
   ```
*  ```py
   symmetric_generalized_shift_eigensolver(
     mat_A: scipy.sparse, mat_B: scipy.sparse, eigenpairs: int, basis_size: int, shift: float,
     selection_rule: str)
     -> (np.ndarray, np.ndarray)
   ```

### Example

```py
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs, eigsh

from pfpyspectra import eigensolver, eigensolverh

# matrix size
size = 100

# number of eigenpairs to compute
nvalues = 2
SIGMA = 1.0 + 0.5j

# Create random matrix
xs = np.random.normal(size=size ** 2).reshape(size, size)
print("xs = ", xs)
print("--------------------")

# Create symmetric matrix
mat = xs + xs.T
print("mat = ", mat)
print("--------------------")
# Compute two eigenpairs selecting the eigenvalues with
# largest magnitude (default).


new_xs = sp.csc_matrix(xs)
print("new_xs = ", xs)
print("--------------------")

# selection_rule = ["LargestMagn",
#                   "LargestReal",
#                   "LargestImag",
#                   "LargestAlge",
#                   "SmallestReal",
#                   "SmallestMagn",
#                   "SmallestImag",
#                   "SmallestAlge",
#                   "BothEnds"]

selection_rule = ["LargestMagn",
                  "LargestReal",
                  "LargestImag",
                  "SmallestReal",
                  ]

# which_rule = ["LM", "LR", "LI", "SR", "SM", "SI"]
which_rule = ["LM", "LR", "LI", "SR"]


for i in range(4):
    print(f"\ntest_rule{i}:{which_rule[i]}")
    # 1 eigensolver

    # 1.1
    eigenvalues, eigenvectors = eigensolver(new_xs, nvalues, selection_rule[i])
    
    print()    
    print("test_noshift:--------------------------------")
    print(eigenvalues)
    # print("--------------------")
    # print(eigenvectors)
    # print("--------------------")

    print("scipy_noshift:ans--------------------------------")
    ans_vals, ans_vecs = eigs(new_xs, nvalues, which=which_rule[i], ncv=6)
    print(ans_vals)
    # print("--------------------")
    # print(ans_vecs)
    # print("--------------------")

    # 1.2

    eigenvalues, eigenvectors = eigensolver(
        new_xs, nvalues, selection_rule[i], shift=SIGMA.real)

    print()

    print("test_shift_real--------------------------------")    
    print(eigenvalues)
    # print("--------------------")
    # print(eigenvectors)
    # print("--------------------")

    print("scipy_shift_real:ans--------------------------------")
    ans_vals, ans_vecs = eigs(new_xs, nvalues, which=which_rule[i], sigma=SIGMA.real)
    print(ans_vals)
    # print("--------------------")
    # print(ans_vecs)
    # print("--------------------")

    # 1.3

    eigenvalues, eigenvectors = eigensolver(
        new_xs, nvalues, selection_rule[i], shift=SIGMA)
    
    print()
    print("test_shift--------------------------------")
    print(eigenvalues)
    # print("--------------------")
    # print(eigenvectors)
    # print("--------------------")

    print("scipy_shift:ans--------------------------------")
    ans_vals, ans_vecs = eigs(new_xs, nvalues, which=which_rule[i], sigma=SIGMA)
    print(ans_vals)
    # print("--------------------")
    # print(ans_vecs)
    # print("--------------------")
```

**All functions return a tuple whith the resulting eigenvalues and eigenvectors.**


## Installation
To install pyspectra, do:
```bash
  git clone git@gitee.com:PerfXLab/spectra4py.git
  cd pyspectra
  bash ./install.sh
```

Run tests (including coverage) with:

```bash
  pytest tests
```

