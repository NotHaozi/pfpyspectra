# pfPySpectra

pfpyspectra based on [pyspectra](https://github.com/NLESC-JCER/pyspectra.git), Python interface to the [C++ Spectra library](https://github.com/yixuan/spectra)

## Eigensolvers
**pfPySpecta** offers two general interfaces to [Spectra](https://github.com/yixuan/spectra): **eigensolver** and **eigensolverh**. For sparse
and symmetric matrices respectively.

These two functions would invoke the most suitable method based on the information provided by the user.

**Note**:

  The [available selection_rules](https://github.com/NLESC-JCER/pyspectra/blob/master/include/Spectra/Util/SelectionRule.h) to compute a portion of the spectrum are:
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
Eigenpairs of a symmetric dense matrix using shift
```py
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs, eigsh

from pfpyspectra import eigensolver, eigensolverh

# matrix size
size = 10

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


# new_xs = sp.csc_matrix(xs)
# print("new_xs = ", xs)
# print("--------------------")

# selection_rule = "LargestMagn"
# eigenvalues, eigenvectors = eigensolver(new_xs, nvalues, selection_rule)
# print(eigenvalues)
# print("--------------------")
# print(eigenvectors)
# print("--------------------")

# print("scipy_noshift:ans--------------------------------")
# ans_vals, ans_vecs = eigs(new_xs, nvalues, which='LM', ncv=6)
# print(ans_vals)
# print("--------------------")
# print(ans_vecs)
# print("--------------------")

# print("scipy_shift_real:ans--------------------------------")
# ans_vals, ans_vecs = eigs(new_xs, nvalues, which='LM', sigma=SIGMA.real)
# print(ans_vals)
# print("--------------------")
# print(ans_vecs)
# print("--------------------")

# print("scipy_shift:ans--------------------------------")
# ans_vals, ans_vecs = eigs(new_xs, nvalues, which='LM', sigma=SIGMA)
# print(ans_vals)
# print("--------------------")
# print(ans_vecs)
# print("--------------------")

# eigenvalues, eigenvectors = eigensolver(new_xs, nvalues, selection_rule, shift=SIGMA.real)
# print(eigenvalues)
# print("--------------------")
# print(eigenvectors)
# print("--------------------")

# eigenvalues, eigenvectors = eigensolver(new_xs, nvalues, selection_rule, shift=SIGMA)
# print(eigenvalues)
# print("--------------------")
# print(eigenvectors)
# print("--------------------")

# Compute two eigenpairs selecting the eigenvalues with
# largest algebraic value
selection_rule = "LargestMagn"

print("sym:scipy_noshift:ans--------------------------------")
ans_vals, ans_vecs = eigsh(mat, nvalues, which='LM', ncv=6)
print(ans_vals)
print("--------------------")
print(ans_vecs)
print("--------------------")

print("sym:my--------------------------------")
symm_eigenvalues, symm_eigenvectors = eigensolverh(
    mat, nvalues, selection_rule)
print(symm_eigenvalues)
print("--------------------")
print(symm_eigenvectors)
print("--------------------")
```

**All functions return a tuple whith the resulting eigenvalues and eigenvectors.**


## Installation
To install pyspectra, do:
```bash
  git clone https://github.com/NotHaozi/pfpyspectra.git
  cd pyspectra
  pip install .
```

Run tests (including coverage) with:

```bash
  pytest tests
```

