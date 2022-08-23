# pfPySpectra

pfpyspectra based on [pyspectra](https://github.com/NLESC-JCER/pyspectra.git), Python interface to the [C++ Spectra library](https://github.com/yixuan/spectra)

## Eigensolvers
**pfPySpecta** offers two general interfaces to [Spectra](https://github.com/yixuan/spectra): **eigensolver** and **eigensolverh**. For general(dense&sparse) and symmetric(dense&sparse) matrices respectively.These two functions would invoke the most suitable method based on the information provided by the user.

## Usage
```python
import numpy as np
import scipy.sparse as sp
from pfpyspectra import eigensolver, eigensolverh

# matrix size
size = 100

# number of eigenpairs to compute
nvalues = 2

# Create random matrix
xs = np.random.normal(size=size ** 2).reshape(size, size)
new_xs=sp.rand(size, size, density=0.1, format='csc')

# Create symmetric matrix
mat = xs + xs.T
new_mat = new_xs + new_xs.T

# Compute two eigenpairs selecting the eigenvalues with
# largest magnitude (default).
eigenvalues, eigenvectors = eigensolver(xs, nvalues)
sprse_eigenvalues, sprse_eigenvectors = eigensolver(new_xs, nvalues)
# Compute two eigenpairs selecting the eigenvalues with
# largest algebraic value
selection_rule = "LargestAlge"
symm_eigenvalues, symm_eigenvectors = eigensolverh(
  mat, nvalues, selection_rule)
sprse_symm_eigenvalues, sprse_symm_eigenvectors = eigensolverh(
  mat, nvalues, selection_rule)
```

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

## Eigensolvers Dense Interface
You can also call directly the dense interface. You would need to import the following module:
```python
import numpy as np
from pfpyspectra import spectra_dense_interface
```
The following functions are available in the spectra_dense_interface:
*  ```py
   general_eigensolver(
    mat: np.ndarray, eigenpairs: int, basis_size: int, selection_rule: str)
    -> (np.ndarray, np.ndarray)
   ```
*  ```py
   general_real_shift_eigensolver(
   mat: np.ndarray, eigenpairs: int, basis_size: int, shift: float, selection_rule: str)
   -> (np.ndarray, np.ndarray)
   ```
*  ```py
   general_complex_shift_eigensolver(
     mat: np.ndarray, eigenpairs: int, basis_size: int,
     shift_real: float, shift_imag: float, selection_rule: str)
     -> (np.ndarray, np.ndarray)
   ```
*  ```py
   symmetric_eigensolver(
     mat: np.ndarray, eigenpairs: int, basis_size: int, selection_rule: str)
     -> (np.ndarray, np.ndarray)
   ```
*  ```py
   symmetric_shift_eigensolver(
     mat: np.ndarray, eigenpairs: int, basis_size: int, shift: float, selection_rule: str)
     -> (np.ndarray, np.ndarray)
   ```
*  ```py
   symmetric_generalized_shift_eigensolver(
     mat_A: np.ndarray, mat_B: np.ndarray, eigenpairs: int, basis_size: int, shift: float,
     selection_rule: str)
     -> (np.ndarray, np.ndarray)
   ```

## Eigensolvers Sparse Interface
You can also call directly the sparse interface. You would need to import the following module:
```python
import scipy as sp
from pfpyspectra import spectra_sparse_interface
```
The following functions are available in the spectra_sparse_interface:
*  ```py
   sparse_general_eigensolver(
    mat: sp.spmatrix, eigenpairs: int, basis_size: int, selection_rule: str)
    -> (np.ndarray, np.ndarray)
   ```
*  ```py
   sparse_general_real_shift_eigensolver(
   mat: sp.spmatrix, eigenpairs: int, basis_size: int, shift: float, selection_rule: str)
   -> (np.ndarray, np.ndarray)
   ```
*  ```py
   sparse_general_complex_shift_eigensolver(
     mat: sp.spmatrix, eigenpairs: int, basis_size: int,
     shift_real: float, shift_imag: float, selection_rule: str)
     -> (np.ndarray, np.ndarray)
   ```
*  ```py
   sparse_symmetric_eigensolver(
     mat: sp.spmatrix, eigenpairs: int, basis_size: int, selection_rule: str)
     -> (np.ndarray, np.ndarray)
   ```
*  ```py
   sparse_symmetric_shift_eigensolver(
     mat: sp.spmatrix, eigenpairs: int, basis_size: int, shift: float, selection_rule: str)
     -> (np.ndarray, np.ndarray)
   ```
*  ```py
   sparse_symmetric_generalized_shift_eigensolver(
     mat_A: sp.spmatrix, mat_B: sp.spmatrix, eigenpairs: int, basis_size: int, shift: float,
     selection_rule: str)
     -> (np.ndarray, np.ndarray)
   ```
## Example
```python
import numpy as np
from pfpyspectra import spectra_dense_interface

size = 100
nvalues = 2 # eigenpairs to compute
search_space = nvalues * 2 # size of the search space
shift = 1.0

# Create random matrix
xs = np.random.normal(size=size ** 2).reshape(size, size)

# Create symmetric matrix
mat = xs + xs.T

# Compute two eigenpairs selecting the eigenvalues with
# largest algebraic value
selection_rule = "LargestAlge"
symm_eigenvalues, symm_eigenvectors = \
  spectra_dense_interface.symmetric_eigensolver(
  mat, nvalues, search_space, selection_rule)

```
> Note: **All functions return a tuple whith the resulting eigenvalues and eigenvectors.**
> For more examples, please see the directory: `pfpyspectra/tests/`


## Installation
To install pyspectra, do:
```bash
  git clone git@gitee.com:PerfXLab/spectra4py.git
  cd pyspectra
  bash ./install.sh
```
## Test
Run tests (including coverage) with:

```bash
  pytest tests/test_dense_pyspectra.py
  pytest tests/test_sparse_pyspectra.py
  pytest tests/test_pyspectra.py
  # also you can just `pytest tests`
```
> **Help:** If you don't pass them all, don't worry, try a few more times.  
> I think that's because of the random parameter problem, It will not affect the use, can you help me?

## License
No. Just for fun!  
Thanks :  
  [pyspectra](https://github.com/NLESC-JCER/pyspectra.git),  
  [C++ Spectra library](https://github.com/yixuan/spectra)