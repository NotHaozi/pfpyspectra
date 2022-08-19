"""Helper functions to tests."""

from typing import Union
import numpy as np
import scipy.sparse as sp


def norm(vs: np.array) -> float:
    """Compute the norm of a vector."""
    return np.sqrt(np.dot(vs, vs))


def create_random_matrix(size: int) -> np.array:
    """Create a numpy random matrix."""
    return np.random.normal(size=size ** 2).reshape(size, size)


def create_random_spmatrix(size: int) -> sp.spmatrix:
    """Create a scipy random spmatrix."""
    return sp.rand(size, size, format='csc')


def create_symmetic_matrix(size: int) -> np.array:
    """Create a numpy symmetric matrix."""
    xs = create_random_matrix(size)
    return xs + xs.T


def create_sparse_symmetic_matrix(size: int) -> sp.spmatrix:
    """Create a sparse symmetric matrix."""
    xs = create_random_spmatrix(size)
    return xs + xs.T


def check_eigenpairs(
        matrix: Union[np.ndarray, sp.spmatrix], eigenvalues: np.ndarray,
        eigenvectors: np.ndarray) -> bool:
    """Check that the eigenvalue equation holds."""
    for i, value in enumerate(eigenvalues):
        # 理论上 AX = lambda X
        # 是否为稀疏矩阵
        if sp.issparse(matrix):
            dense_mat = matrix.toarray()
        else:
            dense_mat = matrix
        residue = np.dot(
            dense_mat, eigenvectors[:, i]) - value * eigenvectors[:, i]
        assert norm(residue) < 1e-8
