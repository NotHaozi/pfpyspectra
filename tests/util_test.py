"""Helper functions to tests."""

import numpy as np
import scipy.sparse as sp


def norm(vs: np.array) -> float:
    """Compute the norm of a vector."""
    return np.sqrt(np.dot(vs, vs))


def create_random_matrix(size: int) -> np.array:
    """Create a numpy random matrix."""
    return np.random.normal(size=size ** 2).reshape(size, size)


def create_random_spmatrix(size: int) -> np.array:
    """Create a numpy random spmatrix."""
    mat = np.random.normal(size=size ** 2).reshape(size, size)
    return sp.csc_matrix(mat)


def create_symmetic_matrix(size: int) -> np.array:
    """Create a numpy symmetric matrix."""
    xs = create_random_matrix(size)
    return xs + xs.T


def check_eigenpairs(
        matrix: sp, eigenvalues: np.ndarray,
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
