"""pyspectra API."""
# TODO: 导入稀疏接口
import spectra_dense_interface
import spectra_sparse_interface

from .__version__ import __version__
from .pfpyspectra import eigensolver, eigensolverh

__author__ = "pf_test"
__email__ = 'pf_test_email'

__all__ = ["__version__", "eigensolver",
           "eigensolverh", "spectra_dense_interface", "spectra_sparse_interface"]
