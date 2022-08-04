"""pyspectra API."""
# TODO: 导入稀疏接口
import spectra_sparse_interface

from .__version__ import __version__
from .pyspectra import eigensolver, eigensolverh

__author__ = "123124124"
__email__ = '213123123@qq.com'


__all__ = ["__version__", "eigensolver",
           "eigensolverh", "spectra_sparse_interface"]
