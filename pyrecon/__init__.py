"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: pyrecon -- spatial reconstruction methods for
hyperbolic conservation laws
"""
from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("pyrecon")
except PackageNotFoundError:
    __version__ = "0.1.0"

from pyrecon.interface import reconstruct_array, list_methods, get_method

__all__ = ["reconstruct_array", "list_methods", "get_method", "__version__"]
#
# :D
#
