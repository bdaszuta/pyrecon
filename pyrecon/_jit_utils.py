"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Numba JIT convenience partials (pynalgo convention)
"""
from functools import partial
from typing import TYPE_CHECKING  # noqa: F401

from numba import jit  # type: ignore[import-untyped]

_SETTINGS_JIT = {
    "nopython": True,
    "nogil": True,
    "cache": True,
    "inline": "never",
    "error_model": "numpy",
    "boundscheck": False,
}

_SETTINGS_JITI = {**_SETTINGS_JIT, "inline": "always"}

JIT = partial(jit, **_SETTINGS_JIT)
JIT.__doc__ = (
    "Numba jit partial: nopython=True, nogil=True, cache=True, "
    "inline='never', error_model='numpy', boundscheck=False."
)
JITI = partial(jit, **_SETTINGS_JITI)
JITI.__doc__ = (
    "Numba jit partial: nopython=True, nogil=True, cache=True, "
    "inline='always', error_model='numpy', boundscheck=False."
)

#
# :D
#
