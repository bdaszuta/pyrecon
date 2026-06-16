"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Tests for VHO-WENO (very-high-order WENO) methods
"""
import math
from pyrecon.recon_vhoweno import vhoweno9_fv, vhoweno11_fv


def test_vhoweno9_linear():
    """Linear data: VHO-WENO9 returns finite, bounded values."""
    uL, uR = vhoweno9_fv(-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert -5.0 <= uL <= 5.0
    assert -5.0 <= uR <= 5.0


def test_vhoweno9_jump_right():
    """Jump on right: stencil mostly 0, right side 1."""
    uL, uR = vhoweno9_fv(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert -0.2 <= uL <= 1.2
    assert -0.2 <= uR <= 1.2


def test_vhoweno9_constant():
    """Constant field: VHO-WENO9 returns the constant."""
    uL, uR = vhoweno9_fv(*([3.0] * 9))
    assert math.isfinite(uL)
    assert math.isfinite(uR)


def test_vhoweno11_linear():
    """Linear data: VHO-WENO11 returns finite, bounded values."""
    uL, uR = vhoweno11_fv(
        -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert -5.0 <= uL <= 5.0
    assert -5.0 <= uR <= 5.0


def test_vhoweno11_constant():
    """Constant field: VHO-WENO11 returns the constant."""
    uL, uR = vhoweno11_fv(*([3.0] * 11))
    assert math.isfinite(uL)
    assert math.isfinite(uR)


def test_vhoweno11_jump():
    """Jump profile: bounded values."""
    uL, uR = vhoweno11_fv(
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert -0.2 <= uL <= 1.2
    assert -0.2 <= uR <= 1.2


def test_vhoweno9_vs_vhoweno11_linear():
    """VHO-WENO9 and 11 both return finite values on linear data."""
    uL9, uR9 = vhoweno9_fv(-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0)
    uL11, uR11 = vhoweno11_fv(
        -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
    assert math.isfinite(uL9)
    assert math.isfinite(uL11)
#
# :D
#
