"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Tests for WENO5-Z p=2 reconstruction
"""
import math
from pyrecon.recon_esweno5 import weno5z_p2_fv as weno5


def test_weno5_constant():
    """Constant data: ES-WENO5 returns exact value."""
    uL, uR = weno5(5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert abs(uR - 5.0) < 1e-14


def test_weno5_linear():
    """Linear data: ES-WENO5 should be exact (5th order on linear data)."""
    uL, uR = weno5(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert abs(uL - 0.5) < 1e-14
    assert abs(uR - (-0.5)) < 1e-14


def test_weno5_jump():
    """Discontinuity: ES-WENO5 suppresses oscillations."""
    uL, uR = weno5(0.0, 0.0, 0.0, 1.0, 1.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert -1e-14 <= uL <= 1.0 + 1e-14
    assert -1e-14 <= uR <= 1.0 + 1e-14


def test_weno5_jump_reversed():
    """Reversed discontinuity."""
    uL, uR = weno5(1.0, 1.0, 1.0, 0.0, 0.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert 0.0 <= uL <= 1.0
    assert 0.0 <= uR <= 1.0


def test_weno5_symmetric():
    """ES-WENO5 should be anti-symmetric for mirrored data."""
    uL1, uR1 = weno5(-2.0, -1.0, 0.0, 1.0, 2.0)
    uL2, uR2 = weno5(2.0, 1.0, 0.0, -1.0, -2.0)
    assert abs(uL1 + uL2) < 1e-14
    assert abs(uR1 + uR2) < 1e-14


def test_weno5_vs_weno5z_on_linear():
    """ES-WENO5 (p=2) and WENO5-Z (p=1) agree on linear data."""
    from pyrecon.recon_weno5 import weno5z_fv as weno5z
    uL_es, uR_es = weno5(-2.0, -1.0, 0.0, 1.0, 2.0)
    uL_z, uR_z = weno5z(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert abs(uL_es - 0.5) < 1e-14
    assert abs(uL_z - 0.5) < 1e-14
    assert abs(uL_es - uL_z) < 1e-14
#
# :D
#
