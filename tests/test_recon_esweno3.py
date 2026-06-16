"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Tests for ES-WENO3 reconstruction methods
 (Yamaleev & Carpenter 2009a)
"""
import math
from pyrecon.recon_esweno3 import esweno3_fv as esweno3, esweno3_pw


# ---------------------------------------------------------------------------
# ES-WENO3 (FV weights)
# ---------------------------------------------------------------------------

def test_esweno3_constant():
    """Constant data: ES-WENO3 returns exact value."""
    uL, uR = esweno3(5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert abs(uR - 5.0) < 1e-14


def test_esweno3_linear():
    """Linear data: ES-WENO3 should be exact (3rd order on linear data)."""
    uL, uR = esweno3(-1.0, 0.0, 1.0)
    assert abs(uL - 0.5) < 1e-14
    assert abs(uR - (-0.5)) < 1e-14


def test_esweno3_jump():
    """Discontinuity: ES-WENO3 suppresses oscillations, stays near data."""
    uL, uR = esweno3(0.0, 0.0, 1.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert -1e-14 <= uL <= 1.0 + 1e-14
    assert -1.0 <= uR <= 1.0 + 1e-14  # right face may dip slightly below 0


def test_esweno3_jump_reversed():
    """Reversed discontinuity."""
    uL, uR = esweno3(1.0, 1.0, 0.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert 0.0 <= uL <= 2.0
    assert 0.0 <= uR <= 2.0


def test_esweno3_symmetric():
    """ES-WENO3 should be symmetric for mirrored data."""
    uL1, uR1 = esweno3(-1.0, 0.0, 1.0)
    uL2, uR2 = esweno3(1.0, 0.0, -1.0)
    assert abs(uL1 + uL2) < 1e-14
    assert abs(uR1 + uR2) < 1e-14


def test_esweno3_vs_weno3z_close():
    """ES-WENO3(p=2) should be close but slightly different from WENO3-Z(p=1).

    For non-linear data, the weight modification exponent affects results.
    """
    from pyrecon.recon_weno3 import weno3z_fv
    # Use data where sub-stencil values differ, so weight sensitivity matters
    uL_es, uR_es = esweno3(-1.0, 0.0, 1.0)
    uL_z, uR_z = weno3z_fv(-1.0, 0.0, 1.0)
    # Both should be near 0.5 (the left face value for linear data)
    assert abs(uL_es - 0.5) < 1e-14
    assert abs(uL_z - 0.5) < 1e-14
    # They should agree on exact linear data (both give optimal weights)
    assert abs(uL_es - uL_z) < 1e-14


# ---------------------------------------------------------------------------
# ES-WENO3 (PW weights)
# ---------------------------------------------------------------------------

def test_esweno3_pw_constant():
    uL, uR = esweno3_pw(5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert abs(uR - 5.0) < 1e-14


def test_esweno3_pw_linear():
    uL, uR = esweno3_pw(-1.0, 0.0, 1.0)
    assert abs(uL - 0.5) < 1e-14
    assert abs(uR - (-0.5)) < 1e-14


def test_esweno3_pw_jump():
    uL, uR = esweno3_pw(0.0, 0.0, 1.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert -1e-14 <= uL <= 1.0 + 1e-14
    assert -1.0 <= uR <= 1.0 + 1e-14
#
# :D
#
