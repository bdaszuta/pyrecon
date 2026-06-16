"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Tests for WENO5-Z-SP sign-preserving reconstruction
"""
import math
from pyrecon.recon_weno5 import weno5z_fv
from pyrecon.recon_weno5z_sp import weno5z_sp_fv, weno5z_sp_pw


# ===================================================================
# Standard reconstruction tests
# ===================================================================

def test_sp_constant():
    uL, uR = weno5z_sp_fv(5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert abs(uR - 5.0) < 1e-14

    uL, uR = weno5z_sp_fv(-3.0, -3.0, -3.0, -3.0, -3.0)
    assert abs(uL - (-3.0)) < 1e-14
    assert abs(uR - (-3.0)) < 1e-14


def test_sp_linear():
    uL, uR = weno5z_sp_fv(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert abs(uL - 0.5) < 1e-14
    assert abs(uR - (-0.5)) < 1e-14


def test_sp_quadratic():
    uL, uR = weno5z_sp_fv(4.0, 1.0, 0.0, 1.0, 4.0)
    assert math.isfinite(uL) and math.isfinite(uR)
    assert 0.0 <= uL <= 4.0
    assert 0.0 <= uR <= 4.0


def test_sp_jump():
    uL, uR = weno5z_sp_fv(0.0, 0.0, 0.0, 1.0, 1.0)
    assert math.isfinite(uL) and math.isfinite(uR)
    assert uL > -1e-12 and uR > -1e-12


# ===================================================================
# Sign-preservation tests
# ===================================================================

def test_sp_positive_preservation():
    """All-positive stencil where weno5z undershoots negative."""
    u = (0.001, 0.01, 0.001, 0.001, 0.01)
    uL_z, uR_z = weno5z_fv(*u)
    uL_sp, uR_sp = weno5z_sp_fv(*u)

    assert uL_z < 0.0, f"weno5z uL should be negative, got {uL_z}"
    assert uL_sp > 0.0, f"weno5z-sp uL should be positive, got {uL_sp}"
    assert uR_sp == uR_z, "uR unchanged when it respects sign"


def test_sp_negative_preservation():
    """All-negative stencil where weno5z overshoots positive."""
    u = (-10.0, -10.0, -1.0, -1.0, -10.0)
    uL_z, uR_z = weno5z_fv(*u)
    uL_sp, uR_sp = weno5z_sp_fv(*u)

    assert uL_z > 0.0, f"weno5z uL should be positive, got {uL_z}"
    assert uL_sp < 0.0, f"weno5z-sp uL should be negative, got {uL_sp}"
    assert uR_sp == uR_z, "uR unchanged when it respects sign"


def test_sp_noop_mixed_sign():
    """Mixed-sign stencil: SP should not alter weno5z output."""
    u = (-1.0, -0.5, 0.0, 0.5, 1.0)
    uL_z, uR_z = weno5z_fv(*u)
    uL_sp, uR_sp = weno5z_sp_fv(*u)

    assert uL_sp == uL_z, (
        f"SP should not alter uL when signs mixed: {uL_sp} != {uL_z}")
    assert uR_sp == uR_z, (
        f"SP should not alter uR when signs mixed: {uR_sp} != {uR_z}")


def test_sp_noop_already_safe():
    """All-positive stencil where weno5z is already safe."""
    u = (1.0, 1.0, 1.0, 1.0, 1.0)
    uL_z, uR_z = weno5z_fv(*u)
    uL_sp, uR_sp = weno5z_sp_fv(*u)

    assert uL_sp == 1.0, f"Constant field should return 1.0, got {uL_sp}"
    assert uR_sp == 1.0, f"Constant field should return 1.0, got {uR_sp}"


def test_sp_preserves_negativity():
    """All-negative stencil: SP preserves sign."""
    u = (-10.0, -10.0, -1.0, -1.0, -10.0)
    uL_sp, uR_sp = weno5z_sp_fv(*u)
    assert uL_sp < 0.0 and uR_sp < 0.0, "SP must preserve negativity"

# ===================================================================
# WENO5-Z-SP (PW)
# ===================================================================

def test_sp_pw_constant():
    uL, uR = weno5z_sp_pw(5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert abs(uR - 5.0) < 1e-14

    uL, uR = weno5z_sp_pw(-3.0, -3.0, -3.0, -3.0, -3.0)
    assert abs(uL - (-3.0)) < 1e-14
    assert abs(uR - (-3.0)) < 1e-14


def test_sp_pw_linear():
    uL, uR = weno5z_sp_pw(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert abs(uL - 0.5) < 1e-14
    assert abs(uR - (-0.5)) < 1e-14


def test_sp_pw_finite():
    uL, uR = weno5z_sp_pw(0.0, 0.0, 0.0, 1.0, 1.0)
    assert math.isfinite(uL) and math.isfinite(uR)
    assert uL > -1e-12 and uR > -1e-12
#
# :D
#
