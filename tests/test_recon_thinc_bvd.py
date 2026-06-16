"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Tests for THINC-BVD reconstruction method
"""
import math
from pyrecon.recon_thinc_bvd import (
    thinc_bvd_fv as thinc_bvd, thinc_bvd_pw,
    _js_smoothness,
)
from pyrecon.utils import thinc_value_L, thinc_value_R


# ---------------------------------------------------------------------------
# THINC face values
# ---------------------------------------------------------------------------

def test_thinc_value_L_constant():
    """Constant data: THINC returns center value."""
    assert thinc_value_L(1.0, 1.0, 1.0) == 1.0


def test_thinc_value_R_constant():
    assert thinc_value_R(2.0, 2.0, 2.0) == 2.0


def test_thinc_value_L_monotonic_increasing():
    """Increasing data: u_im1 < u_i < u_ip1 -> gamma=1, tanh(beta*0.5) > 0."""
    val = thinc_value_L(0.0, 0.5, 1.0)
    # Should be above midpoint due to sharpening
    assert 0.5 <= val <= 1.0


def test_thinc_value_R_monotonic_increasing():
    """Increasing data: u_im1 < u_i < u_ip1 -> tanh(-beta*0.5) for R face."""
    val = thinc_value_R(0.0, 0.5, 1.0)
    assert 0.0 <= val <= 0.5


def test_thinc_value_L_monotonic_decreasing():
    """Decreasing data: gamma=-1."""
    val = thinc_value_L(1.0, 0.5, 0.0)
    assert 0.0 <= val <= 0.5


def test_thinc_value_R_monotonic_decreasing():
    val = thinc_value_R(1.0, 0.5, 0.0)
    assert 0.5 <= val <= 1.0


def test_thinc_value_LR_symmetry():
    """THINC face values obey reflection symmetry for mirrored stencils.

    thinc_value_L(0, 0.5, 1) should equal thinc_value_R(1, 0.5, 0)
    since both represent the "high side" face for their respective
    monotonic profiles. Similarly the "low side" faces match.
    """
    vL_inc = thinc_value_L(0.0, 0.5, 1.0)
    vR_inc = thinc_value_R(0.0, 0.5, 1.0)
    vL_dec = thinc_value_L(1.0, 0.5, 0.0)
    vR_dec = thinc_value_R(1.0, 0.5, 0.0)

    # High sides match
    assert abs(vL_inc - vR_dec) < 1e-14
    # Low sides match
    assert abs(vR_inc - vL_dec) < 1e-14
    # High + low = u_min + u_max = 1.0
    assert abs(vL_inc + vR_inc - 1.0) < 1e-14
    assert abs(vL_dec + vR_dec - 1.0) < 1e-14


# ---------------------------------------------------------------------------
# JS smoothness indicators (13/12 version used in THINC-BVD)
# ---------------------------------------------------------------------------

def test_js_smoothness_constant():
    b0, b1, b2 = _js_smoothness(1.0, 1.0, 1.0, 1.0, 1.0)
    assert b0 == 0.0
    assert b1 == 0.0
    assert b2 == 0.0


def test_js_smoothness_linear():
    b0, b1, b2 = _js_smoothness(-2.0, -1.0, 0.0, 1.0, 2.0)
    # For linear data: b0 = 13/12*0 + 1/4*(-2 - 4*(-1) + 3*0)^2 = 1/4*2^2 = 1
    assert abs(b0 - 1.0) < 1e-14
    assert abs(b1 - 1.0) < 1e-14
    assert abs(b2 - 1.0) < 1e-14


# ---------------------------------------------------------------------------
# THINC-BVD (FV)
# ---------------------------------------------------------------------------

def test_thinc_bvd_constant():
    uL, uR = thinc_bvd(5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert abs(uR - 5.0) < 1e-14


def test_thinc_bvd_linear():
    """Linear data: ratio ~1 -> WENO5z path, exact reconstruction."""
    uL, uR = thinc_bvd(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert abs(uL - 0.5) < 1e-14
    assert abs(uR - (-0.5)) < 1e-14


def test_thinc_bvd_smooth_ratio():
    """Smooth data should use WENO5z (similar to weno5z)."""
    uL, uR = thinc_bvd(0.0, 0.5, 1.0, 1.5, 2.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    # Values should be in reasonable range
    assert 0.5 <= uL <= 2.0
    assert 0.5 <= uR <= 2.0


def test_thinc_bvd_sharp_jump():
    """Large ratio -> THINC path."""
    uL, uR = thinc_bvd(0.0, 0.0, 0.0, 1.0, 1.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)


def test_thinc_bvd_symmetric_jump():
    """Symmetric step should have consistent values."""
    uL, uR = thinc_bvd(1.0, 1.0, 0.0, 0.0, 0.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)


# ---------------------------------------------------------------------------
# THINC-BVD (PW)
# ---------------------------------------------------------------------------

def test_thinc_bvd_pw_constant():
    uL, uR = thinc_bvd_pw(5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert abs(uR - 5.0) < 1e-14


def test_thinc_bvd_pw_linear():
    uL, uR = thinc_bvd_pw(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert abs(uL - 0.5) < 1e-14
    assert abs(uR - (-0.5)) < 1e-14


def test_thinc_bvd_pw_jump():
    uL, uR = thinc_bvd_pw(0.0, 0.0, 1.0, 1.0, 1.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
#
# :D
#
