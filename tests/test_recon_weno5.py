"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Tests for WENO5 reconstruction methods
"""
import math
from pyrecon.recon_weno5 import (
    weno5_fv, weno5z_fv, weno5d_si_fv,
    weno5_pw, weno5z_pw, weno5d_si_pw,
    _js_smoothness, _stencils_fv,
)


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

def test_js_smoothness_constant():
    b0, b1, b2 = _js_smoothness(1.0, 1.0, 1.0, 1.0, 1.0)
    assert b0 == 0.0
    assert b1 == 0.0
    assert b2 == 0.0


def test_js_smoothness_linear():
    b0, b1, b2 = _js_smoothness(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert abs(b0 - 1.0) < 1e-14
    assert abs(b1 - 1.0) < 1e-14
    assert abs(b2 - 1.0) < 1e-14


def test_js_smoothness_quadratic():
    b0, b1, b2 = _js_smoothness(4.0, 1.0, 0.0, 1.0, 4.0)
    assert abs(b0 - 13.0/3.0) < 1e-14
    assert abs(b1 - 13.0/3.0) < 1e-14
    assert abs(b2 - 13.0/3.0) < 1e-14


def test_stencils_linear():
    u0, u1, u2 = _stencils_fv(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert abs(u0 - 0.5) < 1e-14
    assert abs(u1 - 0.5) < 1e-14
    assert abs(u2 - 0.5) < 1e-14


# ---------------------------------------------------------------------------
# WENO5-JS
# ---------------------------------------------------------------------------

def test_weno5_constant():
    uL, uR = weno5_fv(5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert abs(uR - 5.0) < 1e-14


def test_weno5_linear():
    uL, uR = weno5_fv(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert abs(uL - 0.5) < 1e-14
    assert abs(uR - (-0.5)) < 1e-14


def test_weno5_quadratic():
    uL, uR = weno5_fv(4.0, 1.0, 0.0, 1.0, 4.0)
    assert abs(uL - 1.0 / 6.0) < 1e-14
    assert abs(uR - 1.0 / 6.0) < 1e-14


def test_weno5_jump_left_stencil():
    """Jump at i=0 in [0,0,1,1,1]: uL~1 (smooth region), uR~1."""
    uL, uR = weno5_fv(0.0, 0.0, 1.0, 1.0, 1.0)
    # Face i+1/2 is between value-1 cells -> uL approx 1
    assert abs(uL - 1.0) < 0.05
    # Reversed stencil [1,1,1,0,0], face i-1/2 from right -> near 1
    assert abs(uR - 1.0) < 0.05


def test_weno5_jump_right_stencil():
    """Jump at i+1 in [1,1,1,0,0]: both faces finite and well-behaved."""
    uL, uR = weno5_fv(1.0, 1.0, 1.0, 0.0, 0.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    # Values should be within the data range [0, 1]
    assert 0.0 <= uL <= 1.0
    assert 0.0 <= uR <= 1.0


# ---------------------------------------------------------------------------
# WENO5-Z
# ---------------------------------------------------------------------------

def test_weno5z_constant():
    uL, uR = weno5z_fv(5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert abs(uR - 5.0) < 1e-14


def test_weno5z_linear():
    uL, uR = weno5z_fv(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert abs(uL - 0.5) < 1e-14
    assert abs(uR - (-0.5)) < 1e-14


def test_weno5z_jump():
    uL, uR = weno5z_fv(0.0, 0.0, 1.0, 1.0, 1.0)
    assert abs(uL - 1.0) < 0.05
    assert abs(uR - 1.0) < 0.05


def test_weno5z_critical_point():
    uL, uR = weno5z_fv(-8.0, -1.0, 0.0, 1.0, 8.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)


# ---------------------------------------------------------------------------
# WENO5-D-SI
# ---------------------------------------------------------------------------

def test_weno5d_si_constant():
    uL, uR = weno5d_si_fv(5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert abs(uR - 5.0) < 1e-14


def test_weno5d_si_linear():
    uL, uR = weno5d_si_fv(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert abs(uL - 0.5) < 1e-14
    assert abs(uR - (-0.5)) < 1e-14


def test_weno5d_si_jump():
    uL, uR = weno5d_si_fv(0.0, 0.0, 1.0, 1.0, 1.0)
    assert abs(uL - 1.0) < 0.1
    assert abs(uR - 1.0) < 0.1


def test_weno5d_si_scale_invariance():
    uL1, uR1 = weno5d_si_fv(-2.0, -1.0, 0.0, 1.0, 2.0)
    uL2, uR2 = weno5d_si_fv(998.0, 999.0, 1000.0, 1001.0, 1002.0)
    assert abs((uL2 - 1000.0) - uL1) < 1e-12
    assert abs((uR2 - 1000.0) - uR1) < 1e-12

# ---------------------------------------------------------------------------
# WENO5-JS (PW)
# ---------------------------------------------------------------------------

def test_weno5_pw_constant():
    uL, uR = weno5_pw(5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert abs(uR - 5.0) < 1e-14


def test_weno5_pw_linear():
    uL, uR = weno5_pw(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert abs(uL - 0.5) < 1e-14
    assert abs(uR - (-0.5)) < 1e-14


def test_weno5_pw_finite():
    uL, uR = weno5_pw(0.0, 0.0, 1.0, 1.0, 1.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)

# ---------------------------------------------------------------------------
# WENO5-Z (PW)
# ---------------------------------------------------------------------------

def test_weno5z_pw_constant():
    uL, uR = weno5z_pw(5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert abs(uR - 5.0) < 1e-14


def test_weno5z_pw_linear():
    uL, uR = weno5z_pw(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert abs(uL - 0.5) < 1e-14
    assert abs(uR - (-0.5)) < 1e-14


def test_weno5z_pw_finite():
    uL, uR = weno5z_pw(0.0, 0.0, 1.0, 1.0, 1.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)

# ---------------------------------------------------------------------------
# WENO5-D-SI (PW)
# ---------------------------------------------------------------------------

def test_weno5d_si_pw_constant():
    uL, uR = weno5d_si_pw(5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert abs(uR - 5.0) < 1e-14


def test_weno5d_si_pw_linear():
    uL, uR = weno5d_si_pw(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert abs(uL - 0.5) < 1e-14
    assert abs(uR - (-0.5)) < 1e-14


def test_weno5d_si_pw_finite():
    uL, uR = weno5d_si_pw(0.0, 0.0, 1.0, 1.0, 1.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
#
# :D
#
