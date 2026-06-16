"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Tests for WENO-AO(5,3) reconstruction
  Balsara, Garain & Shu, J. Comput. Phys. 326, 780-804 (2016)
"""
import math
from pyrecon.recon_weno_ao53 import (
    weno_ao53_fv, weno_ao53_pw, _js_smoothness, _smoothness_r5_central)

# ---------------------------------------------------------------------------
# Smoothness indicator tests
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


def test_smoothness_r5_central_constant():
    """Constant data: all Legendre moments zero -> beta_r5 = 0."""
    b = _smoothness_r5_central(1.0, 1.0, 1.0, 1.0, 1.0)
    assert abs(b) < 1e-13, f"beta_r5={b} for constant data, expected 0"


def test_smoothness_r5_central_linear():
    """Linear data: beta_r5 should equal JS sub-stencil indicators (=1.0)."""
    b = _smoothness_r5_central(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert abs(b - 1.0) < 1e-13, (
        f"beta_r5={b} for linear data, expected 1.0")


# ---------------------------------------------------------------------------
# Basic reconstruction tests
# ---------------------------------------------------------------------------

def test_weno_ao53_constant():
    uL, uR = weno_ao53_fv(5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14, f"uL={uL}"
    assert abs(uR - 5.0) < 1e-14, f"uR={uR}"


def test_weno_ao53_linear():
    uL, uR = weno_ao53_fv(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert abs(uL - 0.5) < 1e-14, f"uL={uL}"
    assert abs(uR - (-0.5)) < 1e-14, f"uR={uR}"


def test_weno_ao53_quadratic():
    uL, uR = weno_ao53_fv(4.0, 1.0, 0.0, 1.0, 4.0)
    assert abs(uL) < 4.0
    assert abs(uR) < 4.0


def test_weno_ao53_jump():
    uL, uR = weno_ao53_fv(0.0, 0.0, 1.0, 1.0, 1.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)


def test_weno_ao53_shock():
    uL, uR = weno_ao53_fv(1.0, 1.0, 1.0, 0.0, 0.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)


def test_weno_ao53_cubic():
    uL, uR = weno_ao53_fv(-8.0, -1.0, 0.0, 1.0, 8.0)
    assert abs(uL) < 8.0
    assert abs(uR) < 8.0


# ---------------------------------------------------------------------------
# Smooth-limit: Eq (3.9) guarantees P_AO -> P5 when all smoothness
# indicators are equal. Linear data is the only case at h=1 where all
# four smoothness indicators (JS sub-stencil + Legendre central) are
# exactly equal. This verifies the Zhu-Qiu combination formula.
# ---------------------------------------------------------------------------

def test_weno_ao53_smooth_limit_linear():
    """Linear: all smoothness = 1.0 -> tau=0 -> w_k=gamma_k -> P_AO=P5."""
    uL, uR = weno_ao53_fv(-2.0, -1.0, 0.0, 1.0, 2.0)
    # P5(1/2) for u(x)=x is 0.5
    assert abs(uL - 0.5) < 1e-13
    assert abs(uR - (-0.5)) < 1e-13


def test_weno_ao53_smooth_limit_quadratic():
    """Quadratic with FV cell averages of u(x)=x^2 at integer points.
    P5 and all sub-stencils give u_{1/2} = 1/6 for these values.
    """
    uL, uR = weno_ao53_fv(4.0, 1.0, 0.0, 1.0, 4.0)
    assert abs(uL - 1.0/6.0) < 1e-12
    assert abs(uR - 1.0/6.0) < 1e-12


# ---------------------------------------------------------------------------
# Symmetry
# ---------------------------------------------------------------------------

def test_weno_ao53_symmetric():
    """Symmetric data: both faces should give same value."""
    uL, uR = weno_ao53_fv(1.0, 2.0, 3.0, 2.0, 1.0)
    # For symmetric data, the central polynomial gives same value at
    # both faces since the stencil reversal produces the mirror value.
    assert abs(uL - uR) < 1e-13, (
        f"uL={uL}, uR={uR} should be equal for symmetric data")

# ---------------------------------------------------------------------------
# WENO-AO(5,3) (PW)
# ---------------------------------------------------------------------------

def test_weno_ao53_pw_constant():
    uL, uR = weno_ao53_pw(5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14, f"uL={uL}"
    assert abs(uR - 5.0) < 1e-14, f"uR={uR}"


def test_weno_ao53_pw_linear():
    uL, uR = weno_ao53_pw(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert abs(uL - 0.5) < 1e-14, f"uL={uL}"
    assert abs(uR - (-0.5)) < 1e-14, f"uR={uR}"


def test_weno_ao53_pw_finite():
    uL, uR = weno_ao53_pw(0.0, 0.0, 1.0, 1.0, 1.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
#
# :D
#
