"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Tests for AWENO5 reconstruction method (Wang, Don, Wang 2023)
"""
import math
from pyrecon.recon_aweno import (
    aweno5_fv as aweno5,
    _js_smoothness,
    _stencils,
    _scale_independent_sigma,
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


def test_stencils_linear():
    u0, u1, u2 = _stencils(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert abs(u0 - 0.5) < 1e-14
    assert abs(u1 - 0.5) < 1e-14
    assert abs(u2 - 0.5) < 1e-14


# ---------------------------------------------------------------------------
# Scale-independent sigma
# ---------------------------------------------------------------------------

def test_sigma_constant():
    """Constant data: sigma should be 0 (perfectly smooth)."""
    sigma = _scale_independent_sigma(1.0, 1.0, 1.0, 1.0, 1.0)
    assert sigma == 0.0


def test_sigma_linear():
    """Linear data: sigma should be 0."""
    sigma = _scale_independent_sigma(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert sigma == 0.0


def test_sigma_jump():
    """Discontinuity: sigma should be large."""
    sigma = _scale_independent_sigma(0.0, 0.0, 0.0, 1.0, 1.0)
    assert sigma > 0.1


def test_sigma_bounded():
    """sigma should be in [0, 1]."""
    sigma = _scale_independent_sigma(0.0, 0.0, 1.0, 1.0, 1.0)
    assert 0.0 <= sigma <= 1.0


# ---------------------------------------------------------------------------
# AWENO5 reconstruction
# ---------------------------------------------------------------------------

def test_aweno5_constant():
    uL, uR = aweno5(5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert abs(uR - 5.0) < 1e-14


def test_aweno5_linear():
    """Linear data: AWENO5 should be exact (sigma=0 -> linear weights)."""
    uL, uR = aweno5(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert abs(uL - 0.5) < 1e-14
    assert abs(uR - (-0.5)) < 1e-14


def test_aweno5_quadratic():
    """Quadratic data: should be close to exact."""
    uL, uR = aweno5(4.0, 1.0, 0.0, 1.0, 4.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    # For quadratic x^2 at i=0: uL(0.5) = 1/6 ~= 0.1667
    assert abs(uL - 1.0 / 6.0) < 0.05


def test_aweno5_jump():
    """Discontinuity: AWENO5 should suppress oscillations."""
    uL, uR = aweno5(0.0, 0.0, 1.0, 1.0, 1.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert 0.5 <= uL <= 1.5  # Allow some overshoot but not too much
    assert 0.5 <= uR <= 1.5


def test_aweno5_smooth_region():
    """In smooth regions, AWENO5 should be close to linear WENO5."""
    uL, uR = aweno5(0.0, 0.5, 1.0, 1.5, 2.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert 0.5 <= uL <= 2.0
    assert 0.5 <= uR <= 2.0
#
# :D
#
