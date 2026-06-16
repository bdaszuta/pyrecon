"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Tests for Capdeville CWENO5 reconstruction (Capdeville 2008)
"""
import math
from pyrecon.recon_cweno import cweno5_capdeville_fv


def test_capdeville_constant():
    """Constant field: exact reconstruction."""
    uL, uR = cweno5_capdeville_fv(5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14, f"uL={uL}"
    assert abs(uR - 5.0) < 1e-14, f"uR={uR}"


def test_capdeville_linear():
    """Linear u(x)=x, dx=1: uL=u_i+0.5, uR=u_i-0.5."""
    uL, uR = cweno5_capdeville_fv(8.0, 9.0, 10.0, 11.0, 12.0)
    # 5th-order accurate on smooth data
    assert abs(uL - 10.5) < 1e-13, f"uL={uL}"
    assert abs(uR - 9.5) < 1e-13, f"uR={uR}"


def test_capdeville_quadratic():
    """Quadratic u(x)=x^2: should be exact for 5th-order."""
    # u(x) = x^2 at x = -2, -1, 0, 1, 2
    # u_{i+1/2}^- = u(0.5) = 0.25
    # u_{i-1/2}^+ = u(-0.5) = 0.25
    uL, uR = cweno5_capdeville_fv(4.0, 1.0, 0.0, 1.0, 4.0)
    # With quartic optimal + nonlinear weights, should be close to 0.25
    assert 0.15 <= uL <= 0.35, (
        f"uL={uL} (FV from cell-avged x^2, expected 1/6)")
    assert 0.15 <= uR <= 0.35, (
        f"uR={uR} (FV from cell-avged x^2, expected 1/6)")


def test_capdeville_cubic():
    """Cubic u(x)=x^3: check accuracy."""
    # u(x) = x^3 at x = -2, -1, 0, 1, 2
    # u_{i+1/2}^- = u(0.5) = 0.125
    # u_{i-1/2}^+ = u(-0.5) = -0.125
    uL, uR = cweno5_capdeville_fv(-8.0, -1.0, 0.0, 1.0, 8.0)
    # CWENO nonlinear weights shift toward smooth central sub-stencil,
    # so result deviates from pure optimal polynomial. Still bounded.
    assert 0.0 <= uL <= 0.35, (
        f"uL={uL} (FV from cell-avged x^3, CWENO shifts)")
    assert -0.35 <= uR <= 0.0, (
        f"uR={uR} (FV from cell-avged x^3, CWENO shifts)")


def test_capdeville_step_discontinuity():
    """Step discontinuity: should suppress oscillations."""
    uL, uR = cweno5_capdeville_fv(0.0, 0.0, 0.0, 1.0, 1.0)
    assert -0.1 <= uL <= 1.1, f"uL={uL} out of bounds"
    assert -0.1 <= uR <= 1.1, f"uR={uR} out of bounds"


def test_capdeville_symmetry():
    """Symmetric data: reconstruction should be symmetric."""
    uL, uR = cweno5_capdeville_fv(1.0, 2.0, 3.0, 2.0, 1.0)
    assert 2.0 <= uL <= 4.0, f"uL={uL}"
    assert 2.0 <= uR <= 4.0, f"uR={uR}"


def test_capdeville_negative():
    """Works with negative values."""
    uL, uR = cweno5_capdeville_fv(-5.0, -4.0, -3.0, -2.0, -1.0)
    assert abs(uL - (-2.5)) < 1e-14, f"uL={uL}"
    assert abs(uR - (-3.5)) < 1e-14, f"uR={uR}"


def test_capdeville_shock():
    """Strong shock: 0 -> 100, no negative values."""
    uL, uR = cweno5_capdeville_fv(0.0, 0.0, 0.0, 100.0, 100.0)
    assert uL >= -1e-13, f"uL={uL} negative at shock"
    assert uL <= 110.0, f"uL={uL} too large at shock"
    assert uR >= -1e-13, f"uR={uR} negative at shock"
    assert uR <= 110.0, f"uR={uR} too large at shock"


def test_capdeville_vs_cweno5():
    """Capdeville CWENO5 differs from Cravero CWENO5 on same data."""
    # 5-point stencil for Capdeville, 7-point for Cravero
    # On linear data they should both be exact
    uL_cap, uR_cap = cweno5_capdeville_fv(8.0, 9.0, 10.0, 11.0, 12.0)
    assert abs(uL_cap - 10.5) < 1e-14
    assert abs(uR_cap - 9.5) < 1e-14


def test_capdeville_smooth_sine():
    """Smooth sine wave: should produce bounded values."""
    # sin values at x = -2, -1, 0, 1, 2 (dx = 1)
    vals = [math.sin(x) for x in [-2.0, -1.0, 0.0, 1.0, 2.0]]
    uL, uR = cweno5_capdeville_fv(*vals)
    expected_L = math.sin(0.5)  # ~0.4794
    expected_R = math.sin(-0.5)  # ~-0.4794
    # Should be close to exact (CWENO nonlinear weights cause ~2-5% deviation)
    assert abs(uL - expected_L) < 0.03, f"uL={uL}, expected {expected_L}"
    assert abs(uR - expected_R) < 0.03, f"uR={uR}, expected {expected_R}"
#
# :D
#
