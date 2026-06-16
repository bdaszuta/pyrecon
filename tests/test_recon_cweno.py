"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Tests for CWENO3, CWENO5, CWENO-Z,
    and Central WENO reconstruction methods
"""
import math
from pyrecon.recon_cweno import (
    cweno3_fv, cweno5_fv,
    cweno_z3_fv, cweno_z5_fv,
    central_weno_fv,
    _cweno3_sub_L, _cweno3_opt_L, _cweno3_smoothness,
    _cweno5_opt_L, _cweno5_sub_L, _cweno5_smoothness,
    _cw_smoothness,
)


# ===========================================================================
# CWENO3-Z helper tests (old CWENO3 helpers, still used by cweno_z3_fv)
# ===========================================================================

def test_cweno3_smoothness_constant():
    b0, b1, b2 = _cweno3_smoothness(1.0, 1.0, 1.0, 1.0, 1.0)
    assert b0 == 0.0
    assert b1 == 0.0
    assert b2 == 0.0


def test_cweno3_smoothness_linear():
    b0, b1, b2 = _cweno3_smoothness(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert abs(b0) < 1e-14
    assert abs(b1) < 1e-14
    assert abs(b2) < 1e-14


def test_cweno3_smoothness_quadratic():
    b0, b1, b2 = _cweno3_smoothness(4.0, 1.0, 0.0, 1.0, 4.0)
    assert abs(b0 - 4.0) < 1e-14
    assert abs(b1 - 4.0) < 1e-14
    assert abs(b2 - 4.0) < 1e-14


def test_cweno3_opt_linear():
    """Optimal quadratic through 5 points should be exact for linear data."""
    uL = _cweno3_opt_L(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert abs(uL - 0.5) < 1e-14


def test_cweno3_sub_linear():
    """Sub-stencil quadratics should be exact for linear data."""
    u0, u1, u2 = _cweno3_sub_L(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert abs(u0 - 0.5) < 1e-14
    assert abs(u1 - 0.5) < 1e-14
    assert abs(u2 - 0.5) < 1e-14


def test_cweno3_opt_quadratic():
    """Optimal quadratic: FV reconstruction gives 1/6 from cell-averaged x^2."""
    uL = _cweno3_opt_L(4.0, 1.0, 0.0, 1.0, 4.0)
    assert abs(uL - 1.0/6.0) < 1e-14


# ===========================================================================
# CWENO3-Z tests (old CWENO3, now cweno_z3_fv)
# ===========================================================================

def test_cweno_z3_constant():
    uL, uR = cweno_z3_fv(5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert abs(uR - 5.0) < 1e-14


def test_cweno_z3_linear():
    uL, uR = cweno_z3_fv(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert abs(uL - 0.5) < 1e-14
    assert abs(uR - (-0.5)) < 1e-14


def test_cweno_z3_linear_shifted():
    uL, uR = cweno_z3_fv(998.0, 999.0, 1000.0, 1001.0, 1002.0)
    assert abs(uL - 1000.5) < 3e-13
    assert abs(uR - 999.5) < 3e-13


def test_cweno_z3_quadratic():
    """Quadratic x^2: FV reconstruction gives 1/6 from cell-averaged data."""
    uL, uR = cweno_z3_fv(4.0, 1.0, 0.0, 1.0, 4.0)
    assert abs(uL - 1.0/6.0) < 1e-13
    assert abs(uR - 1.0/6.0) < 1e-13


def test_cweno_z3_jump():
    """Jump at i: [0,0,1,1,1]; both faces should be well-behaved."""
    uL, uR = cweno_z3_fv(0.0, 0.0, 1.0, 1.0, 1.0)
    # Face i+1/2 is between two 1's -> near 1
    assert abs(uL - 1.0) < 0.1
    # Face i-1/2 is between 0 and 1 -> should be in range
    assert 0.0 <= uR <= 1.0
    assert math.isfinite(uL)
    assert math.isfinite(uR)


def test_cweno_z3_jump_bound():
    """Values should stay within data range for sharp transitions."""
    uL, uR = cweno_z3_fv(10.0, 10.0, 10.0, 0.0, 0.0)
    assert 0.0 <= uL <= 10.0
    assert 0.0 <= uR <= 10.0
    assert math.isfinite(uL)
    assert math.isfinite(uR)


def test_cweno_z3_cubic():
    """Cubic u(x)=x^3: check finite and within range."""
    uL, uR = cweno_z3_fv(-8.0, -1.0, 0.0, 1.0, 8.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    # For cubic, 3rd order is not exact but should be reasonably close
    assert abs(uL - 0.125) < 2.5
    assert abs(uR - (-0.125)) < 2.5


def test_cweno_z3_symmetric():
    """Symmetric data: uL and uR should be symmetric."""
    uL, uR = cweno_z3_fv(-10.0, -1.0, 0.0, 1.0, 10.0)
    assert abs(uL + uR) < 1e-13


def test_cweno_z3_peak():
    """Peak at center: values stay in range."""
    uL, uR = cweno_z3_fv(1.0, 3.0, 5.0, 3.0, 1.0)
    assert 1.0 <= uL <= 5.0
    assert 1.0 <= uR <= 5.0
    assert math.isfinite(uL)
    assert math.isfinite(uR)


# ===========================================================================
# CWENO5-Z helper tests (old CWENO5 helpers)
# ===========================================================================

def test_cweno5_smoothness_constant():
    b0, b1, b2 = _cweno5_smoothness(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    assert abs(b0) < 1e-12
    assert abs(b1) < 1e-12
    assert abs(b2) < 1e-12


def test_cweno5_smoothness_linear():
    b0, b1, b2 = _cweno5_smoothness(-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
    # For linear data, b0 and b2 have nonzero 1st-derivative contribution
    assert b0 > 0
    assert b1 > 0
    assert b2 > 0


def test_cweno5_opt_linear():
    """Optimal quartic through 7 points should be exact for linear data."""
    uL = _cweno5_opt_L(-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
    assert abs(uL - 0.5) < 1e-14


def test_cweno5_opt_quadratic():
    """Optimal quartic: FV reconstruction gives 1/6 from cell-averaged x^2."""
    uL = _cweno5_opt_L(9.0, 4.0, 1.0, 0.0, 1.0, 4.0, 9.0)
    assert abs(uL - 1.0/6.0) < 1e-13


def test_cweno5_opt_cubic():
    """Optimal quartic: FV reconstruction gives ~0 from cell-averaged x^3."""
    uL = _cweno5_opt_L(-27.0, -8.0, -1.0, 0.0, 1.0, 8.0, 27.0)
    assert abs(uL) < 1e-12


def test_cweno5_opt_quartic():
    """Optimal quartic: FV reconstruction gives -1/30 from cell-averaged x^4."""
    uL = _cweno5_opt_L(81.0, 16.0, 1.0, 0.0, 1.0, 16.0, 81.0)
    assert abs(uL - (-1.0/30.0)) < 1e-12


def test_cweno5_sub_linear():
    """Sub-stencil quadratics should be exact for linear data."""
    u0, u1, u2 = _cweno5_sub_L(-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
    assert abs(u0 - 0.5) < 1e-14
    assert abs(u1 - 0.5) < 1e-14
    assert abs(u2 - 0.5) < 1e-14


# ===========================================================================
# CWENO5-Z tests (old CWENO5, now cweno_z5_fv)
# ===========================================================================

def test_cweno_z5_constant():
    uL, uR = cweno_z5_fv(5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert abs(uR - 5.0) < 1e-14


def test_cweno_z5_linear():
    uL, uR = cweno_z5_fv(-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
    assert abs(uL - 0.5) < 1e-14
    assert abs(uR - (-0.5)) < 1e-14


def test_cweno_z5_linear_shifted():
    uL, uR = cweno_z5_fv(997.0, 998.0, 999.0, 1000.0, 1001.0, 1002.0, 1003.0)
    assert abs(uL - 1000.5) < 2e-13
    assert abs(uR - 999.5) < 5e-13


def test_cweno_z5_quadratic():
    """Quadratic x^2: FV reconstruction gives 1/6 from cell-averaged data."""
    uL, uR = cweno_z5_fv(9.0, 4.0, 1.0, 0.0, 1.0, 4.0, 9.0)
    assert abs(uL - 1.0/6.0) < 1e-13
    assert abs(uR - 1.0/6.0) < 1e-13


def test_cweno_z5_cubic():
    """Cubic x^3: FV reconstruction gives ~0 from cell-averaged data."""
    uL, uR = cweno_z5_fv(-27.0, -8.0, -1.0, 0.0, 1.0, 8.0, 27.0)
    assert abs(uL) < 1e-12
    assert abs(uR) < 1e-12


def test_cweno_z5_jump():
    """Jump: [0,0,0,1,1,1,1]; CWENO5 may produce mild overshoot at jump."""
    uL, uR = cweno_z5_fv(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0)
    # Left face is between two 1's -> near 1 (with possible overshoot)
    assert 0.9 <= uL <= 1.5
    # Right face sees the jump from the other side
    assert 0.0 <= uR <= 1.1
    assert math.isfinite(uL)
    assert math.isfinite(uR)


def test_cweno_z5_jump_bound():
    """Values should be reasonably bounded for sharp transitions."""
    uL, uR = cweno_z5_fv(5.0, 5.0, 5.0, 5.0, 0.0, 0.0, 0.0)
    assert -0.1 <= uL <= 5.1
    assert -0.1 <= uR <= 7.5
    assert math.isfinite(uL)
    assert math.isfinite(uR)


def test_cweno_z5_sinusoid():
    """Smooth sinusoid: check reasonable reconstruction."""
    points = [math.sin(-3.0), math.sin(-2.0), math.sin(-1.0),
              math.sin(0.0), math.sin(1.0), math.sin(2.0), math.sin(3.0)]
    uL, uR = cweno_z5_fv(*points)
    assert abs(uL - math.sin(0.5)) < 0.06
    assert abs(uR - math.sin(-0.5)) < 0.06


def test_cweno_z5_quartic():
    """Quartic x^4: FV reconstruction gives -1/30 from cell-averaged data."""
    uL, uR = cweno_z5_fv(81.0, 16.0, 1.0, 0.0, 1.0, 16.0, 81.0)
    assert abs(uL - (-1.0/30.0)) < 1e-12
    assert abs(uR - (-1.0/30.0)) < 1e-12


# ===========================================================================
# CWENO3 (Cravero 2017 -- full CWENO framework, paper CWENO5)
# ===========================================================================

def test_cweno3_constant():
    """Constant field: reconstruction should be exact."""
    uL, uR = cweno3_fv(5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert abs(uR - 5.0) < 1e-14


def test_cweno3_constant_random():
    """Constant field with arbitrary value."""
    for val in [0.0, 1.0, -3.5, 1e10]:
        uL, uR = cweno3_fv(val, val, val, val, val)
        assert abs(uL - val) < 1e-14 * max(1.0, abs(val))
        assert abs(uR - val) < 1e-14 * max(1.0, abs(val))


def test_cweno3_linear():
    """Linear field: CWENO3 should be exact for linear data."""
    uL, uR = cweno3_fv(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert abs(uL - 0.5) < 1e-14
    assert abs(uR - (-0.5)) < 1e-14


def test_cweno3_linear_shifted():
    """Linear field with offset."""
    uL, uR = cweno3_fv(998.0, 999.0, 1000.0, 1001.0, 1002.0)
    assert abs(uL - 1000.5) < 5e-13
    assert abs(uR - 999.5) < 5e-13


def test_cweno3_quadratic():
    """Quadratic x^2: FV reconstruction gives 1/6 from cell-averaged data."""
    uL, uR = cweno3_fv(4.0, 1.0, 0.0, 1.0, 4.0)
    assert abs(uL - 1.0/6.0) < 1e-13
    assert abs(uR - 1.0/6.0) < 1e-13


def test_cweno3_cubic():
    """Cubic u(x)=x^3: check finite and within range."""
    uL, uR = cweno3_fv(-8.0, -1.0, 0.0, 1.0, 8.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    # Not exact for cubic but should be somewhat close
    assert abs(uL - 0.125) < 0.5
    assert abs(uR - (-0.125)) < 0.5


def test_cweno3_discontinuity_01():
    """Jump [0,0,1,1,1]: uL should be ~1 (between two 1's)."""
    uL, uR = cweno3_fv(0.0, 0.0, 1.0, 1.0, 1.0)
    assert 0.9 <= uL <= 1.0
    assert 0.0 <= uR <= 1.0
    assert math.isfinite(uL)
    assert math.isfinite(uR)


def test_cweno3_discontinuity_10():
    """Jump [1,1,1,0,0]: values stay in range."""
    uL, uR = cweno3_fv(1.0, 1.0, 1.0, 0.0, 0.0)
    assert 0.0 <= uL <= 1.0
    assert 0.0 <= uR <= 1.0
    assert math.isfinite(uL)
    assert math.isfinite(uR)


def test_cweno3_discontinuity_sharp():
    """Sharp transition: values should stay in range."""
    uL, uR = cweno3_fv(10.0, 10.0, 10.0, 0.0, 0.0)
    assert -0.1 <= uL <= 10.1
    assert -0.1 <= uR <= 10.1
    assert math.isfinite(uL)
    assert math.isfinite(uR)


def test_cweno3_symmetric():
    """Symmetric data: uL and uR should be symmetric."""
    uL, uR = cweno3_fv(-10.0, -1.0, 0.0, 1.0, 10.0)
    assert abs(uL + uR) < 1e-13


def test_cweno3_peak():
    """Peak at center: values stay in range."""
    uL, uR = cweno3_fv(1.0, 3.0, 5.0, 3.0, 1.0)
    assert 1.0 <= uL <= 5.0
    assert 1.0 <= uR <= 5.0
    assert math.isfinite(uL)
    assert math.isfinite(uR)


def test_cweno3_weights_sum_to_one():
    """Implicit: the weights w_k sum to 1 by construction."""
    # This is tested indirectly: constant data gives exact reconstruction.
    uL, uR = cweno3_fv(3.0, 3.0, 3.0, 3.0, 3.0)
    assert abs(uL - 3.0) < 1e-14


# ===========================================================================
# CWENO5 (Cravero 2017 -- full CWENO framework, paper CWENO7)
# ===========================================================================

def test_cweno5_constant():
    """Constant field: reconstruction should be exact."""
    uL, uR = cweno5_fv(5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert abs(uR - 5.0) < 1e-14


def test_cweno5_constant_random():
    """Constant field with arbitrary values."""
    for val in [0.0, 1.0, -3.5, 1e10]:
        uL, uR = cweno5_fv(val, val, val, val, val, val, val)
        assert abs(uL - val) < 1e-14 * max(1.0, abs(val))
        assert abs(uR - val) < 1e-14 * max(1.0, abs(val))


def test_cweno5_linear():
    """Linear field: CWENO5 should be exact."""
    uL, uR = cweno5_fv(-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
    assert abs(uL - 0.5) < 1e-14
    assert abs(uR - (-0.5)) < 1e-14


def test_cweno5_linear_shifted():
    """Linear field with offset."""
    uL, uR = cweno5_fv(997.0, 998.0, 999.0, 1000.0, 1001.0, 1002.0, 1003.0)
    assert abs(uL - 1000.5) < 5e-13
    assert abs(uR - 999.5) < 5e-13


def test_cweno5_quadratic():
    """Quadratic x^2: FV reconstruction gives 1/6 from cell-averaged data."""
    uL, uR = cweno5_fv(9.0, 4.0, 1.0, 0.0, 1.0, 4.0, 9.0)
    assert abs(uL - 1.0/6.0) < 1e-13
    assert abs(uR - 1.0/6.0) < 1e-13


def test_cweno5_cubic():
    """Cubic x^3: FV reconstruction gives ~0 from cell-averaged data."""
    uL, uR = cweno5_fv(-27.0, -8.0, -1.0, 0.0, 1.0, 8.0, 27.0)
    assert abs(uL) < 1e-12
    assert abs(uR) < 1e-12


def test_cweno5_quartic():
    """Quartic x^4: CWENO weights shift away from P_opt for high-curvature data.
    Check that reconstruction is finite and within reasonable bounds."""
    uL, uR = cweno5_fv(81.0, 16.0, 1.0, 0.0, 1.0, 16.0, 81.0)
    # Optimal polynomial would give -1/30 ~ -0.0333, but JS indicators
    # detect curvature and shift weights toward sub-stencils.
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    # Should be within data range or near it
    assert -1.0 <= uL <= 1.0
    assert -1.0 <= uR <= 1.0


def test_cweno5_quintic_finite():
    """Quintic x^5: values should be finite."""
    uL, uR = cweno5_fv(-243.0, -32.0, -1.0, 0.0, 1.0, 32.0, 243.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)


def test_cweno5_discontinuity_01():
    """Jump [0,0,0,1,1,1,1]: check behavior."""
    uL, uR = cweno5_fv(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0)
    assert 0.9 <= uL <= 1.0
    assert 0.0 <= uR <= 1.0
    assert math.isfinite(uL)
    assert math.isfinite(uR)


def test_cweno5_discontinuity_10():
    """Jump [1,1,1,1,0,0,0]: values stay in range."""
    uL, uR = cweno5_fv(1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0)
    assert 0.0 <= uL <= 1.0
    assert 0.0 <= uR <= 1.0
    assert math.isfinite(uL)
    assert math.isfinite(uR)


def test_cweno5_discontinuity_sharp():
    """Sharp transition: values should stay in range."""
    uL, uR = cweno5_fv(10.0, 10.0, 10.0, 10.0, 0.0, 0.0, 0.0)
    assert -0.1 <= uL <= 10.1
    assert -0.1 <= uR <= 10.1
    assert math.isfinite(uL)
    assert math.isfinite(uR)


def test_cweno5_symmetric():
    """Symmetric data: check antisymmetry preserved."""
    # u(x) = x^3 data
    uL, uR = cweno5_fv(-27.0, -8.0, -1.0, 0.0, 1.0, 8.0, 27.0)
    assert abs(uL + uR) < 1e-12


def test_cweno5_sinusoid():
    """Smooth sinusoid: check reasonable reconstruction."""
    points = [math.sin(-3.0), math.sin(-2.0), math.sin(-1.0),
              math.sin(0.0), math.sin(1.0), math.sin(2.0), math.sin(3.0)]
    uL, uR = cweno5_fv(*points)
    assert abs(uL - math.sin(0.5)) < 0.03
    assert abs(uR - math.sin(-0.5)) < 0.03


def test_cweno5_weights_sum_to_one():
    """Implicit: weights sum to 1 -> constant data gives exact reconstruction."""
    uL, uR = cweno5_fv(3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0)
    assert abs(uL - 3.0) < 1e-14


# ===========================================================================
# Central WENO helper tests
# ===========================================================================

def test_cw_smoothness_constant():
    b_opt, b_L, b_R = _cw_smoothness(1.0, 1.0, 1.0)
    assert b_opt == 0.0
    assert b_L == 0.0
    assert b_R == 0.0


def test_cw_smoothness_linear():
    b_opt, b_L, b_R = _cw_smoothness(-1.0, 0.0, 1.0)
    assert abs(b_opt) < 1e-14
    assert abs(b_L - 1.0) < 1e-14
    assert abs(b_R - 1.0) < 1e-14


def test_cw_smoothness_quadratic():
    b_opt, b_L, b_R = _cw_smoothness(1.0, 0.0, 1.0)
    assert abs(b_opt - 4.0) < 1e-14
    assert abs(b_L - 1.0) < 1e-14
    assert abs(b_R - 1.0) < 1e-14


# ===========================================================================
# Central WENO tests
# ===========================================================================

def test_central_weno_constant():
    uL, uR = central_weno_fv(5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert abs(uR - 5.0) < 1e-14


def test_central_weno_linear():
    uL, uR = central_weno_fv(-1.0, 0.0, 1.0)
    assert abs(uL - 0.5) < 1e-14
    assert abs(uR - (-0.5)) < 1e-14


def test_central_weno_linear_shifted():
    uL, uR = central_weno_fv(999.0, 1000.0, 1001.0)
    assert abs(uL - 1000.5) < 2e-13
    assert abs(uR - 999.5) < 2e-13


def test_central_weno_quadratic():
    """For quadratic data, central WENO's optimal weight is suppressed
    due to large 2nd derivative, so output deviates from exact quadratic."""
    uL, uR = central_weno_fv(1.0, 0.0, 1.0)
    assert 0.0 <= uL <= 1.0
    assert 0.0 <= uR <= 1.0
    assert math.isfinite(uL)
    assert math.isfinite(uR)


def test_central_weno_jump_left():
    uL, uR = central_weno_fv(0.0, 1.0, 1.0)
    assert abs(uL - 1.0) < 1e-14
    assert 0.5 <= uR <= 1.0
    assert math.isfinite(uR)


def test_central_weno_jump_right():
    uL, uR = central_weno_fv(1.0, 1.0, 0.0)
    assert abs(uR - 1.0) < 1e-14
    assert 0.5 <= uL <= 1.0
    assert math.isfinite(uL)


def test_central_weno_jump_sharp():
    uL, uR = central_weno_fv(10.0, 10.0, 0.0)
    assert 0.0 <= uL <= 10.0
    assert 0.0 <= uR <= 10.0
    assert math.isfinite(uL)
    assert math.isfinite(uR)


def test_central_weno_symmetric():
    uL, uR = central_weno_fv(-1.0, 0.0, 1.0)
    assert abs(uL + uR) < 1e-14


def test_central_weno_cubic_like():
    uL, uR = central_weno_fv(-8.0, -1.0, 0.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert -8.0 <= uL <= 1.0
    assert -8.0 <= uR <= 1.0
#
# :D
#
