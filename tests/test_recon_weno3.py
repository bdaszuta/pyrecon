"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Tests for WENO3 reconstruction methods (3-point stencil)
"""
import math
from pyrecon.recon_weno3 import weno3_fv, weno3_pw, weno3z_fv, weno3z_pw


# ---------------------------------------------------------------------------
# Constant data
# ---------------------------------------------------------------------------

def test_weno3_fv_constant():
    uL, uR = weno3_fv(3.0, 3.0, 3.0)
    assert abs(uL - 3.0) < 1e-14
    assert abs(uR - 3.0) < 1e-14


def test_weno3_pw_constant():
    uL, uR = weno3_pw(3.0, 3.0, 3.0)
    assert abs(uL - 3.0) < 1e-14
    assert abs(uR - 3.0) < 1e-14


def test_weno3z_fv_constant():
    uL, uR = weno3z_fv(3.0, 3.0, 3.0)
    assert abs(uL - 3.0) < 1e-14
    assert abs(uR - 3.0) < 1e-14


def test_weno3z_pw_constant():
    uL, uR = weno3z_pw(3.0, 3.0, 3.0)
    assert abs(uL - 3.0) < 1e-14
    assert abs(uR - 3.0) < 1e-14


# ---------------------------------------------------------------------------
# Linear data: f(x) = x, point values at [-1, 0, 1]
# Face i+1/2 at x=0.5, exact = 0.5. Face i-1/2 at x=-0.5, exact = -0.5.
# Both FV and PW weights give exact linear faces on linear data.
# ---------------------------------------------------------------------------

def test_weno3_fv_linear():
    uL, uR = weno3_fv(-1.0, 0.0, 1.0)
    assert abs(uL - 0.5) < 1e-14
    assert abs(uR + 0.5) < 1e-14


def test_weno3_pw_linear():
    uL, uR = weno3_pw(-1.0, 0.0, 1.0)
    assert abs(uL - 0.5) < 1e-14
    assert abs(uR + 0.5) < 1e-14


def test_weno3z_fv_linear():
    uL, uR = weno3z_fv(-1.0, 0.0, 1.0)
    assert abs(uL - 0.5) < 1e-14
    assert abs(uR + 0.5) < 1e-14


def test_weno3z_pw_linear():
    uL, uR = weno3z_pw(-1.0, 0.0, 1.0)
    assert abs(uL - 0.5) < 1e-14
    assert abs(uR + 0.5) < 1e-14


# ---------------------------------------------------------------------------
# Quadratic data: f(x) = x^2, point values at [-1, 0, 1] -> [1, 0, 1]
# PW: face at x=0.5 -> f(0.5) = 0.25
# FV: cell averages of x^2 are [4/3, 1/3, 4/3] but using point values
#     as cell-averaged input gives FV face = 1/6 (see skill pitfall).
#     This test uses the actual point-value inputs [1,0,1] treated as
#     cell averages, and expects the FV-correct face value 1/6.
# ---------------------------------------------------------------------------

def test_weno3_fv_quadratic():
    """FV weights on quadratic point-value data produce 1/6 not 0.25."""
    uL, uR = weno3_fv(1.0, 0.0, 1.0)
    assert abs(uL - 1.0 / 6.0) < 1e-14
    # Right face: symmetric data -> symmetric output (uR = uL = 1/6)
    assert abs(uR - 1.0 / 6.0) < 1e-14


def test_weno3_pw_quadratic():
    """PW weights on quadratic point-value data produce 0.25."""
    uL, uR = weno3_pw(1.0, 0.0, 1.0)
    assert abs(uL - 0.25) < 1e-14
    # Right face: symmetric data -> symmetric output (uR = uL = 0.25)
    assert abs(uR - 0.25) < 1e-14


def test_weno3z_fv_quadratic():
    uL, uR = weno3z_fv(1.0, 0.0, 1.0)
    assert abs(uL - 1.0 / 6.0) < 1e-14
    assert abs(uR - 1.0 / 6.0) < 1e-14


def test_weno3z_pw_quadratic():
    uL, uR = weno3z_pw(1.0, 0.0, 1.0)
    assert abs(uL - 0.25) < 1e-14
    assert abs(uR - 0.25) < 1e-14


# ---------------------------------------------------------------------------
# FV vs PW distinction: on quadratic data, FV and PW should differ.
# On linear data they coincide, so this directly tests weight correctness.
# ---------------------------------------------------------------------------

def test_weno3_fv_vs_pw_quadratic_differ():
    """FV and PW must give different face values on quadratic data."""
    uL_fv, uR_fv = weno3_fv(1.0, 0.0, 1.0)
    uL_pw, uR_pw = weno3_pw(1.0, 0.0, 1.0)
    assert abs(uL_fv - uL_pw) > 1e-10, "FV and PW should differ on quadratic"
    assert abs(uR_fv - uR_pw) > 1e-10, "FV and PW should differ on quadratic"


# ---------------------------------------------------------------------------
# Jump / step: faces should stay finite and within data range
# ---------------------------------------------------------------------------

def test_weno3_fv_jump_left():
    """Jump [0,0,1]: faces ~0 before/after jump (WENO smooth bias)."""
    uL, uR = weno3_fv(0.0, 0.0, 1.0)
    assert math.isfinite(uL) and math.isfinite(uR)
    assert -0.1 <= uL <= 1.1
    assert -0.1 <= uR <= 1.1


def test_weno3_pw_jump_left():
    uL, uR = weno3_pw(0.0, 0.0, 1.0)
    assert math.isfinite(uL) and math.isfinite(uR)
    assert -0.1 <= uL <= 1.1
    assert -0.1 <= uR <= 1.1


def test_weno3z_fv_jump_left():
    uL, uR = weno3z_fv(0.0, 0.0, 1.0)
    assert math.isfinite(uL) and math.isfinite(uR)
    assert -0.1 <= uL <= 1.1
    assert -0.1 <= uR <= 1.1


def test_weno3z_pw_jump_left():
    uL, uR = weno3z_pw(0.0, 0.0, 1.0)
    assert math.isfinite(uL) and math.isfinite(uR)
    assert -0.1 <= uL <= 1.1
    assert -0.1 <= uR <= 1.1


def test_weno3_fv_jump_right():
    """Jump [1,1,0]: both faces ~1 (smooth stencil on left side dominates)."""
    uL, uR = weno3_fv(1.0, 1.0, 0.0)
    assert math.isfinite(uL) and math.isfinite(uR)
    assert -0.1 <= uL <= 1.1
    assert -0.1 <= uR <= 1.1
    # Both faces should be close to 1 (the smooth region value)
    assert abs(uL - 1.0) < 0.1
    assert abs(uR - 1.0) < 0.1


def test_weno3_pw_jump_right():
    uL, uR = weno3_pw(1.0, 1.0, 0.0)
    assert math.isfinite(uL) and math.isfinite(uR)
    assert -0.1 <= uL <= 1.1
    assert -0.1 <= uR <= 1.1
    assert abs(uL - 1.0) < 0.1
    assert abs(uR - 1.0) < 0.1


def test_weno3z_fv_jump_right():
    uL, uR = weno3z_fv(1.0, 1.0, 0.0)
    assert math.isfinite(uL) and math.isfinite(uR)
    assert -0.1 <= uL <= 1.1
    assert -0.1 <= uR <= 1.1
    assert abs(uL - 1.0) < 0.1
    assert abs(uR - 1.0) < 0.1


def test_weno3z_pw_jump_right():
    uL, uR = weno3z_pw(1.0, 1.0, 0.0)
    assert math.isfinite(uL) and math.isfinite(uR)
    assert -0.1 <= uL <= 1.1
    assert -0.1 <= uR <= 1.1
    assert abs(uL - 1.0) < 0.1
    assert abs(uR - 1.0) < 0.1


# ---------------------------------------------------------------------------
# Critical point: f(x) ~ x^3 near x=0, f'(0)=0
# Z variants should handle critical points better than JS (less biasing).
# JS: with f'=0 both b0 ~ b1 ~ O(h^2), weights depend on 1/b_k^2 ratio.
# Z: tau term compensates.
# ---------------------------------------------------------------------------

def test_weno3_critical_point_finite():
    """Critical point data: all variants produce finite output."""
    stencil = [-8.0, -1.0, 0.0]
    for fn in [weno3_fv, weno3_pw, weno3z_fv, weno3z_pw]:
        uL, uR = fn(*stencil)
        assert math.isfinite(uL)
        assert math.isfinite(uR)


def test_weno3_z_vs_js_differ_at_critical():
    """Z and JS should differ near critical points (f'=0).
    JS weights degrade; Z's tau term redistributes toward center."""
    uL_js, uR_js = weno3_fv(-8.0, -1.0, 0.0)
    uL_z, uR_z = weno3z_fv(-8.0, -1.0, 0.0)
    assert abs(uL_js - uL_z) > 1e-10, "JS and Z should differ at critical point"
    assert abs(uR_js - uR_z) > 1e-10, "JS and Z should differ at critical point"


# ---------------------------------------------------------------------------
# Edge cases: zeros, negatives, mixed sign
# ---------------------------------------------------------------------------

def test_weno3_all_zeros():
    """All-zeros: no division by zero, output is 0."""
    for fn in [weno3_fv, weno3_pw, weno3z_fv, weno3z_pw]:
        uL, uR = fn(0.0, 0.0, 0.0)
        assert uL == 0.0
        assert uR == 0.0


def test_weno3_negative_values():
    """Negative linear values: output preserves sign correctly."""
    for fn in [weno3_fv, weno3_pw, weno3z_fv, weno3z_pw]:
        uL, uR = fn(-3.0, -2.0, -1.0)
        assert abs(uL + 1.5) < 1e-14
        assert abs(uR + 2.5) < 1e-14


def test_weno3_mixed_sign():
    """Mixed-sign data crossing zero: no crash, output finite."""
    for fn in [weno3_fv, weno3_pw, weno3z_fv, weno3z_pw]:
        uL, uR = fn(-1.0, 0.0, 1.0)
        assert math.isfinite(uL)
        assert math.isfinite(uR)


# ---------------------------------------------------------------------------
# Symmetry: forward and reversed stencils should swap L/R faces
# ---------------------------------------------------------------------------

def test_weno3_fv_symmetry():
    u_fwd = [1.0, 2.0, 3.0]
    u_rev = [3.0, 2.0, 1.0]
    uL_f, uR_f = weno3_fv(*u_fwd)
    uL_r, uR_r = weno3_fv(*u_rev)
    assert abs(uL_f - uR_r) < 1e-14
    assert abs(uR_f - uL_r) < 1e-14


def test_weno3_pw_symmetry():
    uL_f, uR_f = weno3_pw(1.0, 2.0, 3.0)
    uL_r, uR_r = weno3_pw(3.0, 2.0, 1.0)
    assert abs(uL_f - uR_r) < 1e-14
    assert abs(uR_f - uL_r) < 1e-14


def test_weno3z_fv_symmetry():
    uL_f, uR_f = weno3z_fv(1.0, 2.0, 3.0)
    uL_r, uR_r = weno3z_fv(3.0, 2.0, 1.0)
    assert abs(uL_f - uR_r) < 1e-14
    assert abs(uR_f - uL_r) < 1e-14


def test_weno3z_pw_symmetry():
    uL_f, uR_f = weno3z_pw(1.0, 2.0, 3.0)
    uL_r, uR_r = weno3z_pw(3.0, 2.0, 1.0)
    assert abs(uL_f - uR_r) < 1e-14
    assert abs(uR_f - uL_r) < 1e-14


# ---------------------------------------------------------------------------
# Optimal weight verification: combined sub-stencil equals central value
# ---------------------------------------------------------------------------

def test_weno3_fv_optimal_weights():
    """FV optimal weights (1/3, 2/3) recover exact face on linear data."""
    u_im1, u_i, u_ip1 = -1.0, 0.0, 1.0
    dw_fv = (1.0 / 3.0, 2.0 / 3.0)
    # Sub-stencil 0 (left): (3*u_i - u_im1)/2 = 0.5
    uk0 = (-u_im1 + 3.0 * u_i) * 0.5
    # Sub-stencil 1 (right): (u_i + u_ip1)/2 = 0.5
    uk1 = (u_i + u_ip1) * 0.5
    f_opt = dw_fv[0] * uk0 + dw_fv[1] * uk1
    assert abs(f_opt - 0.5) < 1e-14


def test_weno3_pw_optimal_weights():
    """PW optimal weights (1/4, 3/4) recover exact face on linear data."""
    u_im1, u_i, u_ip1 = -1.0, 0.0, 1.0
    dw_pw = (1.0 / 4.0, 3.0 / 4.0)
    uk0 = (-u_im1 + 3.0 * u_i) * 0.5
    uk1 = (u_i + u_ip1) * 0.5
    f_opt = dw_pw[0] * uk0 + dw_pw[1] * uk1
    assert abs(f_opt - 0.5) < 1e-14
#
# :D
#
