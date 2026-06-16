"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Tests for WENO5-M (Mapped WENO) reconstruction
"""
import math
from pyrecon.recon_weno5m import (
    weno5m_fv, weno5m_pw, _henrick_map, _js_smoothness,
)


# ---------------------------------------------------------------------------
# Henrick mapping function
# ---------------------------------------------------------------------------

def test_henrick_map_endpoints():
    """g_k(0) = 0, g_k(1) = 1 for all d_k."""
    for dk in (1.0/10.0, 3.0/5.0, 3.0/10.0):
        assert _henrick_map(0.0, dk) == 0.0
        assert abs(_henrick_map(1.0, dk) - 1.0) < 1e-15


def test_henrick_map_fixed_point():
    """g_k(d_k) = d_k (fixed point at optimal weight)."""
    for dk in (1.0/10.0, 3.0/5.0, 3.0/10.0):
        assert abs(_henrick_map(dk, dk) - dk) < 1e-15


def test_henrick_map_monotonic():
    """Mapping should be monotonic on [0, 1]."""
    for dk in (1.0/10.0, 3.0/5.0, 3.0/10.0):
        prev = _henrick_map(0.0, dk)
        for omega in (0.1, 0.3, 0.5, 0.7, 0.9):
            curr = _henrick_map(omega, dk)
            assert curr >= prev, f"Non-monotonic at dk={dk}, omega={omega}"
            prev = curr


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


# ---------------------------------------------------------------------------
# WENO5-M
# ---------------------------------------------------------------------------

def test_weno5m_constant():
    uL, uR = weno5m_fv(5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert abs(uR - 5.0) < 1e-14


def test_weno5m_linear():
    uL, uR = weno5m_fv(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert abs(uL - 0.5) < 1e-14
    assert abs(uR - (-0.5)) < 1e-14


def test_weno5m_quadratic():
    uL, uR = weno5m_fv(4.0, 1.0, 0.0, 1.0, 4.0)
    assert abs(uL - 1.0 / 6.0) < 1e-14
    assert abs(uR - 1.0 / 6.0) < 1e-14


def test_weno5m_jump_left():
    """Jump at i=0 in [0,0,1,1,1]: both faces should be ~1."""
    uL, uR = weno5m_fv(0.0, 0.0, 1.0, 1.0, 1.0)
    assert abs(uL - 1.0) < 0.05
    assert abs(uR - 1.0) < 0.05


def test_weno5m_jump_right():
    """Jump at i+1 in [1,1,1,0,0]: values within range."""
    uL, uR = weno5m_fv(1.0, 1.0, 1.0, 0.0, 0.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert 0.0 <= uL <= 1.0
    assert 0.0 <= uR <= 1.0


def test_weno5m_critical_point():
    """Critical point (f'=0) at i: u(x) = sin(pi*x) around x=0.

    At x=0, f'(0)=pi*cos(0)=pi != 0 (not critical), so use
    cos-like data. For f(x)=cos(pi*x/4) on [-2,-1,0,1,2]:
    f(0)=1, f'(0)=0. This tests critical point behavior.

    Values: cos(-pi/2)=0, cos(-pi/4)=0.7071, cos(0)=1,
            cos(pi/4)=0.7071, cos(pi/2)=0

    Using u(x) = 1 - x^2 (f'(0)=0 at x=0):
    u(-2)= -3, u(-1)=0, u(0)=1, u(1)=0, u(2)=-3
    """
    # Use parabola u(x)=1-x^2: f'=0 at x=0 (critical point)
    # u_im2=u(-2)= -3, u_im1=u(-1)=0, u_i=u(0)=1, u_ip1=u(1)=0, u_ip2=u(2)=-3
    uL, uR = weno5m_fv(-3.0, 0.0, 1.0, 0.0, -3.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    # Face at i+1/2 is x=0.5, u(0.5)=1-0.25=0.75
    # FV optimal: uL ~ 0.75 with some ENO bias
    assert 0.5 <= uL <= 1.0
    assert 0.5 <= uR <= 1.0
    # Should be close to optimal for smooth critical-point data
    assert abs(uL - 0.75) < 0.15
    assert abs(uR - 0.75) < 0.15


def test_weno5m_symmetry():
    """WENO-M should be approximately anti-symmetric under negation."""
    uL1, uR1 = weno5m_fv(-2.0, -1.0, 0.0, 1.0, 2.0)
    uL2, uR2 = weno5m_fv(2.0, 1.0, 0.0, -1.0, -2.0)
    assert abs(uL1 + uL2) < 0.01
    assert abs(uR1 + uR2) < 0.01

# ---------------------------------------------------------------------------
# WENO5-M (PW)
# ---------------------------------------------------------------------------

def test_weno5m_pw_constant():
    uL, uR = weno5m_pw(5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert abs(uR - 5.0) < 1e-14


def test_weno5m_pw_linear():
    uL, uR = weno5m_pw(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert abs(uL - 0.5) < 1e-14
    assert abs(uR - (-0.5)) < 1e-14


def test_weno5m_pw_finite():
    uL, uR = weno5m_pw(0.0, 0.0, 1.0, 1.0, 1.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
#
# :D
#
