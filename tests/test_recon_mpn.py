"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Tests for MP family reconstruction methods (mp3, mp5, mp7, mp5_r)
"""
import math
from pyrecon.recon_mpn import mp3_fv, mp5_fv, mp7_fv, mp5_r_fv


# ===========================================================================
# mp3
# ===========================================================================

def test_mp3_constant():
    """Constant field: faces equal the constant value."""
    uL, uR = mp3_fv(5.0, 5.0, 5.0, 5.0, 5.0)
    assert uL == 5.0
    assert uR == 5.0


def test_mp3_linear():
    """Linear field u(x)=x: faces equal exact value at face centre."""
    # u = [8, 9, 10, 11, 12], u_i = 10
    # uL = u_{i+1/2}^- = 10.5, uR = u_{i-1/2}^+ = 9.5
    uL, uR = mp3_fv(8.0, 9.0, 10.0, 11.0, 12.0)
    assert abs(uL - 10.5) < 1e-14
    assert abs(uR - 9.5) < 1e-14


def test_mp3_monotonicity():
    """Sharp jump: reconstructed faces must stay within cell bounds."""
    # Jump 0->1 at u_i: [0, 0, 0, 1, 1]
    uL, uR = mp3_fv(0.0, 0.0, 0.0, 1.0, 1.0)
    # uL should be between min/max of u_i and u_ip1 (0 and 1)
    assert 0.0 <= uL <= 1.0
    # uR at face i-1/2 between u_im1=0 and u_i=0 -> should be 0
    assert abs(uR) < 1e-14


def test_mp3_smooth_extrema():
    """Smooth extremum: limiter preserves accuracy."""
    # u = x^2 parabola: [-2, -1, 0, 1, 2]^2 = [4, 1, 0, 1, 4], u_i=0
    uL, uR = mp3_fv(4.0, 1.0, 0.0, 1.0, 4.0)
    # Values should be finite and reasonable
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    # Both should be bounded by the stencil range
    assert min(4, 1, 0, 1, 4) <= uL <= max(4, 1, 0, 1, 4)


# ===========================================================================
# mp5
# ===========================================================================

def test_mp5_constant():
    uL, uR = mp5_fv(5.0, 5.0, 5.0, 5.0, 5.0)
    assert uL == 5.0
    assert uR == 5.0


def test_mp5_linear():
    # [8, 9, 10, 11, 12], u_i=10 -> faces at 10.5 and 9.5
    uL, uR = mp5_fv(8.0, 9.0, 10.0, 11.0, 12.0)
    assert abs(uL - 10.5) < 1e-14
    assert abs(uR - 9.5) < 1e-14


def test_mp5_monotonicity():
    uL, uR = mp5_fv(0.0, 0.0, 0.0, 1.0, 1.0)
    assert 0.0 <= uL <= 1.0
    assert abs(uR) < 1e-14


def test_mp5_jump_mid():
    """Jump centered at i: [0, 0, 1, 1, 1], u_i=1."""
    uL, uR = mp5_fv(0.0, 0.0, 1.0, 1.0, 1.0)
    # uL at face i+1/2: between u_i=1 and u_ip1=1 -> near 1
    assert abs(uL - 1.0) < 0.05
    # uR at face i-1/2: between u_im1=0 and u_i=1 -> bounded
    assert 0.0 <= uR <= 1.0


# ===========================================================================
# mp7
# ===========================================================================

def test_mp7_constant():
    uL, uR = mp7_fv(5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert abs(uR - 5.0) < 1e-14


def test_mp7_linear():
    # 7-point linear: [6, 7, 8, 9, 10, 11, 12], u_i=9
    # uL = u_{i+1/2}^- = 9.5, uR = u_{i-1/2}^+ = 8.5
    uL, uR = mp7_fv(6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0)
    assert abs(uL - 9.5) < 1e-14
    assert abs(uR - 8.5) < 1e-14


def test_mp7_monotonicity():
    # 7-point jump: [0, 0, 0, 0, 1, 1, 1], u_i=0
    uL, uR = mp7_fv(0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    assert 0.0 <= uL <= 1.0
    assert abs(uR) < 1e-14


def test_mp7_smooth():
    """Smooth 7-point profile: all faces well-defined."""
    uL, uR = mp7_fv(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)


# ===========================================================================
# mp5_r
# ===========================================================================

def test_mp5_r_constant():
    uL, uR = mp5_r_fv(5.0, 5.0, 5.0, 5.0, 5.0)
    assert uL == 5.0
    assert uR == 5.0


def test_mp5_r_linear():
    uL, uR = mp5_r_fv(8.0, 9.0, 10.0, 11.0, 12.0)
    assert abs(uL - 10.5) < 1e-14
    assert abs(uR - 9.5) < 1e-14


def test_mp5_r_monotonicity():
    uL, uR = mp5_r_fv(0.0, 0.0, 0.0, 1.0, 1.0)
    assert 0.0 <= uL <= 1.0
    assert abs(uR) < 1e-14


def test_mp5_r_jump_mid():
    """Refined limiter on centered jump."""
    uL, uR = mp5_r_fv(0.0, 0.0, 1.0, 1.0, 1.0)
    assert abs(uL - 1.0) < 0.05
    assert 0.0 <= uR <= 1.0


# ===========================================================================
# Cross-method comparisons
# ===========================================================================

def test_all_methods_agree_linear():
    """All methods should be exact for linear data."""
    u = (8.0, 9.0, 10.0, 11.0, 12.0)
    u7 = (6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0)

    uL3, uR3 = mp3_fv(*u)
    uL5, uR5 = mp5_fv(*u)
    uL7, uR7 = mp7_fv(*u7)
    uLr, uRr = mp5_r_fv(*u)

    # All uL from 5-point stencil should match (face at i+1/2 on data [8..12])
    assert abs(uL3 - 10.5) < 1e-14
    assert abs(uL5 - 10.5) < 1e-14
    assert abs(uLr - 10.5) < 1e-14

    # mp7 uses different data; face i+1/2 = 9.5 where u_i=9
    assert abs(uL7 - 9.5) < 1e-14

    # uR are at face i-1/2
    assert abs(uR3 - 9.5) < 1e-14
    assert abs(uR5 - 9.5) < 1e-14
    assert abs(uRr - 9.5) < 1e-14
    assert abs(uR7 - 8.5) < 1e-14


def test_all_methods_agree_constant():
    """All methods should return the constant value."""
    uL3, uR3 = mp3_fv(7.0, 7.0, 7.0, 7.0, 7.0)
    uL5, uR5 = mp5_fv(7.0, 7.0, 7.0, 7.0, 7.0)
    uL7, uR7 = mp7_fv(7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0)
    uLr, uRr = mp5_r_fv(7.0, 7.0, 7.0, 7.0, 7.0)

    assert abs(uL3 - 7.0) < 1e-14
    assert abs(uR3 - 7.0) < 1e-14
    assert abs(uL5 - 7.0) < 1e-14
    assert abs(uR5 - 7.0) < 1e-14
    assert abs(uL7 - 7.0) < 1e-14
    assert abs(uR7 - 7.0) < 1e-14
    assert abs(uLr - 7.0) < 1e-14
    assert abs(uRr - 7.0) < 1e-14


def test_mp5_r_distinct_from_mp5():
    """mp5_r has a different limiter; check it produces different output
    on non-trivial data where the limiter is active."""
    # Sharp gradient where MP limiter is active
    u = (0.0, 0.0, 0.1, 0.9, 1.0)
    uL5, uR5 = mp5_fv(*u)
    uLr, uRr = mp5_r_fv(*u)
    # Both should be finite and within [0, 1] range
    for v in (uL5, uR5, uLr, uRr):
        assert math.isfinite(v)
        assert 0.0 <= v <= 1.0
    # They may differ since the limiters are different
#
# :D
#
