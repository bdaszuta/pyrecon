"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Tests for CENO3 and CENO5 reconstruction methods
"""
import math
from pyrecon.recon_ceno import ceno3_fv, ceno5_fv, _ceno3lim


# ---------------------------------------------------------------------------
# ceno3lim tests
# ---------------------------------------------------------------------------

def test_ceno3lim_mixed_signs() -> None:
    """When candidates have mixed signs, limiter returns 0."""
    assert _ceno3lim(1.0, -1.0, 1.0) == 0.0
    assert _ceno3lim(-1.0, 1.0, -1.0) == 0.0
    assert _ceno3lim(1.0, 1.0, -1.0) == 0.0


def test_ceno3lim_all_positive() -> None:
    """When all positive, pick candidate with min abs after alpha weighting."""
    # d = (2, 1, 3), weighted: (2, 0.7, 3) -> min is d1=1
    result = _ceno3lim(2.0, 1.0, 3.0)
    assert abs(result - 1.0) < 1e-15


def test_ceno3lim_all_negative() -> None:
    """When all negative, pick candidate with min abs after alpha weighting."""
    # d = (-2, -1, -3), weighted: (2, 0.7, 3) -> min abs is d1=-1
    result = _ceno3lim(-2.0, -1.0, -3.0)
    assert abs(result - (-1.0)) < 1e-15


def test_ceno3lim_alpha_tiebreak() -> None:
    """Unweighted values equal: alpha weighting breaks tie toward center."""
    # d = (1.0, 1.0, 1.0), weighted: (1.0, 0.7, 1.0) -> min is d1=1.0
    result = _ceno3lim(1.0, 1.0, 1.0)
    assert abs(result - 1.0) < 1e-15


def test_ceno3lim_zero_candidates() -> None:
    """All zeros -> returns 0."""
    assert _ceno3lim(0.0, 0.0, 0.0) == 0.0


# ---------------------------------------------------------------------------
# ceno3 tests (5-point stencil)
# ---------------------------------------------------------------------------

def test_ceno3_constant() -> None:
    """Constant field: uL = uR = constant."""
    uL, uR = ceno3_fv(5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert abs(uR - 5.0) < 1e-14


def test_ceno3_linear_increasing() -> None:
    """Linear u(x)=x: uL = u_i + 0.5, uR = u_i - 0.5."""
    # u: -2, -1, 0, 1, 2 at dx=1
    uL, uR = ceno3_fv(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert abs(uL - 0.5) < 1e-14
    assert abs(uR - (-0.5)) < 1e-14


def test_ceno3_linear_shifted() -> None:
    """Linear field with offset."""
    uL, uR = ceno3_fv(998.0, 999.0, 1000.0, 1001.0, 1002.0)
    assert abs(uL - 1000.5) < 1e-14
    assert abs(uR - 999.5) < 1e-14


def test_ceno3_quadratic() -> None:
    """Quadratic u(x)=x^2: uL = uR = 1/6 (FV recon from cell avgs)."""
    # u: 4, 1, 0, 1, 4 at x=-2,-1,0,1,2 (cell-averaged)
    uL, uR = ceno3_fv(4.0, 1.0, 0.0, 1.0, 4.0)
    # 5th-order FV linear reconstruction on these cell averages gives 1/6.
    # CENO3 reduces to the same value on smooth data.
    assert abs(uL - 1.0/6.0) < 1e-14
    assert abs(uR - 1.0/6.0) < 1e-14


def test_ceno3_jump() -> None:
    """Jump at i=0: stencil [0,0,1,1,1], should be ~1 at face."""
    uL, uR = ceno3_fv(0.0, 0.0, 1.0, 1.0, 1.0)
    assert abs(uL - 1.0) < 0.1
    assert abs(uR - 1.0) < 0.1


def test_ceno3_peak() -> None:
    """Peak/valley: smoother than pure minmod."""
    # Peak at center
    uL, uR = ceno3_fv(1.0, 3.0, 5.0, 3.0, 1.0)
    # Both faces should be in [1, 5] range
    assert 1.0 <= uL <= 5.0
    assert 1.0 <= uR <= 5.0
    assert math.isfinite(uL)
    assert math.isfinite(uR)


def test_ceno3_sharp_step() -> None:
    """Step: sharp transition, values should stay in range."""
    uL, uR = ceno3_fv(10.0, 10.0, 10.0, 0.0, 0.0)
    assert 0.0 <= uL <= 10.0
    assert 0.0 <= uR <= 10.0
    assert math.isfinite(uL)
    assert math.isfinite(uR)


# ---------------------------------------------------------------------------
# ceno5 tests (7-point stencil)
# ---------------------------------------------------------------------------

def test_ceno5_constant() -> None:
    """Constant field."""
    uL, uR = ceno5_fv(5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert abs(uR - 5.0) < 1e-14


def test_ceno5_linear_increasing() -> None:
    """Linear u(x)=x."""
    # u: -3, -2, -1, 0, 1, 2, 3
    uL, uR = ceno5_fv(-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
    assert abs(uL - 0.5) < 1e-14
    assert abs(uR - (-0.5)) < 1e-14


def test_ceno5_linear_shifted() -> None:
    """Linear field with large offset."""
    uL, uR = ceno5_fv(997.0, 998.0, 999.0, 1000.0, 1001.0, 1002.0, 1003.0)
    assert abs(uL - 1000.5) < 1e-14
    assert abs(uR - 999.5) < 1e-14


def test_ceno5_quadratic() -> None:
    """Quadratic u(x)=x^2: uL = uR = 1/6 (FV recon from cell avgs)."""
    # u: 9, 4, 1, 0, 1, 4, 9 at x=-3,-2,-1,0,1,2,3 (cell-averaged)
    uL, uR = ceno5_fv(9.0, 4.0, 1.0, 0.0, 1.0, 4.0, 9.0)
    # 7th-order FV linear reconstruction on these cell averages gives 1/6.
    # CENO5 reduces to the same value on smooth data.
    assert abs(uL - 1.0/6.0) < 1e-14
    assert abs(uR - 1.0/6.0) < 1e-14


def test_ceno5_cubic() -> None:
    """Cubic u(x)=x^3: CENO5 exactly reconstructs with true cell averages."""
    # True cell averages of x^3: x_i^3 + x_i/4
    uL, uR = ceno5_fv(-27.75, -8.5, -1.25, 0.0, 1.25, 8.5, 27.75)
    assert abs(uL - 0.125) < 5e-14
    assert abs(uR - (-0.125)) < 5e-14


def test_ceno5_quartic() -> None:
    """Quartic u(x)=x^4: CENO5 reconstructs with 5th-order accuracy."""
    # True cell averages of x^4: x_i^4 + 0.5*x_i^2 + 0.0125
    ca_3 = 81.0 + 4.5 + 0.0125
    ca_2 = 16.0 + 2.0 + 0.0125
    ca_1 = 1.0 + 0.5 + 0.0125
    ca_0 = 0.0125
    uL, uR = ceno5_fv(ca_3, ca_2, ca_1, ca_0, ca_1, ca_2, ca_3)
    assert abs(uL - 0.0625) < 1e-13
    assert abs(uR - 0.0625) < 1e-13


def test_ceno5_jump_left_stencil() -> None:
    """Jump at i=0: [0,0,0,1,1,1,1].

    Left face u_{i+1/2}^- sees mostly 1s -> near 1.
    Right face u_{i-1/2}^+ (reversed stencil) sees the jump from
    the other side, centered at u_i=1 surrounded by 0s -> the
    ceno3lim picks the min-magnitude same-sign candidate,
    giving uR=0.8 (all candidates same-sign, center-biased).
    """
    uL, uR = ceno5_fv(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0)
    assert abs(uL - 1.05) < 1e-10
    assert abs(uR - 0.80) < 1e-10


def test_ceno5_jump_right_stencil() -> None:
    """Jump at i: [1,1,1,1,0,0,0].

    Left face u_{i+1/2}^-: forward stencil sees mostly 0s after the
    jump, CENO limiting keeps it near 0.
    Right face u_{i-1/2}^+: reversed stencil centered at u_i=1 with
    1s to the left and 0s to the right. CENO5 picks the min-magnitude
    same-sign candidate, giving uR ~1.039 (slightly overshooting).
    """
    uL, uR = ceno5_fv(1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0)
    assert 0.0 <= uL <= 1.0
    # uR ~1.039 is a known slight overshoot from CENO limiting
    assert 0.0 <= uR <= 1.1
    assert math.isfinite(uL)
    assert math.isfinite(uR)


def test_ceno5_sinusoid() -> None:
    """Smooth sinusoid, check reasonable reconstruction."""
    import math as m
    points = [m.sin(-3.0), m.sin(-2.0), m.sin(-1.0),
              m.sin(0.0), m.sin(1.0), m.sin(2.0), m.sin(3.0)]
    uL, uR = ceno5_fv(*points)
    # Face at +0.5: sin(0.5) ~ 0.479
    assert abs(uL - m.sin(0.5)) < 0.02
    assert abs(uR - m.sin(-0.5)) < 0.02


def test_ceno5_symmetric() -> None:
    """Symmetric field: uL and uR should be symmetric."""
    uL, uR = ceno5_fv(-10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0)
    assert abs(uL + uR) < 1e-14  # antisymmetric data -> symmetric faces
#
# :D
#
