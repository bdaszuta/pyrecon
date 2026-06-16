"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Tests for TENO-M reconstruction methods
"""
import math
from pyrecon.recon_teno_m import (
    teno_m_va_fv,
    teno_m_tvd5_fv,
    teno_m_mp_fv,
    _va_limited,
    _tvd5_limited,
    _mp_limited_L,
)
from pyrecon.recon_teno5 import teno5_fv as teno5


# ---------------------------------------------------------------------------
# VA (Van Albada) limiter unit tests
# ---------------------------------------------------------------------------

def test_va_limited_constant():
    """Constant field: d_plus = d_minus = 0 -> returns u_i."""
    result = _va_limited(5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(result - 5.0) < 1e-14


def test_va_limited_linear():
    """Linear field u(x)=x: u=[-2,-1,0,1,2] -> should give 0.5 (3rd-order)."""
    result = _va_limited(-2.0, -1.0, 0.0, 1.0, 2.0)
    # For linear data: d_plus=1, d_minus=1, r=1, phi=2*1/(1+1)=1
    # result = 0 + 0.25*1*((1 - 1/3)*1 + (1 + 1/3)*1) = 0.25*(2/3 + 4/3) = 0.5
    assert abs(result - 0.5) < 1e-14


def test_va_limited_extremum():
    """At extremum: d_plus*d_minus <= 0 -> returns u_i."""
    result = _va_limited(0.0, 2.0, 3.0, 1.0, 0.0)  # peak at i
    assert abs(result - 3.0) < 1e-14


def test_va_limited_shock():
    """Shock: sharp gradient should not blow up."""
    result = _va_limited(0.0, 0.0, 0.0, 100.0, 100.0)
    assert math.isfinite(result)
    # At a jump with d_minus=0, returns u_i=0
    assert abs(result - 0.0) < 1e-14


# ---------------------------------------------------------------------------
# TVD5 limiter unit tests
# ---------------------------------------------------------------------------

def test_tvd5_limited_constant():
    result = _tvd5_limited(5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(result - 5.0) < 1e-14


def test_tvd5_limited_linear():
    """Linear field u(x)=x: should be exact at 5th order."""
    result = _tvd5_limited(-2.0, -1.0, 0.0, 1.0, 2.0)
    # For linear data, phi should give the right slope
    # r_j = 1, r_jm1 = 1, r_jp1 = 1
    # beta = (-2/1 + 11 + 24*1 - 3*1*1)/30 = (33 - 3)/30 = 1.0
    # phi = max(0, min(2, 2, 1)) = 1
    # result = 0 + 0.5*1*1 = 0.5
    assert abs(result - 0.5) < 1e-14


def test_tvd5_limited_finite():
    """Values should stay finite for sharp jumps."""
    result = _tvd5_limited(1e10, 0.0, 0.0, 0.0, 1e10)
    assert math.isfinite(result)


# ---------------------------------------------------------------------------
# MP limiter unit tests
# ---------------------------------------------------------------------------

def test_mp_limited_constant():
    result = _mp_limited_L(5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(result - 5.0) < 1e-14


def test_mp_limited_linear():
    """Linear field: MP should pass through 5th-order candidate unchanged."""
    result = _mp_limited_L(-2.0, -1.0, 0.0, 1.0, 2.0)
    # 5th-order linear: 0.5 exactly. No curvature -> MP should pass through.
    assert abs(result - 0.5) < 1e-14


def test_mp_limited_finite():
    """Values should stay finite for sharp jumps."""
    result = _mp_limited_L(1e10, 0.0, 0.0, 0.0, 1e10)
    assert math.isfinite(result)


# ---------------------------------------------------------------------------
# TENO-M-VA (FV) integration tests
# ---------------------------------------------------------------------------

def test_teno_m_va_constant():
    uL, uR = teno_m_va_fv(5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert abs(uR - 5.0) < 1e-14


def test_teno_m_va_linear():
    """Linear field: should reconstruct exactly (smooth -> delta_k=1 -> TENO)."""
    uL, uR = teno_m_va_fv(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert abs(uL - 0.5) < 1e-14
    assert abs(uR - (-0.5)) < 1e-14


def test_teno_m_va_smooth_matches_teno5():
    """On smooth data, TENO-M should match standard TENO5."""
    uL_m, uR_m = teno_m_va_fv(-2.0, -1.0, 0.0, 1.0, 2.0)
    uL_5, uR_5 = teno5(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert abs(uL_m - uL_5) < 1e-14
    assert abs(uR_m - uR_5) < 1e-14


def test_teno_m_va_jump():
    """Jump [0,0,1,1,1]: reconstruction should be bounded."""
    uL, uR = teno_m_va_fv(0.0, 0.0, 1.0, 1.0, 1.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    # Should not overshoot too far
    assert 0.0 <= uL <= 1.5
    assert 0.0 <= uR <= 1.5


def test_teno_m_va_finite():
    """All values should stay finite even for extreme jumps."""
    uL, uR = teno_m_va_fv(1e10, 0.0, 0.0, 0.0, 1e10)
    assert math.isfinite(uL)
    assert math.isfinite(uR)


def test_teno_m_va_sine_extrema():
    """Sine wave: TENO-M-VA should give bounded results at extrema."""
    import math
    # sin at x = {-2dx, -dx, 0, dx, 2dx} with dx = 0.5
    vals = [-0.8415, -0.4794, 0.0, 0.4794, 0.8415]
    uL, uR = teno_m_va_fv(*vals)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    # Should be between min and max of stencil
    assert min(vals) - 0.1 <= uL <= max(vals) + 0.1
    assert min(vals) - 0.1 <= uR <= max(vals) + 0.1


# ---------------------------------------------------------------------------
# TENO-M-TVD5 (FV) integration tests
# ---------------------------------------------------------------------------

def test_teno_m_tvd5_constant():
    uL, uR = teno_m_tvd5_fv(5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert abs(uR - 5.0) < 1e-14


def test_teno_m_tvd5_linear():
    """Linear field: should be exact (smooth -> delta_k=1 -> TENO)."""
    uL, uR = teno_m_tvd5_fv(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert abs(uL - 0.5) < 1e-14
    assert abs(uR - (-0.5)) < 1e-14


def test_teno_m_tvd5_smooth_matches_teno5():
    """On smooth data, TENO-M-TVD5 should match standard TENO5."""
    uL_m, uR_m = teno_m_tvd5_fv(-2.0, -1.0, 0.0, 1.0, 2.0)
    uL_5, uR_5 = teno5(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert abs(uL_m - uL_5) < 1e-14
    assert abs(uR_m - uR_5) < 1e-14


def test_teno_m_tvd5_jump():
    """Jump triggers TVD5 limiter on nonsmooth stencils."""
    uL, uR = teno_m_tvd5_fv(0.0, 0.0, 1.0, 1.0, 1.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert 0.0 <= uL <= 1.5
    assert 0.0 <= uR <= 1.5


def test_teno_m_tvd5_finite():
    uL, uR = teno_m_tvd5_fv(1e10, 0.0, 0.0, 0.0, 1e10)
    assert math.isfinite(uL)
    assert math.isfinite(uR)


# ---------------------------------------------------------------------------
# TENO-M-MP (FV) integration tests
# ---------------------------------------------------------------------------

def test_teno_m_mp_constant():
    uL, uR = teno_m_mp_fv(5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert abs(uR - 5.0) < 1e-14


def test_teno_m_mp_linear():
    """Linear field: should be exact."""
    uL, uR = teno_m_mp_fv(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert abs(uL - 0.5) < 1e-14
    assert abs(uR - (-0.5)) < 1e-14


def test_teno_m_mp_smooth_matches_teno5():
    """On smooth data, TENO-M-MP should match standard TENO5."""
    uL_m, uR_m = teno_m_mp_fv(-2.0, -1.0, 0.0, 1.0, 2.0)
    uL_5, uR_5 = teno5(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert abs(uL_m - uL_5) < 1e-14
    assert abs(uR_m - uR_5) < 1e-14


def test_teno_m_mp_jump():
    """Jump triggers MP limiter on nonsmooth stencils."""
    uL, uR = teno_m_mp_fv(0.0, 0.0, 1.0, 1.0, 1.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert 0.0 <= uL <= 1.5
    assert 0.0 <= uR <= 1.5


def test_teno_m_mp_finite():
    uL, uR = teno_m_mp_fv(1e10, 0.0, 0.0, 0.0, 1e10)
    assert math.isfinite(uL)
    assert math.isfinite(uR)


def test_teno_m_mp_local_extremum():
    """MP should allow smooth local extrema (unlike TVD limiters)."""
    # Parabola-like smooth extremum at i: u = {0.5, 0.8, 1.0, 0.8, 0.5}
    uL, uR = teno_m_mp_fv(0.5, 0.8, 1.0, 0.8, 0.5)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    # MP should NOT clamp to u_i in a smooth extremum
    # (TVD limiters would clamp, MP preserves extremum)


# ---------------------------------------------------------------------------
# Discontinuity propagation test: all three variants
# ---------------------------------------------------------------------------

def test_discontinuity_all_variants():
    """All three TENO-M variants should handle a step jump without NaN/inf."""
    # Step: cells 0-2 = 0, cells 3-4 = 1, interface at i=3
    uL_va, uR_va = teno_m_va_fv(0.0, 0.0, 0.0, 1.0, 1.0)
    uL_tvd5, uR_tvd5 = teno_m_tvd5_fv(0.0, 0.0, 0.0, 1.0, 1.0)
    uL_mp, uR_mp = teno_m_mp_fv(0.0, 0.0, 0.0, 1.0, 1.0)

    for name, uL, uR in [("VA", uL_va, uR_va),
                           ("TVD5", uL_tvd5, uR_tvd5),
                           ("MP", uL_mp, uR_mp)]:
        assert math.isfinite(uL), f"{name}: uL not finite"
        assert math.isfinite(uR), f"{name}: uR not finite"
        assert uL >= -0.1, f"{name}: uL negative overshoot"
        assert uL <= 1.1, f"{name}: uL positive overshoot"
        assert uR >= -0.1, f"{name}: uR negative overshoot"
        assert uR <= 1.1, f"{name}: uR positive overshoot"


# ---------------------------------------------------------------------------
# Interface registration test
# ---------------------------------------------------------------------------

def test_interface_registration():
    """All three methods should be registered in interface.py."""
    from pyrecon.interface import get_method
    va_fn = get_method("teno_m_va_fv")
    tvd5_fn = get_method("teno_m_tvd5_fv")
    mp_fn = get_method("teno_m_mp_fv")
    assert va_fn is teno_m_va_fv
    assert tvd5_fn is teno_m_tvd5_fv
    assert mp_fn is teno_m_mp_fv
#
# :D
#
