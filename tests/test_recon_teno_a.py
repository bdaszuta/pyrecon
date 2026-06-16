"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Tests for TENO-A reconstruction methods
"""
import math
from pyrecon.recon_teno_a import (
    teno_a_fv as teno_a, teno_a_pw,
    _scale_sensor_m, _adaptive_CT,
    _teno_a_cutoff,
)


# ---------------------------------------------------------------------------
# Scale sensor tests
# ---------------------------------------------------------------------------

def test_scale_sensor_m_constant():
    """Constant field: all b_k = 0 -> ratio ~ 1 -> m ~ 1"""
    m = _scale_sensor_m(0.0, 0.0, 0.0)
    assert 0.99 <= m <= 1.01, f"m should be ~1 for constant field, got {m}"


def test_scale_sensor_m_smooth_linear():
    """Linear field: b_k all equal but non-zero -> m ~ 1"""
    # For linear data, all smoothness indicators equal ~ 1
    m = _scale_sensor_m(1.0, 1.0, 1.0)
    assert 0.99 <= m <= 1.01, f"m should be ~1 for smooth linear field, got {m}"


def test_scale_sensor_m_discontinuity():
    """Discontinuity: large ratio -> m ~ 0"""
    m = _scale_sensor_m(1e-10, 1.0, 1.0)
    assert m < 0.1, f"m should be near 0 for discontinuity, got {m}"


def test_scale_sensor_m_range():
    """m should always be in [0, 1]"""
    # Test with various values
    test_cases = [
        (0.0, 0.0, 0.0),
        (1e-10, 1e-5, 1e-10),
        (0.0, 1.0, 100.0),
        (1000.0, 1.0, 0.0),
        (1e10, 1e10, 1e-20),
    ]
    for b0, b1, b2 in test_cases:
        m = _scale_sensor_m(b0, b1, b2)
        assert 0.0 <= m <= 1.0, f"m={m} out of range for ({b0},{b1},{b2})"


# ---------------------------------------------------------------------------
# Adaptive CT tests
# ---------------------------------------------------------------------------

def test_adaptive_CT_smooth():
    """Smooth flow (m=1): CT should be at minimum."""
    CT = _adaptive_CT(1.0)
    # CT_MIN is 1e-7
    assert CT < 1e-5, f"CT should be small for smooth flow, got {CT}"


def test_adaptive_CT_discontinuity():
    """Discontinuity (m=0): CT should be at maximum."""
    CT = _adaptive_CT(0.0)
    # CT_MAX is 1e-4
    assert CT > 1e-5, f"CT should be large for discontinuity, got {CT}"


def test_adaptive_CT_monotonic():
    """CT should be monotonic decreasing with m."""
    ct_prev = _adaptive_CT(0.0)
    for m in [0.2, 0.4, 0.6, 0.8, 1.0]:
        ct = _adaptive_CT(m)
        assert ct <= ct_prev + 1e-15, (
            f"CT not monotonic: m={m}, ct={ct}, prev={ct_prev}")
        ct_prev = ct


# ---------------------------------------------------------------------------
# TENO-A cutoff tests
# ---------------------------------------------------------------------------

def test_teno_a_cutoff_constant():
    """Constant field with small CT: all stencils should pass."""
    d0, d1, d2 = _teno_a_cutoff(0.0, 0.0, 0.0, 1e-6)
    assert d0 == 1.0
    assert d1 == 1.0
    assert d2 == 1.0


def test_teno_a_cutoff_discontinuity():
    """Discontinuity with large CT: only smooth stencil passes."""
    d0, d1, d2 = _teno_a_cutoff(1e-10, 1.0, 1.0, 1e-4)
    assert d0 == 1.0
    assert d1 == 0.0
    assert d2 == 0.0


# ---------------------------------------------------------------------------
# TENO-A public API (FV)
# ---------------------------------------------------------------------------

def test_teno_a_constant():
    uL, uR = teno_a(5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert abs(uR - 5.0) < 1e-14


def test_teno_a_linear():
    """Linear field: TENO-A should be exact."""
    uL, uR = teno_a(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert abs(uL - 0.5) < 1e-14
    assert abs(uR - (-0.5)) < 1e-14


def test_teno_a_jump():
    """Jump at i: should be bounded."""
    uL, uR = teno_a(0.0, 0.0, 1.0, 1.0, 1.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert 0.0 <= uL <= 1.5
    assert 0.0 <= uR <= 1.5


def test_teno_a_finite():
    """Extreme values should remain finite."""
    uL, uR = teno_a(1e10, 0.0, 0.0, 0.0, 1e10)
    assert math.isfinite(uL)
    assert math.isfinite(uR)


# ---------------------------------------------------------------------------
# TENO-A public API (PW)
# ---------------------------------------------------------------------------

def test_teno_a_pw_constant():
    uL, uR = teno_a_pw(5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert abs(uR - 5.0) < 1e-14


def test_teno_a_pw_linear():
    uL, uR = teno_a_pw(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert abs(uL - 0.5) < 1e-14
    assert abs(uR - (-0.5)) < 1e-14


def test_teno_a_pw_jump():
    uL, uR = teno_a_pw(0.0, 0.0, 1.0, 1.0, 1.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
#
# :D
#
