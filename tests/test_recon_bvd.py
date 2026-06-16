"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Tests for BVD reconstruction method (Sun, Inaba, Xiao 2016)
"""
import math
from pyrecon.recon_bvd import bvd_fv as bvd
from pyrecon.recon_bvd import bvd_tbv_fv as bvd_tbv

# ---------------------------------------------------------------------------
# BVD reconstruction
# ---------------------------------------------------------------------------

def test_bvd_constant():
    """Constant data: BVD should return exact value."""
    uL, uR = bvd(5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert abs(uR - 5.0) < 1e-14


def test_bvd_linear():
    """Linear data: WENO5-Z path should be exact (TBV=0)."""
    uL, uR = bvd(-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
    # WENO5-Z on linear data gives exact face values
    assert abs(uL - 0.5) < 1e-14
    assert abs(uR - (-0.5)) < 1e-14


def test_bvd_jump():
    """Discontinuity: BVD should select THINC and be finite."""
    uL, uR = bvd(0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)


def test_bvd_jump_reversed():
    """Reversed discontinuity."""
    uL, uR = bvd(1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)


def test_bvd_smooth_extrema():
    """Smooth data with extremum: BVD should be finite."""
    uL, uR = bvd(0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)


def test_bvd_monotonic():
    """Monotonic data: should work."""
    uL, uR = bvd(0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert 0.0 <= uL <= 3.0
    assert 0.0 <= uR <= 3.0


def test_bvd_returns_pair():
    """BVD should return a pair (uL, uR) of floats."""
    result = bvd(0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], float)
    assert isinstance(result[1], float)


# ---------------------------------------------------------------------------
# Phase E: 4-way per-interface BVD -- profile tests
# ---------------------------------------------------------------------------

_TOL = 1e-12


def test_bvd_step_up():
    """Step profile: left=0, right=1. BVD should use THINC at the jump."""
    uL, uR = bvd(0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert -_TOL <= uL <= 1.0 + _TOL, f"uL={uL} out of [-eps, 1+eps]"
    assert -_TOL <= uR <= 1.0 + _TOL, f"uR={uR} out of [-eps, 1+eps]"


def test_bvd_step_down():
    """Step profile: left=1, right=0. BVD should use THINC at the jump."""
    uL, uR = bvd(1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert -_TOL <= uL <= 1.0 + _TOL, f"uL={uL} out of [-eps, 1+eps]"
    assert -_TOL <= uR <= 1.0 + _TOL, f"uR={uR} out of [-eps, 1+eps]"


def test_bvd_contact_gentle():
    """Gentle contact: gradual transition 0->1 over 3 cells."""
    uL, uR = bvd(0.0, 0.0, 0.0, 0.25, 0.75, 1.0, 1.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert -_TOL <= uL <= 1.0 + _TOL, f"uL={uL} out of bounds"
    assert -_TOL <= uR <= 1.0 + _TOL, f"uR={uR} out of bounds"


def test_bvd_contact_sharp():
    """Sharp contact: jump 0.1->0.9 in one cell."""
    uL, uR = bvd(0.0, 0.0, 0.1, 0.5, 0.9, 1.0, 1.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert -_TOL <= uL <= 1.0 + _TOL, f"uL={uL} out of bounds"
    assert -_TOL <= uR <= 1.0 + _TOL, f"uR={uR} out of bounds"


def test_bvd_smooth_sine():
    """Smooth sine-like profile: monotonic but curved."""
    uL, uR = bvd(0.0, 0.1, 0.2, 0.5, 0.8, 0.9, 1.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert -_TOL <= uL <= 1.0 + _TOL, f"uL={uL} out of bounds"
    assert -_TOL <= uR <= 1.0 + _TOL, f"uR={uR} out of bounds"


def test_bvd_smooth_gaussian():
    """Smooth hump: Gaussian-like profile with an extremum at cell i."""
    uL, uR = bvd(0.0, 0.2, 0.6, 1.0, 0.6, 0.2, 0.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert 0.5 - _TOL <= uL <= 1.5 + _TOL, f"uL={uL} out of expected range"
    assert 0.5 - _TOL <= uR <= 1.5 + _TOL, f"uR={uR} out of expected range"


def test_bvd_extrema_guard_blocks_thinc():
    """Extrema guard: when (u_{i+1}-u_i)*(u_i-u_{i-1}) <= 0, THINC excluded."""
    uL, uR = bvd(0.0, 0.5, 0.5, 1.0, 0.5, 0.5, 0.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert 0.5 - _TOL <= uL <= 1.5 + _TOL
    assert 0.5 - _TOL <= uR <= 1.5 + _TOL


def test_bvd_four_way_smooth():
    """4-way BVD on smooth linear data: all BV=0 -> WENO wins (tiebreak)."""
    uL, uR = bvd(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert 4.0 - _TOL <= uL <= 6.0 + _TOL
    assert 3.0 - _TOL <= uR <= 5.0 + _TOL


def test_bvd_boundedness():
    """BVD face values should not exceed the input data range."""
    data = [0.0, 0.5, 1.0, 2.0, 1.5, 1.0, 0.5]
    uL, uR = bvd(*data)
    lo = min(data)
    hi = max(data)
    assert lo - _TOL <= uL <= hi + _TOL, f"uL={uL} outside [{lo},{hi}]"
    assert lo - _TOL <= uR <= hi + _TOL, f"uR={uR} outside [{lo},{hi}]"


# ---------------------------------------------------------------------------
# TBV variant tests (backward compatibility)
# ---------------------------------------------------------------------------


def test_bvd_tbv_constant():
    """Constant data: TBV variant should return exact value."""
    uL, uR = bvd_tbv(5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert abs(uR - 5.0) < 1e-14


def test_bvd_tbv_linear():
    """Linear data: TBV variant should match WENO5-Z."""
    uL, uR = bvd_tbv(-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
    assert abs(uL - 0.5) < 1e-14
    assert abs(uR - (-0.5)) < 1e-14


def test_bvd_tbv_jump():
    """Discontinuity: TBV variant should be finite."""
    uL, uR = bvd_tbv(0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
#
# :D
#
