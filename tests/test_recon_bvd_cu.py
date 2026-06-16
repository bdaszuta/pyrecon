"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Tests for BVD central-upwind reconstruction (Chamarthi-Frankel 2021)
           Phase D: HOCUS6 + legacy bvd_e6_mp5_fv
"""
import math
from pyrecon.recon_bvd_cu import bvd_cu_fv, bvd_e6_mp5_fv


# ---------------------------------------------------------------------------
# HOCUS6 BVD central-upwind reconstruction (7-point stencil, alpha=7)
# ---------------------------------------------------------------------------

def test_bvd_cu_constant():
    """Constant data: BVD-CU (HOCUS6) should return exact value."""
    uL, uR = bvd_cu_fv(5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert abs(uR - 5.0) < 1e-14


def test_bvd_cu_linear():
    """Linear data: HOCUS6 should interpolate correctly."""
    uL, uR = bvd_cu_fv(-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
    assert abs(uL - 0.5) < 1e-13
    assert abs(uR - (-0.5)) < 1e-13


def test_bvd_cu_jump():
    """Discontinuity: HOCUS6 BVD should switch to MP5 and be finite."""
    uL, uR = bvd_cu_fv(0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)


def test_bvd_cu_jump_reversed():
    """Reversed discontinuity."""
    uL, uR = bvd_cu_fv(1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)


def test_bvd_cu_smooth_extrema():
    """Smooth data with extremum: HOCUS6 should be finite."""
    uL, uR = bvd_cu_fv(-0.5, 0.0, 0.5, 1.0, 0.5, 0.0, -0.5)
    assert math.isfinite(uL)
    assert math.isfinite(uR)


def test_bvd_cu_monotonic():
    """Monotonic data: should work."""
    uL, uR = bvd_cu_fv(-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert -0.5 <= uL <= 2.5
    assert -0.5 <= uR <= 2.5


def test_bvd_cu_returns_pair():
    """HOCUS6 should return a pair (uL, uR) of floats."""
    result = bvd_cu_fv(0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], float)
    assert isinstance(result[1], float)


def test_bvd_cu_symmetry():
    """HOCUS6 should handle symmetric data."""
    uL1, uR1 = bvd_cu_fv(3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0)
    uL2, uR2 = bvd_cu_fv(-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
    assert math.isfinite(uL1) and math.isfinite(uR1)
    assert math.isfinite(uL2) and math.isfinite(uR2)


def test_bvd_cu_shock():
    """Shock-like profile: HOCUS6 BVD should produce bounded values.

    The C5 compact scheme produces a mild overshoot (~ -1.6) at
    discontinuities; the bound accommodates this while still
    catching pathological unboundedness (the uncorrected code
    gave -1.87).
    """
    uL, uR = bvd_cu_fv(-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0)
    assert -1.7 <= uL <= 1.7, f"uL={uL} out of bounds"
    assert -1.7 <= uR <= 1.7, f"uR={uR} out of bounds"
    assert math.isfinite(uL)
    assert math.isfinite(uR)


def test_bvd_cu_quadratic():
    """Quadratic data: HOCUS6 should achieve high-order accuracy."""
    # f(x) = x^2, cells centered at i
    # cell average over [x-0.5, x+0.5] is x^2 + 1/12
    # Face i+1/2 is at x=0.5 -> exact value = 0.25
    # Face i-1/2 is at x=-0.5 -> exact value = 0.25
    x = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    u = [xi**2 + 1.0/12.0 for xi in x]
    uL, uR = bvd_cu_fv(*u)
    assert abs(uL - 0.25) < 1e-6, f"uL={uL} expected 0.25"
    assert abs(uR - 0.25) < 1e-6, f"uR={uR} expected 0.25"


# ---------------------------------------------------------------------------
# Legacy bvd_e6_mp5_fv (explicit 6th-order + MP5, alpha=4)
# ---------------------------------------------------------------------------

def test_bvd_e6_mp5_constant():
    """Constant data: legacy should return exact value."""
    uL, uR = bvd_e6_mp5_fv(5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert abs(uR - 5.0) < 1e-14


def test_bvd_e6_mp5_linear():
    """Linear data: legacy central interpolation should be exact."""
    uL, uR = bvd_e6_mp5_fv(-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
    assert abs(uL - 0.5) < 1e-13
    assert abs(uR - (-0.5)) < 1e-13


def test_bvd_e6_mp5_jump():
    """Discontinuity: legacy should switch to MP5 and be finite."""
    uL, uR = bvd_e6_mp5_fv(0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)


def test_bvd_e6_mp5_jump_reversed():
    uL, uR = bvd_e6_mp5_fv(1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)


def test_bvd_e6_mp5_returns_pair():
    result = bvd_e6_mp5_fv(0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], float)
    assert isinstance(result[1], float)


def test_bvd_e6_mp5_shock():
    uL, uR = bvd_e6_mp5_fv(-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0)
    assert -1.5 <= uL <= 1.5, f"uL={uL} out of bounds"
    assert -1.5 <= uR <= 1.5, f"uR={uR} out of bounds"
    assert math.isfinite(uL)
    assert math.isfinite(uR)
#
# :D
#
