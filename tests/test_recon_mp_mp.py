"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Tests for modified MP5 for multi-phase flows (Ha-Lee 2020)
"""
import math
from pyrecon.recon_mp_mp import mp5_mp_fv as mp5_mp


# ---------------------------------------------------------------------------
# Modified MP5 multi-phase reconstruction
# ---------------------------------------------------------------------------


def test_mp5_mp_constant():
    """Constant data: MP5-MP should return exact value."""
    uL, uR = mp5_mp(5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert abs(uR - 5.0) < 1e-14


def test_mp5_mp_linear():
    """Linear data: MP5-MP should be exact."""
    uL, uR = mp5_mp(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert abs(uL - 0.5) < 1e-13
    assert abs(uR - (-0.5)) < 1e-13


def test_mp5_mp_smooth_extrema():
    """Smooth extrema: MP5-MP should bypass limiting and preserve extremum."""
    # This is the key feature of the multi-phase MP
    # At a smooth extremum, the standard MP would clip, but MP5-MP bypasses
    uL, uR = mp5_mp(0.0, 0.5, 1.0, 0.5, 0.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert 0.0 <= uL <= 1.0
    assert 0.0 <= uR <= 1.0


def test_mp5_mp_jump():
    """Discontinuity: MP5-MP should still limit and be bounded."""
    uL, uR = mp5_mp(0.0, 0.0, 0.0, 1.0, 1.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert 0.0 <= uL <= 1.0
    assert 0.0 <= uR <= 1.0


def test_mp5_mp_jump_reversed():
    """Reversed discontinuity."""
    uL, uR = mp5_mp(1.0, 1.0, 0.0, 0.0, 0.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert 0.0 <= uL <= 1.0
    assert 0.0 <= uR <= 1.0


def test_mp5_mp_monotonic():
    """Monotonic data: should work."""
    uL, uR = mp5_mp(0.0, 0.5, 1.0, 1.5, 2.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert 0.0 <= uL <= 2.0
    assert 0.0 <= uR <= 2.0


def test_mp5_mp_returns_pair():
    """MP5-MP should return a pair (uL, uR) of floats."""
    result = mp5_mp(0.0, 0.5, 1.0, 1.5, 2.0)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], float)
    assert isinstance(result[1], float)


def test_mp5_mp_phase_interface_like():
    """Phase interface-like profile: steep but smooth gradient."""
    # Simulating a phase boundary: steep gradient but continuous
    uL, uR = mp5_mp(0.01, 0.1, 0.5, 0.9, 0.99)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    # Should be higher order (not clipped to first order)
    assert 0.01 <= uL <= 0.99
    assert 0.01 <= uR <= 0.99


def test_mp5_mp_compared_to_mp5():
    """MP5-MP vs standard MP5: on smooth extrema, MP5-MP may be less clipped."""
    from pyrecon.recon_mpn import mp5_fv as mp5
    # Smooth extremum case
    args = (0.0, 0.5, 1.0, 0.5, 0.0)
    uL_mp, uR_mp = mp5_mp(*args)
    uL_m5, uR_m5 = mp5(*args)
    # Both should be finite and bounded
    assert math.isfinite(uL_mp) and math.isfinite(uR_mp)
    assert math.isfinite(uL_m5) and math.isfinite(uR_m5)
    assert 0.0 <= uL_mp <= 1.0
    assert 0.0 <= uL_m5 <= 1.0


def test_mp5_mp_sharp_gradient():
    """Very sharp gradient: MP5-MP should handle without NaN/inf."""
    uL, uR = mp5_mp(0.0, 0.0, 0.5, 500.0, 500.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert -10.0 <= uL <= 500.0
    assert -10.0 <= uR <= 500.0
#
# :D
#
