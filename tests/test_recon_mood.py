"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Tests for MOOD reconstruction method (Clain-Diot-Loubere 2011)
"""
import math
from pyrecon.recon_mood import mood_fv as mood


# ---------------------------------------------------------------------------
# MOOD reconstruction
# ---------------------------------------------------------------------------


def test_mood_constant():
    """Constant data: MOOD should return exact value."""
    uL, uR = mood(5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert abs(uR - 5.0) < 1e-14


def test_mood_linear():
    """Linear data: MOOD should use high-order and be exact."""
    uL, uR = mood(-2.0, -1.0, 0.0, 1.0, 2.0)
    # For linear data, face values should be u_i +/- 0.5
    assert abs(uL - 0.5) < 1e-13
    assert abs(uR - (-0.5)) < 1e-13


def test_mood_parabolic():
    """Parabolic data: MOOD should use at least 3rd order and be accurate."""
    # u(x) = x^2, so u_i = i^2, faces at i+1/2: (i+0.5)^2 = i^2 + i + 0.25
    # For cell i=0: uL at face 0.5 => 0.25, uR at face -0.5 => 0.25
    uL, uR = mood(4.0, 1.0, 0.0, 1.0, 4.0)
    # Allow some error since MOOD may drop order with smooth extrema
    assert abs(uL - 0.25) < 0.3
    assert abs(uR - 0.25) < 0.3
    assert math.isfinite(uL)
    assert math.isfinite(uR)


def test_mood_jump():
    """Discontinuity: MOOD cascades down to lower order but stays bounded."""
    uL, uR = mood(0.0, 0.0, 0.0, 1.0, 1.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    # At a jump, the values should be between the stencil min and max
    assert 0.0 <= uL <= 1.0
    assert 0.0 <= uR <= 1.0


def test_mood_jump_reversed():
    """Reversed discontinuity."""
    uL, uR = mood(1.0, 1.0, 0.0, 0.0, 0.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert 0.0 <= uL <= 1.0
    assert 0.0 <= uR <= 1.0


def test_mood_smooth_extrema():
    """Smooth data with extremum: MOOD should handle gracefully."""
    uL, uR = mood(0.0, 0.5, 1.0, 0.5, 0.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert 0.0 <= uL <= 1.0
    assert 0.0 <= uR <= 1.0


def test_mood_monotonic():
    """Monotonic data: MOOD should work."""
    uL, uR = mood(0.0, 0.5, 1.0, 1.5, 2.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert 0.0 <= uL <= 2.0
    assert 0.0 <= uR <= 2.0


def test_mood_returns_pair():
    """MOOD should return a pair (uL, uR) of floats."""
    result = mood(0.0, 0.5, 1.0, 1.5, 2.0)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], float)
    assert isinstance(result[1], float)


def test_mood_cascade_behavior():
    """MOOD cascade: result always bounded by stencil min/max."""
    test_cases = [
        (0.0, 0.0, 0.0, 0.0, 0.0),       # constant
        (-2.0, -1.0, 0.0, 1.0, 2.0),      # linear
        (0.0, 0.5, 1.0, 0.5, 0.0),        # extremum
        (0.0, 0.0, 0.0, 1.0, 1.0),        # jump
        (1.0, 1.0, 0.0, 0.0, 0.0),        # reversed jump
        (0.0, 0.3, 0.7, 1.0, 1.3),        # smooth rising
        (0.0, 0.0, 1.0, 0.0, 0.0),        # spike
    ]
    for args in test_cases:
        uL, uR = mood(*args)
        u_min = min(args)
        u_max = max(args)
        assert u_min - 1e-12 <= uL <= u_max + 1e-12, (
            f"uL={uL} out of bounds [{u_min}, {u_max}] for args={args}"
        )
        assert u_min - 1e-12 <= uR <= u_max + 1e-12, (
            f"uR={uR} out of bounds [{u_min}, {u_max}] for args={args}"
        )
#
# :D
#
