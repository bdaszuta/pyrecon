"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Tests for geno reconstruction method.
"""
import math
from pyrecon.recon_geno import geno5_fv


# ---------------------------------------------------------------------------
# Constant field
# ---------------------------------------------------------------------------

def test_geno5_constant():
    """Constant field: exact reconstruction regardless of chi."""
    uL, uR = geno5_fv(5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14, f"uL={uL}"
    assert abs(uR - 5.0) < 1e-14, f"uR={uR}"


# ---------------------------------------------------------------------------
# Linear field
# ---------------------------------------------------------------------------

def test_geno5_linear():
    """Linear u(x)=x, dx=1: smooth -> chi~1, should give 5th-order central.

    For u(x)=x with x in {-2,-1,0,1,2}:
      q_H = (2*(-2) - 13*(-1) + 47*0 + 27*1 - 3*2) / 60
          = (-4 + 13 + 0 + 27 - 6) / 60 = 30/60 = 0.5
    So uL = 0.5, uR = -0.5.
    """
    uL, uR = geno5_fv(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert abs(uL - 0.5) < 1e-13, f"uL={uL}"
    assert abs(uR - (-0.5)) < 1e-13, f"uR={uR}"


def test_geno5_linear_shifted():
    """Linear field shifted: same slope, different offset.

    u(x)=x+100, same result offset.
    """
    uL, uR = geno5_fv(98.0, 99.0, 100.0, 101.0, 102.0)
    assert abs(uL - 100.5) < 1e-13, f"uL={uL}"
    assert abs(uR - 99.5) < 1e-13, f"uR={uR}"


# ---------------------------------------------------------------------------
# Quadratic field
# ---------------------------------------------------------------------------

def test_geno5_quadratic():
    """Quadratic u(x)=x^2: x in {-2,-1,0,1,2}.

    q_H = (2*4 - 13*1 + 47*0 + 27*1 - 3*4) / 60
        = (8 - 13 + 0 + 27 - 12) / 60 = 10/60 = 1/6
    All stencil polynomials also give 1/6 for quadratic.
    """
    uL, uR = geno5_fv(4.0, 1.0, 0.0, 1.0, 4.0)
    assert abs(uL - 1.0 / 6.0) < 1e-13, f"uL={uL}"
    assert abs(uR - 1.0 / 6.0) < 1e-13, f"uR={uR}"


# ---------------------------------------------------------------------------
# Step discontinuity
# ---------------------------------------------------------------------------

def test_geno5_step_left():
    """Step at i=0: [0,0,1,1,1]. Face i+1/2 is between two 1-cells -> uL~1."""
    uL, uR = geno5_fv(0.0, 0.0, 1.0, 1.0, 1.0)
    # Left face is in smooth region (1,1) -> should be very close to 1
    assert abs(uL - 1.0) < 0.1, f"uL={uL} deviates from 1.0"
    # Reversed stencil [1,1,1,0,0]: face in smooth region -> close to 1
    assert abs(uR - 1.0) < 0.1, f"uR={uR} deviates from 1.0"


def test_geno5_step_right():
    """Step at i+1: [1,1,1,0,0]. Both faces should be well-behaved."""
    uL, uR = geno5_fv(1.0, 1.0, 1.0, 0.0, 0.0)
    assert math.isfinite(uL), f"uL={uL} not finite"
    assert math.isfinite(uR), f"uR={uR} not finite"
    # Values should be within or near the data range [0, 1]
    assert -0.2 <= uL <= 1.2, f"uL={uL} out of bounds"
    assert -0.2 <= uR <= 1.2, f"uR={uR} out of bounds"


def test_geno5_step_center():
    """Centered step: [0,0,0,1,1]."""
    uL, uR = geno5_fv(0.0, 0.0, 0.0, 1.0, 1.0)
    # Both faces should be finite and within reasonable range
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert -0.2 <= uL <= 1.2, f"uL={uL}"
    assert -0.2 <= uR <= 1.2, f"uR={uR}"


# ---------------------------------------------------------------------------
# Symmetry
# ---------------------------------------------------------------------------

def test_geno5_symmetry():
    """Symmetric data: [1,2,3,2,1] -> both faces should be near 3."""
    uL, uR = geno5_fv(1.0, 2.0, 3.0, 2.0, 1.0)
    assert 2.0 <= uL <= 4.0, f"uL={uL} outside symmetric range"
    assert 2.0 <= uR <= 4.0, f"uR={uR} outside symmetric range"


# ---------------------------------------------------------------------------
# Negative values
# ---------------------------------------------------------------------------

def test_geno5_negative_linear():
    """Negative linear field."""
    uL, uR = geno5_fv(-5.0, -4.0, -3.0, -2.0, -1.0)
    assert abs(uL - (-2.5)) < 1e-13, f"uL={uL}"
    assert abs(uR - (-3.5)) < 1e-13, f"uR={uR}"


# ---------------------------------------------------------------------------
# Strong shock
# ---------------------------------------------------------------------------

def test_geno5_strong_shock():
    """Strong shock: 0 -> 100, should not produce negative values."""
    uL, uR = geno5_fv(0.0, 0.0, 0.0, 100.0, 100.0)
    assert uL >= -1e-12, f"uL={uL} negative at shock"
    assert uR >= -1e-12, f"uR={uR} negative at shock"
    assert uL <= 110.0, f"uL={uL} too large at shock"
    assert uR <= 110.0, f"uR={uR} too large at shock"


# ---------------------------------------------------------------------------
# D-weight centering (Section 3.3: central sub-stencil gets C_0=8 boost)
# ---------------------------------------------------------------------------

def test_geno5_eno_center_boost():
    """ENO reconstruction: D-boost must go to center stencil (Paper 3.3).

    The paper specifies that p_1 (the central sub-stencil) receives the
    boosted weight d_1 = C_0 = 8, while all other sub-stencils get d_k = 1.
    On the 5-point stencil, stencil index 1 (st1) is always the physical
    center stencil, regardless of face orientation.
    """
    from pyrecon.recon_geno import _eno_reconstruction

    # Left, center, right stencil values all distinct
    st_left = 0.0
    st_center = 10.0
    st_right = 0.0

    # Equal smoothness: D-weights are the only discriminator
    b = 1.0  # nonzero to avoid EPS floor effects

    eno = _eno_reconstruction(st_left, st_center, st_right, b, b, b)

    # Left + right weights = 1+1=2, center weight = 8, total = 10
    # Expected: (1*st_left + 8*st_center + 1*st_right) / 10 = 8.0
    expected_center_boost = 8.0
    assert abs(eno - expected_center_boost) < 1e-13, (
        f"ENO={eno}, expected {expected_center_boost} "
        f"(center stencil must get D=8 boost per paper Section 3.3)"
    )


def test_geno5_eno_center_boost_reversed():
    """ENO D-boost stays on the center stencil after argument reversal.

    When called with reversed arguments for the right face, the center
    stencil is still at index position 1 in the stencil order.
    """
    from pyrecon.recon_geno import _eno_reconstruction

    # Simulate right-face call: reversed physical stencil positions.
    # st0 = right stencil, st1 = center stencil, st2 = left stencil
    st_right = 0.0
    st_center = 10.0
    st_left = 0.0

    # Smoothness indicators are also swapped by _js_smoothness under
    # reversal, but with equal values it doesn't matter.
    b = 1.0

    eno = _eno_reconstruction(st_right, st_center, st_left, b, b, b)

    # The center stencil (index 1) should still get the D=8 boost.
    expected = 8.0
    assert abs(eno - expected) < 1e-13, (
        f"ENO={eno}, expected {expected} "
        f"(after reversal, center stencil at index 1 still needs D=8)"
    )
#
# :D
#
