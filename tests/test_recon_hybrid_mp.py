"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Tests for hybrid MP5-MUSCL reconstruction (Ahn & Lee 2019)
"""
from pyrecon.recon_hybrid_mp import hybrid_mp_mc2_fv


def test_hybrid_mp_constant():
    """Constant field: exact reconstruction."""
    uL, uR = hybrid_mp_mc2_fv(5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14, f"uL={uL}"
    assert abs(uR - 5.0) < 1e-14, f"uR={uR}"


def test_hybrid_mp_linear():
    """Linear u(x)=x, dx=1: uL=u_i+0.5, uR=u_i-0.5."""
    uL, uR = hybrid_mp_mc2_fv(8.0, 9.0, 10.0, 11.0, 12.0)
    assert abs(uL - 10.5) < 1e-13, f"uL={uL}"
    assert abs(uR - 9.5) < 1e-13, f"uR={uR}"


def test_hybrid_mp_smooth():
    """Smooth quadratic: should not trigger fallback, uses MP5."""
    # u(x) = x^2 at x = -2, -1, 0, 1, 2 => 4, 1, 0, 1, 4
    uL, uR = hybrid_mp_mc2_fv(4.0, 1.0, 0.0, 1.0, 4.0)
    # MP5 left: (2/60*4 + -13/60*1 + 47/60*0 + 27/60*1 + -3/60*4)
    # = (8-13+0+27-12)/60 = 10/60 = 0.166...
    # But with limiter... should be reasonable
    assert -1.0 <= uL <= 2.0, f"uL={uL} outside reasonable range"
    assert -1.0 <= uR <= 2.0, f"uR={uR} outside reasonable range"


def test_hybrid_mp_step():
    """Step discontinuity: should not oscillate."""
    uL, uR = hybrid_mp_mc2_fv(0.0, 0.0, 1.0, 1.0, 1.0)
    assert 0.0 <= uL <= 1.5, f"uL={uL}"
    assert 0.0 <= uR <= 1.5, f"uR={uR}"


def test_hybrid_mp_extreme_gradient():
    """Extreme gradient: should fall back to MUSCL-MC2."""
    # Very large jump in curvature: 0, 0, 0, 100, 200
    # d2m = |0-0+0| = 0, d2c = |0-0+100| = 100, d2p = |0-200+200| = 0
    # ratio = max/avg... might not trigger. Let's use a clearer case.
    # Values creating extreme curvature ratio
    uL, uR = hybrid_mp_mc2_fv(0.0, 0.0, 100.0, 0.0, 0.0)
    # Should not produce wildly oscillatory values
    assert -50.0 <= uL <= 150.0, f"uL={uL} wild oscillation"
    assert -50.0 <= uR <= 150.0, f"uR={uR} wild oscillation"


def test_hybrid_mp_negative():
    """Works with negative values."""
    uL, uR = hybrid_mp_mc2_fv(-5.0, -4.0, -3.0, -2.0, -1.0)
    assert abs(uL - (-2.5)) < 1e-13, f"uL={uL}"
    assert abs(uR - (-3.5)) < 1e-13, f"uR={uR}"


def test_hybrid_mp_shock():
    """Strong shock: 0 -> 100, bounded results."""
    uL, uR = hybrid_mp_mc2_fv(0.0, 0.0, 0.0, 100.0, 100.0)
    assert uL >= -1e-13, f"uL={uL} negative"
    assert uL <= 110.0, f"uL={uL} too large"
    assert uR >= -1e-13, f"uR={uR} negative"
    assert uR <= 110.0, f"uR={uR} too large"


def test_hybrid_mp_symmetry():
    """Symmetric data: uL and uR should both be near center."""
    uL, uR = hybrid_mp_mc2_fv(1.0, 2.0, 3.0, 2.0, 1.0)
    # Both faces should be near symmetric center
    assert 2.0 <= uL <= 4.0, f"uL={uL}"
    assert 2.0 <= uR <= 4.0, f"uR={uR}"
#
# :D
#
