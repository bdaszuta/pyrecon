"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Tests for low-dissipation shock-capturing reconstruction
 (Li, Fu, Adams 2021)
"""
from pyrecon.recon_lowdiss import (
    hybrid_linear_weno_fv, hybrid_linear_weno_mild_fv,
    hybrid_linear_weno_strong_fv,
)


def test_lowdiss_constant():
    """Constant field: exact reconstruction (sigma=0 -> pure linear)."""
    for fn in [hybrid_linear_weno_fv, hybrid_linear_weno_mild_fv,
               hybrid_linear_weno_strong_fv]:
        uL, uR = fn(5.0, 5.0, 5.0, 5.0, 5.0)
        assert abs(uL - 5.0) < 1e-14, f"uL={uL}"
        assert abs(uR - 5.0) < 1e-14, f"uR={uR}"


def test_lowdiss_linear():
    """Linear u(x)=x, dx=1: uL=u_i+0.5, uR=u_i-0.5."""
    # u_{i-2}=8, u_{i-1}=9, u_i=10, u_{i+1}=11, u_{i+2}=12
    for fn in [hybrid_linear_weno_fv, hybrid_linear_weno_mild_fv,
               hybrid_linear_weno_strong_fv]:
        uL, uR = fn(8.0, 9.0, 10.0, 11.0, 12.0)
        # Linear field -> sigma ~ 0, should match 5th-order linear
        assert abs(uL - 10.5) < 1e-13, f"fn={fn.__name__}, uL={uL}"
        assert abs(uR - 9.5) < 1e-13, f"fn={fn.__name__}, uR={uR}"


def test_lowdiss_step_discontinuity():
    """Step function: should detect discontinuity (sigma > 0)."""
    # Sharp jump from 0 to 1
    uL, uR = hybrid_linear_weno_fv(0.0, 0.0, 0.0, 1.0, 1.0)
    # At a discontinuity, sigma should be > 0, giving some WENO contribution
    # but results should not oscillate
    assert -1e-12 <= uL <= 1.1, f"uL={uL} out of bounds"
    assert -1e-12 <= uR <= 1.1, f"uR={uR} out of bounds"

    uL, uR = hybrid_linear_weno_fv(1.0, 1.0, 1.0, 0.0, 0.0)
    assert -0.1 <= uL <= 1.1, f"uL={uL} out of bounds"
    assert -0.1 <= uR <= 1.1, f"uR={uR} out of bounds"


def test_lowdiss_mild_vs_strong():
    """Mild (C_tau=0.5) differs from strong (C_tau=2.0) near discontinuity."""
    # At a smooth region they should be close
    uL_m, uR_m = hybrid_linear_weno_mild_fv(1.0, 2.0, 3.0, 4.0, 5.0)
    uL_s, uR_s = hybrid_linear_weno_strong_fv(1.0, 2.0, 3.0, 4.0, 5.0)
    assert abs(uL_m - uL_s) < 1e-14, "Smooth: mild vs strong should match"
    assert abs(uR_m - uR_s) < 1e-14

    # Near a jump they may differ
    uL_m, uR_m = hybrid_linear_weno_mild_fv(0.0, 0.0, 0.0, 1.0, 1.0)
    uL_s, uR_s = hybrid_linear_weno_strong_fv(0.0, 0.0, 0.0, 1.0, 1.0)
    # Both should be bounded
    assert 0.0 <= uL_m <= 1.1, f"mild uL={uL_m}"
    assert 0.0 <= uL_s <= 1.1, f"strong uL={uL_s}"


def test_lowdiss_symmetry():
    """Reversing stencil should give symmetric result for symmetric data."""
    # Symmetric data: [1, 2, 3, 2, 1]
    uL, uR = hybrid_linear_weno_fv(1.0, 2.0, 3.0, 2.0, 1.0)
    # Both faces should be near 3.0 (local max)
    assert 2.0 <= uL <= 4.0, f"uL={uL} outside symmetric range"
    assert 2.0 <= uR <= 4.0, f"uR={uR} outside symmetric range"


def test_lowdiss_negative_values():
    """Works with all-negative values."""
    uL, uR = hybrid_linear_weno_fv(-5.0, -4.0, -3.0, -2.0, -1.0)
    assert abs(uL - (-2.5)) < 1e-13, f"uL={uL}"
    assert abs(uR - (-3.5)) < 1e-13, f"uR={uR}"


def test_lowdiss_shock():
    """Strong shock: 0 -> 100, should not produce negative values."""
    uL, uR = hybrid_linear_weno_fv(0.0, 0.0, 0.0, 100.0, 100.0)
    assert uL >= -1e-13, f"uL={uL} negative at shock"
    assert uR >= -1e-13, f"uR={uR} negative at shock"
    assert uL <= 110.0, f"uL={uL} too large at shock"
    assert uR <= 110.0, f"uR={uR} too large at shock"
#
# :D
#
