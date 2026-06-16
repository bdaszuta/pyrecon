"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Tests for TENO Hybrid reconstruction methods
"""
import math
from pyrecon.recon_teno_hybrid import (
    teno_hybrid_fv as teno_hybrid,
    teno_hybrid_pw,
    _discontinuity_indicator_sigma,
)


# ---------------------------------------------------------------------------
# Discontinuity indicator tests
# ---------------------------------------------------------------------------

def test_sigma_constant():
    """Constant field: all b_k = 0 -> sigma ~ 1"""
    sigma = _discontinuity_indicator_sigma(0.0, 0.0, 0.0)
    # With all b_k=0, gamma_k all equal (1^6=1), chi_k=1/3, sigma=1
    assert abs(sigma - 1.0) < 1e-12, f"sigma should be ~1, got {sigma}"


def test_sigma_smooth_linear():
    """Linear field: all b_k equal non-zero -> sigma ~ 1"""
    sigma = _discontinuity_indicator_sigma(1.0, 1.0, 1.0)
    assert abs(sigma - 1.0) < 1e-12, (
        f"sigma should be ~1 for smooth, got {sigma}")


def test_sigma_discontinuity():
    """Discontinuity: one b_k much smaller -> sigma >> 1"""
    sigma = _discontinuity_indicator_sigma(1e-10, 1.0, 1.0)
    assert sigma > 10.0, f"sigma should be large for discontinuity, got {sigma}"


def test_sigma_range():
    """sigma should always be >= 1"""
    test_cases = [
        (0.0, 0.0, 0.0),
        (1.0, 2.0, 3.0),
        (1e-10, 1e-5, 1e-10),
        (100.0, 1.0, 0.0),
        (1e10, 1e10, 1e-20),
    ]
    for b0, b1, b2 in test_cases:
        sigma = _discontinuity_indicator_sigma(b0, b1, b2)
        assert sigma >= 1.0 - 1e-12, f"sigma={sigma} < 1 for ({b0},{b1},{b2})"


# ---------------------------------------------------------------------------
# TENO Hybrid FV tests
# ---------------------------------------------------------------------------

def test_teno_hybrid_constant():
    uL, uR = teno_hybrid(5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert abs(uR - 5.0) < 1e-14


def test_teno_hybrid_linear():
    """Linear field: sigma ~ 1 < threshold -> TENO used -> exact."""
    uL, uR = teno_hybrid(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert abs(uL - 0.5) < 1e-14
    assert abs(uR - (-0.5)) < 1e-14


def test_teno_hybrid_jump():
    """Jump: sigma > threshold -> MC2 fallback."""
    uL, uR = teno_hybrid(0.0, 0.0, 1.0, 1.0, 1.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert 0.0 <= uL <= 1.5
    assert 0.0 <= uR <= 1.5


def test_teno_hybrid_finite():
    """Extreme values should remain finite."""
    uL, uR = teno_hybrid(1e10, 0.0, 0.0, 0.0, 1e10)
    assert math.isfinite(uL)
    assert math.isfinite(uR)


# ---------------------------------------------------------------------------
# TENO Hybrid PW tests
# ---------------------------------------------------------------------------

def test_teno_hybrid_pw_constant():
    uL, uR = teno_hybrid_pw(5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert abs(uR - 5.0) < 1e-14


def test_teno_hybrid_pw_linear():
    uL, uR = teno_hybrid_pw(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert abs(uL - 0.5) < 1e-14
    assert abs(uR - (-0.5)) < 1e-14


def test_teno_hybrid_pw_jump():
    uL, uR = teno_hybrid_pw(0.0, 0.0, 1.0, 1.0, 1.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
#
# :D
#
