"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Tests for CTENO reconstruction methods
"""
import math
from pyrecon.recon_cteno import (
    cteno5_fv, cteno5z_fv,
    _central_fv, _gamma_cteno5, _gamma_cteno5z,
)


# ---------------------------------------------------------------------------
# Central stencil polynomial
# ---------------------------------------------------------------------------

def test_central_linear():
    """Central polynomial for u(x)=x: u(0.5) = 0.5"""
    pc = _central_fv(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert abs(pc - 0.5) < 1e-14


def test_central_constant():
    """Central polynomial for u(x)=const should return const"""
    pc = _central_fv(5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(pc - 5.0) < 1e-14


def test_central_quadratic():
    """Central polynomial for u(x)=x^2 (FV convention, point inputs)."""
    pc = _central_fv(4.0, 1.0, 0.0, 1.0, 4.0)
    assert abs(pc - 1.0 / 6.0) < 1e-14


# ---------------------------------------------------------------------------
# Scale-separation functions
# ---------------------------------------------------------------------------

def test_gamma_cteno5_smooth():
    """CTENO5 gamma for smooth field (all SI equal)"""
    SI = [1.0, 1.0, 1.0, 1.0]
    g = _gamma_cteno5(SI)
    assert len(g) == 4
    for gi in g:
        assert gi > 0
    # All equal -> all chi ~ 0.25, central should pass
    ratio = g[0] / g[1]
    assert abs(ratio - 1.0) < 1e-12


def test_gamma_cteno5z_smooth():
    """CTENO5Z gamma for smooth field (tau=0)"""
    SI = [1.0, 1.0, 1.0, 1.0]
    g = _gamma_cteno5z(SI)
    assert len(g) == 4
    for gi in g:
        assert abs(gi - 1.0) < 1e-14


def test_gamma_cteno5_discontinuous():
    """CTENO5 gamma when one SI is zero (discontinuity)"""
    SI = [1.0, 1.0, 0.5, 1e-10]
    g = _gamma_cteno5(SI)
    # The stencil with near-zero SI should dominate gamma
    assert g[3] > g[0] * 1e6


# ---------------------------------------------------------------------------
# CTENO5-FV
# ---------------------------------------------------------------------------

def test_cteno5_constant():
    uL, uR = cteno5_fv(5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert abs(uR - 5.0) < 1e-14


def test_cteno5_linear():
    """For u(x)=x, face at i+1/2 = 0.5, i-1/2 = -0.5"""
    uL, uR = cteno5_fv(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert abs(uL - 0.5) < 1e-14
    assert abs(uR - (-0.5)) < 1e-14


def test_cteno5_quadratic():
    """For u(x)=x^2, FV reconstruction with point inputs gives 1/6."""
    uL, uR = cteno5_fv(4.0, 1.0, 0.0, 1.0, 4.0)
    assert abs(uL - 1.0 / 6.0) < 1e-14
    assert abs(uR - 1.0 / 6.0) < 1e-14


def test_cteno5_cubic():
    """For u(x)=x^3, middle directional stencil passes: uL = 0.5."""
    uL, uR = cteno5_fv(-8.0, -1.0, 0.0, 1.0, 8.0)
    assert abs(uL - 0.5) < 1e-14
    assert abs(uR - (-0.5)) < 1e-14


def test_cteno5_quartic():
    """For u(x)=x^4, middle directional stencil passes: uL = uR = 1/6."""
    uL, uR = cteno5_fv(16.0, 1.0, 0.0, 1.0, 16.0)
    assert abs(uL - 1.0 / 6.0) < 1e-14
    assert abs(uR - 1.0 / 6.0) < 1e-14


def test_cteno5_jump_left_stencil():
    """Jump at i=0 in stencil [0,0,1,1,1]: faces in smooth region -> near 1"""
    uL, uR = cteno5_fv(0.0, 0.0, 1.0, 1.0, 1.0)
    assert abs(uL - 1.0) < 0.05
    assert abs(uR - 1.0) < 0.05
    assert math.isfinite(uL)
    assert math.isfinite(uR)


def test_cteno5_jump_right_stencil():
    """Jump at i+1 in [1,1,1,0,0]: both faces well-behaved in [0,1]"""
    uL, uR = cteno5_fv(1.0, 1.0, 1.0, 0.0, 0.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert 0.0 <= uL <= 1.0
    assert 0.0 <= uR <= 1.0


def test_cteno5_sharp_jump():
    """Sharp jump [1,1,0,0,0]: value at face should be near 1 or 0"""
    uL, uR = cteno5_fv(1.0, 1.0, 0.0, 0.0, 0.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    # uL should be close to 0 (forward-looking into zeros)
    assert 0.0 <= uL <= 1.0
    # uR should be close to 1 (backward-looking into ones)
    assert 0.0 <= uR <= 1.0


# ---------------------------------------------------------------------------
# CTENO5Z-FV
# ---------------------------------------------------------------------------

def test_cteno5z_constant():
    uL, uR = cteno5z_fv(5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert abs(uR - 5.0) < 1e-14


def test_cteno5z_linear():
    uL, uR = cteno5z_fv(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert abs(uL - 0.5) < 1e-14
    assert abs(uR - (-0.5)) < 1e-14


def test_cteno5z_quadratic():
    uL, uR = cteno5z_fv(4.0, 1.0, 0.0, 1.0, 4.0)
    assert abs(uL - 1.0 / 6.0) < 1e-14
    assert abs(uR - 1.0 / 6.0) < 1e-14


def test_cteno5z_jump():
    uL, uR = cteno5z_fv(0.0, 0.0, 1.0, 1.0, 1.0)
    assert abs(uL - 1.0) < 0.05
    assert abs(uR - 1.0) < 0.05
    assert math.isfinite(uL)
    assert math.isfinite(uR)


def test_cteno5z_right_jump():
    uL, uR = cteno5z_fv(1.0, 1.0, 1.0, 0.0, 0.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert 0.0 <= uL <= 1.0
    assert 0.0 <= uR <= 1.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_cteno5_all_zeros():
    """All-zero stencil should return zero"""
    uL, uR = cteno5_fv(0.0, 0.0, 0.0, 0.0, 0.0)
    assert abs(uL) < 1e-14
    assert abs(uR) < 1e-14


def test_cteno5z_all_zeros():
    uL, uR = cteno5z_fv(0.0, 0.0, 0.0, 0.0, 0.0)
    assert abs(uL) < 1e-14
    assert abs(uR) < 1e-14


def test_cteno5_negative_values():
    """Negative constant field"""
    uL, uR = cteno5_fv(-3.0, -3.0, -3.0, -3.0, -3.0)
    assert abs(uL + 3.0) < 1e-14
    assert abs(uR + 3.0) < 1e-14


def test_cteno5z_negative_values():
    uL, uR = cteno5z_fv(-3.0, -3.0, -3.0, -3.0, -3.0)
    assert abs(uL + 3.0) < 1e-14
    assert abs(uR + 3.0) < 1e-14


def test_cteno5_large_values():
    """Large values should not overflow"""
    uL, uR = cteno5_fv(1e10, 1e10, 1e10, 1e10, 1e10)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert abs(uL - 1e10) / 1e10 < 1e-14


def test_cteno5z_large_values():
    uL, uR = cteno5z_fv(1e10, 1e10, 1e10, 1e10, 1e10)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert abs(uL - 1e10) / 1e10 < 1e-14


def test_cteno5_symmetric():
    """Symmetric stencil should give symmetric result"""
    uL, uR = cteno5_fv(-1.0, -1.0, 0.0, 1.0, 1.0)
    # L should be > 0 (trending upward), R should be < 0 (trending downward)
    assert uL > 0.0
    assert uR < 0.0
    assert math.isfinite(uL)
    assert math.isfinite(uR)


def test_cteno5z_symmetric():
    uL, uR = cteno5z_fv(-1.0, -1.0, 0.0, 1.0, 1.0)
    assert uL > 0.0
    assert uR < 0.0
    assert math.isfinite(uL)
    assert math.isfinite(uR)
#
# :D
#
