"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Tests for VHO-TENO-AA reconstruction (Fu 2021)
"""
import math
from pyrecon.recon_vho_teno import (
    vho_teno8_aa_pw, vho_teno10_aa_pw,
    _js_small, _js_large,
    _JS_LARGE_6, _JS_LARGE_8, _JS_LARGE_10,
    _stencils_small_L, _stencil_large,
    _FLUX_S3, _FLUX_S4, _FLUX_S5,
    _scale_separation, _adaptive_CT_full, _select_face,
    _D_SMALL,
)


# ---------------------------------------------------------------------------
# Smoothness indicator tests
# ---------------------------------------------------------------------------

def test_js_small_constant():
    """Constant field: all JS indicators equal zero."""
    b0, b1, b2 = _js_small(1.0, 1.0, 1.0, 1.0, 1.0)
    assert b0 == 0.0
    assert b1 == 0.0
    assert b2 == 0.0


def test_js_small_linear():
    """Linear field f(x)=x: JS indicators equal 1.0."""
    b0, b1, b2 = _js_small(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert abs(b0 - 1.0) < 1e-14
    assert abs(b1 - 1.0) < 1e-14
    assert abs(b2 - 1.0) < 1e-14


def test_js_small_nonnegative():
    """JS indicators must be non-negative for arbitrary input."""
    b0, b1, b2 = _js_small(-3.0, 1.5, -0.5, 2.0, -1.0)
    assert b0 >= -1e-14
    assert b1 >= -1e-14
    assert b2 >= -1e-14


def test_js_large_s3_linear():
    """S3 (6pt) JS on linear data = 1.0."""
    M, d = _JS_LARGE_6
    b = _js_large(M, d, (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0))
    assert abs(b - 1.0) < 1e-12


def test_js_large_s4_linear():
    """S4 (8pt) JS on linear data = 1.0."""
    M, d = _JS_LARGE_8
    b = _js_large(M, d, (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0))
    assert abs(b - 1.0) < 1e-12


def test_js_large_s5_nonnegative():
    """S5 (10pt) JS must be non-negative."""
    M, d = _JS_LARGE_10
    vals = tuple(float(i ** 2 - 5 * i) for i in range(10))
    b = _js_large(M, d, vals)
    assert b >= -1e-12


# ---------------------------------------------------------------------------
# Stencil flux tests
# ---------------------------------------------------------------------------

def test_small_flux_exact_linear():
    """Small-stencil fluxes exact on linear data."""
    f0, f1, f2 = _stencils_small_L(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert abs(f0 - 0.5) < 1e-14
    assert abs(f1 - 0.5) < 1e-14
    assert abs(f2 - 0.5) < 1e-14


def test_large_flux_s3_linear():
    """S3 flux exact on linear data."""
    f = _stencil_large(_FLUX_S3, (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0))
    assert abs(f - 0.5) < 1e-12


def test_large_flux_s4_linear():
    """S4 flux exact on linear data."""
    f = _stencil_large(_FLUX_S4,
                       (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0))
    assert abs(f - 0.5) < 1e-12


def test_large_flux_s5_linear():
    """S5 flux exact on linear data."""
    f = _stencil_large(_FLUX_S5,
                       (-4.0, -3.0, -2.0, -1.0, 0.0,
                        1.0, 2.0, 3.0, 4.0, 5.0))
    assert abs(f - 0.5) < 1e-12


# ---------------------------------------------------------------------------
# Scale separation tests
# ---------------------------------------------------------------------------

def test_scale_separation_equal():
    """Equal betas -> equal chi."""
    chi = _scale_separation([1.0, 1.0, 1.0, 1.0])
    assert len(chi) == 4
    for c in chi:
        assert abs(c - 0.25) < 1e-14


def test_scale_separation_sums_to_one():
    """chi values must sum to 1.0."""
    chi = _scale_separation([0.1, 1.0, 10.0, 100.0])
    assert abs(sum(chi) - 1.0) < 1e-14


# ---------------------------------------------------------------------------
# Adaptive C_T tests
# ---------------------------------------------------------------------------

def test_adaptive_ct_smooth():
    """Smooth linear data -> C_T ~ 1e-14."""
    CT = _adaptive_CT_full(-2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
    assert 1e-15 <= CT <= 1e-13


def test_adaptive_ct_jump():
    """Jump -> C_T increases to ~1e-7."""
    CT = _adaptive_CT_full(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    assert 1e-8 <= CT <= 1e-6


# ---------------------------------------------------------------------------
# Recursive selection tests
# ---------------------------------------------------------------------------

def test_select_large_stencil_chosen():
    """Largest stencil should be selected on smooth linear data."""
    CT = 1e-14
    u = _select_face(
        betas_small=(1.0, 1.0, 1.0),
        betas_large=[1.0, 1.0],
        f_small=(0.5, 0.5, 0.5),
        f_large=[99.0, 0.5],
        optimw_small=_D_SMALL,
        C_T=CT,
    )
    assert abs(u - 0.5) < 1e-14


def test_select_fallback_small():
    """When large stencils fail, fall back to donor cell."""
    CT = 0.5
    u = _select_face(
        betas_small=(1.0, 1.0, 1.0),
        betas_large=[1.0, 1.0],
        f_small=(1.0, 2.0, 3.0),
        f_large=[99.0, 99.0],
        optimw_small=_D_SMALL,
        C_T=CT,
    )
    assert abs(u - 2.0) < 1e-14


# ---------------------------------------------------------------------------
# Public API tests
# ---------------------------------------------------------------------------

def test_vho_teno8_aa_constant():
    """Constant field -> exact reconstruction (9-arg)."""
    uL, uR = vho_teno8_aa_pw(
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    assert abs(uL - 1.0) < 1e-14
    assert abs(uR - 1.0) < 1e-14


def test_vho_teno8_aa_linear():
    """Linear field -> exact reconstruction (9-arg)."""
    uL, uR = vho_teno8_aa_pw(
        -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0)
    assert abs(uL - 0.5) < 1e-12
    assert abs(uR - (-0.5)) < 1e-12


def test_vho_teno8_aa_jump():
    """Step function -> bounded output (9-arg)."""
    uL, uR = vho_teno8_aa_pw(
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert 0.0 <= uL <= 1.1
    assert 0.0 <= uR <= 1.1


def test_vho_teno8_aa_finite():
    """Large values remain finite (9-arg)."""
    uL, uR = vho_teno8_aa_pw(
        0.0, 0.0, 0.0, 1e10, 0.0, 0.0, 0.0, 0.0, 1e10)
    assert math.isfinite(uL)
    assert math.isfinite(uR)


def test_vho_teno10_aa_constant():
    """10-point constant field (11-arg)."""
    uL, uR = vho_teno10_aa_pw(
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    assert abs(uL - 1.0) < 1e-14
    assert abs(uR - 1.0) < 1e-14


def test_vho_teno10_aa_linear():
    """10-point linear field (11-arg)."""
    uL, uR = vho_teno10_aa_pw(
        -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
    assert abs(uL - 0.5) < 1e-12
    assert abs(uR - (-0.5)) < 1e-12


def test_vho_teno10_aa_jump():
    """10-point jump -> bounded (11-arg)."""
    uL, uR = vho_teno10_aa_pw(
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert 0.0 <= uL <= 1.1
    assert 0.0 <= uR <= 1.1


# ---------------------------------------------------------------------------
# Optimal weights check
# ---------------------------------------------------------------------------

def test_dsmall_weights_sum():
    """Small-stencil optimal weights sum to 1."""
    total = _D_SMALL[0] + _D_SMALL[1] + _D_SMALL[2]
    assert abs(total - 1.0) < 1e-14
#
# :D
#
