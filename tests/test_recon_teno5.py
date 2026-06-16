"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Tests for TENO5 reconstruction methods
"""
import math
from pyrecon.recon_teno5 import (
    teno5_fv as teno5, teno5_mc2_fv as teno5_mc2, teno5_koren_fv as teno5_koren,
    teno5_pw, teno5_mc2_pw, teno5_koren_pw,
    _teno_B0, _teno_B1, _teno_B2,
    _teno5_cutoff,
)


# ---------------------------------------------------------------------------
# TENO smoothness indicators
# ---------------------------------------------------------------------------

def test_teno_B0_constant():
    assert _teno_B0(1.0, 1.0, 1.0) == 0.0


def test_teno_B0_linear():
    # Linear data: u = c*x
    # u_im1=-1, u_i=0, u_ip1=1 -> im1-ip1=-2, im1-2*i+ip1=0 -> B0=1/4*4=1
    b0 = _teno_B0(-1.0, 0.0, 1.0)
    assert abs(b0 - 1.0) < 1e-14


def test_teno_B0_quadratic():
    # u(x) = x^2: u_im1=1, u_i=0, u_ip1=1 -> im1-ip1=0, im1-2*i+ip1=2
    # B0 = 1/4*0 + 13/12*4 = 13/3 = 4.333...
    b0 = _teno_B0(1.0, 0.0, 1.0)
    assert abs(b0 - 13.0 / 3.0) < 1e-14


def test_teno_B1_linear():
    # u(x) = x: u_i=-1, u_ip1=0, u_ip2=1
    # 3*i - 4*ip1 + ip2 = -3 - 0 + 1 = -2
    # i - 2*ip1 + ip2 = -1 + 0 + 1 = 0
    # B1 = 1/4*4 = 1
    b1 = _teno_B1(-1.0, 0.0, 1.0)
    assert abs(b1 - 1.0) < 1e-14


def test_teno_B2_linear():
    # u(x) = x: u_im2=-2, u_im1=-1, u_i=0
    # im2 - 4*im1 + 3*i = -2 + 4 + 0 = 2
    # im2 - 2*im1 + i = -2 + 2 + 0 = 0
    # B2 = 1/4*4 = 1
    b2 = _teno_B2(-2.0, -1.0, 0.0)
    assert abs(b2 - 1.0) < 1e-14


def test_teno5_cutoff_constant():
    """Constant field: all b_k = 0 -> tau = 0 -> gamma_k all equal."""
    d0, d1, d2 = _teno5_cutoff(0.0, 0.0, 0.0)
    # With b_k=0, the cutoff uses EPSL in denominator
    # gamma_k = (1 + 0/(0+EPSL))^6 = 1^6 = 1
    # chi_k = 1/3 = 0.333... > C_T = 1e-5
    assert d0 == 1.0
    assert d1 == 1.0
    assert d2 == 1.0


def test_teno5_cutoff_one_smooth():
    """One smooth stencil, two rough ones -> only smooth one passes."""
    # b0 smooth, b1 and b2 large and DIFFERENT so tau = |b1-b2| > 0
    d0, d1, d2 = _teno5_cutoff(1e-10, 1.0, 2.0)
    # tau = |b1 - b2| = 1.0  (Takagi et al. 2022)
    # gamma0 = (1 + 1.0/1e-10)^6 ~ 1e60 (huge, dominates)
    # gamma1 = (1 + 1.0/1.0)^6 = 2^6 = 64
    # gamma2 = (1 + 1.0/2.0)^6 = 1.5^6 = 11.4
    # chi0 ~= 1, chi1 ~= 0, chi2 ~= 0
    assert d0 == 1.0
    assert d1 == 0.0
    assert d2 == 0.0


# ---------------------------------------------------------------------------
# TENO5 (FV)
# ---------------------------------------------------------------------------

def test_teno5_constant():
    uL, uR = teno5(5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert abs(uR - 5.0) < 1e-14


def test_teno5_linear():
    """Linear field u(x)=x with dx=1: uL should be u_i + 0.5."""
    uL, uR = teno5(-2.0, -1.0, 0.0, 1.0, 2.0)
    # For linear data, all smoothness indicators are equal (non-zero).
    # TENO5 should reconstruct exactly: uL = u_{i+1/2}^- = 0.5
    assert abs(uL - 0.5) < 1e-14
    assert abs(uR - (-0.5)) < 1e-14


def test_teno5_jump():
    """Jump at i: stencil [0, 0, 1, 1, 1] -> uL ~ 1."""
    uL, uR = teno5(0.0, 0.0, 1.0, 1.0, 1.0)
    assert abs(uL - 1.0) < 0.05
    assert abs(uR - 1.0) < 0.05


def test_teno5_finite():
    """All values should stay finite even for sharp jumps."""
    uL, uR = teno5(1e10, 0.0, 0.0, 0.0, 1e10)
    assert math.isfinite(uL)
    assert math.isfinite(uR)


# ---------------------------------------------------------------------------
# TENO5-MC2 (FV)
# ---------------------------------------------------------------------------

def test_teno5_mc2_constant():
    uL, uR = teno5_mc2(5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert abs(uR - 5.0) < 1e-14


def test_teno5_mc2_linear():
    uL, uR = teno5_mc2(-2.0, -1.0, 0.0, 1.0, 2.0)
    # All stencils should pass cutoff -> TENO reconstruction
    assert abs(uL - 0.5) < 1e-14
    assert abs(uR - (-0.5)) < 1e-14


def test_teno5_mc2_jump():
    """Jump triggers MC2 fallback."""
    uL, uR = teno5_mc2(0.0, 0.0, 1.0, 1.0, 1.0)
    # MC2 fallback should give bounded values
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert 0.0 <= uL <= 1.5 or abs(uL - 1.0) < 0.5
    assert 0.0 <= uR <= 1.5 or abs(uR - 1.0) < 0.5


# ---------------------------------------------------------------------------
# TENO5-Koren (FV)
# ---------------------------------------------------------------------------

def test_teno5_koren_constant():
    uL, uR = teno5_koren(5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert abs(uR - 5.0) < 1e-14


def test_teno5_koren_linear():
    uL, uR = teno5_koren(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert abs(uL - 0.5) < 1e-14
    assert abs(uR - (-0.5)) < 1e-14


def test_teno5_koren_jump():
    """Jump triggers Koren fallback."""
    uL, uR = teno5_koren(0.0, 0.0, 1.0, 1.0, 1.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)


# ---------------------------------------------------------------------------
# TENO5 (PW)
# ---------------------------------------------------------------------------

def test_teno5_pw_constant():
    uL, uR = teno5_pw(5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert abs(uR - 5.0) < 1e-14


def test_teno5_pw_linear():
    uL, uR = teno5_pw(-2.0, -1.0, 0.0, 1.0, 2.0)
    # PW should also be exact for linear data
    assert abs(uL - 0.5) < 1e-14
    assert abs(uR - (-0.5)) < 1e-14


def test_teno5_mc2_pw_linear():
    uL, uR = teno5_mc2_pw(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert abs(uL - 0.5) < 1e-14
    assert abs(uR - (-0.5)) < 1e-14


def test_teno5_koren_pw_linear():
    uL, uR = teno5_koren_pw(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert abs(uL - 0.5) < 1e-14
    assert abs(uR - (-0.5)) < 1e-14
#
# :D
#
