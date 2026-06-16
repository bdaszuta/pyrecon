"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Tests for weno5_ext reconstruction method.
"""
import math
from pyrecon.recon_weno5_ext import (
    weno5zcplus_fv, weno5z_ns_fv, weno5zp_fv,
    weno5_ha_js_fv, weno5cz_fv, weno5_bc_fv,
)


# ---------------------------------------------------------------------------
# WENO5-ZC+
# ---------------------------------------------------------------------------

def test_weno5zcplus_constant():
    uL, uR = weno5zcplus_fv(5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert abs(uR - 5.0) < 1e-14


def test_weno5zcplus_linear():
    uL, uR = weno5zcplus_fv(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert abs(uL - 0.5) < 1e-14
    assert abs(uR - (-0.5)) < 1e-14


def test_weno5zcplus_quadratic():
    # Cell-averaged x^2: j^2 + 1/12
    oo12 = 1.0 / 12.0
    uL, uR = weno5zcplus_fv(
        4.0 + oo12, 1.0 + oo12, 0.0 + oo12, 1.0 + oo12, 4.0 + oo12)
    assert abs(uL - 0.25) < 1e-13
    assert abs(uR - 0.25) < 1e-13


def test_weno5zcplus_jump():
    uL, uR = weno5zcplus_fv(0.0, 0.0, 1.0, 1.0, 1.0)
    assert abs(uL - 1.0) < 0.05
    assert abs(uR - 1.0) < 0.05


def test_weno5zcplus_shock():
    uL, uR = weno5zcplus_fv(1.0, 1.0, 1.0, 0.0, 0.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert 0.0 <= uL <= 1.0
    assert 0.0 <= uR <= 1.0


# ---------------------------------------------------------------------------
# WENO5-Z-NS
# ---------------------------------------------------------------------------

def test_weno5z_ns_constant():
    uL, uR = weno5z_ns_fv(5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert abs(uR - 5.0) < 1e-14


def test_weno5z_ns_linear():
    uL, uR = weno5z_ns_fv(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert abs(uL - 0.5) < 1e-13
    assert abs(uR - (-0.5)) < 1e-13


def test_weno5z_ns_jump():
    uL, uR = weno5z_ns_fv(0.0, 0.0, 1.0, 1.0, 1.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert abs(uL - 1.0) < 0.05
    assert abs(uR - 1.0) < 0.05


def test_weno5z_ns_shock():
    uL, uR = weno5z_ns_fv(1.0, 1.0, 1.0, 0.0, 0.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert 0.0 <= uL <= 1.0
    assert 0.0 <= uR <= 1.0


def test_weno5z_ns_symmetry():
    uL1, uR1 = weno5z_ns_fv(-2.0, -1.0, 0.0, 2.0, 1.0)
    uL2, uR2 = weno5z_ns_fv(2.0, 1.0, 0.0, -2.0, -1.0)
    assert abs(uL1 + uL2) < 0.01
    assert abs(uR1 + uR2) < 0.01


# ---------------------------------------------------------------------------
# WENO5-Z+ (Hong et al. 2020)
# ---------------------------------------------------------------------------

def test_weno5zp_constant():
    uL, uR = weno5zp_fv(5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert abs(uR - 5.0) < 1e-14


def test_weno5zp_linear():
    uL, uR = weno5zp_fv(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert abs(uL - 0.5) < 1e-14
    assert abs(uR - (-0.5)) < 1e-14


def test_weno5zp_quadratic():
    oo12 = 1.0 / 12.0
    uL, uR = weno5zp_fv(
        4.0 + oo12, 1.0 + oo12, 0.0 + oo12, 1.0 + oo12, 4.0 + oo12)
    assert abs(uL - 0.25) < 1e-13
    assert abs(uR - 0.25) < 1e-13


def test_weno5zp_jump():
    uL, uR = weno5zp_fv(0.0, 0.0, 1.0, 1.0, 1.0)
    assert abs(uL - 1.0) < 0.05
    assert abs(uR - 1.0) < 0.05


def test_weno5zp_shock():
    uL, uR = weno5zp_fv(1.0, 1.0, 1.0, 0.0, 0.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert 0.0 <= uL <= 1.0
    assert 0.0 <= uR <= 1.0


def test_weno5zp_symmetry():
    """WENO-Z+ should preserve symmetry under data reversal."""
    uL1, uR1 = weno5zp_fv(-2.0, -1.0, 0.0, 1.0, 2.0)
    uL2, uR2 = weno5zp_fv(2.0, 1.0, 0.0, -1.0, -2.0)
    assert abs(uL1 + uL2) < 0.01
    assert abs(uR1 + uR2) < 0.01


# ---------------------------------------------------------------------------
# WENO5-Ha-JS (Ha et al. 2013 smoothness + JS weights)
# ---------------------------------------------------------------------------

def test_weno5_ha_js_constant():
    uL, uR = weno5_ha_js_fv(5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert abs(uR - 5.0) < 1e-14


def test_weno5_ha_js_linear():
    uL, uR = weno5_ha_js_fv(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert abs(uL - 0.5) < 1e-13
    assert abs(uR - (-0.5)) < 1e-13


def test_weno5_ha_js_jump():
    uL, uR = weno5_ha_js_fv(0.0, 0.0, 1.0, 1.0, 1.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert abs(uL - 1.0) < 0.05
    assert abs(uR - 1.0) < 0.05


def test_weno5_ha_js_shock():
    uL, uR = weno5_ha_js_fv(1.0, 1.0, 1.0, 0.0, 0.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert 0.0 <= uL <= 1.0
    assert 0.0 <= uR <= 1.0


# ---------------------------------------------------------------------------
# WENO5-CZ (Barreto et al. 2023 centered Z)
# ---------------------------------------------------------------------------

def test_weno5cz_constant():
    uL, uR = weno5cz_fv(5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert abs(uR - 5.0) < 1e-14


def test_weno5cz_linear():
    uL, uR = weno5cz_fv(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert abs(uL - 0.5) < 1e-14
    assert abs(uR - (-0.5)) < 1e-14


def test_weno5cz_jump():
    uL, uR = weno5cz_fv(0.0, 0.0, 1.0, 1.0, 1.0)
    assert abs(uL - 1.0) < 0.05
    assert abs(uR - 1.0) < 0.05


def test_weno5cz_shock():
    uL, uR = weno5cz_fv(1.0, 1.0, 1.0, 0.0, 0.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert 0.0 <= uL <= 1.0
    assert 0.0 <= uR <= 1.0


# ---------------------------------------------------------------------------
# WENO5-BC (Barreto et al. 2023 biased-centering)
# ---------------------------------------------------------------------------

def test_weno5_bc_constant():
    uL, uR = weno5_bc_fv(5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert abs(uR - 5.0) < 1e-14


def test_weno5_bc_linear():
    uL, uR = weno5_bc_fv(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert abs(uL - 0.5) < 1e-14
    assert abs(uR - (-0.5)) < 1e-14


def test_weno5_bc_jump():
    uL, uR = weno5_bc_fv(0.0, 0.0, 1.0, 1.0, 1.0)
    assert abs(uL - 1.0) < 0.05
    assert abs(uR - 1.0) < 0.05


def test_weno5_bc_shock():
    uL, uR = weno5_bc_fv(1.0, 1.0, 1.0, 0.0, 0.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert 0.0 <= uL <= 1.0
    assert 0.0 <= uR <= 1.0


def test_weno5_bc_central_boost():
    """In smooth regions, BC should behave similarly to Z."""
    uL_bc, uR_bc = weno5_bc_fv(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert abs(uL_bc - 0.5) < 1e-12
    assert abs(uR_bc - (-0.5)) < 1e-12
#
# :D
#
