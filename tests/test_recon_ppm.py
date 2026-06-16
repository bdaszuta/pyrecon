"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Tests for PPM and PPMX reconstruction methods
"""
import math
from pyrecon.recon_ppm import ppm_fv as ppm, ppmx_fv as ppmx


# ============================================================================
# Standard PPM
# ============================================================================

def test_ppm_constant():
    uL, uR = ppm(5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert abs(uR - 5.0) < 1e-14


def test_ppm_linear():
    uL, uR = ppm(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert abs(uL - 0.5) < 1e-14
    assert abs(uR - (-0.5)) < 1e-14


def test_ppm_monotonic():
    uL, uR = ppm(0.0, 1.0, 2.0, 3.0, 4.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert uL >= 1.0
    assert uR >= 0.0


def test_ppm_extrema():
    """At local maximum, PPM should not overshoot."""
    uL, uR = ppm(1.0, 3.0, 5.0, 3.0, 1.0)
    assert uL <= 5.0 + 1e-14
    assert uR <= 5.0 + 1e-14


def test_ppm_discontinuity():
    uL, uR = ppm(0.0, 0.0, 1.0, 1.0, 1.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)


# ============================================================================
# PPMX
# ============================================================================

def test_ppmx_constant():
    """Constant field should reconstruct exactly."""
    uL, uR = ppmx(5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert abs(uR - 5.0) < 1e-14


def test_ppmx_linear():
    """Linear field u(x)=x with dx=1 should be exact 4th-order interp."""
    uL, uR = ppmx(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert abs(uL - 0.5) < 1e-14
    assert abs(uR - (-0.5)) < 1e-14


def test_ppmx_quadratic():
    """Quadratic field u(x) = x^2: values 4, 1, 0, 1, 4."""
    uL, uR = ppmx(4.0, 1.0, 0.0, 1.0, 4.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert abs(uL - 0.25) < 1.0
    assert abs(uR - 0.25) < 1.0


def test_ppmx_extrema_preserving():
    """At local maximum, PPMX should preserve extremum (no overshoot)."""
    uL, uR = ppmx(1.0, 3.0, 5.0, 3.0, 1.0)
    assert uL <= 5.0 + 1e-14
    assert uR <= 5.0 + 1e-14
    assert uL >= min(1.0, 3.0, 5.0, 3.0, 1.0) - 1e-14
    assert uR >= min(1.0, 3.0, 5.0, 3.0, 1.0) - 1e-14


def test_ppmx_monotonic():
    """Monotonically increasing data should produce monotonic faces."""
    uL, uR = ppmx(0.0, 1.0, 2.0, 3.0, 4.0)
    assert uL >= 1.0
    assert uR >= 0.0
    assert uL <= 3.0
    assert uR <= 2.0
    assert uR < 2.0 < uL


def test_ppmx_discontinuity():
    """Sharp jump should be handled without NaN/inf."""
    uL, uR = ppmx(0.0, 0.0, 1.0, 1.0, 1.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)


def test_ppmx_negative_values():
    """Works with negative values."""
    uL, uR = ppmx(-5.0, -3.0, -2.0, -1.0, 0.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert uL < 0.0
    assert uR < 0.0
#
# :D
#
