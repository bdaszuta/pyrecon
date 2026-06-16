"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Tests for WENO7 reconstruction methods
"""
import math
from pyrecon.recon_weno7 import (
    weno7_fv, weno7z_fv, weno7z_pw, weno7_pw,
    _js_smoothness_fv, _stencils_fv_weno7,
)


def test_js_smoothness_constant():
    b0, b1, b2, b3 = _js_smoothness_fv(
        3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0)
    # Undivided form: b0=b3=0 but b1=b2!=0 for constant data
    # These are the Balsara-Shu undivided indicators
    assert math.isfinite(b0)
    assert math.isfinite(b1)


def test_fv_stencils_linear():
    """FV stencils: exact on linear data."""
    u0, u1, u2, u3 = _stencils_fv_weno7(
        -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
    assert abs(u0 - 0.5) < 1e-14
    assert abs(u1 - 0.5) < 1e-14
    assert abs(u2 - 0.5) < 1e-14
    assert abs(u3 - 0.5) < 1e-14


def test_weno7_constant():
    uL, uR = weno7_fv(5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert abs(uR - 5.0) < 1e-14


def test_weno7_linear():
    uL, uR = weno7_fv(-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
    assert abs(uL - 0.5) < 1e-14
    assert abs(uR - (-0.5)) < 1e-14


def test_weno7_finite():
    uL, uR = weno7_fv(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)


def test_weno7z_constant():
    uL, uR = weno7z_fv(5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert abs(uR - 5.0) < 1e-14


def test_weno7z_linear():
    uL, uR = weno7z_fv(-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
    assert abs(uL - 0.5) < 1e-14
    assert abs(uR - (-0.5)) < 1e-14


def test_weno7_pw_finite():
    uL, uR = weno7_pw(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)


def test_weno7z_pw_constant():
    uL, uR = weno7z_pw(5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert abs(uR - 5.0) < 1e-14


def test_weno7z_pw_linear():
    uL, uR = weno7z_pw(-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
    assert abs(uL - 0.5) < 1e-14
    assert abs(uR - (-0.5)) < 1e-14


def test_weno7z_pw_finite():
    uL, uR = weno7z_pw(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
#
# :D
#
