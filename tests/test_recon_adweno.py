"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Tests for anti-diffusive WENO5 reconstruction
"""
import math
from pyrecon.recon_adweno import adweno5_fv as adweno5


def test_adweno5_constant():
    uL, uR = adweno5(5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert abs(uR - 5.0) < 1e-14


def test_adweno5_linear():
    """Curvature is zero on linear data -> no correction -> == WENO5-Z."""
    uL, uR = adweno5(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert abs(uL - 0.5) < 1e-14
    assert abs(uR - (-0.5)) < 1e-14


def test_adweno5_quadratic():
    """Constant curvature -> correction is uniform, still well-behaved."""
    uL, uR = adweno5(4.0, 1.0, 0.0, 1.0, 4.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert 0.0 <= uL <= 4.0
    assert 0.0 <= uR <= 4.0


def test_adweno5_jump():
    """Sharp jump: curvature correction sharpens the contact."""
    uL, uR = adweno5(0.0, 0.0, 0.0, 1.0, 1.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
#
# :D
#
