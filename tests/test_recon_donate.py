"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Tests for donate (first-order donor cell) reconstruction
"""
from pyrecon.recon_donate import donate_fv


def test_donate_constant_field():
    """Constant field: face values equal cell-centered value."""
    u_im2, u_im1, u_i, u_ip1, u_ip2 = 3.0, 3.0, 3.0, 3.0, 3.0
    uL, uR = donate_fv(u_im2, u_im1, u_i, u_ip1, u_ip2)
    assert uL == 3.0
    assert uR == 3.0


def test_donate_varying_field():
    """Donate returns the center value regardless of neighbors."""
    uL, uR = donate_fv(1.0, 2.0, 5.0, 7.0, 9.0)
    assert uL == 5.0
    assert uR == 5.0


def test_donate_negative_values():
    """Donate works with negative values."""
    uL, uR = donate_fv(-1.0, -0.5, -3.0, 0.0, 2.0)
    assert uL == -3.0
    assert uR == -3.0


def test_donate_zero():
    """Donate with zero field."""
    uL, uR = donate_fv(0.0, 0.0, 0.0, 0.0, 0.0)
    assert uL == 0.0
    assert uR == 0.0
#
# :D
#
