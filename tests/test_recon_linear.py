"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Tests for linear reconstruction method.
"""
from pyrecon.recon_linear import lin_vl_fv, lin_mc2_fv


def test_lin_vl_constant():
    uL, uR = lin_vl_fv(5.0, 5.0, 5.0, 5.0, 5.0)
    assert uL == 5.0
    assert uR == 5.0


def test_lin_vl_linear():
    """Linear u(x)=x, dx=1: uL=u_i+0.5, uR=u_i-0.5."""
    uL, uR = lin_vl_fv(8.0, 9.0, 10.0, 11.0, 12.0)
    assert uL == 10.5  # u_i + 0.5
    assert uR == 9.5   # u_i - 0.5


def test_lin_vl_extrema():
    """At local max, slopes have opposite signs -> van Leer gives 0."""
    uL, uR = lin_vl_fv(1.0, 3.0, 5.0, 3.0, 1.0)
    # Forward: dul=3-5=-2, dur=5-3=2, du2=-4<=0 -> dum=0, uL=5
    assert uL == 5.0
    # Reversed: dul=3-5=-2, dur=5-3=2, du2=-4<=0 -> dum=0, uR=5
    assert uR == 5.0


def test_lin_vl_sign_change():
    """Slopes have same sign -> van Leer gives harmonic mean."""
    uL, uR = lin_vl_fv(0.0, 2.0, 1.0, 3.0, 5.0)
    # Forward: dul=3-1=2, dur=1-2=-1, du2=-2<=0 -> dum=0, uL=1
    assert uL == 1.0
    # Reversed: (3, 1, 2) -> dul=2-1=1, dur=1-3=-2, du2=-2<=0 -> dum=0, uR=1
    assert uR == 1.0


def test_lin_mc2_constant():
    uL, uR = lin_mc2_fv(5.0, 5.0, 5.0, 5.0, 5.0)
    assert uL == 5.0
    assert uR == 5.0


def test_lin_mc2_linear():
    uL, uR = lin_mc2_fv(8.0, 9.0, 10.0, 11.0, 12.0)
    assert uL == 10.5
    assert uR == 9.5


def test_lin_mc2_sharp():
    """MC2 with sharp gradient on one side."""
    uL, uR = lin_mc2_fv(0.0, 0.0, 0.0, 10.0, 10.0)
    # Forward: MC2(0, 10) -> 0.5*(sign(1,0)+sign(1,10))*min(0, 5) = 0, uL=0
    assert uL == 0.0
    # Reversed: MC2(0-10, 0-0) = MC2(-10, 0) = 0, uR=0+0.5*0=0
    assert uR == 0.0
#
# :D
#
