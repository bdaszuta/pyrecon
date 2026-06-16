"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Tests for Lag6 reconstruction methods -- FV and PW variants
"""
from pyrecon.recon_lag6 import lag6_fv, lag6_pw


def test_lag6_fv_constant():
    uL, uR = lag6_fv(5.0, 5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert uL == uR


def test_lag6_fv_linear():
    # Cell-averaged: u(j)=j for linear data
    uL, uR = lag6_fv(-2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
    assert abs(uL - 0.5) < 1e-14
    assert uL == uR


def test_lag6_fv_quadratic():
    # Cell averages of x^2: j^2 + 1/12
    oo12 = 1.0 / 12.0
    vals = [(-2)**2 + oo12, (-1)**2 + oo12, 0**2 + oo12,
            1**2 + oo12, 2**2 + oo12, 3**2 + oo12]
    uL, uR = lag6_fv(*vals)
    assert abs(uL - 0.25) < 1e-14  # face at x=0.5: 0.5^2 = 0.25
    assert uL == uR


def test_lag6_fv_cubic():
    # Cell averages of x^3: j^3 + j/4
    vals = [(-2)**3 + (-2)/4, (-1)**3 + (-1)/4, 0**3 + 0/4,
            1**3 + 1/4, 2**3 + 2/4, 3**3 + 3/4]
    uL, uR = lag6_fv(*vals)
    assert abs(uL - 0.125) < 2e-14  # face at x=0.5: 0.5^3 = 0.125
    assert uL == uR


def test_lag6_fv_quartic():
    # Cell averages of x^4: j^4 + j^2/2 + 1/80
    oo80 = 1.0 / 80.0
    vals = [(-2)**4 + (-2)**2/2 + oo80, (-1)**4 + (-1)**2/2 + oo80,
            0**4 + 0**2/2 + oo80,
            1**4 + 1**2/2 + oo80, 2**4 + 2**2/2 + oo80,
            3**4 + 3**2/2 + oo80]
    uL, uR = lag6_fv(*vals)
    assert abs(uL - 0.0625) < 1e-13  # face at x=0.5: 0.5^4 = 0.0625
    assert uL == uR


def test_lag6_fv_coefficient_sum():
    total = (1.0 - 8.0 + 37.0 + 37.0 - 8.0 + 1.0) / 60.0
    assert abs(total - 1.0) < 1e-15


def test_lag6_fv_finite():
    import math as m
    # Cell-averaged exp on [-2.5, 3.5]
    pts = [
        (m.exp(-1.5) - m.exp(-2.5)),  # cell avg over [-2.5, -1.5]
        (m.exp(-0.5) - m.exp(-1.5)),
        (m.exp(0.5) - m.exp(-0.5)),
        (m.exp(1.5) - m.exp(0.5)),
        (m.exp(2.5) - m.exp(1.5)),
        (m.exp(3.5) - m.exp(2.5)),
    ]
    uL, uR = lag6_fv(*pts)
    assert m.isfinite(uL)
    assert uL == uR


# PW variant tests
def test_lag6_pw_constant():
    uL, uR = lag6_pw(5.0, 5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert uL == uR


def test_lag6_pw_linear():
    uL, uR = lag6_pw(-2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
    assert abs(uL - 0.5) < 1e-14
    assert uL == uR


def test_lag6_pw_coefficient_sum():
    total = (3.0 - 25.0 + 150.0 + 150.0 - 25.0 + 3.0) / 256.0
    assert abs(total - 1.0) < 1e-15
#
# :D
#
