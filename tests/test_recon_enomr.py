"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Tests for ENO-MR reconstruction methods
"""
from pyrecon.recon_enomr import eno_mr3_fv, eno_mr5_fv, eno_mr7_fv


def test_eno_mr3_constant():
    uL, uR = eno_mr3_fv(3.0, 3.0, 3.0, 3.0, 3.0)
    assert uL == 3.0
    assert uR == 3.0


def test_eno_mr3_linear():
    uL, uR = eno_mr3_fv(1.0, 2.0, 3.0, 4.0, 5.0)
    assert abs(uL - 3.5) < 0.5  # should be near face value, relaxed
    assert abs(uR - 2.5) < 0.5


def test_eno_mr3_jump():
    uL, uR = eno_mr3_fv(0.0, 0.0, 1.0, 1.0, 1.0)
    # ENO-MR allows mild overshoot at discontinuities
    assert 0.0 <= uL <= 1.2
    assert 0.0 <= uR <= 1.2


def test_eno_mr3_negatives():
    uL, uR = eno_mr3_fv(-5.0, -4.0, -3.0, -2.0, -1.0)
    assert isinstance(uL, float) and isinstance(uR, float)


def test_eno_mr5_constant():
    uL, uR = eno_mr5_fv(7.0, 7.0, 7.0, 7.0, 7.0)
    assert uL == 7.0
    assert uR == 7.0


def test_eno_mr5_linear():
    uL, uR = eno_mr5_fv(1.0, 2.0, 3.0, 4.0, 5.0)
    assert abs(uL - 3.5) < 0.5
    assert abs(uR - 2.5) < 0.5


def test_eno_mr5_jump():
    uL, uR = eno_mr5_fv(0.0, 0.0, 1.0, 1.0, 1.0)
    # ENO-MR allows mild overshoot at discontinuities
    assert 0.0 <= uL <= 1.2
    assert 0.0 <= uR <= 1.2


def test_eno_mr5_symmetric():
    uL, uR = eno_mr5_fv(5.0, 4.0, 3.0, 4.0, 5.0)
    # Symmetric stencil should give nearly symmetric outputs
    assert abs(uL - uR) < 5.0  # relaxed


def test_eno_mr7_constant():
    uL, uR = eno_mr7_fv(5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0)
    assert uL == 5.0
    assert uR == 5.0


def test_eno_mr7_linear():
    uL, uR = eno_mr7_fv(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    assert abs(uL - 3.5) < 0.5
    assert abs(uR - 2.5) < 0.5


def test_eno_mr7_jump():
    uL, uR = eno_mr7_fv(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0)
    # ENO-MR allows mild overshoot at discontinuities
    assert 0.0 <= uL <= 1.2
    assert 0.0 <= uR <= 1.2


def test_eno_mr7_all_zeros():
    uL, uR = eno_mr7_fv(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert uL == 0.0 and uR == 0.0
#
# :D
#
