"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Tests for WENO-C / WENO-ZC reconstruction methods
"""
import pytest
from pyrecon.recon_wenoc import wenoc5_fv, wenoc5z_fv, wenoc7_fv, wenoc7z_fv


def test_wenoc5_constant():
    uL, uR = wenoc5_fv(2.0, 2.0, 2.0, 2.0, 2.0)
    assert uL == pytest.approx(2.0)
    assert uR == pytest.approx(2.0)


def test_wenoc5_linear():
    uL, uR = wenoc5_fv(1.0, 2.0, 3.0, 4.0, 5.0)
    assert uL == pytest.approx(3.5, abs=0.5)
    assert uR == pytest.approx(2.5, abs=0.5)


def test_wenoc5z_constant():
    uL, uR = wenoc5z_fv(5.0, 5.0, 5.0, 5.0, 5.0)
    assert uL == pytest.approx(5.0)
    assert uR == pytest.approx(5.0)


def test_wenoc5z_linear():
    uL, uR = wenoc5z_fv(1.0, 2.0, 3.0, 4.0, 5.0)
    assert uL == pytest.approx(3.5, abs=0.5)


def test_wenoc5_jump():
    uL, uR = wenoc5_fv(0.0, 0.0, 1.0, 1.0, 1.0)
    assert 0.0 <= uL <= 1.0


def test_wenoc5z_jump():
    uL, uR = wenoc5z_fv(0.0, 0.0, 1.0, 1.0, 1.0)
    assert 0.0 <= uL <= 1.0


def test_wenoc5_all_zeros():
    uL, uR = wenoc5_fv(0.0, 0.0, 0.0, 0.0, 0.0)
    assert uL == 0.0 and uR == 0.0


def test_wenoc7_constant():
    uL, uR = wenoc7_fv(3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0)
    assert uL == pytest.approx(3.0)
    assert uR == pytest.approx(3.0)


def test_wenoc7_linear():
    uL, uR = wenoc7_fv(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    assert uL == pytest.approx(3.5, abs=0.5)
    assert uR == pytest.approx(2.5, abs=0.5)


def test_wenoc7z_constant():
    uL, uR = wenoc7z_fv(10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0)
    assert uL == pytest.approx(10.0)
    assert uR == pytest.approx(10.0)


def test_wenoc7z_linear():
    uL, uR = wenoc7z_fv(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    assert uL == pytest.approx(3.5, abs=0.5)


def test_wenoc7_jump():
    uL, uR = wenoc7_fv(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0)
    assert 0.0 <= uL <= 1.0


def test_wenoc7z_jump():
    uL, uR = wenoc7z_fv(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0)
    assert 0.0 <= uL <= 1.0
#
# :D
#
