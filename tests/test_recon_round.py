"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Tests for ROUND reconstruction (Deng 2023 unified framework)
"""
import math
from pyrecon.recon_round import (
    round_a_fv as round_a, round_b_fv as round_b, round_c_fv as round_c,
)


# --- ROUND-A ---

def test_round_a_constant():
    uL, uR = round_a(5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert abs(uR - 5.0) < 1e-14


def test_round_a_bounded_linear():
    """Linear data: ROUND-A stays within data bounds (nonlinear mapping)."""
    uL, uR = round_a(-1.0, 0.0, 1.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    # ROUND-A is a nonlinear shape-preserving scheme, not exact on linear data
    assert -1.0 <= uL <= 1.0
    assert -1.0 <= uR <= 1.0


def test_round_a_jump():
    uL, uR = round_a(0.0, 0.0, 1.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert 0.0 <= uL <= 1.0
    assert 0.0 <= uR <= 1.0


def test_round_a_symmetric():
    uL1, uR1 = round_a(-1.0, 0.0, 1.0)
    uL2, uR2 = round_a(1.0, 0.0, -1.0)
    assert abs(uL1 + uL2) < 1e-14
    assert abs(uR1 + uR2) < 1e-14


def test_round_a_nonlinear():
    """Nonlinear data: ROUND-A stays bounded."""
    uL, uR = round_a(-2.0, 1.0, -1.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)


# --- ROUND-B ---

def test_round_b_constant():
    uL, uR = round_b(5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert abs(uR - 5.0) < 1e-14


def test_round_b_bounded_linear():
    uL, uR = round_b(-1.0, 0.0, 1.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert -1.0 <= uL <= 1.0
    assert -1.0 <= uR <= 1.0


def test_round_b_jump():
    uL, uR = round_b(0.0, 0.0, 1.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert 0.0 <= uL <= 1.0


def test_round_b_symmetric():
    uL1, uR1 = round_b(-1.0, 0.0, 1.0)
    uL2, uR2 = round_b(1.0, 0.0, -1.0)
    assert abs(uL1 + uL2) < 1e-14


# --- ROUND-C ---

def test_round_c_constant():
    uL, uR = round_c(5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert abs(uR - 5.0) < 1e-14


def test_round_c_bounded_linear():
    uL, uR = round_c(-1.0, 0.0, 1.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert -1.0 <= uL <= 1.0
    assert -1.0 <= uR <= 1.0


def test_round_c_jump():
    uL, uR = round_c(0.0, 0.0, 1.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert 0.0 <= uL <= 1.0


# --- Cross-variant comparison ---

def test_round_variants_symmetric():
    """All ROUND variants give anti-symmetric results for mirrored data."""
    data = (-1.0, 0.0, 1.0)
    data_rev = (1.0, 0.0, -1.0)
    for fn in [round_a, round_b, round_c]:
        uL1, uR1 = fn(*data)
        uL2, uR2 = fn(*data_rev)
        assert abs(uL1 + uL2) < 1e-14
        assert abs(uR1 + uR2) < 1e-14


def test_round_variants_bounded():
    """All variants stay within max/min of data."""
    data = (-3.0, 1.0, -2.0)
    for fn in [round_a, round_b, round_c]:
        uL, uR = fn(*data)
        assert min(data) - 1e-12 <= uL <= max(data) + 1e-12
        assert min(data) - 1e-12 <= uR <= max(data) + 1e-12
#
# :D
#
