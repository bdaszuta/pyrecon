"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Tests for LS-WENO5-H reconstruction (log-space hybrid)
"""
import math
from pyrecon.recon_lsweno5h import lsweno5h_fv


# --- Constant data ---

def test_lsweno5h_constant():
    """Constant field: all face values equal the constant."""
    u = [3.0, 3.0, 3.0, 3.0, 3.0]
    uL, uR = lsweno5h_fv(*u)
    assert abs(uL - 3.0) < 1e-13
    assert abs(uR - 3.0) < 1e-13


# --- Linear data ---

def test_lsweno5h_linear():
    """Linear field: WENO5-Z default path produces exact linear faces."""
    u = [-2.0, -1.0, 0.0, 1.0, 2.0]
    uL, uR = lsweno5h_fv(*u)
    assert abs(uL - 0.5) < 1e-12
    assert abs(uR + 0.5) < 1e-12


# --- Jump / step ---

def test_lsweno5h_jump():
    """Step function: no NaN, no Inf, reasonable magnitude."""
    u = [0.0, 0.0, 0.0, 1.0, 1.0]
    uL, uR = lsweno5h_fv(*u)
    assert not (math.isnan(uL) or math.isinf(uL))
    assert not (math.isnan(uR) or math.isinf(uR))
    # WENO overshoot at jumps is expected; just check not extreme
    assert abs(uL) < 10.0
    assert abs(uR) < 10.0


def test_lsweno5h_jump_reversed():
    """Step down: no NaN, no Inf."""
    u = [1.0, 1.0, 1.0, 0.0, 0.0]
    uL, uR = lsweno5h_fv(*u)
    assert not (math.isnan(uL) or math.isinf(uL))
    assert not (math.isnan(uR) or math.isinf(uR))
    assert abs(uL) < 10.0
    assert abs(uR) < 10.0


# --- Sharp exponential drop (core feature) ---

def test_lsweno5h_sharp_drop_left_face():
    """Across a 2-cell exponential drop, face values stay positive."""
    u = [1e-3, 1e-3, 1e-3, 1e-14, 1e-14]
    uL, uR = lsweno5h_fv(*u)
    assert uL > 0.0, f"uL={uL} should be positive (log-space fallback)"
    assert uR > 0.0, f"uR={uR} should be positive"


def test_lsweno5h_sharp_drop_right_face():
    """Sharp drop from right side."""
    u = [1e-14, 1e-14, 1e-3, 1e-3, 1e-3]
    uL, uR = lsweno5h_fv(*u)
    assert uL > 0.0, f"uL={uL} should be positive"
    assert uR > 0.0, f"uR={uR} should be positive"


# --- Floor / zero data ---

def test_lsweno5h_all_zeros():
    """All-zeros field: should not crash or produce NaN."""
    u = [0.0, 0.0, 0.0, 0.0, 0.0]
    uL, uR = lsweno5h_fv(*u)
    assert not (math.isnan(uL) or math.isnan(uR))
    assert uL >= 0.0
    assert uR >= 0.0


def test_lsweno5h_near_floor():
    """All cells at floor value: should not crash."""
    u = [1e-40, 1e-40, 1e-40, 1e-40, 1e-40]
    uL, uR = lsweno5h_fv(*u)
    assert not (math.isnan(uL) or math.isnan(uR))
    assert uL >= 0.0
    assert uR >= 0.0


# --- Negative values ---

def test_lsweno5h_negative_values():
    """Negative input: WENO5-Z default path handles sign normally."""
    u = [-5.0, -4.0, -3.0, -2.0, -1.0]
    uL, uR = lsweno5h_fv(*u)
    assert not (math.isnan(uL) or math.isnan(uR))


def test_lsweno5h_mixed_sign():
    """Mixed-sign data crossing zero: WENO5-Z default path."""
    u = [-2.0, -1.0, 0.0, 1.0, 2.0]
    uL, uR = lsweno5h_fv(*u)
    assert not (math.isnan(uL) or math.isnan(uR))


# --- Symmetry ---

def test_lsweno5h_symmetry():
    """Reversed stencil should give swapped L/R faces on smooth data."""
    u_fwd = [1.0, 2.0, 3.0, 4.0, 5.0]
    u_rev = [5.0, 4.0, 3.0, 2.0, 1.0]
    uL_fwd, uR_fwd = lsweno5h_fv(*u_fwd)
    uL_rev, uR_rev = lsweno5h_fv(*u_rev)
    assert abs(uL_fwd - uR_rev) < 1e-10
    assert abs(uR_fwd - uL_rev) < 1e-10
#
# :D
#
