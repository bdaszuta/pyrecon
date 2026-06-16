"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Tests for LS-WENO5-HP reconstruction (physics-informed hybrid)
"""
import math
from pyrecon.recon_lsweno5hp import lsweno5hp_fv


# --- Constant data ---

def test_lsweno5hp_constant():
    """Constant field: all face values equal the constant."""
    u = [3.0, 3.0, 3.0, 3.0, 3.0]
    uL, uR = lsweno5hp_fv(*u)
    assert abs(uL - 3.0) < 1e-13
    assert abs(uR - 3.0) < 1e-13


# --- Linear data ---

def test_lsweno5hp_linear():
    """Linear field: WENO5-Z default path produces exact linear faces."""
    u = [-2.0, -1.0, 0.0, 1.0, 2.0]
    uL, uR = lsweno5hp_fv(*u)
    assert abs(uL - 0.5) < 1e-12
    assert abs(uR + 0.5) < 1e-12


# --- Jump / step ---

def test_lsweno5hp_jump():
    """Step function: no NaN, no Inf, reasonable magnitude."""
    u = [0.0, 0.0, 0.0, 1.0, 1.0]
    uL, uR = lsweno5hp_fv(*u)
    assert not (math.isnan(uL) or math.isinf(uL))
    assert not (math.isnan(uR) or math.isinf(uR))
    assert abs(uL) < 10.0
    assert abs(uR) < 10.0


def test_lsweno5hp_jump_reversed():
    """Step down: no NaN, no Inf."""
    u = [1.0, 1.0, 1.0, 0.0, 0.0]
    uL, uR = lsweno5hp_fv(*u)
    assert not (math.isnan(uL) or math.isinf(uL))
    assert not (math.isnan(uR) or math.isinf(uR))
    assert abs(uL) < 10.0
    assert abs(uR) < 10.0


# --- Sharp exponential drop (two-region fit path) ---

def test_lsweno5hp_sharp_drop_left_face():
    """Across a 2-cell exponential drop, face values stay positive."""
    u = [1e-3, 1e-3, 1e-3, 1e-14, 1e-14]
    uL, uR = lsweno5hp_fv(*u)
    assert uL > 0.0, f"uL={uL} should be positive (two-region fit)"
    assert uR > 0.0, f"uR={uR} should be positive"


def test_lsweno5hp_sharp_drop_right_face():
    """Sharp drop from right side."""
    u = [1e-14, 1e-14, 1e-3, 1e-3, 1e-3]
    uL, uR = lsweno5hp_fv(*u)
    assert uL > 0.0, f"uL={uL} should be positive"
    assert uR > 0.0, f"uR={uR} should be positive"


# --- Two-region fit: plateau detection guard ---

def test_lsweno5hp_no_plateau_guard():
    """When u_im2 is much larger than u_i (ratio > 5), the two-region
    fit should NOT trigger on the plateau path.  Without the plateau
    guard, this would misfit and could produce garbage."""
    u = [5e-12, 1e-13, 1e-3, 1e-13, 1e-26]
    uL, uR = lsweno5hp_fv(*u)
    assert not (math.isnan(uL) or math.isnan(uR))
    assert uL > 0.0, f"uL={uL} should be positive (guard prevented bad fit)"
    assert uR > 0.0, f"uR={uR} should be positive"


# --- Slightly perturbed interior (tests 2-point linear estimate) ---

def test_lsweno5hp_perturbed_interior():
    """Small perturbation on interior plateau: two-region fit uses
    2-point linear which captures perturbation trend."""
    u = [0.95e-3, 1.05e-3, 1.0e-3, 1e-14, 1e-24]
    uL, uR = lsweno5hp_fv(*u)
    assert uL > 0.0, f"uL={uL} should be positive"
    assert uR > 0.0, f"uR={uR} should be positive"


# --- Floor data ---

def test_lsweno5hp_all_zeros():
    """All-zeros field: should not crash or produce NaN."""
    u = [0.0, 0.0, 0.0, 0.0, 0.0]
    uL, uR = lsweno5hp_fv(*u)
    assert not (math.isnan(uL) or math.isnan(uR))
    assert uL >= 0.0
    assert uR >= 0.0


def test_lsweno5hp_near_floor():
    """All cells at floor value: should not crash."""
    u = [1e-40, 1e-40, 1e-40, 1e-40, 1e-40]
    uL, uR = lsweno5hp_fv(*u)
    assert not (math.isnan(uL) or math.isnan(uR))
    assert uL >= 0.0
    assert uR >= 0.0


# --- Negative values ---

def test_lsweno5hp_negative_values():
    """Negative input: WENO5-Z default path handles sign normally."""
    u = [-5.0, -4.0, -3.0, -2.0, -1.0]
    uL, uR = lsweno5hp_fv(*u)
    assert not (math.isnan(uL) or math.isnan(uR))


def test_lsweno5hp_mixed_sign():
    """Mixed-sign data crossing zero: WENO5-Z default path."""
    u = [-2.0, -1.0, 0.0, 1.0, 2.0]
    uL, uR = lsweno5hp_fv(*u)
    assert not (math.isnan(uL) or math.isnan(uR))


# --- Symmetry ---

def test_lsweno5hp_symmetry():
    """Reversed stencil should give swapped L/R faces on smooth data."""
    u_fwd = [1.0, 2.0, 3.0, 4.0, 5.0]
    u_rev = [5.0, 4.0, 3.0, 2.0, 1.0]
    uL_fwd, uR_fwd = lsweno5hp_fv(*u_fwd)
    uL_rev, uR_rev = lsweno5hp_fv(*u_rev)
    assert abs(uL_fwd - uR_rev) < 1e-10
    assert abs(uR_fwd - uL_rev) < 1e-10
#
# :D
#
