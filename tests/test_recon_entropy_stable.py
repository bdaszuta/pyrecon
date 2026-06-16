"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Tests for entropy-stable scalar reconstruction
 (Duan & Tang 2020 adapted)
"""
import math
from pyrecon.recon_entropy_stable import (
    es_scalar_quad_fv as es_scalar_quad,
    es_scalar_log_fv as es_scalar_log,
    es_scalar_cubic_fv as es_scalar_cubic,
    es_scalar_quad_pw,
    es_scalar_log_pw,
    es_scalar_cubic_pw,
)


# --- Quadratic entropy (identity transform) ---

def test_es_quad_constant():
    uL, uR = es_scalar_quad(5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert abs(uR - 5.0) < 1e-14


def test_es_quad_linear():
    uL, uR = es_scalar_quad(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert abs(uL - 0.5) < 1e-14
    assert abs(uR - (-0.5)) < 1e-14


def test_es_quad_jump():
    uL, uR = es_scalar_quad(0.0, 0.0, 0.0, 1.0, 1.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    # WENO5-Z may produce slight overshoots at discontinuities
    assert -0.01 <= uL <= 1.01
    assert -0.01 <= uR <= 1.01


def test_es_quad_equals_weno5z():
    """Quadratic entropy (v=u) should equal WENO5-Z."""
    from pyrecon.recon_weno5 import weno5z_fv as weno5z
    data = (-2.0, -1.0, 0.0, 1.0, 2.0)
    uL_es, uR_es = es_scalar_quad(*data)
    uL_wz, uR_wz = weno5z(*data)
    assert abs(uL_es - uL_wz) < 1e-14
    assert abs(uR_es - uR_wz) < 1e-14


# --- Logarithmic entropy ---

def test_es_log_constant():
    """Log entropy on constant positive data."""
    uL, uR = es_scalar_log(2.0, 2.0, 2.0, 2.0, 2.0)
    assert abs(uL - 2.0) < 1e-12
    assert abs(uR - 2.0) < 1e-12


def test_es_log_linear_positive():
    """Log entropy on linear positive data."""
    uL, uR = es_scalar_log(1.0, 2.0, 3.0, 4.0, 5.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert uL > 0 and uR > 0


def test_es_log_finite():
    """Log entropy produces finite values."""
    uL, uR = es_scalar_log(0.1, 0.5, 1.0, 0.5, 0.1)
    assert math.isfinite(uL)
    assert math.isfinite(uR)


# --- Cubic entropy ---

def test_es_cubic_constant():
    uL, uR = es_scalar_cubic(5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-12
    assert abs(uR - 5.0) < 1e-12


def test_es_cubic_linear():
    """Cubic entropy: nonlinear transform, not exact on linear u-data."""
    uL, uR = es_scalar_cubic(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    # The nonlinear entropy transform v=u^3 means we don't get exact
    # linear interpolation in u-space. But values should be close to
    # the face values (0.5 and -0.5).
    assert abs(uL - 0.5) < 1.0
    assert abs(uR - (-0.5)) < 1.0


def test_es_cubic_jump():
    uL, uR = es_scalar_cubic(0.0, 0.0, 0.0, 1.0, 1.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    # Nonlinear entropy transform may produce slight overshoots
    assert -0.01 <= uL <= 1.01
    assert -0.01 <= uR <= 1.01


def test_es_cubic_symmetric():
    uL1, uR1 = es_scalar_cubic(-1.0, 0.0, 1.0, 0.0, -1.0)
    uL2, uR2 = es_scalar_cubic(1.0, 0.0, -1.0, 0.0, 1.0)
    assert abs(uL1 + uL2) < 1e-12

# --- Entropy-stable scalar (PW) ---

def test_es_quad_pw_constant():
    uL, uR = es_scalar_quad_pw(5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert abs(uR - 5.0) < 1e-14


def test_es_quad_pw_linear():
    uL, uR = es_scalar_quad_pw(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert abs(uL - 0.5) < 1e-14
    assert abs(uR - (-0.5)) < 1e-14


def test_es_quad_pw_finite():
    uL, uR = es_scalar_quad_pw(0.0, 0.0, 0.0, 1.0, 1.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)


def test_es_log_pw_constant():
    uL, uR = es_scalar_log_pw(2.0, 2.0, 2.0, 2.0, 2.0)
    assert abs(uL - 2.0) < 1e-12
    assert abs(uR - 2.0) < 1e-12


def test_es_log_pw_positive():
    uL, uR = es_scalar_log_pw(1.0, 2.0, 3.0, 4.0, 5.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert uL > 0 and uR > 0


def test_es_log_pw_finite():
    uL, uR = es_scalar_log_pw(0.1, 0.5, 1.0, 0.5, 0.1)
    assert math.isfinite(uL)
    assert math.isfinite(uR)


def test_es_cubic_pw_constant():
    uL, uR = es_scalar_cubic_pw(5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-12
    assert abs(uR - 5.0) < 1e-12


def test_es_cubic_pw_linear():
    uL, uR = es_scalar_cubic_pw(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert abs(uL - 0.5) < 1.0
    assert abs(uR - (-0.5)) < 1.0


def test_es_cubic_pw_finite():
    uL, uR = es_scalar_cubic_pw(0.0, 0.0, 0.0, 1.0, 1.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
#
# :D
#
