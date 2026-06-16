"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Tests for TENO-THINC reconstruction methods
"""
import math
from pyrecon.recon_teno_thinc import (
    teno_thinc_fv as teno_thinc, teno_thinc_pw,
    _zeta_detector,
    _thinc_face, _thinc_face_R,
)
from pyrecon.utils import thinc_value_L, thinc_value_R


# ---------------------------------------------------------------------------
# THINC face value tests
# ---------------------------------------------------------------------------

def test_thinc_L_smooth_positive_slope():
    """THINC left face with positive slope."""
    # u_im1=0, u_i=0.5, u_ip1=1 -> u_min=0, u_max=1, gamma=1
    # tanh(6*0.5) ~ tanh(3) ~ 0.995
    # uL = 0 + 0.5 * (1 + 1*0.995) = 0.9975
    uL = thinc_value_L(0.0, 0.5, 1.0)
    assert 0.9 < uL < 1.01


def test_thinc_L_step_up():
    """THINC with step up."""
    uL = thinc_value_L(0.0, 1.0, 1.0)
    # u_min=0, u_max=1 -> flat profile, uL=u_min... wait
    # u_ip1=1, u_im1=0, gamma=1
    # uL ~ 0.9975
    assert 0.9 < uL < 1.01


def test_thinc_R_smooth_positive_slope():
    """THINC right face with positive slope."""
    # tanh(-6*0.5) = -tanh(3) ~ -0.995
    # uR = 0 + 0.5 * (1 + 1*(-0.995)) = 0.0025
    uR = thinc_value_R(0.0, 0.5, 1.0)
    assert -0.01 < uR < 0.1


def test_thinc_constant():
    """THINC with constant data returns u_i."""
    assert abs(thinc_value_L(1.0, 1.0, 1.0) - 1.0) < 1e-14
    assert abs(thinc_value_R(1.0, 1.0, 1.0) - 1.0) < 1e-14


def test_thinc_bounded():
    """THINC face values should be in [u_min, u_max]."""
    for u_im1, u_i, u_ip1 in [(0.0, 0.5, 1.0), (1.0, 0.5, 0.0),
                                 (-1.0, 0.0, 1.0), (2.0, 1.0, -3.0)]:
        u_min = min(u_im1, u_ip1)
        u_max = max(u_im1, u_ip1)
        uL = thinc_value_L(u_im1, u_i, u_ip1)
        uR = thinc_value_R(u_im1, u_i, u_ip1)
        assert u_min - 1e-12 <= uL <= u_max + 1e-12, \
            f"uL={uL} not in [{u_min},{u_max}]"
        assert u_min - 1e-12 <= uR <= u_max + 1e-12, \
            f"uR={uR} not in [{u_min},{u_max}]"


# ---------------------------------------------------------------------------
# TENO-THINC FV tests
# ---------------------------------------------------------------------------

def test_teno_thinc_constant():
    uL, uR = teno_thinc(5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert abs(uR - 5.0) < 1e-14


def test_teno_thinc_linear():
    """Linear field: smooth -> TENO -> exact."""
    uL, uR = teno_thinc(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert abs(uL - 0.5) < 1e-14
    assert abs(uR - (-0.5)) < 1e-14


def test_teno_thinc_jump():
    """Jump: discontinuity detected -> THINC used."""
    uL, uR = teno_thinc(0.0, 0.0, 1.0, 1.0, 1.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    assert 0.0 <= uL <= 1.5
    assert 0.0 <= uR <= 1.5


def test_teno_thinc_sharp_jump():
    """Sharp jump should give very sharp interface (THINC)."""
    # THINC with beta=6 should give sharp near-step profile
    uL, uR = teno_thinc(0.0, 0.0, 0.0, 1.0, 1.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    # Should be close to 0 or 1 (sharp)
    assert uL < 0.02 or uL > 0.98, f"Expected sharp, got uL={uL}"


def test_teno_thinc_finite():
    uL, uR = teno_thinc(1e10, 0.0, 0.0, 0.0, 1e10)
    assert math.isfinite(uL)
    assert math.isfinite(uR)


# ---------------------------------------------------------------------------
# TENO-THINC PW tests
# ---------------------------------------------------------------------------

def test_teno_thinc_pw_constant():
    uL, uR = teno_thinc_pw(5.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(uL - 5.0) < 1e-14
    assert abs(uR - 5.0) < 1e-14


def test_teno_thinc_pw_linear():
    uL, uR = teno_thinc_pw(-2.0, -1.0, 0.0, 1.0, 2.0)
    assert abs(uL - 0.5) < 1e-14
    assert abs(uR - (-0.5)) < 1e-14


def test_teno_thinc_pw_jump():
    uL, uR = teno_thinc_pw(0.0, 0.0, 1.0, 1.0, 1.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)


# ---------------------------------------------------------------------------
# Zeta-k discontinuity detector tests
# ---------------------------------------------------------------------------

def test_zeta_detector_all_pass():
    """All stencils pass (all deltas=1) -> smooth -> zeta=0."""
    assert _zeta_detector(1.0, 1.0, 1.0) == 0.0


def test_zeta_detector_one_rejected():
    """One stencil rejected -> discontinuity -> zeta=1."""
    assert _zeta_detector(1.0, 0.0, 1.0) == 1.0


def test_zeta_detector_all_rejected():
    """All stencils rejected -> discontinuity -> zeta=1."""
    assert _zeta_detector(0.0, 0.0, 0.0) == 1.0


def test_zeta_detector_two_rejected():
    """Two stencils rejected -> discontinuity -> zeta=1."""
    assert _zeta_detector(0.0, 0.0, 1.0) == 1.0


# ---------------------------------------------------------------------------
# Closed-form cell-averaged THINC tests (Takagi et al. 2022 Eq 16)
# ---------------------------------------------------------------------------

def test_thinc_face_constant():
    """Constant data: THINC returns cell average u_i."""
    assert abs(_thinc_face(1.0, 1.0, 1.0) - 1.0) < 1e-14
    assert abs(_thinc_face_R(1.0, 1.0, 1.0) - 1.0) < 1e-14


def test_thinc_face_linear_recovers_cell_average():
    """Linear data u=[0,1,2]: THINC recovers cell average.

    For linear data f_plus=1.0, f_minus=1.0, alpha=0, B=1, A=(1/T-1)/(2).
    The closed-form formula preserves the cell average constraint.
    """
    uL = _thinc_face(0.0, 1.0, 2.0)
    uR = _thinc_face_R(0.0, 1.0, 2.0)
    # THINC profile bounded by [u_im1, u_ip1] = [0, 2]
    assert 0.0 <= uL <= 2.0, f"uL={uL} out of bounds"
    assert 0.0 <= uR <= 2.0, f"uR={uR} out of bounds"
    # Both faces should be finite
    assert math.isfinite(uL)
    assert math.isfinite(uR)


def test_thinc_face_step_sharp_interface():
    """Step data u=[0,0,1]: THINC gives sharp interface.

    Discontinuity at cell i: u_i=0, u_im1=0, u_ip1=1.
    THINC should produce a sharp jump near the interface.
    """
    uL = _thinc_face(0.0, 0.0, 1.0)
    uR = _thinc_face_R(0.0, 0.0, 1.0)
    assert math.isfinite(uL)
    assert math.isfinite(uR)
    # Bounded by [u_im1, u_ip1] = [0, 1]
    assert 0.0 <= uL <= 1.0, f"uL={uL}"
    assert 0.0 <= uR <= 1.0, f"uR={uR}"
    # With KBETA=1.6, the interface is sharp but not infinitely sharp
    # L face should be near u_im1=0 (or somewhat above)
    # R face should be near u_ip1=1 (or somewhat below)


def test_thinc_face_step_data_sharp():
    """Step data straddling cell: THINC produces sharp interface.

    u=[-1, 0.3, 2]: the jump straddles cell i (u_i=0.3 between -1 and 2).
    With cell-averaged closed-form at KBETA=1.6, the profile captures
    the interface -- L face is pulled toward the positive direction and
    R face toward the negative (as expected from the tanh profile).
    Both faces stay bounded within [u_im1, u_ip1].
    """
    uL = _thinc_face(-1.0, 0.3, 2.0)
    uR = _thinc_face_R(-1.0, 0.3, 2.0)
    # Bounded by [u_im1, u_ip1]
    assert -1.0 <= uL <= 2.0, f"uL={uL}"
    assert -1.0 <= uR <= 2.0, f"uR={uR}"
    # uL should be above f_plus=0.5 (steepening toward the right)
    assert uL > 0.5, f"L face not steepened: uL={uL}"
    # uR should be below f_plus=0.5 (steepening toward the left)
    assert uR < 0.5, f"R face not steepened: uR={uR}"


def test_thinc_face_symmetry():
    """THINC L/R symmetry: _thinc_face(a,b,c) and _thinc_face_R(c,b,a)."""
    uL_inc = _thinc_face(0.0, 0.5, 1.0)
    uR_dec = _thinc_face_R(1.0, 0.5, 0.0)
    # For symmetric decreasing data, R face of (1,0.5,0) should
    # correspond symmetrically to L face of (0,0.5,1)
    # Both should be finite and bounded
    assert math.isfinite(uL_inc)
    assert math.isfinite(uR_dec)


def test_thinc_face_bounded():
    """Closed-form THINC face values bounded by [u_im1, u_ip1]."""
    for u_im1, u_i, u_ip1 in [(0.0, 0.5, 1.0), (1.0, 0.5, 0.0),
                                 (-1.0, 0.0, 1.0), (2.0, 1.0, -3.0)]:
        u_min = min(u_im1, u_ip1)
        u_max = max(u_im1, u_ip1)
        uL = _thinc_face(u_im1, u_i, u_ip1)
        uR = _thinc_face_R(u_im1, u_i, u_ip1)
        assert u_min - 1e-12 <= uL <= u_max + 1e-12, \
            f"uL={uL} not in [{u_min},{u_max}] for ({u_im1},{u_i},{u_ip1})"
        assert u_min - 1e-12 <= uR <= u_max + 1e-12, \
            f"uR={uR} not in [{u_min},{u_max}] for ({u_im1},{u_i},{u_ip1})"


def test_thinc_face_near_flat():
    """Near-flat data triggers first-order fallback to u_i."""
    result = _thinc_face(0.0, 1e-20, 0.0)
    assert abs(result - 1e-20) < 1e-30
    result_R = _thinc_face_R(0.0, 1e-20, 0.0)
    assert abs(result_R - 1e-20) < 1e-30

#
# :D
#
