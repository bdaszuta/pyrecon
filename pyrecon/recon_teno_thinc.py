"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: TENO-THINC reconstruction
"""
import math
from pyrecon._jit_utils import JIT, TYPE_CHECKING
from pyrecon.recon_teno5 import (
    _teno_B0, _teno_B1, _teno_B2,
    _teno5_cutoff,
    _teno5_stencils_L_fv, _teno5_stencils_R_fv,
    _teno5_stencils_L_pw, _teno5_stencils_R_pw,
    _DTENO_FV, _DTENO_PW,
    _EPSL,
)

# ---------------------------------------------------------------------------
# TENO-THINC constants
# ---------------------------------------------------------------------------

# THINC steepness parameter (Takagi et al. 2022: beta = 1.6)
_KBETA = 1.6
_TANH_HALF_BETA = math.tanh(0.5 * _KBETA)


# ---------------------------------------------------------------------------
# Zeta-k discontinuity detector (Takagi et al. 2022, Eqs 18-20)
# ---------------------------------------------------------------------------

def _zeta_detector(delta0, delta1, delta2):
    """Zeta-k discontinuity detector from TENO5 delta flags.

    Returns 1.0 if any stencil is rejected (discontinuity present),
    0.0 if all stencils pass (smooth region).

    Reference: Takagi et al. (2022), Eqs. 18-20.
    """
    if delta0 == 0.0 or delta1 == 0.0 or delta2 == 0.0:
        return 1.0
    return 0.0

# ---------------------------------------------------------------------------
# THINC closed-form cell-averaged reconstruction (Takagi et al. 2022 Eq 16)
# ---------------------------------------------------------------------------

def _thinc_face(u_im1, u_i, u_ip1):
    """THINC left face value (df/du > 0) via closed-form cell-averaged.

    Reference: Takagi et al. (2022), Eq. 16.
    """
    f_plus = 0.5 * (u_im1 + u_ip1)
    f_minus = 0.5 * (u_ip1 - u_im1)
    if abs(f_minus) < 1e-40:
        return u_i
    if (u_ip1 - u_i) * (u_i - u_im1) <= 1e-15:
        return u_i
    alpha = (u_i - f_plus) / f_minus
    B = math.exp(math.copysign(1.0, alpha) * abs(alpha) * _KBETA)
    A = (B / _TANH_HALF_BETA - 1.0) / (B + 1.0)
    num = A + _TANH_HALF_BETA
    den = 1.0 + A * _TANH_HALF_BETA
    return f_plus + f_minus * num / den


def _thinc_face_R(u_im1, u_i, u_ip1):
    """THINC right face value (df/du < 0) via closed-form cell-averaged.

    Reference: Takagi et al. (2022), Eq. 16.
    """
    f_plus = 0.5 * (u_im1 + u_ip1)
    f_minus = 0.5 * (u_im1 - u_ip1)
    if abs(f_minus) < 1e-40:
        return u_i
    if (u_i - u_im1) * (u_ip1 - u_i) <= 1e-15:
        return u_i
    alpha = (u_i - f_plus) / f_minus
    B = math.exp(math.copysign(1.0, alpha) * abs(alpha) * _KBETA)
    A = (B / _TANH_HALF_BETA - 1.0) / (B + 1.0)
    num = A + _TANH_HALF_BETA
    den = 1.0 + A * _TANH_HALF_BETA
    return f_plus + f_minus * num / den


# ---------------------------------------------------------------------------
# TENO-THINC core
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Inlined core: teno_thinc_fv (FV weights)
# ---------------------------------------------------------------------------

def _teno_thinc_LR_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    r"""TENO-THINC paired L+R reconstruction (FV weights).

    Note: the zeta discontinuity detector uses left-face delta flags
    only. If a discontinuity only affects right-face stencils, THINC
    will not trigger for the right face.
    """
    # Smoothness indicators
    b0 = _teno_B0(u_im1, u_i, u_ip1)
    b1 = _teno_B1(u_i, u_ip1, u_ip2)
    b2 = _teno_B2(u_im2, u_im1, u_i)

    # Compute delta flags via TENO5 cutoff
    dL0, dL1, dL2 = _teno5_cutoff(b0, b1, b2)

    # Zeta-k discontinuity detector from delta flags
    zeta = _zeta_detector(dL0, dL1, dL2)

    if zeta > 0.0:
        # Discontinuity detected: use closed-form cell-averaged THINC
        uL = _thinc_face(u_im1, u_i, u_ip1)
        uR = _thinc_face_R(u_im1, u_i, u_ip1)
        return uL, uR

    # Smooth region: use TENO5 reconstruction
    # Left face cutoff
    use_teno_L = not (dL0 == 0.0 and dL1 == 0.0 and dL2 == 0.0)

    # Right face cutoff (recompute B1, B2 for right face)
    b1_R = _teno_B1(u_i, u_im1, u_im2)
    b2_R = _teno_B2(u_ip2, u_ip1, u_i)
    dR0, dR1, dR2 = _teno5_cutoff(b0, b1_R, b2_R)
    use_teno_R = not (dR0 == 0.0 and dR1 == 0.0 and dR2 == 0.0)

    uR_computed = False

    # Left face TENO
    if use_teno_L:
        denom = _DTENO_FV[0] * dL0 + _DTENO_FV[1] * dL1 + _DTENO_FV[2] * dL2
        inv_denom = 1.0 / max(denom, _EPSL)
        ukL0, ukL1, ukL2 = _teno5_stencils_L_fv(u_im2, u_im1, u_i, u_ip1, u_ip2)
        uL = inv_denom * (_DTENO_FV[0] * dL0 * ukL0 +
                          _DTENO_FV[1] * dL1 * ukL1 +
                          _DTENO_FV[2] * dL2 * ukL2)
    else:
        # Fallback to THINC if all stencils rejected
        uL = _thinc_face(u_im1, u_i, u_ip1)
        uR = _thinc_face_R(u_im1, u_i, u_ip1)
        uR_computed = True

    # Right face TENO
    if use_teno_R:
        denom = _DTENO_FV[0] * dR0 + _DTENO_FV[1] * dR1 + _DTENO_FV[2] * dR2
        inv_denom = 1.0 / max(denom, _EPSL)
        ukR0, ukR1, ukR2 = _teno5_stencils_R_fv(u_im2, u_im1, u_i, u_ip1, u_ip2)
        uR = inv_denom * (_DTENO_FV[0] * dR0 * ukR0 +
                          _DTENO_FV[1] * dR1 * ukR1 +
                          _DTENO_FV[2] * dR2 * ukR2)
        uR_computed = True
    elif not uR_computed:
        uR = _thinc_face_R(u_im1, u_i, u_ip1)

    return uL, uR


# ---------------------------------------------------------------------------
# Inlined core: teno_thinc_pw (PW weights)
# ---------------------------------------------------------------------------

def _teno_thinc_LR_pw(u_im2, u_im1, u_i, u_ip1, u_ip2):
    r"""TENO-THINC paired L+R reconstruction (PW weights).

    Note: the zeta discontinuity detector uses left-face delta flags
    only. If a discontinuity only affects right-face stencils, THINC
    will not trigger for the right face.
    """
    # Smoothness indicators
    b0 = _teno_B0(u_im1, u_i, u_ip1)
    b1 = _teno_B1(u_i, u_ip1, u_ip2)
    b2 = _teno_B2(u_im2, u_im1, u_i)

    # Compute delta flags via TENO5 cutoff
    dL0, dL1, dL2 = _teno5_cutoff(b0, b1, b2)

    # Zeta-k discontinuity detector from delta flags
    zeta = _zeta_detector(dL0, dL1, dL2)

    if zeta > 0.0:
        # Discontinuity detected: use closed-form cell-averaged THINC
        uL = _thinc_face(u_im1, u_i, u_ip1)
        uR = _thinc_face_R(u_im1, u_i, u_ip1)
        return uL, uR

    # Smooth region: use TENO5 reconstruction
    # Left face cutoff
    use_teno_L = not (dL0 == 0.0 and dL1 == 0.0 and dL2 == 0.0)

    # Right face cutoff (recompute B1, B2 for right face)
    b1_R = _teno_B1(u_i, u_im1, u_im2)
    b2_R = _teno_B2(u_ip2, u_ip1, u_i)
    dR0, dR1, dR2 = _teno5_cutoff(b0, b1_R, b2_R)
    use_teno_R = not (dR0 == 0.0 and dR1 == 0.0 and dR2 == 0.0)

    uR_computed = False

    # Left face TENO
    if use_teno_L:
        denom = _DTENO_PW[0] * dL0 + _DTENO_PW[1] * dL1 + _DTENO_PW[2] * dL2
        inv_denom = 1.0 / max(denom, _EPSL)
        ukL0, ukL1, ukL2 = _teno5_stencils_L_pw(u_im2, u_im1, u_i, u_ip1, u_ip2)
        uL = inv_denom * (_DTENO_PW[0] * dL0 * ukL0 +
                          _DTENO_PW[1] * dL1 * ukL1 +
                          _DTENO_PW[2] * dL2 * ukL2)
    else:
        # Fallback to THINC if all stencils rejected
        uL = _thinc_face(u_im1, u_i, u_ip1)
        uR = _thinc_face_R(u_im1, u_i, u_ip1)
        uR_computed = True

    # Right face TENO
    if use_teno_R:
        denom = _DTENO_PW[0] * dR0 + _DTENO_PW[1] * dR1 + _DTENO_PW[2] * dR2
        inv_denom = 1.0 / max(denom, _EPSL)
        ukR0, ukR1, ukR2 = _teno5_stencils_R_pw(u_im2, u_im1, u_i, u_ip1, u_ip2)
        uR = inv_denom * (_DTENO_PW[0] * dR0 * ukR0 +
                          _DTENO_PW[1] * dR1 * ukR1 +
                          _DTENO_PW[2] * dR2 * ukR2)
        uR_computed = True
    elif not uR_computed:
        uR = _thinc_face_R(u_im1, u_i, u_ip1)

    return uL, uR


# ---------------------------------------------------------------------------
# Public API: TENO-THINC (FV)
# ---------------------------------------------------------------------------

def teno_thinc_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """TENO-THINC reconstruction (FV weights) at i+1/2.

    Zeta-k discontinuity detection from TENO5 delta flags:
      - zeta=1 (discontinuity): closed-form cell-averaged THINC
      - zeta=0 (smooth): TENO5 for low dissipation

    Reference: Takagi et al. (2022).
    """
    return _teno_thinc_LR_fv(u_im2, u_im1, u_i, u_ip1, u_ip2)


# ---------------------------------------------------------------------------
# Public API: TENO-THINC (PW)
# ---------------------------------------------------------------------------

def teno_thinc_pw(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """TENO-THINC reconstruction (PW weights) at i+1/2.

    Zeta-k discontinuity detection from TENO5 delta flags:
      - zeta=1 (discontinuity): closed-form cell-averaged THINC
      - zeta=0 (smooth): TENO5 for low dissipation

    Reference: Takagi et al. (2022).
    """
    return _teno_thinc_LR_pw(u_im2, u_im1, u_i, u_ip1, u_ip2)
# ---------------------------------------------------------------------------
# JIT compilation (pynalgo convention)
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    reveal_locals()  # noqa: F821
else:
    _zeta_detector = JIT(_zeta_detector)
    _thinc_face = JIT(_thinc_face)
    _thinc_face_R = JIT(_thinc_face_R)
    _teno_thinc_LR_fv = JIT(_teno_thinc_LR_fv)
    _teno_thinc_LR_pw = JIT(_teno_thinc_LR_pw)
    teno_thinc_fv = JIT(teno_thinc_fv)
    teno_thinc_pw = JIT(teno_thinc_pw)
#
# :D
#
