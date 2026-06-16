"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: TENO-A reconstruction: Adaptive-dissipation TENO5
    (custom beta-ratio sensor + linear CT).
    Not Fu et al. 2018 adaptive-order TENO-A.
"""
from pyrecon.recon_teno5 import (
    _teno_B0, _teno_B1, _teno_B2,
    _teno5_stencils_L_fv, _teno5_stencils_R_fv,
    _teno5_stencils_L_pw, _teno5_stencils_R_pw,
    _DTENO_FV, _DTENO_PW,
    _EPSL, _mc2_fallback_LR,
)
from pyrecon._jit_utils import JIT, TYPE_CHECKING

_EPSL_Z = 1e-12

# ---------------------------------------------------------------------------
# TENO-A constants
# ---------------------------------------------------------------------------

# Adaptive CT range: CT in [CT_MIN, CT_MAX]
_CT_MIN = 1e-7  # small CT -> more stencils pass -> higher order in smooth flow
_CT_MAX = 1e-4  # larger CT -> selective -> dissipative near discontinuities

# Reference smoothness ratio (above this we consider flow "non-smooth")
_R_REF = 100.0

# Exponent for the scale sensor mapping
_P_M = 2.0  # sharpness of CT transition


# ---------------------------------------------------------------------------
# TENO-A scale sensor
# ---------------------------------------------------------------------------

def _scale_sensor_m(b0, b1, b2):
    """Compute scale sensor m from TENO smoothness indicators.

    m in [0, 1]: 1 = perfectly smooth (all bk equal), 0 = discontinuity.

    Uses the ratio of max to min smoothness indicator:
      r = (b_max + eps) / (b_min + eps)
      m = max(0, 1 - (r-1)/(R_ref-1))^p

    For smooth flow (r ~= 1): m ~= 1
    For discontinuities (r >> 1): m -> 0
    """
    b_min = min(b0, min(b1, b2))
    b_max = max(b0, max(b1, b2))

    # Ratio of max to min smoothness indicator
    ratio = (b_max + _EPSL) / (b_min + _EPSL)

    # Map ratio to [0,1] sensor
    m_raw = max(0.0, 1.0 - (ratio - 1.0) / max(_R_REF - 1.0, 1e-10))
    m = m_raw ** _P_M
    return m


# ---------------------------------------------------------------------------
# Adaptive CT from scale sensor
# ---------------------------------------------------------------------------

def _adaptive_CT(m):
    """Compute adaptive cut-off parameter from scale sensor m.

    CT = CT_MIN + (CT_MAX - CT_MIN) * (1 - m)
    Smooth (m~1) -> CT ~ CT_MIN (less dissipation)
    Discontinuity (m~0) -> CT ~ CT_MAX (more dissipation)
    """
    return _CT_MIN + (_CT_MAX - _CT_MIN) * (1.0 - m)


# ---------------------------------------------------------------------------
# TENO-A cutoff with adaptive CT
# ---------------------------------------------------------------------------

def _teno_a_cutoff(b0, b1, b2, CT):
    """Compute binary stencil flags with adaptive CT.

    Same chi_k/delta_k logic as _teno5_cutoff. Uses Fu et al. tau
    (|b0-b1|+|b0-b2|) with epsilon 1e-12 (differs from recon_teno5.py
    which uses Takagi simplified tau |b1-b2| with epsilon 1e-40).

    Returns (d0, d1, d2).

    Reference: Fu et al., J. Comput. Phys. 305, 333-359 (2016).
    """
    # Fu et al. convention (differs from recon_teno5.py Takagi convention)
    tau = abs(b0 - b1) + abs(b0 - b2)

    x0 = 1.0 + tau / (b0 + _EPSL_Z)
    x1 = 1.0 + tau / (b1 + _EPSL_Z)
    x2 = 1.0 + tau / (b2 + _EPSL_Z)

    # gamma = x^6 = (x^2)^3
    g0_2 = x0 * x0
    g1_2 = x1 * x1
    g2_2 = x2 * x2
    g0 = g0_2 * g0_2 * g0_2
    g1 = g1_2 * g1_2 * g1_2
    g2 = g2_2 * g2_2 * g2_2

    inv_sum_g = 1.0 / (g0 + g1 + g2)
    chi0 = g0 * inv_sum_g
    chi1 = g1 * inv_sum_g
    chi2 = g2 * inv_sum_g

    d0 = 1.0 if chi0 >= CT else 0.0
    d1 = 1.0 if chi1 >= CT else 0.0
    d2 = 1.0 if chi2 >= CT else 0.0
    return d0, d1, d2


# ---------------------------------------------------------------------------
# Inlined core: teno_a_fv (FV weights)
# ---------------------------------------------------------------------------

def _teno_a_LR_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    r"""TENO-A paired L+R reconstruction (FV weights).

    Note: the adaptive threshold CT is derived from left-face smoothness
    indicators only and reused for both faces. The right-face smoothness
    structure has no influence on CT.
    """
    # Left-face smoothness indicators
    b0_L = _teno_B0(u_im1, u_i, u_ip1)
    b1_L = _teno_B1(u_i, u_ip1, u_ip2)
    b2_L = _teno_B2(u_im2, u_im1, u_i)

    # Scale sensor (same for L and R, based on L indicators)
    m = _scale_sensor_m(b0_L, b1_L, b2_L)
    CT = _adaptive_CT(m)

    # Left face cutoff
    dL0, dL1, dL2 = _teno_a_cutoff(b0_L, b1_L, b2_L, CT)

    # Right-face smoothness indicators
    b1_R = _teno_B1(u_i, u_im1, u_im2)
    b2_R = _teno_B2(u_ip2, u_ip1, u_i)
    dR0, dR1, dR2 = _teno_a_cutoff(b0_L, b1_R, b2_R, CT)

    # Check if any stencils pass
    use_teno_L = not (dL0 == 0.0 and dL1 == 0.0 and dL2 == 0.0)
    use_teno_R = not (dR0 == 0.0 and dR1 == 0.0 and dR2 == 0.0)

    uR_computed = False

    # Left face
    if use_teno_L:
        denom = _DTENO_FV[0] * dL0 + _DTENO_FV[1] * dL1 + _DTENO_FV[2] * dL2
        inv_denom = 1.0 / max(denom, _EPSL)
        ukL0, ukL1, ukL2 = _teno5_stencils_L_fv(u_im2, u_im1, u_i, u_ip1, u_ip2)
        uL = inv_denom * (_DTENO_FV[0] * dL0 * ukL0 +
                          _DTENO_FV[1] * dL1 * ukL1 +
                          _DTENO_FV[2] * dL2 * ukL2)
    else:
        uL, uR = _mc2_fallback_LR(u_im1, u_i, u_ip1)
        uR_computed = True

    # Right face
    if use_teno_R:
        denom = _DTENO_FV[0] * dR0 + _DTENO_FV[1] * dR1 + _DTENO_FV[2] * dR2
        inv_denom = 1.0 / max(denom, _EPSL)
        ukR0, ukR1, ukR2 = _teno5_stencils_R_fv(u_im2, u_im1, u_i, u_ip1, u_ip2)
        uR = inv_denom * (_DTENO_FV[0] * dR0 * ukR0 +
                          _DTENO_FV[1] * dR1 * ukR1 +
                          _DTENO_FV[2] * dR2 * ukR2)
        uR_computed = True
    elif not uR_computed:
        _, uR = _mc2_fallback_LR(u_im1, u_i, u_ip1)

    return uL, uR


# ---------------------------------------------------------------------------
# Inlined core: teno_a_pw (PW weights)
# ---------------------------------------------------------------------------

def _teno_a_LR_pw(u_im2, u_im1, u_i, u_ip1, u_ip2):
    r"""TENO-A paired L+R reconstruction (PW weights).

    Note: the adaptive threshold CT is derived from left-face smoothness
    indicators only and reused for both faces. The right-face smoothness
    structure has no influence on CT.
    """
    # Left-face smoothness indicators
    b0_L = _teno_B0(u_im1, u_i, u_ip1)
    b1_L = _teno_B1(u_i, u_ip1, u_ip2)
    b2_L = _teno_B2(u_im2, u_im1, u_i)

    # Scale sensor (same for L and R, based on L indicators)
    m = _scale_sensor_m(b0_L, b1_L, b2_L)
    CT = _adaptive_CT(m)

    # Left face cutoff
    dL0, dL1, dL2 = _teno_a_cutoff(b0_L, b1_L, b2_L, CT)

    # Right-face smoothness indicators
    b1_R = _teno_B1(u_i, u_im1, u_im2)
    b2_R = _teno_B2(u_ip2, u_ip1, u_i)
    dR0, dR1, dR2 = _teno_a_cutoff(b0_L, b1_R, b2_R, CT)

    # Check if any stencils pass
    use_teno_L = not (dL0 == 0.0 and dL1 == 0.0 and dL2 == 0.0)
    use_teno_R = not (dR0 == 0.0 and dR1 == 0.0 and dR2 == 0.0)

    uR_computed = False

    # Left face
    if use_teno_L:
        denom = _DTENO_PW[0] * dL0 + _DTENO_PW[1] * dL1 + _DTENO_PW[2] * dL2
        inv_denom = 1.0 / max(denom, _EPSL)
        ukL0, ukL1, ukL2 = _teno5_stencils_L_pw(u_im2, u_im1, u_i, u_ip1, u_ip2)
        uL = inv_denom * (_DTENO_PW[0] * dL0 * ukL0 +
                          _DTENO_PW[1] * dL1 * ukL1 +
                          _DTENO_PW[2] * dL2 * ukL2)
    else:
        uL, uR = _mc2_fallback_LR(u_im1, u_i, u_ip1)
        uR_computed = True

    # Right face
    if use_teno_R:
        denom = _DTENO_PW[0] * dR0 + _DTENO_PW[1] * dR1 + _DTENO_PW[2] * dR2
        inv_denom = 1.0 / max(denom, _EPSL)
        ukR0, ukR1, ukR2 = _teno5_stencils_R_pw(u_im2, u_im1, u_i, u_ip1, u_ip2)
        uR = inv_denom * (_DTENO_PW[0] * dR0 * ukR0 +
                          _DTENO_PW[1] * dR1 * ukR1 +
                          _DTENO_PW[2] * dR2 * ukR2)
        uR_computed = True
    elif not uR_computed:
        _, uR = _mc2_fallback_LR(u_im1, u_i, u_ip1)

    return uL, uR


# ---------------------------------------------------------------------------
# Public API: TENO-A (FV)
# ---------------------------------------------------------------------------

def teno_a_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """TENO-A reconstruction (FV weights) at i+1/2.

    Adaptive cut-off parameter CT based on local scale sensor m.
    Smooth regions: small CT -> more stencils pass -> lower numerical
    dissipation (closer to the optimal linear 5th-order scheme).
    Discontinuities: large CT -> selective stencils -> robust MC2 fallback.

    """
    return _teno_a_LR_fv(u_im2, u_im1, u_i, u_ip1, u_ip2)


# ---------------------------------------------------------------------------
# Public API: TENO-A (PW)
# ---------------------------------------------------------------------------

def teno_a_pw(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """TENO-A reconstruction (PW weights) at i+1/2.

    Adaptive cut-off parameter CT based on local scale sensor m.
    Smooth regions: small CT -> more stencils pass -> lower numerical
    dissipation (closer to the optimal linear 5th-order scheme).
    Discontinuities: large CT -> selective stencils -> robust MC2 fallback.
    """
    return _teno_a_LR_pw(u_im2, u_im1, u_i, u_ip1, u_ip2)
# ---------------------------------------------------------------------------
# JIT compilation (pynalgo convention)
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    reveal_locals()  # noqa: F821
else:
    _scale_sensor_m = JIT(_scale_sensor_m)
    _adaptive_CT = JIT(_adaptive_CT)
    _teno_a_cutoff = JIT(_teno_a_cutoff)
    _teno_a_LR_fv = JIT(_teno_a_LR_fv)
    _teno_a_LR_pw = JIT(_teno_a_LR_pw)
    teno_a_fv = JIT(teno_a_fv)
    teno_a_pw = JIT(teno_a_pw)
#
# :D
#
