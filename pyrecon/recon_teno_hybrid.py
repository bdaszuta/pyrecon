"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: TENO Hybrid reconstruction with discontinuity indicator
"""
from pyrecon.recon_teno5 import (
    _teno_B0, _teno_B1, _teno_B2,
    _teno5_cutoff,
    _teno5_stencils_L_fv, _teno5_stencils_R_fv,
    _teno5_stencils_L_pw, _teno5_stencils_R_pw,
    _DTENO_FV, _DTENO_PW,
    _EPSL, _mc2_fallback_LR,
)
from pyrecon._jit_utils import JIT, TYPE_CHECKING


# ---------------------------------------------------------------------------
# TENO Hybrid constants
# ---------------------------------------------------------------------------

# Discontinuity indicator threshold
# sigma > threshold -> non-smooth -> switch to dissipative fallback
_SIGMA_THRESHOLD = 1.5

# Number of stencils for the indicator
_N_STENCIL = 3


# ---------------------------------------------------------------------------
# Discontinuity indicator (Fu 2019, CiCP)
# ---------------------------------------------------------------------------

def _discontinuity_indicator_sigma(b0, b1, b2):
    """Compute the TENO-based discontinuity indicator sigma.

    Uses the normalized smoothness indicators from the TENO cutoff.

    Compute gamma_k = (1 + tau/(beta_k + epsilon))^6
    chi_k = gamma_k / Sigma gamma_k
    sigma = max(chi_k) / (min_j(chi_j) + epsilon)

    sigma ~= 1 for smooth flow (all chi_k similar)
    sigma >> 1 for discontinuities (one chi_k dominates)

    Returns sigma (float).

    Reference: Fu (2019), Eqs. 4.1-4.3.
    """
    tau = abs(b0 - b1) + abs(b0 - b2)

    x0 = 1.0 + tau / (b0 + _EPSL)
    x1 = 1.0 + tau / (b1 + _EPSL)
    x2 = 1.0 + tau / (b2 + _EPSL)

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

    chi_max = max(chi0, max(chi1, chi2))
    chi_min = min(chi0, min(chi1, chi2))

    sigma = chi_max / (chi_min + _EPSL)
    return sigma


# ---------------------------------------------------------------------------
# TENO Hybrid core
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Inlined core: teno_hybrid_fv (FV weights)
# ---------------------------------------------------------------------------

def _teno_hybrid_LR_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    r"""TENO Hybrid paired L+R reconstruction (FV weights).

    Note: the discontinuity indicator uses Fu et al. tau
    (:math:`|\beta_0-\beta_1|+|\beta_0-\beta_2|`) for broader
    sensitivity, while the TENO5 cutoff uses Takagi simplified tau
    (:math:`|\beta_1-\beta_2|`) for sharper stencil selection.
    """
    # Left-face smoothness indicators
    b0_L = _teno_B0(u_im1, u_i, u_ip1)
    b1_L = _teno_B1(u_i, u_ip1, u_ip2)
    b2_L = _teno_B2(u_im2, u_im1, u_i)

    # Discontinuity indicator
    sigma = _discontinuity_indicator_sigma(b0_L, b1_L, b2_L)

    if sigma > _SIGMA_THRESHOLD:
        # Non-smooth: use MC2 fallback for robustness
        return _mc2_fallback_LR(u_im1, u_i, u_ip1)

    # Smooth: standard TENO5 reconstruction
    # Left face
    dL0, dL1, dL2 = _teno5_cutoff(b0_L, b1_L, b2_L)
    use_teno_L = not (dL0 == 0.0 and dL1 == 0.0 and dL2 == 0.0)

    # Right-face smoothness indicators
    b1_R = _teno_B1(u_i, u_im1, u_im2)
    b2_R = _teno_B2(u_ip2, u_ip1, u_i)
    dR0, dR1, dR2 = _teno5_cutoff(b0_L, b1_R, b2_R)
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
# Inlined core: teno_hybrid_pw (PW weights)
# ---------------------------------------------------------------------------

def _teno_hybrid_LR_pw(u_im2, u_im1, u_i, u_ip1, u_ip2):
    r"""TENO Hybrid paired L+R reconstruction (PW weights).

    Note: the discontinuity indicator uses Fu et al. tau
    (:math:`|\beta_0-\beta_1|+|\beta_0-\beta_2|`) for broader
    sensitivity, while the TENO5 cutoff uses Takagi simplified tau
    (:math:`|\beta_1-\beta_2|`) for sharper stencil selection.
    """
    # Left-face smoothness indicators
    b0_L = _teno_B0(u_im1, u_i, u_ip1)
    b1_L = _teno_B1(u_i, u_ip1, u_ip2)
    b2_L = _teno_B2(u_im2, u_im1, u_i)

    # Discontinuity indicator
    sigma = _discontinuity_indicator_sigma(b0_L, b1_L, b2_L)

    if sigma > _SIGMA_THRESHOLD:
        # Non-smooth: use MC2 fallback for robustness
        return _mc2_fallback_LR(u_im1, u_i, u_ip1)

    # Smooth: standard TENO5 reconstruction
    # Left face
    dL0, dL1, dL2 = _teno5_cutoff(b0_L, b1_L, b2_L)
    use_teno_L = not (dL0 == 0.0 and dL1 == 0.0 and dL2 == 0.0)

    # Right-face smoothness indicators
    b1_R = _teno_B1(u_i, u_im1, u_im2)
    b2_R = _teno_B2(u_ip2, u_ip1, u_i)
    dR0, dR1, dR2 = _teno5_cutoff(b0_L, b1_R, b2_R)
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
# Public API: TENO Hybrid (FV)
# ---------------------------------------------------------------------------

def teno_hybrid_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """TENO Hybrid reconstruction (FV weights) at i+1/2.

    Discontinuity indicator sigma switches between high-order TENO and
    MC2 fallback for robustness near shocks.

    """
    return _teno_hybrid_LR_fv(u_im2, u_im1, u_i, u_ip1, u_ip2)


# ---------------------------------------------------------------------------
# Public API: TENO Hybrid (PW)
# ---------------------------------------------------------------------------

def teno_hybrid_pw(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """TENO Hybrid reconstruction (PW weights) at i+1/2.

    Discontinuity indicator sigma switches between high-order TENO and
    MC2 fallback for robustness near shocks.
    """
    return _teno_hybrid_LR_pw(u_im2, u_im1, u_i, u_ip1, u_ip2)
# ---------------------------------------------------------------------------
# JIT compilation (pynalgo convention)
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    reveal_locals()  # noqa: F821
else:
    _discontinuity_indicator_sigma = JIT(_discontinuity_indicator_sigma)
    _teno_hybrid_LR_fv = JIT(_teno_hybrid_LR_fv)
    _teno_hybrid_LR_pw = JIT(_teno_hybrid_LR_pw)
    teno_hybrid_fv = JIT(teno_hybrid_fv)
    teno_hybrid_pw = JIT(teno_hybrid_pw)
#
# :D
#
