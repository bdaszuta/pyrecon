"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: TENO5 reconstruction methods
"""
import math
from pyrecon._jit_utils import JIT, TYPE_CHECKING

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_K_1_4 = 1.0 / 4.0
_K_13_12 = 13.0 / 12.0
_K_1_6 = 1.0 / 6.0
_K_1_8 = 1.0 / 8.0

# TENO5 optimal linear weights: {central, forward, backward}
_DTENO_FV = (6.0 / 10.0, 3.0 / 10.0, 1.0 / 10.0)
_DTENO_PW = (5.0 / 8.0, 5.0 / 16.0, 1.0 / 16.0)

# TENO5 parameters
_C_T = 1e-5
_Q_TENO = 6.0
_EPSL = 1e-40

# ---------------------------------------------------------------------------
# TENO smoothness indicators (Eq. 10)
# ---------------------------------------------------------------------------

def _teno_B0(u_im1, u_i, u_ip1):
    r"""Central sub-stencil smoothness indicator.

    B0 = :math:`1/4*(u_{i-1} - u_{i+1})^2 + 13/12*(u_{i-1} - 2*u_i + u_{i+1})^2`
    """
    return (_K_1_4 * (u_im1 - u_ip1) ** 2 +
            _K_13_12 * (u_im1 - 2.0 * u_i + u_ip1) ** 2)


def _teno_B1(u_i, u_ip1, u_ip2):
    r"""Forward sub-stencil smoothness indicator.

    .. math::
       B1 = 1/4*(3*u_i - 4*u_{i+1} + u_{i+2})^2 +
            13/12*(u_i - 2*u_{i+1} + u_{i+2})^2
    """
    return (_K_1_4 * (3.0 * u_i - 4.0 * u_ip1 + u_ip2) ** 2 +
            _K_13_12 * (u_i - 2.0 * u_ip1 + u_ip2) ** 2)


def _teno_B2(u_im2, u_im1, u_i):
    r"""Backward sub-stencil smoothness indicator.

    .. math::
       B2 = 1/4*(u_{i-2} - 4*u_{i-1} + 3*u_i)^2 +
            13/12*(u_{i-2} - 2*u_{i-1} + u_i)^2
    """
    return (_K_1_4 * (u_im2 - 4.0 * u_im1 + 3.0 * u_i) ** 2 +
            _K_13_12 * (u_im2 - 2.0 * u_im1 + u_i) ** 2)


# ---------------------------------------------------------------------------
# TENO5 cutoff (Takagi et al. 2022 simplified tau)
# ---------------------------------------------------------------------------

def _teno5_cutoff(b0, b1, b2):
    """Compute binary stencil flags (delta_k) via TENO sharp cutoff.

    tau = |B1 - B2|
    gamma_k = (1 + tau/(B_k + eps))^q   (q=6)
    chi_k = gamma_k / sum(gamma_k)
    delta_k = 1 if chi_k >= C_T else 0

    Returns (d0, d1, d2).

    Reference: Takagi et al. (2022), Eq. 13.
    """
    tau = abs(b1 - b2)  # Takagi et al. 2022: tau_5 = |beta_{1,3} - beta_{2,3}|

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

    d0 = 1.0 if chi0 >= _C_T else 0.0
    d1 = 1.0 if chi1 >= _C_T else 0.0
    d2 = 1.0 if chi2 >= _C_T else 0.0
    return d0, d1, d2


# ---------------------------------------------------------------------------
# MC2 fallback (3-point, paired L+R)
# ---------------------------------------------------------------------------

def _mc2_fallback_LR(a, b, c):
    """MC2 limiter on 3-point stencil {a, b, c}.

    Computes both uL (right face of cell containing b) and
    uR (left face of cell containing b) for use as fallback
    in TENO5 variants.

    """
    dl = c - b
    dr = b - a
    if dl * dr <= 0.0:
        return b, b

    sgn = 1.0 if dl > 0.0 else -1.0
    adl = abs(dl)
    adr = abs(dr)
    adc = 0.5 * abs(c - a)
    du = sgn * min(2.0 * adl, min(2.0 * adr, adc))
    uL = b + 0.5 * du
    uR = b - 0.5 * du
    return uL, uR


# ---------------------------------------------------------------------------
# Koren limiter fallback (3-point, paired L+R)
# ---------------------------------------------------------------------------

def _koren_fallback_LR(a, b, c):
    """Two-sided symmetric Koren-type limiter reconstruction pair.

    Computes both uL (right face) and uR (left face) with
    symmetric forward/backward phi ratios:
    phi = max(0, min(2*r, (1+2*r)/3, 2)) with r = (c-b)/(b-a+eps).
    """
    dl = c - b
    dr = b - a
    if dl * dr <= 0.0:
        return b, b

    r_fwd = dl / (dr + _EPSL)
    r_bwd = dr / (dl + _EPSL)

    phi_fwd = max(0.0,
                  min(2.0 * r_fwd,
                      min((1.0 + 2.0 * r_fwd) / 3.0, 2.0)))
    phi_bwd = max(0.0,
                  min(2.0 * r_bwd,
                      min((1.0 + 2.0 * r_bwd) / 3.0, 2.0)))

    slope = math.copysign(
        min(phi_fwd * abs(dr), phi_bwd * abs(dl)), dl)

    uL = b + 0.5 * slope
    uR = b - 0.5 * slope
    return uL, uR


# ---------------------------------------------------------------------------
# TENO5 stencil polynomials (FV)
# ---------------------------------------------------------------------------

def _teno5_stencils_L_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    r"""Left-face FV stencil polynomials for TENO5.

    Returns (uk0, uk1, uk2):
      uk0 (central)  = :math:`1/6 * (-u_{i-1} + 5*u_i + 2*u_{i+1})`
      uk1 (forward)  = :math:`1/6 * (2*u_i + 5*u_{i+1} - u_{i+2})`
      uk2 (backward) = :math:`1/6 * (2*u_{i-2} - 7*u_{i-1} + 11*u_i)`
    """
    uk0 = _K_1_6 * (-u_im1 + 5.0 * u_i + 2.0 * u_ip1)
    uk1 = _K_1_6 * (2.0 * u_i + 5.0 * u_ip1 - u_ip2)
    uk2 = _K_1_6 * (2.0 * u_im2 - 7.0 * u_im1 + 11.0 * u_i)
    return uk0, uk1, uk2


def _teno5_stencils_R_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    r"""Right-face FV stencil polynomials for TENO5 (reversed stencil).

    Returns (uk0, uk1, uk2):
      uk0 (central)  = :math:`1/6 * (-u_{i+1} + 5*u_i + 2*u_{i-1})`
      uk1 (forward)  = :math:`1/6 * (2*u_i + 5*u_{i-1} - u_{i-2})`
      uk2 (backward) = :math:`1/6 * (2*u_{i+2} - 7*u_{i+1} + 11*u_i)`
    """
    uk0 = _K_1_6 * (-u_ip1 + 5.0 * u_i + 2.0 * u_im1)
    uk1 = _K_1_6 * (2.0 * u_i + 5.0 * u_im1 - u_im2)
    uk2 = _K_1_6 * (2.0 * u_ip2 - 7.0 * u_ip1 + 11.0 * u_i)
    return uk0, uk1, uk2


# ---------------------------------------------------------------------------
# TENO5 stencil polynomials (PW)
# ---------------------------------------------------------------------------

def _teno5_stencils_L_pw(u_im2, u_im1, u_i, u_ip1, u_ip2):
    r"""Left-face PW stencil polynomials for TENO5.

    Returns (uk0, uk1, uk2):
      uk0 (central)  = :math:`1/8 * (-u_{i-1} + 6*u_i + 3*u_{i+1})`
      uk1 (forward)  = :math:`1/8 * (3*u_i + 6*u_{i+1} - u_{i+2})`
      uk2 (backward) = :math:`1/8 * (3*u_{i-2} - 10*u_{i-1} + 15*u_i)`
    """
    uk0 = _K_1_8 * (-u_im1 + 6.0 * u_i + 3.0 * u_ip1)
    uk1 = _K_1_8 * (3.0 * u_i + 6.0 * u_ip1 - u_ip2)
    uk2 = _K_1_8 * (3.0 * u_im2 - 10.0 * u_im1 + 15.0 * u_i)
    return uk0, uk1, uk2


def _teno5_stencils_R_pw(u_im2, u_im1, u_i, u_ip1, u_ip2):
    r"""Right-face PW stencil polynomials for TENO5 (reversed stencil).

    Returns (uk0, uk1, uk2):
      uk0 (central)  = :math:`1/8 * (-u_{i+1} + 6*u_i + 3*u_{i-1})`
      uk1 (forward)  = :math:`1/8 * (3*u_i + 6*u_{i-1} - u_{i-2})`
      uk2 (backward) = :math:`1/8 * (3*u_{i+2} - 10*u_{i+1} + 15*u_i)`
    """
    uk0 = _K_1_8 * (-u_ip1 + 6.0 * u_i + 3.0 * u_im1)
    uk1 = _K_1_8 * (3.0 * u_i + 6.0 * u_im1 - u_im2)
    uk2 = _K_1_8 * (3.0 * u_ip2 - 10.0 * u_ip1 + 15.0 * u_i)
    return uk0, uk1, uk2


# ---------------------------------------------------------------------------
# Inlined core: teno5_fv (FV weights, cutoff_teno5, mc2 fallback)
# ---------------------------------------------------------------------------

def _teno5_LR_core_teno5_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """TENO5 core (FV weights, cutoff_teno5, MC2 fallback)."""
    # Left-face smoothness indicators
    b0_L = _teno_B0(u_im1, u_i, u_ip1)
    b1_L = _teno_B1(u_i, u_ip1, u_ip2)
    b2_L = _teno_B2(u_im2, u_im1, u_i)
    dL0, dL1, dL2 = _teno5_cutoff(b0_L, b1_L, b2_L)

    # Right-face smoothness indicators (B0 symmetric, B1/B2 recomputed)
    b1_R = _teno_B1(u_i, u_im1, u_im2)
    b2_R = _teno_B2(u_ip2, u_ip1, u_i)
    dR0, dR1, dR2 = _teno5_cutoff(b0_L, b1_R, b2_R)

    # cutoff_teno5: use TENO unless ALL stencils rejected
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
# Inlined core: teno5_mc2_fv (FV weights, cutoff_teno5_mc2, mc2 fallback)
# ---------------------------------------------------------------------------

def _teno5_LR_core_teno5_mc2_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """TENO5-MC2 core (FV weights, cutoff_teno5_mc2, MC2 fallback)."""
    b0_L = _teno_B0(u_im1, u_i, u_ip1)
    b1_L = _teno_B1(u_i, u_ip1, u_ip2)
    b2_L = _teno_B2(u_im2, u_im1, u_i)
    dL0, dL1, dL2 = _teno5_cutoff(b0_L, b1_L, b2_L)

    b1_R = _teno_B1(u_i, u_im1, u_im2)
    b2_R = _teno_B2(u_ip2, u_ip1, u_i)
    dR0, dR1, dR2 = _teno5_cutoff(b0_L, b1_R, b2_R)

    # cutoff_teno5_mc2: use TENO only if ALL stencils pass
    use_teno_L = (dL0 > 0.0 and dL1 > 0.0 and dL2 > 0.0)
    use_teno_R = (dR0 > 0.0 and dR1 > 0.0 and dR2 > 0.0)

    uR_computed = False

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
# Inlined core: teno5_koren_fv (FV weights, cutoff_teno5_mc2, koren fallback)
# ---------------------------------------------------------------------------

def _teno5_LR_core_teno5_koren_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """TENO5-Koren core (FV weights, cutoff_teno5_mc2, Koren fallback)."""
    b0_L = _teno_B0(u_im1, u_i, u_ip1)
    b1_L = _teno_B1(u_i, u_ip1, u_ip2)
    b2_L = _teno_B2(u_im2, u_im1, u_i)
    dL0, dL1, dL2 = _teno5_cutoff(b0_L, b1_L, b2_L)

    b1_R = _teno_B1(u_i, u_im1, u_im2)
    b2_R = _teno_B2(u_ip2, u_ip1, u_i)
    dR0, dR1, dR2 = _teno5_cutoff(b0_L, b1_R, b2_R)

    # cutoff_teno5_mc2: use TENO only if ALL stencils pass
    use_teno_L = (dL0 > 0.0 and dL1 > 0.0 and dL2 > 0.0)
    use_teno_R = (dR0 > 0.0 and dR1 > 0.0 and dR2 > 0.0)

    uR_computed = False

    if use_teno_L:
        denom = _DTENO_FV[0] * dL0 + _DTENO_FV[1] * dL1 + _DTENO_FV[2] * dL2
        inv_denom = 1.0 / max(denom, _EPSL)
        ukL0, ukL1, ukL2 = _teno5_stencils_L_fv(u_im2, u_im1, u_i, u_ip1, u_ip2)
        uL = inv_denom * (_DTENO_FV[0] * dL0 * ukL0 +
                          _DTENO_FV[1] * dL1 * ukL1 +
                          _DTENO_FV[2] * dL2 * ukL2)
    else:
        uL, uR = _koren_fallback_LR(u_im1, u_i, u_ip1)
        uR_computed = True

    if use_teno_R:
        denom = _DTENO_FV[0] * dR0 + _DTENO_FV[1] * dR1 + _DTENO_FV[2] * dR2
        inv_denom = 1.0 / max(denom, _EPSL)
        ukR0, ukR1, ukR2 = _teno5_stencils_R_fv(u_im2, u_im1, u_i, u_ip1, u_ip2)
        uR = inv_denom * (_DTENO_FV[0] * dR0 * ukR0 +
                          _DTENO_FV[1] * dR1 * ukR1 +
                          _DTENO_FV[2] * dR2 * ukR2)
        uR_computed = True
    elif not uR_computed:
        _, uR = _koren_fallback_LR(u_im1, u_i, u_ip1)

    return uL, uR


# ---------------------------------------------------------------------------
# Inlined core: teno5_pw (PW weights, cutoff_teno5, mc2 fallback)
# ---------------------------------------------------------------------------

def _teno5_LR_core_teno5_pw(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """TENO5 core (PW weights, cutoff_teno5, MC2 fallback)."""
    b0_L = _teno_B0(u_im1, u_i, u_ip1)
    b1_L = _teno_B1(u_i, u_ip1, u_ip2)
    b2_L = _teno_B2(u_im2, u_im1, u_i)
    dL0, dL1, dL2 = _teno5_cutoff(b0_L, b1_L, b2_L)

    b1_R = _teno_B1(u_i, u_im1, u_im2)
    b2_R = _teno_B2(u_ip2, u_ip1, u_i)
    dR0, dR1, dR2 = _teno5_cutoff(b0_L, b1_R, b2_R)

    # cutoff_teno5: use TENO unless ALL stencils rejected
    use_teno_L = not (dL0 == 0.0 and dL1 == 0.0 and dL2 == 0.0)
    use_teno_R = not (dR0 == 0.0 and dR1 == 0.0 and dR2 == 0.0)

    uR_computed = False

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
# Inlined core: teno5_mc2_pw (PW weights, cutoff_teno5_mc2, mc2 fallback)
# ---------------------------------------------------------------------------

def _teno5_LR_core_teno5_mc2_pw(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """TENO5-MC2 core (PW weights, cutoff_teno5_mc2, MC2 fallback)."""
    b0_L = _teno_B0(u_im1, u_i, u_ip1)
    b1_L = _teno_B1(u_i, u_ip1, u_ip2)
    b2_L = _teno_B2(u_im2, u_im1, u_i)
    dL0, dL1, dL2 = _teno5_cutoff(b0_L, b1_L, b2_L)

    b1_R = _teno_B1(u_i, u_im1, u_im2)
    b2_R = _teno_B2(u_ip2, u_ip1, u_i)
    dR0, dR1, dR2 = _teno5_cutoff(b0_L, b1_R, b2_R)

    # cutoff_teno5_mc2: use TENO only if ALL stencils pass
    use_teno_L = (dL0 > 0.0 and dL1 > 0.0 and dL2 > 0.0)
    use_teno_R = (dR0 > 0.0 and dR1 > 0.0 and dR2 > 0.0)

    uR_computed = False

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
# Inlined core: teno5_koren_pw (PW weights, cutoff_teno5_mc2, koren fallback)
# ---------------------------------------------------------------------------

def _teno5_LR_core_teno5_koren_pw(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """TENO5-Koren core (PW weights, cutoff_teno5_mc2, Koren fallback)."""
    b0_L = _teno_B0(u_im1, u_i, u_ip1)
    b1_L = _teno_B1(u_i, u_ip1, u_ip2)
    b2_L = _teno_B2(u_im2, u_im1, u_i)
    dL0, dL1, dL2 = _teno5_cutoff(b0_L, b1_L, b2_L)

    b1_R = _teno_B1(u_i, u_im1, u_im2)
    b2_R = _teno_B2(u_ip2, u_ip1, u_i)
    dR0, dR1, dR2 = _teno5_cutoff(b0_L, b1_R, b2_R)

    # cutoff_teno5_mc2: use TENO only if ALL stencils pass
    use_teno_L = (dL0 > 0.0 and dL1 > 0.0 and dL2 > 0.0)
    use_teno_R = (dR0 > 0.0 and dR1 > 0.0 and dR2 > 0.0)

    uR_computed = False

    if use_teno_L:
        denom = _DTENO_PW[0] * dL0 + _DTENO_PW[1] * dL1 + _DTENO_PW[2] * dL2
        inv_denom = 1.0 / max(denom, _EPSL)
        ukL0, ukL1, ukL2 = _teno5_stencils_L_pw(u_im2, u_im1, u_i, u_ip1, u_ip2)
        uL = inv_denom * (_DTENO_PW[0] * dL0 * ukL0 +
                          _DTENO_PW[1] * dL1 * ukL1 +
                          _DTENO_PW[2] * dL2 * ukL2)
    else:
        uL, uR = _koren_fallback_LR(u_im1, u_i, u_ip1)
        uR_computed = True

    if use_teno_R:
        denom = _DTENO_PW[0] * dR0 + _DTENO_PW[1] * dR1 + _DTENO_PW[2] * dR2
        inv_denom = 1.0 / max(denom, _EPSL)
        ukR0, ukR1, ukR2 = _teno5_stencils_R_pw(u_im2, u_im1, u_i, u_ip1, u_ip2)
        uR = inv_denom * (_DTENO_PW[0] * dR0 * ukR0 +
                          _DTENO_PW[1] * dR1 * ukR1 +
                          _DTENO_PW[2] * dR2 * ukR2)
        uR_computed = True
    elif not uR_computed:
        _, uR = _koren_fallback_LR(u_im1, u_i, u_ip1)

    return uL, uR


def _cutoff_teno5(dL, dR):
    """TENO5 cutoff: use TENO unless ALL stencils rejected.

    Reference implementation; cores inline the logic for performance.
    """
    use_L = not (dL[0] == 0.0 and dL[1] == 0.0 and dL[2] == 0.0)
    use_R = not (dR[0] == 0.0 and dR[1] == 0.0 and dR[2] == 0.0)
    return use_L, use_R


def _cutoff_teno5_mc2(dL, dR):
    """TENO5-MC2 cutoff: use TENO only if ALL stencils pass.

    Reference implementation; cores inline the logic for performance.
    """
    use_L = (dL[0] > 0.0 and dL[1] > 0.0 and dL[2] > 0.0)
    use_R = (dR[0] > 0.0 and dR[1] > 0.0 and dR[2] > 0.0)
    return use_L, use_R


# ---------------------------------------------------------------------------
# Public API: TENO5 (FV)
# ---------------------------------------------------------------------------

def teno5_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """TENO5 reconstruction (FV weights) at i+1/2.

    MC2 fallback only when ALL three stencils are rejected.

    """
    return _teno5_LR_core_teno5_fv(u_im2, u_im1, u_i, u_ip1, u_ip2)


def teno5_mc2_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """TENO5-MC2 reconstruction (FV weights) at i+1/2.

    MC2 fallback when ANY stencil is rejected.

    """
    return _teno5_LR_core_teno5_mc2_fv(u_im2, u_im1, u_i, u_ip1, u_ip2)


def teno5_koren_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """TENO5-Koren reconstruction (FV weights) at i+1/2.

    Koren limiter fallback when ANY stencil is rejected.

    """
    return _teno5_LR_core_teno5_koren_fv(u_im2, u_im1, u_i, u_ip1, u_ip2)


# ---------------------------------------------------------------------------
# Public API: TENO5 (PW)
# ---------------------------------------------------------------------------

def teno5_pw(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """TENO5 reconstruction (PW weights) at i+1/2.

    MC2 fallback only when ALL three stencils are rejected.

    """
    return _teno5_LR_core_teno5_pw(u_im2, u_im1, u_i, u_ip1, u_ip2)


def teno5_mc2_pw(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """TENO5-MC2 reconstruction (PW weights) at i+1/2.

    MC2 fallback when ANY stencil is rejected.

    """
    return _teno5_LR_core_teno5_mc2_pw(u_im2, u_im1, u_i, u_ip1, u_ip2)


def teno5_koren_pw(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """TENO5-Koren reconstruction (PW weights) at i+1/2.

    Koren limiter fallback when ANY stencil is rejected.

    """
    return _teno5_LR_core_teno5_koren_pw(u_im2, u_im1, u_i, u_ip1, u_ip2)
# ---------------------------------------------------------------------------
# JIT compilation (pynalgo convention)
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    reveal_locals()  # noqa: F821
else:
    _teno_B0 = JIT(_teno_B0)
    _teno_B1 = JIT(_teno_B1)
    _teno_B2 = JIT(_teno_B2)
    _teno5_cutoff = JIT(_teno5_cutoff)
    _mc2_fallback_LR = JIT(_mc2_fallback_LR)
    _koren_fallback_LR = JIT(_koren_fallback_LR)
    _teno5_stencils_L_fv = JIT(_teno5_stencils_L_fv)
    _teno5_stencils_R_fv = JIT(_teno5_stencils_R_fv)
    _teno5_stencils_L_pw = JIT(_teno5_stencils_L_pw)
    _teno5_stencils_R_pw = JIT(_teno5_stencils_R_pw)
    _teno5_LR_core_teno5_fv = JIT(_teno5_LR_core_teno5_fv)
    _teno5_LR_core_teno5_mc2_fv = JIT(_teno5_LR_core_teno5_mc2_fv)
    _teno5_LR_core_teno5_koren_fv = JIT(_teno5_LR_core_teno5_koren_fv)
    _teno5_LR_core_teno5_pw = JIT(_teno5_LR_core_teno5_pw)
    _teno5_LR_core_teno5_mc2_pw = JIT(_teno5_LR_core_teno5_mc2_pw)
    _teno5_LR_core_teno5_koren_pw = JIT(_teno5_LR_core_teno5_koren_pw)
    _cutoff_teno5 = JIT(_cutoff_teno5)
    _cutoff_teno5_mc2 = JIT(_cutoff_teno5_mc2)
    teno5_fv = JIT(teno5_fv)
    teno5_mc2_fv = JIT(teno5_mc2_fv)
    teno5_koren_fv = JIT(teno5_koren_fv)
    teno5_pw = JIT(teno5_pw)
    teno5_mc2_pw = JIT(teno5_mc2_pw)
    teno5_koren_pw = JIT(teno5_koren_pw)
#
# :D
#
