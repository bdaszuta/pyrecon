"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: WENO-AO(5,3) -- Adaptive Order WENO reconstruction
  Balsara, Garain & Shu, J. Comput. Phys. 326, 780-804 (2016)
"""
from pyrecon.recon_weno5 import _js_smoothness
from pyrecon._jit_utils import JIT, TYPE_CHECKING

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# FV stencil scale
_K_1_6 = 1.0 / 6.0

# PW stencil scales
_K_1_8 = 1.0 / 8.0          # PW sub-stencil scale
_K_1_128 = 1.0 / 128.0      # PW central stencil scale

# FV central stencil scale
_K_1_60 = 1.0 / 60.0        # FV central stencil scale

# WENO-AO(5,3) linear weight parameters (Eq 3.5)
# Gamma_Hi in [0.85, 0.95]; lower -> more robust for strong shocks
_GAMMA_HI = 0.85
_GAMMA_LO = 0.85

# Sub-stencil linear weights from Eq (3.5) with Gamma_Hi=0.85, Gamma_Lo=0.85
_GAMMA_1 = (1.0 - _GAMMA_HI) * (1.0 - _GAMMA_LO) / 2.0
_GAMMA_2 = (1.0 - _GAMMA_HI) * _GAMMA_LO
_GAMMA_3 = (1.0 - _GAMMA_HI) * (1.0 - _GAMMA_LO) / 2.0

# WENO-AO epsilon (paper: typically 1e-12)
_EPSL_AO = 1e-12

# Legendre smoothness indicator coefficients for r=5 central stencil (Eq 2.19)
_LEG_COEFF_A = 1.0 / 10.0        # ux3/10
_LEG_COEFF_B = 13.0 / 3.0        # 13/3
_LEG_COEFF_C = 123.0 / 455.0     # 123/455
_LEG_COEFF_D = 781.0 / 20.0      # 781/20
_LEG_COEFF_E = 1421461.0 / 2275.0  # 1421461/2275
_LEG_SCALE_UX = 1.0 / 120.0      # ux denominator
_LEG_SCALE_UX2 = 1.0 / 56.0      # ux2 denominator
_LEG_SCALE_UX3_UX4 = 1.0 / 12.0  # ux3 denominator (u_im2, u_im1, u_ip1, u_ip2)
_LEG_SCALE_UX4_CENTER = 1.0 / 24.0  # ux4 denominator (all 5 values)


# ---------------------------------------------------------------------------
# 3rd-order sub-stencil polynomials (FV)
# ---------------------------------------------------------------------------

def _sub_stencils_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """3rd-order sub-stencil polynomials (FV, left-biased).

    Used as small-stencil reconstructions in WENO-AO(5,3).

    Returns (u0, u1, u2):
      u0 = (2*u_im2 - 7*u_im1 + 11*u_i) / 6
      u1 = (-u_im1 + 5*u_i + 2*u_ip1) / 6
      u2 = (2*u_i + 5*u_ip1 - u_ip2) / 6
    """
    u0 = _K_1_6 * (2.0 * u_im2 - 7.0 * u_im1 + 11.0 * u_i)
    u1 = _K_1_6 * (-u_im1 + 5.0 * u_i + 2.0 * u_ip1)
    u2 = _K_1_6 * (2.0 * u_i + 5.0 * u_ip1 - u_ip2)
    return u0, u1, u2


# ---------------------------------------------------------------------------
# 3rd-order sub-stencil polynomials (PW)
# ---------------------------------------------------------------------------

def _sub_stencils_pw(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """3rd-order sub-stencil polynomials (PW, left-biased).

    Used as small-stencil reconstructions in WENO-AO(5,3).

    Returns (u0, u1, u2):
      u0 = (3*u_im2 - 10*u_im1 + 15*u_i) / 8
      u1 = (-u_im1 + 6*u_i + 3*u_ip1) / 8
      u2 = (3*u_i + 6*u_ip1 - u_ip2) / 8
    """
    u0 = _K_1_8 * (3.0 * u_im2 - 10.0 * u_im1 + 15.0 * u_i)
    u1 = _K_1_8 * (-u_im1 + 6.0 * u_i + 3.0 * u_ip1)
    u2 = _K_1_8 * (3.0 * u_i + 6.0 * u_ip1 - u_ip2)
    return u0, u1, u2


# ---------------------------------------------------------------------------
# 5th-order central stencil polynomial (FV)
# ---------------------------------------------------------------------------

def _central_stencil_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """5th-order full-stencil polynomial (FV) at i+1/2.

    In AO nomenclature, the "central" stencil is the full 5-point
    reconstruction window (vs. 3-point small stencils). The coefficients
    are upwind-biased at the face: (2, -13, 47, 27, -3)/60.
      (2*u_im2 - 13*u_im1 + 47*u_i + 27*u_ip1 - 3*u_ip2) / 60
    """
    return _K_1_60 * (2.0 * u_im2 - 13.0 * u_im1 + 47.0 * u_i +
                      27.0 * u_ip1 - 3.0 * u_ip2)


# ---------------------------------------------------------------------------
# 5th-order central stencil polynomial (PW)
# ---------------------------------------------------------------------------

def _central_stencil_pw(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """5th-order full-stencil polynomial (PW) at i+1/2.

    In AO nomenclature, the "central" stencil is the full 5-point
    reconstruction window (vs. 3-point small stencils). The coefficients
    are upwind-biased at the face: (3, -20, 90, 60, -5)/128.
      (3*u_im2 - 20*u_im1 + 90*u_i + 60*u_ip1 - 5*u_ip2) / 128
    """
    return _K_1_128 * (3.0 * u_im2 - 20.0 * u_im1 + 90.0 * u_i +
                       60.0 * u_ip1 - 5.0 * u_ip2)


# ---------------------------------------------------------------------------
# r=5 central stencil smoothness indicator (Legendre basis, Eq 2.19)
# ---------------------------------------------------------------------------

def _smoothness_r5_central(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """Smoothness indicator for the r=5 central stencil Sr5_3.

    Computes Legendre coefficients then the smoothness indicator.

    Returns beta_r5.

    Reference: Balsara, Garain & Shu (2016), Eqs. 2.16, 2.19.
    """
    # Legendre coefficients (Eq 2.16)
    ux = _LEG_SCALE_UX * (
        -82.0 * u_im1 + 11.0 * u_im2 + 82.0 * u_ip1 - 11.0 * u_ip2)
    ux2 = _LEG_SCALE_UX2 * (
        40.0 * u_im1 - 3.0 * u_im2 - 74.0 * u_i +
        40.0 * u_ip1 - 3.0 * u_ip2)
    ux3 = _LEG_SCALE_UX3_UX4 * (
        2.0 * u_im1 - u_im2 - 2.0 * u_ip1 + u_ip2)
    ux4 = _LEG_SCALE_UX4_CENTER * (
        -4.0 * u_im1 + u_im2 + 6.0 * u_i - 4.0 * u_ip1 + u_ip2)

    # Smoothness indicator (Eq 2.19)
    term1 = ux + _LEG_COEFF_A * ux3
    term2 = ux2 + _LEG_COEFF_C * ux4
    return (term1 * term1 +
            _LEG_COEFF_B * term2 * term2 +
            _LEG_COEFF_D * ux3 * ux3 +
            _LEG_COEFF_E * ux4 * ux4)


# ---------------------------------------------------------------------------
# WENO-AO(5,3) shared core (Balsara, Garain & Shu 2016, Section 3.1)
# ---------------------------------------------------------------------------

def _weno_ao53_LR_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """WENO-AO(5,3) paired L+R reconstruction (FV).

    Reference: Balsara, Garain & Shu, J. Comput. Phys. 326, 780-804 (2016),
    Eqs. (3.5)-(3.9).
    """
    # Sub-stencil smoothness indicators (JS, 3-point)
    b0, b1, b2 = _js_smoothness(u_im2, u_im1, u_i, u_ip1, u_ip2)

    # Central stencil smoothness (Legendre, Eq 2.19)
    b_opt = _smoothness_r5_central(u_im2, u_im1, u_i, u_ip1, u_ip2)

    # Linear weights (Eq 3.5)
    g_opt = _GAMMA_HI
    g_1 = _GAMMA_1
    g_2 = _GAMMA_2
    g_3 = _GAMMA_3

    # tau (Eq 3.6): average deviation of sub-stencil sm from central
    tau = (abs(b_opt - b0) + abs(b_opt - b1) + abs(b_opt - b2)) / 3.0

    # Non-linear weights (Eq 3.7a) -- p=2 with epsilon=1e-12
    tau2 = tau * tau
    w_opt = g_opt * (1.0 + tau2 / ((b_opt + _EPSL_AO) ** 2))
    w_1 = g_1 * (1.0 + tau2 / ((b0 + _EPSL_AO) ** 2))
    w_2 = g_2 * (1.0 + tau2 / ((b1 + _EPSL_AO) ** 2))
    w_3 = g_3 * (1.0 + tau2 / ((b2 + _EPSL_AO) ** 2))

    # Normalized non-linear weights (Eq 3.8)
    inv_sum = 1.0 / (w_opt + w_1 + w_2 + w_3)
    wb_opt = w_opt * inv_sum
    wb_1 = w_1 * inv_sum
    wb_2 = w_2 * inv_sum
    wb_3 = w_3 * inv_sum

    # Pre-compute subtraction scaling factor for Eq (3.9)
    sf = wb_opt / g_opt

    # --- Left face (i+1/2) ---
    p_opt_L = _central_stencil_fv(u_im2, u_im1, u_i, u_ip1, u_ip2)
    p_1_L, p_2_L, p_3_L = _sub_stencils_fv(u_im2, u_im1, u_i, u_ip1, u_ip2)

    # Reconstruction combination (Eq 3.9):
    uL = (sf * p_opt_L +
          (wb_1 - sf * g_1) * p_1_L +
          (wb_2 - sf * g_2) * p_2_L +
          (wb_3 - sf * g_3) * p_3_L)

    # --- Right face (i-1/2): mirror the stencil ---
    w_opt_R = g_opt * (1.0 + tau2 / ((b_opt + _EPSL_AO) ** 2))
    w_1_R = g_1 * (1.0 + tau2 / ((b2 + _EPSL_AO) ** 2))
    w_2_R = g_2 * (1.0 + tau2 / ((b1 + _EPSL_AO) ** 2))
    w_3_R = g_3 * (1.0 + tau2 / ((b0 + _EPSL_AO) ** 2))
    inv_sum_R = 1.0 / (w_opt_R + w_1_R + w_2_R + w_3_R)
    wb_opt_R = w_opt_R * inv_sum_R
    wb_1_R = w_1_R * inv_sum_R
    wb_2_R = w_2_R * inv_sum_R
    wb_3_R = w_3_R * inv_sum_R

    sf_R = wb_opt_R / g_opt

    p_opt_R = _central_stencil_fv(u_ip2, u_ip1, u_i, u_im1, u_im2)
    p_1_R, p_2_R, p_3_R = _sub_stencils_fv(u_ip2, u_ip1, u_i, u_im1, u_im2)

    uR = (sf_R * p_opt_R +
          (wb_1_R - sf_R * g_1) * p_1_R +
          (wb_2_R - sf_R * g_2) * p_2_R +
          (wb_3_R - sf_R * g_3) * p_3_R)

    return uL, uR


def _weno_ao53_LR_pw(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """WENO-AO(5,3) paired L+R reconstruction (PW).

    Reference: Balsara, Garain & Shu, J. Comput. Phys. 326, 780-804 (2016),
    Eqs. (3.5)-(3.9).
    """
    # Sub-stencil smoothness indicators (JS, 3-point)
    b0, b1, b2 = _js_smoothness(u_im2, u_im1, u_i, u_ip1, u_ip2)

    # Central stencil smoothness (Legendre, Eq 2.19)
    b_opt = _smoothness_r5_central(u_im2, u_im1, u_i, u_ip1, u_ip2)

    # Linear weights (Eq 3.5)
    g_opt = _GAMMA_HI
    g_1 = _GAMMA_1
    g_2 = _GAMMA_2
    g_3 = _GAMMA_3

    # tau (Eq 3.6): average deviation of sub-stencil sm from central
    tau = (abs(b_opt - b0) + abs(b_opt - b1) + abs(b_opt - b2)) / 3.0

    # Non-linear weights (Eq 3.7a) -- p=2 with epsilon=1e-12
    tau2 = tau * tau
    w_opt = g_opt * (1.0 + tau2 / ((b_opt + _EPSL_AO) ** 2))
    w_1 = g_1 * (1.0 + tau2 / ((b0 + _EPSL_AO) ** 2))
    w_2 = g_2 * (1.0 + tau2 / ((b1 + _EPSL_AO) ** 2))
    w_3 = g_3 * (1.0 + tau2 / ((b2 + _EPSL_AO) ** 2))

    # Normalized non-linear weights (Eq 3.8)
    inv_sum = 1.0 / (w_opt + w_1 + w_2 + w_3)
    wb_opt = w_opt * inv_sum
    wb_1 = w_1 * inv_sum
    wb_2 = w_2 * inv_sum
    wb_3 = w_3 * inv_sum

    # Pre-compute subtraction scaling factor for Eq (3.9)
    sf = wb_opt / g_opt

    # --- Left face (i+1/2) ---
    p_opt_L = _central_stencil_pw(u_im2, u_im1, u_i, u_ip1, u_ip2)
    p_1_L, p_2_L, p_3_L = _sub_stencils_pw(u_im2, u_im1, u_i, u_ip1, u_ip2)

    # Reconstruction combination (Eq 3.9):
    uL = (sf * p_opt_L +
          (wb_1 - sf * g_1) * p_1_L +
          (wb_2 - sf * g_2) * p_2_L +
          (wb_3 - sf * g_3) * p_3_L)

    # --- Right face (i-1/2): mirror the stencil ---
    w_opt_R = g_opt * (1.0 + tau2 / ((b_opt + _EPSL_AO) ** 2))
    w_1_R = g_1 * (1.0 + tau2 / ((b2 + _EPSL_AO) ** 2))
    w_2_R = g_2 * (1.0 + tau2 / ((b1 + _EPSL_AO) ** 2))
    w_3_R = g_3 * (1.0 + tau2 / ((b0 + _EPSL_AO) ** 2))
    inv_sum_R = 1.0 / (w_opt_R + w_1_R + w_2_R + w_3_R)
    wb_opt_R = w_opt_R * inv_sum_R
    wb_1_R = w_1_R * inv_sum_R
    wb_2_R = w_2_R * inv_sum_R
    wb_3_R = w_3_R * inv_sum_R

    sf_R = wb_opt_R / g_opt

    p_opt_R = _central_stencil_pw(u_ip2, u_ip1, u_i, u_im1, u_im2)
    p_1_R, p_2_R, p_3_R = _sub_stencils_pw(u_ip2, u_ip1, u_i, u_im1, u_im2)

    uR = (sf_R * p_opt_R +
          (wb_1_R - sf_R * g_1) * p_1_R +
          (wb_2_R - sf_R * g_2) * p_2_R +
          (wb_3_R - sf_R * g_3) * p_3_R)

    return uL, uR


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def weno_ao53_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """WENO-AO(5,3) reconstruction at i+1/2 (FV).

    5-point window combining 5th-order central stencil with
    3rd-order WENO sub-stencils.

    Reference: Balsara, Garain & Shu, J. Comput. Phys. 326, 780-804 (2016).
    """
    return _weno_ao53_LR_fv(u_im2, u_im1, u_i, u_ip1, u_ip2)


def weno_ao53_pw(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """WENO-AO(5,3) reconstruction at i+1/2 (PW).

    5-point window combining 5th-order central stencil with
    3rd-order WENO sub-stencils.

    Reference: Balsara, Garain & Shu, J. Comput. Phys. 326, 780-804 (2016).
    """
    return _weno_ao53_LR_pw(u_im2, u_im1, u_i, u_ip1, u_ip2)
# ---------------------------------------------------------------------------
# JIT compilation (pynalgo convention)
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    reveal_locals()  # noqa: F821
else:
    _sub_stencils_fv = JIT(_sub_stencils_fv)
    _sub_stencils_pw = JIT(_sub_stencils_pw)
    _central_stencil_fv = JIT(_central_stencil_fv)
    _central_stencil_pw = JIT(_central_stencil_pw)
    _smoothness_r5_central = JIT(_smoothness_r5_central)
    _weno_ao53_LR_fv = JIT(_weno_ao53_LR_fv)
    _weno_ao53_LR_pw = JIT(_weno_ao53_LR_pw)
    weno_ao53_fv = JIT(weno_ao53_fv)
    weno_ao53_pw = JIT(weno_ao53_pw)
#
# :D
#
