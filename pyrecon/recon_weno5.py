"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: WENO5 reconstruction methods
"""
import math
from pyrecon._jit_utils import JIT, TYPE_CHECKING

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OPTIMW_FV = (1/10, 3/5, 3/10)
_EPSL = 1e-40
_EPSL_Z = 1e-12
_K_1_6 = 1.0 / 6.0
_K_1_8 = 1.0 / 8.0
# Pointwise optimal weights
_OPTIMW_PW = (1.0 / 16.0, 5.0 / 8.0, 5.0 / 16.0)

# WENO5-D-SI parameters (Don et al. 2022)
_W5D_SI_EPSL = 1e-12
_W5D_SI_P = 2
_W5D_SI_S = 1
_W5D_SI_MU_0 = 1e-40


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

_JS3_DENOM = 6.0
_JS3_M = (
    ((8, -19, 11), (-19, 50, -31), (11, -31, 20)),
    ((8, -13, 5),  (-13, 26, -13), (5, -13, 8)),
    ((20, -31, 11), (-31, 50, -19), (11, -19, 8)),
)


def _is3(vals, k):
    """Jiang-Shu smoothness for one 3-point sub-stencil."""
    m = _JS3_M[k]
    beta = 0.0
    for i in range(3):
        row = m[i]
        vi = vals[i]
        for j in range(3):
            beta += row[j] * vi * vals[j]
    return beta / _JS3_DENOM


def _js_smoothness(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """Jiang-Shu smoothness for 3-point sub-stencils. Returns (b0,b1,b2)."""
    b0 = _is3((u_im2, u_im1, u_i), 0)
    b1 = _is3((u_im1, u_i, u_ip1), 1)
    b2 = _is3((u_i, u_ip1, u_ip2), 2)
    return b0, b1, b2


def _stencils_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """WENO5 FV candidate stencil polynomials (left-biased)."""
    u0 = _K_1_6 * (2.0 * u_im2 - 7.0 * u_im1 + 11.0 * u_i)
    u1 = _K_1_6 * (-u_im1 + 5.0 * u_i + 2.0 * u_ip1)
    u2 = _K_1_6 * (2.0 * u_i + 5.0 * u_ip1 - u_ip2)
    return u0, u1, u2


def _stencils_pw(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """WENO5 PW candidate stencil polynomials (left-biased)."""
    u0 = _K_1_8 * (3.0 * u_im2 - 10.0 * u_im1 + 15.0 * u_i)
    u1 = _K_1_8 * (-u_im1 + 6.0 * u_i + 3.0 * u_ip1)
    u2 = _K_1_8 * (3.0 * u_i + 6.0 * u_ip1 - u_ip2)
    return u0, u1, u2


# ---------------------------------------------------------------------------
# WENO5-JS cores (FV + PW)
# ---------------------------------------------------------------------------

def _weno5_js_LR_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """WENO5-JS paired L+R reconstruction (FV weights)."""
    b0, b1, b2 = _js_smoothness(u_im2, u_im1, u_i, u_ip1, u_ip2)

    d0, d1, d2 = _OPTIMW_FV
    aL0 = d0 / ((_EPSL + b0) ** 2)
    aL1 = d1 / ((_EPSL + b1) ** 2)
    aL2 = d2 / ((_EPSL + b2) ** 2)
    inv_sum_L = 1.0 / (aL0 + aL1 + aL2)

    ukL0, ukL1, ukL2 = _stencils_fv(u_im2, u_im1, u_i, u_ip1, u_ip2)
    uL = inv_sum_L * (aL0 * ukL0 + aL1 * ukL1 + aL2 * ukL2)

    aR0 = d0 / ((_EPSL + b2) ** 2)
    aR1 = d1 / ((_EPSL + b1) ** 2)
    aR2 = d2 / ((_EPSL + b0) ** 2)
    inv_sum_R = 1.0 / (aR0 + aR1 + aR2)

    ukR0, ukR1, ukR2 = _stencils_fv(u_ip2, u_ip1, u_i, u_im1, u_im2)
    uR = inv_sum_R * (aR0 * ukR0 + aR1 * ukR1 + aR2 * ukR2)

    return uL, uR


def _weno5_js_LR_pw(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """WENO5-JS paired L+R reconstruction (PW weights)."""
    b0, b1, b2 = _js_smoothness(u_im2, u_im1, u_i, u_ip1, u_ip2)

    d0, d1, d2 = _OPTIMW_PW
    aL0 = d0 / ((_EPSL + b0) ** 2)
    aL1 = d1 / ((_EPSL + b1) ** 2)
    aL2 = d2 / ((_EPSL + b2) ** 2)
    inv_sum_L = 1.0 / (aL0 + aL1 + aL2)

    ukL0, ukL1, ukL2 = _stencils_pw(u_im2, u_im1, u_i, u_ip1, u_ip2)
    uL = inv_sum_L * (aL0 * ukL0 + aL1 * ukL1 + aL2 * ukL2)

    aR0 = d0 / ((_EPSL + b2) ** 2)
    aR1 = d1 / ((_EPSL + b1) ** 2)
    aR2 = d2 / ((_EPSL + b0) ** 2)
    inv_sum_R = 1.0 / (aR0 + aR1 + aR2)

    ukR0, ukR1, ukR2 = _stencils_pw(u_ip2, u_ip1, u_i, u_im1, u_im2)
    uR = inv_sum_R * (aR0 * ukR0 + aR1 * ukR1 + aR2 * ukR2)

    return uL, uR


# ---------------------------------------------------------------------------
# WENO5-Z cores (FV + PW)
# ---------------------------------------------------------------------------

def _weno5z_LR_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """WENO5-Z paired L+R reconstruction (FV weights)."""
    b0, b1, b2 = _js_smoothness(u_im2, u_im1, u_i, u_ip1, u_ip2)
    tau = abs(b0 - b2)

    d0, d1, d2 = _OPTIMW_FV
    aL0 = d0 * (1.0 + tau / (_EPSL_Z + b0))
    aL1 = d1 * (1.0 + tau / (_EPSL_Z + b1))
    aL2 = d2 * (1.0 + tau / (_EPSL_Z + b2))
    inv_sum_L = 1.0 / (aL0 + aL1 + aL2)

    ukL0, ukL1, ukL2 = _stencils_fv(u_im2, u_im1, u_i, u_ip1, u_ip2)
    uL = inv_sum_L * (aL0 * ukL0 + aL1 * ukL1 + aL2 * ukL2)

    aR0 = d0 * (1.0 + tau / (_EPSL_Z + b2))
    aR1 = d1 * (1.0 + tau / (_EPSL_Z + b1))
    aR2 = d2 * (1.0 + tau / (_EPSL_Z + b0))
    inv_sum_R = 1.0 / (aR0 + aR1 + aR2)

    ukR0, ukR1, ukR2 = _stencils_fv(u_ip2, u_ip1, u_i, u_im1, u_im2)
    uR = inv_sum_R * (aR0 * ukR0 + aR1 * ukR1 + aR2 * ukR2)

    return uL, uR


def _weno5z_LR_pw(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """WENO5-Z paired L+R reconstruction (PW weights)."""
    b0, b1, b2 = _js_smoothness(u_im2, u_im1, u_i, u_ip1, u_ip2)
    tau = abs(b0 - b2)

    d0, d1, d2 = _OPTIMW_PW
    aL0 = d0 * (1.0 + tau / (_EPSL_Z + b0))
    aL1 = d1 * (1.0 + tau / (_EPSL_Z + b1))
    aL2 = d2 * (1.0 + tau / (_EPSL_Z + b2))
    inv_sum_L = 1.0 / (aL0 + aL1 + aL2)

    ukL0, ukL1, ukL2 = _stencils_pw(u_im2, u_im1, u_i, u_ip1, u_ip2)
    uL = inv_sum_L * (aL0 * ukL0 + aL1 * ukL1 + aL2 * ukL2)

    aR0 = d0 * (1.0 + tau / (_EPSL_Z + b2))
    aR1 = d1 * (1.0 + tau / (_EPSL_Z + b1))
    aR2 = d2 * (1.0 + tau / (_EPSL_Z + b0))
    inv_sum_R = 1.0 / (aR0 + aR1 + aR2)

    ukR0, ukR1, ukR2 = _stencils_pw(u_ip2, u_ip1, u_i, u_im1, u_im2)
    uR = inv_sum_R * (aR0 * ukR0 + aR1 * ukR1 + aR2 * ukR2)

    return uL, uR


# ---------------------------------------------------------------------------
# WENO5-D-SI cores (FV + PW)
# ---------------------------------------------------------------------------

def _weno5d_si_LR_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    r"""WENO5-D-SI paired L+R reconstruction (FV weights).

    Uses :math:`\tau_5 = |\beta_0 - \beta_2|` (Borges-type tau).

    Reference: Don et al., J. Comput. Phys. (2022).
    """
    b0, b1, b2 = _js_smoothness(u_im2, u_im1, u_i, u_ip1, u_ip2)

    phi = math.sqrt(abs(b0 - 2.0 * b1 + b2))
    tau = abs(b0 - b2)

    xi = (abs(u_im2) + abs(u_im1) + abs(u_i) +
          abs(u_ip1) + abs(u_ip2)) / 5.0
    mu = xi + _W5D_SI_MU_0
    mu2 = mu * mu
    Phi = min(1.0, phi / mu)
    eps_mu2 = _W5D_SI_EPSL * mu2

    Z0 = (tau / (b0 + eps_mu2)) ** _W5D_SI_P
    Z1 = (tau / (b1 + eps_mu2)) ** _W5D_SI_P
    Z2 = (tau / (b2 + eps_mu2)) ** _W5D_SI_P

    d0, d1, d2 = _OPTIMW_FV
    aL0 = d0 * (1.0 + Phi * Z0) ** _W5D_SI_S
    aL1 = d1 * (1.0 + Phi * Z1) ** _W5D_SI_S
    aL2 = d2 * (1.0 + Phi * Z2) ** _W5D_SI_S
    inv_sum_L = 1.0 / (aL0 + aL1 + aL2)

    ukL0, ukL1, ukL2 = _stencils_fv(u_im2, u_im1, u_i, u_ip1, u_ip2)
    uL = inv_sum_L * (aL0 * ukL0 + aL1 * ukL1 + aL2 * ukL2)

    aR0 = d0 * (1.0 + Phi * Z2) ** _W5D_SI_S
    aR1 = d1 * (1.0 + Phi * Z1) ** _W5D_SI_S
    aR2 = d2 * (1.0 + Phi * Z0) ** _W5D_SI_S
    inv_sum_R = 1.0 / (aR0 + aR1 + aR2)

    ukR0, ukR1, ukR2 = _stencils_fv(u_ip2, u_ip1, u_i, u_im1, u_im2)
    uR = inv_sum_R * (aR0 * ukR0 + aR1 * ukR1 + aR2 * ukR2)

    return uL, uR


def _weno5d_si_LR_pw(u_im2, u_im1, u_i, u_ip1, u_ip2):
    r"""WENO5-D-SI paired L+R reconstruction (PW weights).

    Uses :math:`\tau_5 = |\beta_0 - \beta_2|` (Borges-type tau).

    Reference: Don et al., J. Comput. Phys. (2022).
    """
    b0, b1, b2 = _js_smoothness(u_im2, u_im1, u_i, u_ip1, u_ip2)

    phi = math.sqrt(abs(b0 - 2.0 * b1 + b2))
    tau = abs(b0 - b2)

    xi = (abs(u_im2) + abs(u_im1) + abs(u_i) +
          abs(u_ip1) + abs(u_ip2)) / 5.0
    mu = xi + _W5D_SI_MU_0
    mu2 = mu * mu
    Phi = min(1.0, phi / mu)
    eps_mu2 = _W5D_SI_EPSL * mu2

    Z0 = (tau / (b0 + eps_mu2)) ** _W5D_SI_P
    Z1 = (tau / (b1 + eps_mu2)) ** _W5D_SI_P
    Z2 = (tau / (b2 + eps_mu2)) ** _W5D_SI_P

    d0, d1, d2 = _OPTIMW_PW
    aL0 = d0 * (1.0 + Phi * Z0) ** _W5D_SI_S
    aL1 = d1 * (1.0 + Phi * Z1) ** _W5D_SI_S
    aL2 = d2 * (1.0 + Phi * Z2) ** _W5D_SI_S
    inv_sum_L = 1.0 / (aL0 + aL1 + aL2)

    ukL0, ukL1, ukL2 = _stencils_pw(u_im2, u_im1, u_i, u_ip1, u_ip2)
    uL = inv_sum_L * (aL0 * ukL0 + aL1 * ukL1 + aL2 * ukL2)

    aR0 = d0 * (1.0 + Phi * Z2) ** _W5D_SI_S
    aR1 = d1 * (1.0 + Phi * Z1) ** _W5D_SI_S
    aR2 = d2 * (1.0 + Phi * Z0) ** _W5D_SI_S
    inv_sum_R = 1.0 / (aR0 + aR1 + aR2)

    ukR0, ukR1, ukR2 = _stencils_pw(u_ip2, u_ip1, u_i, u_im1, u_im2)
    uR = inv_sum_R * (aR0 * ukR0 + aR1 * ukR1 + aR2 * ukR2)

    return uL, uR


# ---------------------------------------------------------------------------
# Public API: FV variants
# ---------------------------------------------------------------------------

def weno5_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """WENO5-JS reconstruction (FV weights) at i+1/2."""
    return _weno5_js_LR_fv(u_im2, u_im1, u_i, u_ip1, u_ip2)


def weno5z_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """WENO5-Z reconstruction (FV weights) at i+1/2."""
    return _weno5z_LR_fv(u_im2, u_im1, u_i, u_ip1, u_ip2)


def weno5d_si_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """WENO5-D-SI reconstruction (FV weights) at i+1/2."""
    return _weno5d_si_LR_fv(u_im2, u_im1, u_i, u_ip1, u_ip2)


# ---------------------------------------------------------------------------
# Public API: PW variants
# ---------------------------------------------------------------------------

def weno5_pw(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """WENO5-JS reconstruction (PW weights) at i+1/2."""
    return _weno5_js_LR_pw(u_im2, u_im1, u_i, u_ip1, u_ip2)


def weno5z_pw(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """WENO5-Z reconstruction (PW weights) at i+1/2."""
    return _weno5z_LR_pw(u_im2, u_im1, u_i, u_ip1, u_ip2)


def weno5d_si_pw(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """WENO5-D-SI reconstruction (PW weights) at i+1/2."""
    return _weno5d_si_LR_pw(u_im2, u_im1, u_i, u_ip1, u_ip2)

# ---------------------------------------------------------------------------
# JIT compilation (pynalgo convention)
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    reveal_locals()  # noqa: F821
else:
    _is3 = JIT(_is3)
    _js_smoothness = JIT(_js_smoothness)
    _stencils_fv = JIT(_stencils_fv)
    _stencils_pw = JIT(_stencils_pw)
    _weno5_js_LR_fv = JIT(_weno5_js_LR_fv)
    _weno5_js_LR_pw = JIT(_weno5_js_LR_pw)
    _weno5z_LR_fv = JIT(_weno5z_LR_fv)
    _weno5z_LR_pw = JIT(_weno5z_LR_pw)
    _weno5d_si_LR_fv = JIT(_weno5d_si_LR_fv)
    _weno5d_si_LR_pw = JIT(_weno5d_si_LR_pw)
    weno5_fv = JIT(weno5_fv)
    weno5z_fv = JIT(weno5z_fv)
    weno5d_si_fv = JIT(weno5d_si_fv)
    weno5_pw = JIT(weno5_pw)
    weno5z_pw = JIT(weno5z_pw)
    weno5d_si_pw = JIT(weno5d_si_pw)
#
# :D
#
