"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: WENO5-M reconstruction: Mapped WENO (Henrick, Aslam & Powers 2005)
"""
from pyrecon.recon_weno5 import (
    _js_smoothness, _stencils_fv, _stencils_pw,
    _OPTIMW_FV, _OPTIMW_PW)
from pyrecon._jit_utils import JIT, TYPE_CHECKING

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Epsilon for JS weights
_EPSL = 1e-40

# Power for JS unnormalized weights
_P = 2


# ---------------------------------------------------------------------------
# Henrick mapping function
# ---------------------------------------------------------------------------

def _henrick_map(omega, dk):
    """Henrick mapping function g_k(omega).

    g_k(omega) = omega * (dk + dk^2 - 3*dk*omega + omega^2)
                 / (dk^2 + omega * (1 - 2*dk))

    Maps [0,1] -> [0,1] with:
      g_k(0) = 0,  g_k(1) = 1
      g_k(dk) = dk,  g_k'(dk) = 0

    Args:
        omega: normalized JS weight (scalar)
        dk: optimal linear weight for this substencil
    Returns:
        mapped weight value

    Reference: Henrick et al. (2005).
    """
    dk2 = dk * dk
    num = omega * (dk + dk2 - 3.0 * dk * omega + omega * omega)
    den = dk2 + omega * (1.0 - 2.0 * dk)
    return num / den


# ---------------------------------------------------------------------------
# WENO5-M cores (FV + PW)
# ---------------------------------------------------------------------------

def _weno5m_LR_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """WENO5-M paired L+R reconstruction (FV weights)."""
    b0, b1, b2 = _js_smoothness(u_im2, u_im1, u_i, u_ip1, u_ip2)

    d0, d1, d2 = _OPTIMW_FV
    a0 = d0 / ((_EPSL + b0) ** _P)
    a1 = d1 / ((_EPSL + b1) ** _P)
    a2 = d2 / ((_EPSL + b2) ** _P)
    inv_sum = 1.0 / (a0 + a1 + a2)
    omega0 = a0 * inv_sum
    omega1 = a1 * inv_sum
    omega2 = a2 * inv_sum

    aM0 = _henrick_map(omega0, d0)
    aM1 = _henrick_map(omega1, d1)
    aM2 = _henrick_map(omega2, d2)

    inv_sum_M = 1.0 / (aM0 + aM1 + aM2)

    ukL0, ukL1, ukL2 = _stencils_fv(u_im2, u_im1, u_i, u_ip1, u_ip2)
    uL = inv_sum_M * (aM0 * ukL0 + aM1 * ukL1 + aM2 * ukL2)

    a0_R = d0 / ((_EPSL + b2) ** _P)
    a1_R = d1 / ((_EPSL + b1) ** _P)
    a2_R = d2 / ((_EPSL + b0) ** _P)
    inv_sum_R = 1.0 / (a0_R + a1_R + a2_R)
    omega0_R = a0_R * inv_sum_R
    omega1_R = a1_R * inv_sum_R
    omega2_R = a2_R * inv_sum_R

    aM0_R = _henrick_map(omega0_R, d0)
    aM1_R = _henrick_map(omega1_R, d1)
    aM2_R = _henrick_map(omega2_R, d2)
    inv_sum_MR = 1.0 / (aM0_R + aM1_R + aM2_R)

    ukR0, ukR1, ukR2 = _stencils_fv(u_ip2, u_ip1, u_i, u_im1, u_im2)
    uR = inv_sum_MR * (aM0_R * ukR0 + aM1_R * ukR1 + aM2_R * ukR2)

    return uL, uR


def _weno5m_LR_pw(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """WENO5-M paired L+R reconstruction (PW weights)."""
    b0, b1, b2 = _js_smoothness(u_im2, u_im1, u_i, u_ip1, u_ip2)

    d0, d1, d2 = _OPTIMW_PW
    a0 = d0 / ((_EPSL + b0) ** _P)
    a1 = d1 / ((_EPSL + b1) ** _P)
    a2 = d2 / ((_EPSL + b2) ** _P)
    inv_sum = 1.0 / (a0 + a1 + a2)
    omega0 = a0 * inv_sum
    omega1 = a1 * inv_sum
    omega2 = a2 * inv_sum

    aM0 = _henrick_map(omega0, d0)
    aM1 = _henrick_map(omega1, d1)
    aM2 = _henrick_map(omega2, d2)

    inv_sum_M = 1.0 / (aM0 + aM1 + aM2)

    ukL0, ukL1, ukL2 = _stencils_pw(u_im2, u_im1, u_i, u_ip1, u_ip2)
    uL = inv_sum_M * (aM0 * ukL0 + aM1 * ukL1 + aM2 * ukL2)

    a0_R = d0 / ((_EPSL + b2) ** _P)
    a1_R = d1 / ((_EPSL + b1) ** _P)
    a2_R = d2 / ((_EPSL + b0) ** _P)
    inv_sum_R = 1.0 / (a0_R + a1_R + a2_R)
    omega0_R = a0_R * inv_sum_R
    omega1_R = a1_R * inv_sum_R
    omega2_R = a2_R * inv_sum_R

    aM0_R = _henrick_map(omega0_R, d0)
    aM1_R = _henrick_map(omega1_R, d1)
    aM2_R = _henrick_map(omega2_R, d2)
    inv_sum_MR = 1.0 / (aM0_R + aM1_R + aM2_R)

    ukR0, ukR1, ukR2 = _stencils_pw(u_ip2, u_ip1, u_i, u_im1, u_im2)
    uR = inv_sum_MR * (aM0_R * ukR0 + aM1_R * ukR1 + aM2_R * ukR2)

    return uL, uR


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def weno5m_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """WENO5-M (Mapped WENO) reconstruction (FV) at i+1/2."""
    return _weno5m_LR_fv(u_im2, u_im1, u_i, u_ip1, u_ip2)


def weno5m_pw(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """WENO5-M (Mapped WENO) reconstruction (PW) at i+1/2."""
    return _weno5m_LR_pw(u_im2, u_im1, u_i, u_ip1, u_ip2)
# ---------------------------------------------------------------------------
# JIT compilation (pynalgo convention)
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    reveal_locals()  # noqa: F821
else:
    _henrick_map = JIT(_henrick_map)
    _weno5m_LR_fv = JIT(_weno5m_LR_fv)
    _weno5m_LR_pw = JIT(_weno5m_LR_pw)
    weno5m_fv = JIT(weno5m_fv)
    weno5m_pw = JIT(weno5m_pw)
#
# :D
#
