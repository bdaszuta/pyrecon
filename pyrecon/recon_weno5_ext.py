"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: WENO5 extension variants
"""
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

from pyrecon.recon_weno5 import _js_smoothness, _OPTIMW_FV, _stencils_fv
from pyrecon._jit_utils import JIT, TYPE_CHECKING

# Epsilon
_EPSL = 1e-40

# WENO5-ZC+ centering coefficients
_C_ZCP = (9/8, 9/4, 9/8)

# WENO5-NS xi parameter (Ha et al. 2013)
_K_NS_XI = 0.4


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

def _ns_smoothness(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """NS smoothness indicators. Returns (b0, b1, b2).

    Reference: Ha et al. (2013).
    """
    b0 = (_K_NS_XI * abs(u_im2 - 3.0 * u_im1 + 2.0 * u_i) +
          abs(u_im2 - 2.0 * u_im1 + u_i))
    b1 = (_K_NS_XI * abs(u_ip1 - u_i) +
          abs(u_im1 - 2.0 * u_i + u_ip1))
    b2 = (_K_NS_XI * abs(u_ip1 - u_i) +
          abs(u_i - 2.0 * u_ip1 + u_ip2))
    return b0, b1, b2


# ---------------------------------------------------------------------------
# WENO5-ZC+ (Barreto et al. 2023)
# ---------------------------------------------------------------------------

def weno5zcplus_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """WENO5-ZC+ at i+1/2.

    Reference: Barreto et al. (2023).
    """
    b0, b1, b2 = _js_smoothness(u_im2, u_im1, u_i, u_ip1, u_ip2)

    tau = abs(b0 - b2)
    bbar = (b0 + b1 + b2) / 3.0
    tau_factor = (tau / (tau + bbar + _EPSL)) ** 2

    aL0 = _OPTIMW_FV[0] * (
        1.0 + _C_ZCP[0] * (tau / (_EPSL + b0))**2 * tau_factor
        + b0 / (tau + bbar + _EPSL))
    aL1 = _OPTIMW_FV[1] * (
        1.0 + _C_ZCP[1] * (tau / (_EPSL + b1))**2 * tau_factor
        + b1 / (tau + bbar + _EPSL))
    aL2 = _OPTIMW_FV[2] * (
        1.0 + _C_ZCP[2] * (tau / (_EPSL + b2))**2 * tau_factor
        + b2 / (tau + bbar + _EPSL))
    inv_sum_L = 1.0 / (aL0 + aL1 + aL2)

    ukL0, ukL1, ukL2 = _stencils_fv(u_im2, u_im1, u_i, u_ip1, u_ip2)
    uL = inv_sum_L * (aL0 * ukL0 + aL1 * ukL1 + aL2 * ukL2)

    aR0 = _OPTIMW_FV[0] * (
        1.0 + _C_ZCP[0] * (tau / (_EPSL + b2))**2 * tau_factor
        + b2 / (tau + bbar + _EPSL))
    aR1 = _OPTIMW_FV[1] * (
        1.0 + _C_ZCP[1] * (tau / (_EPSL + b1))**2 * tau_factor
        + b1 / (tau + bbar + _EPSL))
    aR2 = _OPTIMW_FV[2] * (
        1.0 + _C_ZCP[2] * (tau / (_EPSL + b0))**2 * tau_factor
        + b0 / (tau + bbar + _EPSL))
    inv_sum_R = 1.0 / (aR0 + aR1 + aR2)

    ukR0, ukR1, ukR2 = _stencils_fv(u_ip2, u_ip1, u_i, u_im1, u_im2)
    uR = inv_sum_R * (aR0 * ukR0 + aR1 * ukR1 + aR2 * ukR2)

    return uL, uR


# ---------------------------------------------------------------------------
# WENO5-Z-NS (Ha et al. 2013)
# ---------------------------------------------------------------------------

def weno5z_ns_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """WENO5-Z-NS at i+1/2.

    Reference: Ha et al. (2013).
    """
    bL0, bL1, bL2 = _ns_smoothness(u_im2, u_im1, u_i, u_ip1, u_ip2)

    L11 = u_ip1 - u_i
    L113 = abs(L11) ** 3
    denom = 1.0 + L113    # always >= 1.0 (L113 >= 0); guard is defensive no-op
    if abs(denom) < _EPSL:
        g = 1.0 / _EPSL
    else:
        g = L113 / denom

    db_L = bL0 - bL2
    # g reused for both faces (symmetric first-difference measure)
    zetaL = 0.5 * (db_L ** 2 + g ** 2)

    aL0 = _OPTIMW_FV[0] * (1.0 + zetaL / ((_EPSL + bL0) ** 2))
    aL1 = _OPTIMW_FV[1] * (1.0 + zetaL / ((_EPSL + bL1) ** 2))
    aL2 = _OPTIMW_FV[2] * (1.0 + zetaL / ((_EPSL + bL2) ** 2))
    inv_sum_L = 1.0 / (aL0 + aL1 + aL2)

    ukL0, ukL1, ukL2 = _stencils_fv(u_im2, u_im1, u_i, u_ip1, u_ip2)
    uL = inv_sum_L * (aL0 * ukL0 + aL1 * ukL1 + aL2 * ukL2)

    bR0, bR1, bR2 = _ns_smoothness(u_ip2, u_ip1, u_i, u_im1, u_im2)
    db_R = bR0 - bR2
    zetaR = 0.5 * (db_R ** 2 + g ** 2)

    aR0 = _OPTIMW_FV[0] * (1.0 + zetaR / ((_EPSL + bR0) ** 2))
    aR1 = _OPTIMW_FV[1] * (1.0 + zetaR / ((_EPSL + bR1) ** 2))
    aR2 = _OPTIMW_FV[2] * (1.0 + zetaR / ((_EPSL + bR2) ** 2))
    inv_sum_R = 1.0 / (aR0 + aR1 + aR2)

    ukR0, ukR1, ukR2 = _stencils_fv(u_ip2, u_ip1, u_i, u_im1, u_im2)
    uR = inv_sum_R * (aR0 * ukR0 + aR1 * ukR1 + aR2 * ukR2)

    return uL, uR


# ---------------------------------------------------------------------------
# WENO5-Z+ (Hong, Ye & Ye 2020)
# ---------------------------------------------------------------------------

def weno5zp_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    r"""WENO5-Z+ at :math:`i+1/2` (FV).

    Uses Z-weights with exponent p=2 for stronger dissipation near
    discontinuities, extending Borges (2008) WENO-Z.

    Reference: Hong, Ye & Ye, J. Comput. Phys. (2020).
    """
    b0, b1, b2 = _js_smoothness(u_im2, u_im1, u_i, u_ip1, u_ip2)
    tau = abs(b0 - b2)

    zetaL0 = (tau / (_EPSL + b0)) ** 2
    zetaL1 = (tau / (_EPSL + b1)) ** 2
    zetaL2 = (tau / (_EPSL + b2)) ** 2

    aL0 = _OPTIMW_FV[0] * (1.0 + zetaL0)
    aL1 = _OPTIMW_FV[1] * (1.0 + zetaL1)
    aL2 = _OPTIMW_FV[2] * (1.0 + zetaL2)
    inv_sum_L = 1.0 / (aL0 + aL1 + aL2)

    ukL0, ukL1, ukL2 = _stencils_fv(u_im2, u_im1, u_i, u_ip1, u_ip2)
    uL = inv_sum_L * (aL0 * ukL0 + aL1 * ukL1 + aL2 * ukL2)

    zetaR0 = (tau / (_EPSL + b2)) ** 2
    zetaR1 = (tau / (_EPSL + b1)) ** 2
    zetaR2 = (tau / (_EPSL + b0)) ** 2

    aR0 = _OPTIMW_FV[0] * (1.0 + zetaR0)
    aR1 = _OPTIMW_FV[1] * (1.0 + zetaR1)
    aR2 = _OPTIMW_FV[2] * (1.0 + zetaR2)
    inv_sum_R = 1.0 / (aR0 + aR1 + aR2)

    ukR0, ukR1, ukR2 = _stencils_fv(u_ip2, u_ip1, u_i, u_im1, u_im2)
    uR = inv_sum_R * (aR0 * ukR0 + aR1 * ukR1 + aR2 * ukR2)

    return uL, uR


# ---------------------------------------------------------------------------
# WENO5-Ha-JS (Ha et al. 2013)
# ---------------------------------------------------------------------------

def weno5_ha_js_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """WENO5-Ha-JS at i+1/2.

    Reference: Ha et al. (2013).
    """
    bL0, bL1, bL2 = _ns_smoothness(u_im2, u_im1, u_i, u_ip1, u_ip2)

    aL0 = _OPTIMW_FV[0] / ((_EPSL + bL0) ** 2)
    aL1 = _OPTIMW_FV[1] / ((_EPSL + bL1) ** 2)
    aL2 = _OPTIMW_FV[2] / ((_EPSL + bL2) ** 2)
    inv_sum_L = 1.0 / (aL0 + aL1 + aL2)

    ukL0, ukL1, ukL2 = _stencils_fv(u_im2, u_im1, u_i, u_ip1, u_ip2)
    uL = inv_sum_L * (aL0 * ukL0 + aL1 * ukL1 + aL2 * ukL2)

    bR0, bR1, bR2 = _ns_smoothness(u_ip2, u_ip1, u_i, u_im1, u_im2)

    aR0 = _OPTIMW_FV[0] / ((_EPSL + bR0) ** 2)
    aR1 = _OPTIMW_FV[1] / ((_EPSL + bR1) ** 2)
    aR2 = _OPTIMW_FV[2] / ((_EPSL + bR2) ** 2)
    inv_sum_R = 1.0 / (aR0 + aR1 + aR2)

    ukR0, ukR1, ukR2 = _stencils_fv(u_ip2, u_ip1, u_i, u_im1, u_im2)
    uR = inv_sum_R * (aR0 * ukR0 + aR1 * ukR1 + aR2 * ukR2)

    return uL, uR


# ---------------------------------------------------------------------------
# WENO5-CZ (Barreto et al. 2023); function name "cz" matches registry key
# but internally the algorithm variable is _C_ZC (centering coefficients).
# ---------------------------------------------------------------------------

def weno5cz_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """WENO5-ZC at i+1/2.

    Reference: Barreto et al. (2023).
    """
    b0, b1, b2 = _js_smoothness(u_im2, u_im1, u_i, u_ip1, u_ip2)

    tau = abs(b0 - b2)
    bbar = (b0 + b1 + b2) / 3.0
    tau_factor = (tau / (tau + bbar + _EPSL)) ** 2
    _C_ZC = (3/4, 3/2, 3/4)

    aL0 = _OPTIMW_FV[0] * (
        1.0 + _C_ZC[0] * (tau / (_EPSL + b0))**2 * tau_factor)
    aL1 = _OPTIMW_FV[1] * (
        1.0 + _C_ZC[1] * (tau / (_EPSL + b1))**2 * tau_factor)
    aL2 = _OPTIMW_FV[2] * (
        1.0 + _C_ZC[2] * (tau / (_EPSL + b2))**2 * tau_factor)
    inv_sum_L = 1.0 / (aL0 + aL1 + aL2)

    ukL0, ukL1, ukL2 = _stencils_fv(u_im2, u_im1, u_i, u_ip1, u_ip2)
    uL = inv_sum_L * (aL0 * ukL0 + aL1 * ukL1 + aL2 * ukL2)

    aR0 = _OPTIMW_FV[0] * (
        1.0 + _C_ZC[0] * (tau / (_EPSL + b2))**2 * tau_factor)
    aR1 = _OPTIMW_FV[1] * (
        1.0 + _C_ZC[1] * (tau / (_EPSL + b1))**2 * tau_factor)
    aR2 = _OPTIMW_FV[2] * (
        1.0 + _C_ZC[2] * (tau / (_EPSL + b0))**2 * tau_factor)
    inv_sum_R = 1.0 / (aR0 + aR1 + aR2)

    ukR0, ukR1, ukR2 = _stencils_fv(u_ip2, u_ip1, u_i, u_im1, u_im2)
    uR = inv_sum_R * (aR0 * ukR0 + aR1 * ukR1 + aR2 * ukR2)

    return uL, uR


# ---------------------------------------------------------------------------
# WENO5-BC (Barreto et al. 2023) -- FV adaptation of centered JS
# ("BC" = Blended-Centered)
#
# The (tau/(tau+beta_bar))^2 gate makes the centering vanish as O(h^4) in
# smooth regions (where tau << beta_bar), recovering 5th-order optimal
# weights. Near discontinuities tau ~ beta_bar so centering engages.
# ---------------------------------------------------------------------------

def weno5_bc_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """WENO5-BC FV at i+1/2.

    Centered JS with a smoothness gate: the centering coefficients
    (3/4, 3/2, 3/4) are blended toward 1 in smooth regions via a
    (tau/(tau+beta_bar))^2 factor, recovering full 5th-order convergence on smooth profiles.

    Reference: Barreto et al. (2023).
    """
    b0, b1, b2 = _js_smoothness(u_im2, u_im1, u_i, u_ip1, u_ip2)
    tau = abs(b0 - b2)
    bbar = (b0 + b1 + b2) / 3.0
    gate = (tau / (tau + bbar + _EPSL)) ** 2
    _C_CC = (3/4, 3/2, 3/4)

    def _alpha(k, beta_k):
        return (_OPTIMW_FV[k] / ((_EPSL + beta_k) ** 2) *
                (1.0 + (_C_CC[k] - 1.0) * gate))

    aL0 = _alpha(0, b0)
    aL1 = _alpha(1, b1)
    aL2 = _alpha(2, b2)
    inv_sum_L = 1.0 / (aL0 + aL1 + aL2)

    ukL0, ukL1, ukL2 = _stencils_fv(u_im2, u_im1, u_i, u_ip1, u_ip2)
    uL = inv_sum_L * (aL0 * ukL0 + aL1 * ukL1 + aL2 * ukL2)

    aR0 = _alpha(0, b2)
    aR1 = _alpha(1, b1)
    aR2 = _alpha(2, b0)
    inv_sum_R = 1.0 / (aR0 + aR1 + aR2)

    ukR0, ukR1, ukR2 = _stencils_fv(u_ip2, u_ip1, u_i, u_im1, u_im2)
    uR = inv_sum_R * (aR0 * ukR0 + aR1 * ukR1 + aR2 * ukR2)

    return uL, uR
# ---------------------------------------------------------------------------
# JIT compilation (pynalgo convention)
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    reveal_locals()  # noqa: F821
else:
    _ns_smoothness = JIT(_ns_smoothness)
    weno5zcplus_fv = JIT(weno5zcplus_fv)
    weno5z_ns_fv = JIT(weno5z_ns_fv)
    weno5zp_fv = JIT(weno5zp_fv)
    weno5_ha_js_fv = JIT(weno5_ha_js_fv)
    weno5cz_fv = JIT(weno5cz_fv)
    weno5_bc_fv = JIT(weno5_bc_fv)
#
# :D
#
