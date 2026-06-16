"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: WENO5-Z with p=2 -- WENO-Z type weights at power 2.
           (Not an energy-stable method; filename retained for
            import compatibility.)
"""
from pyrecon._jit_utils import JIT, TYPE_CHECKING
from pyrecon.recon_weno5 import _js_smoothness

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_K_1_6 = 1.0 / 6.0

# Optimal linear weights (FV)
_OW5_FV = (1.0 / 10.0, 3.0 / 5.0, 3.0 / 10.0)

# WENO-Z exponent p=2 for stronger dissipation than p=1.
_P = 2

_EPSL = 1e-40
_EPSL_Z = 1e-12


# ---------------------------------------------------------------------------
# WENO5-Z p=2 tau indicator
# ---------------------------------------------------------------------------

def _esweno_tau5(b0, b1, b2):
    """Global smoothness indicator (WENO-Z tau for 5-point stencil).

    tau5 = |b0 - b2|  -- L/R symmetric.
    """
    return abs(b0 - b2)


# ---------------------------------------------------------------------------
# Stencil polynomials
# ---------------------------------------------------------------------------

def _stencils_L(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """Left-face (i+1/2) stencil polynomials for 5-point WENO.

    Returns (uk0, uk1, uk2).
    """
    uk0 = _K_1_6 * (2.0 * u_im2 - 7.0 * u_im1 + 11.0 * u_i)
    uk1 = _K_1_6 * (-u_im1 + 5.0 * u_i + 2.0 * u_ip1)
    uk2 = _K_1_6 * (2.0 * u_i + 5.0 * u_ip1 - u_ip2)
    return uk0, uk1, uk2


def _stencils_R(u_ip2, u_ip1, u_i, u_im1, u_im2):
    """Right-face (i-1/2) stencil polynomials (reversed arguments)."""
    return _stencils_L(u_ip2, u_ip1, u_i, u_im1, u_im2)


# ---------------------------------------------------------------------------
# WENO5-Z p=2 paired L+R reconstruction core
# ---------------------------------------------------------------------------

def _esweno5_LR(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """WENO5-Z p=2 paired L+R reconstruction.

    alpha_k = d_k * (1 + (tau / (eps + beta_k))^p)  with p=2.
    """
    b0, b1, b2 = _js_smoothness(u_im2, u_im1, u_i, u_ip1, u_ip2)
    tau = _esweno_tau5(b0, b1, b2)

    aL0 = _OW5_FV[0] * (1.0 + (tau / (_EPSL_Z + b0)) ** _P)
    aL1 = _OW5_FV[1] * (1.0 + (tau / (_EPSL_Z + b1)) ** _P)
    aL2 = _OW5_FV[2] * (1.0 + (tau / (_EPSL_Z + b2)) ** _P)
    inv_sum_L = 1.0 / (aL0 + aL1 + aL2)

    ukL0, ukL1, ukL2 = _stencils_L(u_im2, u_im1, u_i, u_ip1, u_ip2)
    uL = inv_sum_L * (aL0 * ukL0 + aL1 * ukL1 + aL2 * ukL2)

    aR0 = _OW5_FV[0] * (1.0 + (tau / (_EPSL_Z + b2)) ** _P)
    aR1 = _OW5_FV[1] * (1.0 + (tau / (_EPSL_Z + b1)) ** _P)
    aR2 = _OW5_FV[2] * (1.0 + (tau / (_EPSL_Z + b0)) ** _P)
    inv_sum_R = 1.0 / (aR0 + aR1 + aR2)

    ukR0, ukR1, ukR2 = _stencils_R(u_ip2, u_ip1, u_i, u_im1, u_im2)
    uR = inv_sum_R * (aR0 * ukR0 + aR1 * ukR1 + aR2 * ukR2)

    return uL, uR


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def weno5z_p2_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """WENO5-Z with exponent p=2 (FV weights).

    Use weighted essentially non-oscillatory method with exponent p=2
    (ratio squared) for stronger dissipation near discontinuities.

    Reference: Borges et al. (2008).
    """
    return _esweno5_LR(u_im2, u_im1, u_i, u_ip1, u_ip2)
# ---------------------------------------------------------------------------
# JIT compilation (pynalgo convention)
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    reveal_locals()  # noqa: F821
else:
    _esweno_tau5 = JIT(_esweno_tau5)
    _stencils_L = JIT(_stencils_L)
    _stencils_R = JIT(_stencils_R)
    _esweno5_LR = JIT(_esweno5_LR)
    weno5z_p2_fv = JIT(weno5z_p2_fv)
#
# :D
#
