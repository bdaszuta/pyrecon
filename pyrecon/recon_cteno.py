"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: CTENO reconstruction methods
"""
from pyrecon._jit_utils import JIT, TYPE_CHECKING
from pyrecon.recon_weno5 import _js_smoothness, _stencils_fv

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Sharp cutoff threshold (CTENO literature value)
_C_T = 1e-6

# Epsilon for smoothness denominator (double-precision safe)
_EPSL = 1e-12

# Scale-separation exponent for CTENO5
_POW = 6

_K_1_60 = 1.0 / 60.0


# ---------------------------------------------------------------------------
# Central stencil polynomial
# ---------------------------------------------------------------------------

def _central_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    r"""Optimal 5th-order WENO polynomial at face i+1/2 (FV).

    .. math::
       p_{\text{central}} = (2 u_{i-2} - 13 u_{i-1} + 47 u_i + 27 u_{i+1} - 3 u_{i+2}) / 60
    """
    return _K_1_60 * (2.0 * u_im2 - 13.0 * u_im1 +
                      47.0 * u_i + 27.0 * u_ip1 - 3.0 * u_ip2)


# ---------------------------------------------------------------------------
# Scale-separation functions (retained for testing; cores inline the logic)
# ---------------------------------------------------------------------------

def _gamma_cteno5(SI):
    """CTENO5 scale separation: gamma_k = 1/(SI_k + eps)^6"""
    return [1.0 / (si + _EPSL) ** _POW for si in SI]


def _gamma_cteno5z(SI):
    """CTENO5Z scale separation: gamma_k = 1 + tau/(SI_k + eps)

    tau = (avg|SI_central - SI_k|)^6
    """
    bc = SI[0]
    tau = (abs(bc - SI[1]) + abs(bc - SI[2]) + abs(bc - SI[3])) / 3.0
    tau = tau ** _POW
    return [1.0 + tau / (si + _EPSL) for si in SI]


# ---------------------------------------------------------------------------
# CTENO5 core (gamma = 1/(SI+eps)^6)
# ---------------------------------------------------------------------------

def _cteno5_core(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """Core CTENO5 reconstruction for a single face.

    Scale separation: gamma_k = 1/(SI_k + eps)^6
    """
    p0, p1, p2 = _stencils_fv(u_im2, u_im1, u_i, u_ip1, u_ip2)
    pc = _central_fv(u_im2, u_im1, u_i, u_ip1, u_ip2)
    b0, b1, b2 = _js_smoothness(u_im2, u_im1, u_i, u_ip1, u_ip2)
    bc = max(b0, b1, b2)

    SI = [bc, b0, b1, b2]
    g = [1.0 / (si + _EPSL) ** _POW for si in SI]

    g_sum = g[0] + g[1] + g[2] + g[3]
    inv_g_sum = 1.0 / g_sum
    chi = [gk * inv_g_sum for gk in g]

    delta = [1.0 if c >= _C_T else 0.0 for c in chi]

    if delta[0] == 1.0:
        return pc

    n_pass = delta[1] + delta[2] + delta[3]
    if n_pass > 0.0:
        result = (delta[1] * p0 + delta[2] * p1 + delta[3] * p2) / n_pass
    else:
        result = p1

    return result


# ---------------------------------------------------------------------------
# CTENO5Z core (gamma = 1 + tau/(SI+eps), tau^6 scaling)
# ---------------------------------------------------------------------------

def _cteno5z_core(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """Core CTENO5Z reconstruction for a single face.

    WENOZ-inspired scale separation: gamma_k = 1 + tau/(SI_k + eps)
    where tau = (avg|SI_central - SI_k|)^6
    """
    p0, p1, p2 = _stencils_fv(u_im2, u_im1, u_i, u_ip1, u_ip2)
    pc = _central_fv(u_im2, u_im1, u_i, u_ip1, u_ip2)
    b0, b1, b2 = _js_smoothness(u_im2, u_im1, u_i, u_ip1, u_ip2)
    bc = max(b0, b1, b2)

    SI = [bc, b0, b1, b2]
    tau = (abs(bc - b0) + abs(bc - b1) + abs(bc - b2)) / 3.0
    tau = tau ** _POW
    g = [1.0 + tau / (si + _EPSL) for si in SI]

    g_sum = g[0] + g[1] + g[2] + g[3]
    inv_g_sum = 1.0 / g_sum
    chi = [gk * inv_g_sum for gk in g]

    delta = [1.0 if c >= _C_T else 0.0 for c in chi]

    if delta[0] == 1.0:
        return pc

    n_pass = delta[1] + delta[2] + delta[3]
    if n_pass > 0.0:
        result = (delta[1] * p0 + delta[2] * p1 + delta[3] * p2) / n_pass
    else:
        result = p1

    return result


# ---------------------------------------------------------------------------
# Public API: CTENO5 FV
# ---------------------------------------------------------------------------

def cteno5_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """CTENO5 reconstruction (FV) at i+1/2.

    Uses 1/(SI+eps)^6 scale separation with hard binary cutoff.
    """
    uL = _cteno5_core(u_im2, u_im1, u_i, u_ip1, u_ip2)
    uR = _cteno5_core(u_ip2, u_ip1, u_i, u_im1, u_im2)
    return uL, uR


# ---------------------------------------------------------------------------
# Public API: CTENO5Z FV
# ---------------------------------------------------------------------------

def cteno5z_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """CTENO5Z reconstruction (FV) at i+1/2.

    Uses WENOZ-inspired tau-based gamma with hard binary cutoff.
    """
    uL = _cteno5z_core(u_im2, u_im1, u_i, u_ip1, u_ip2)
    uR = _cteno5z_core(u_ip2, u_ip1, u_i, u_im1, u_im2)
    return uL, uR
# ---------------------------------------------------------------------------
# JIT compilation (pynalgo convention)
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    reveal_locals()  # noqa: F821
else:
    _central_fv = JIT(_central_fv)
    _cteno5_core = JIT(_cteno5_core)
    _cteno5z_core = JIT(_cteno5z_core)
    cteno5_fv = JIT(cteno5_fv)
    cteno5z_fv = JIT(cteno5z_fv)
#
# :D
#
