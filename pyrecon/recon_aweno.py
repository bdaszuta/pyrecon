"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: AWENO reconstruction method (Wang, Don, Wang 2023)
"""
import math
from pyrecon._jit_utils import JIT, TYPE_CHECKING

from pyrecon.recon_weno5 import _js_smoothness

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_K_1_6 = 1.0 / 6.0

# Optimal linear weights (FV)
_OPTIMW = (1.0 / 10.0, 3.0 / 5.0, 3.0 / 10.0)

_EPSL = 1e-40
_SIGMA_MAX = 1.0  # maximum blending factor


# ---------------------------------------------------------------------------
# Smoothness indicators and stencils
# ---------------------------------------------------------------------------

def _stencils(u_0, u_1, u_2, u_3, u_4):
    """WENO5 candidate stencil polynomials (left-biased)."""
    u0 = _K_1_6 * (2.0 * u_0 - 7.0 * u_1 + 11.0 * u_2)
    u1 = _K_1_6 * (-u_1 + 5.0 * u_2 + 2.0 * u_3)
    u2 = _K_1_6 * (2.0 * u_2 + 5.0 * u_3 - u_4)
    return u0, u1, u2


# ---------------------------------------------------------------------------
# Scale-independent smoothness measure (Wang et al. 2023)
# ---------------------------------------------------------------------------

def _scale_independent_sigma(u_im2, u_im1, u_i, u_ip1, u_ip2):
    r"""Compute the scale-independent smoothness measure sigma.

    Based on the AWENO paper, sigma is computed from a WENO-D-style
    curvature measure :math:`\phi = \sqrt{|\beta_0 - 2\beta_1 + \beta_2|}`
    normalized by the local solution scale.

    sigma = min(1, phi / mu) where:
      phi = sqrt(|b0 - 2*b1 + b2|)  (curvature measure)
      mu = (|u_im2| + |u_im1| + |u_i| + |u_ip1| + |u_ip2|) / 5
    """
    b0, b1, b2 = _js_smoothness(u_im2, u_im1, u_i, u_ip1, u_ip2)

    # Curvature-like measure
    phi = math.sqrt(abs(b0 - 2.0 * b1 + b2))

    # Local scale
    mu = (abs(u_im2) + abs(u_im1) + abs(u_i) +
          abs(u_ip1) + abs(u_ip2)) / 5.0 + _EPSL

    sigma = min(_SIGMA_MAX, phi / mu)
    return sigma


# ---------------------------------------------------------------------------
# AWENO reconstruction
# ---------------------------------------------------------------------------

def aweno5_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    r"""AWENO5 reconstruction at :math:`i+1/2` (FV).

    Adaptive blending between linear optimal weights and WENO-Z
    nonlinear weights based on scale-independent smoothness.

    For smooth regions (sigma small): near-linear optimal weights
    For non-smooth regions (sigma large): WENO-Z nonlinear weights

    Weights: :math:`\omega_k = (1 - \sigma) d_k + \sigma \omega_k^{\mathrm{WENO-Z}}`

    """
    b0, b1, b2 = _js_smoothness(u_im2, u_im1, u_i, u_ip1, u_ip2)

    # Scale-independent smoothness
    sigma = _scale_independent_sigma(u_im2, u_im1, u_i, u_ip1, u_ip2)

    # WENO-Z tau
    tau = abs(b0 - b2)

    # WENO-Z nonlinear weights
    aL0_z = _OPTIMW[0] * (1.0 + tau / (_EPSL + b0))
    aL1_z = _OPTIMW[1] * (1.0 + tau / (_EPSL + b1))
    aL2_z = _OPTIMW[2] * (1.0 + tau / (_EPSL + b2))
    inv_sum_L_z = 1.0 / (aL0_z + aL1_z + aL2_z)
    wL0_z = inv_sum_L_z * aL0_z
    wL1_z = inv_sum_L_z * aL1_z
    wL2_z = inv_sum_L_z * aL2_z

    # Adaptive blending: near linear weights when sigma is small,
    # near WENO-Z weights when sigma is large
    # omega_k = sigma * w_k^{WENO-Z} + (1 - sigma) * d_k
    wL0 = sigma * wL0_z + (1.0 - sigma) * _OPTIMW[0]
    wL1 = sigma * wL1_z + (1.0 - sigma) * _OPTIMW[1]
    wL2 = sigma * wL2_z + (1.0 - sigma) * _OPTIMW[2]

    # Left face
    ukL0, ukL1, ukL2 = _stencils(u_im2, u_im1, u_i, u_ip1, u_ip2)
    uL = wL0 * ukL0 + wL1 * ukL1 + wL2 * ukL2

    # Right face WENO-Z weights (b swapped)
    aR0_z = _OPTIMW[0] * (1.0 + tau / (_EPSL + b2))
    aR1_z = _OPTIMW[1] * (1.0 + tau / (_EPSL + b1))
    aR2_z = _OPTIMW[2] * (1.0 + tau / (_EPSL + b0))
    inv_sum_R_z = 1.0 / (aR0_z + aR1_z + aR2_z)
    wR0_z = inv_sum_R_z * aR0_z
    wR1_z = inv_sum_R_z * aR1_z
    wR2_z = inv_sum_R_z * aR2_z

    wR0 = sigma * wR0_z + (1.0 - sigma) * _OPTIMW[0]
    wR1 = sigma * wR1_z + (1.0 - sigma) * _OPTIMW[1]
    wR2 = sigma * wR2_z + (1.0 - sigma) * _OPTIMW[2]

    ukR0, ukR1, ukR2 = _stencils(u_ip2, u_ip1, u_i, u_im1, u_im2)
    uR = wR0 * ukR0 + wR1 * ukR1 + wR2 * ukR2

    return uL, uR
# ---------------------------------------------------------------------------
# JIT compilation (pynalgo convention)
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    reveal_locals()  # noqa: F821
else:
    _stencils = JIT(_stencils)
    _scale_independent_sigma = JIT(_scale_independent_sigma)
    aweno5_fv = JIT(aweno5_fv)
#
# :D
#
