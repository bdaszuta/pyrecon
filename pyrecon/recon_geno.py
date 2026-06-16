"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: GENO5 -- Gradient-based Essentially Non-Oscillatory 5th-order
"""
import math
from pyrecon._jit_utils import JIT, TYPE_CHECKING

from pyrecon.recon_weno5 import _js_smoothness, _stencils_fv

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Path function steepness
_C = 20.0
_CHI_DENOM = math.tanh(_C)  # tanh(20) ~ 1.0

# Central boost factor: de-prioritize off-center stencils (Paper Sec 3.3)
# Index 1 is always the physical center stencil regardless of face orientation.
_D0 = 1.0
_D1 = 8.0
_D2 = 1.0

# Epsilon for smoothness denominators
_EPS = 1e-15

# Exponent for structured alpha
_R = 2

# Exponent for ENO weights
_RL = 2


# ---------------------------------------------------------------------------
# Upwind-biased 5th-order linear reconstruction
# ---------------------------------------------------------------------------

def _upwind5th(u_im2, u_im1, u_i, u_ip1, u_ip2):
    r"""5th-order upwind-biased optimal polynomial at cell face.

    Returns the interpolated value at the cell face using a 5-point
    upwind-biased stencil. This is the unlimited high-order candidate
    :math:`q^H`.
    """
    return (2.0 * u_im2 - 13.0 * u_im1 + 47.0 * u_i +
            27.0 * u_ip1 - 3.0 * u_ip2) / 60.0


# ---------------------------------------------------------------------------
# Ultimate smoothness indicator and path function
# ---------------------------------------------------------------------------

def _compute_alpha(b0, b1, b2):
    """Compute the GENO5 smoothness blending parameter alpha in [0, 1].

    Uses ultimate smoothness indicators from sub-stencil JS indicators:
      IS_H = max(b0, b1, b2)  -- least smooth (worst case)
      IS_L = min(b0, b1, b2)  -- smoothest sub-stencil
      IS_tau = |(b0 + b2)/2 - b1|  -- higher-order gap

    alpha = 2 * alpha_H / (alpha_H + alpha_L)
      where alpha_K = 1 + (IS_tau / (IS_K + eps))^r

    alpha ~ 1 for smooth flows, alpha << 1 near discontinuities.
    """
    is_h = max(b0, b1, b2)
    is_l = min(b0, b1, b2)
    is_tau = abs((b0 + b2) * 0.5 - b1)

    ratio_h = is_tau / (is_h + _EPS)
    ratio_l = is_tau / (is_l + _EPS)

    alpha_h = 1.0 + ratio_h ** _R
    alpha_l = 1.0 + ratio_l ** _R

    return 2.0 * alpha_h / (alpha_h + alpha_l)


def _compute_chi(alpha):
    """Compute the path function chi = tanh(C*alpha) / tanh(C).

    chi in [0, 1]: ~1 for smooth flows (high-order dominates),
                    ~0 near discontinuities (ENO dominates).
    """
    return math.tanh(_C * alpha) / _CHI_DENOM


# ---------------------------------------------------------------------------
# Lower-order centrally-biased WENO-weighted ENO candidate
# ---------------------------------------------------------------------------

def _eno_reconstruction(st0, st1, st2, b0, b1, b2):
    """Lower-order ENO via WENO-style weighted combination.

    Uses inverse-smoothness weights with a central-stencil boost
    (_D0=1, _D1=8, _D2=1). This de-prioritizes off-center stencils
    relative to the central one, creating a more centered ENO candidate.
    """
    wt0 = _D0 / (b0 + _EPS) ** _RL
    wt1 = _D1 / (b1 + _EPS) ** _RL
    wt2 = _D2 / (b2 + _EPS) ** _RL
    wt_sum = wt0 + wt1 + wt2

    return (wt0 * st0 + wt1 * st1 + wt2 * st2) / wt_sum


# ---------------------------------------------------------------------------
# Single-face GENO5 reconstruction
# ---------------------------------------------------------------------------

def _geno5_single(u_im2, u_im1, u_i, u_ip1, u_ip2):
    r"""Compute GENO5 reconstruction for a single face (left-biased).

    Returns the face value :math:`u_{i+1/2}`.
    """
    # High-order upwind reconstruction
    q_h = _upwind5th(u_im2, u_im1, u_i, u_ip1, u_ip2)

    # Sub-stencil polynomials and smoothness indicators
    st0, st1, st2 = _stencils_fv(u_im2, u_im1, u_i, u_ip1, u_ip2)
    b0, b1, b2 = _js_smoothness(u_im2, u_im1, u_i, u_ip1, u_ip2)

    # Lower-order ENO
    q_l = _eno_reconstruction(st0, st1, st2, b0, b1, b2)

    # Path function blend
    alpha = _compute_alpha(b0, b1, b2)
    chi = _compute_chi(alpha)

    return chi * q_h + (1.0 - chi) * q_l


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def geno5_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    r"""GENO5 reconstruction (FV weights) at :math:`i+1/2`.

    Parameters
    ----------
    u_im2, u_im1, u_i, u_ip1, u_ip2 : float
        5-point stencil values.

    Returns
    -------
    (uL, uR) where:
        uL = :math:`u_{i+1/2}^-`  (left state at right face)
        uR = :math:`u_{i-1/2}^+`  (right state at left face)
    """
    # Left face: forward stencil
    uL = _geno5_single(u_im2, u_im1, u_i, u_ip1, u_ip2)

    # Right face: reversed args
    uR = _geno5_single(u_ip2, u_ip1, u_i, u_im1, u_im2)

    return uL, uR
# ---------------------------------------------------------------------------
# JIT compilation (pynalgo convention)
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    reveal_locals()  # noqa: F821
else:
    _upwind5th = JIT(_upwind5th)
    _compute_alpha = JIT(_compute_alpha)
    _compute_chi = JIT(_compute_chi)
    _eno_reconstruction = JIT(_eno_reconstruction)
    _geno5_single = JIT(_geno5_single)
    geno5_fv = JIT(geno5_fv)
#
# :D
#
