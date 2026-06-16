"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Hybrid linear-WENO-Z reconstruction with convex blend:
 5th-order linear + WENO5-Z nonlinear correction, gated by a
 scale-independent smoothness sensor (C_tau parameter).
"""
import math
from pyrecon._jit_utils import JIT, TYPE_CHECKING

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# FV optimal linear weights (coincidentally same as PW: 1/10, 3/5, 3/10)
_OW5 = (1/10, 3/5, 3/10)

# FV 5-point linear stencil at x_{i+1/2} (exact fractions)
# [1/30, -13/60, 47/60, 9/20, -1/20]
_LIN5_FV = (1/30, -13/60, 47/60, 9/20, -1/20)

# FV 3-point sub-stencil coefficients at x_{i+1/2}
# Stencil [-2,-1,0]: [1/3, -7/6, 11/6]
# Stencil [-1,0,1]:  [-1/6, 5/6, 1/3]
# Stencil [0,1,2]:   [1/3, 5/6, -1/6]
_SUB3_COEFFS = (
    (1/3, -7/6, 11/6),
    (-1/6, 5/6, 1/3),
    (1/3, 5/6, -1/6),
)

# Closed-form JS smoothness matrices for 3-point FV stencils.
# Each IS = u^T * M * u / 6
_JS3_DENOM = 6.0
_JS3_M = (
    # Stencil 0: [-2,-1,0]
    ((8, -19, 11),
     (-19, 50, -31),
     (11, -31, 20)),
    # Stencil 1: [-1,0,1]
    ((8, -13, 5),
     (-13, 26, -13),
     (5, -13, 8)),
    # Stencil 2: [0,1,2]
    ((20, -31, 11),
     (-31, 50, -19),
     (11, -19, 8)),
)

# Default dissipation control parameter
_DEFAULT_C_TAU = 1.0
_EPSL = 1e-40
_EPSL_Z = 1e-12


# ---------------------------------------------------------------------------
# Smoothness indicators
# ---------------------------------------------------------------------------

def _js_smoothness(u, stencil_idx):
    """FV Jiang-Shu smoothness indicator for 3-point sub-stencil."""
    m = _JS3_M[stencil_idx]
    beta = 0.0
    for i in range(3):
        row = m[i]
        vi = u[i]
        for j in range(3):
            beta += row[j] * vi * u[j]
    return beta / _JS3_DENOM


# ---------------------------------------------------------------------------
# Stencils
# ---------------------------------------------------------------------------

def _linear_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    r"""5th-order FV linear reconstruction at :math:`x_{i+1/2}`."""
    return (_LIN5_FV[0] * u_im2 + _LIN5_FV[1] * u_im1 +
            _LIN5_FV[2] * u_i    + _LIN5_FV[3] * u_ip1 +
            _LIN5_FV[4] * u_ip2)


def _substencil_values(u, k):
    r"""FV sub-stencil value at :math:`x_{i+1/2}` for stencil index k."""
    c = _SUB3_COEFFS[k]
    return c[0] * u[0] + c[1] * u[1] + c[2] * u[2]


# ---------------------------------------------------------------------------
# Scale-independent nonlinear dissipation indicator
# ---------------------------------------------------------------------------

def _compute_sigma(b0, b1, b2, u_im2, u_im1, u_i, u_ip1, u_ip2, c_tau):
    """Compute sigma in [0, 1] from scale-independent smoothness measure."""
    phi = math.sqrt(abs(b0 - 2.0 * b1 + b2))
    mu = (abs(u_im2) + abs(u_im1) + abs(u_i) +
          abs(u_ip1) + abs(u_ip2)) / 5.0 + _EPSL
    return min(1.0, c_tau * phi / mu)


# ---------------------------------------------------------------------------
# WENO-Z nonlinear weights
# ---------------------------------------------------------------------------

def _wenoz_weights(b0, b1, b2, left_face=True):
    r"""WENO-Z normalized weights.

    When ``left_face=True``, weights are computed in standard upwind
    order (:math:`b_0` = most-upwind stencil). For right-face
    reconstruction, pass the reversed smoothness indicators with
    ``left_face=True`` -- the data ordering already accounts for the
    face direction.
    """
    tau = abs(b0 - b2)
    if left_face:
        a0 = _OW5[0] * (1.0 + tau / (_EPSL_Z + b0))
        a1 = _OW5[1] * (1.0 + tau / (_EPSL_Z + b1))
        a2 = _OW5[2] * (1.0 + tau / (_EPSL_Z + b2))
    else:
        a0 = _OW5[0] * (1.0 + tau / (_EPSL_Z + b2))
        a1 = _OW5[1] * (1.0 + tau / (_EPSL_Z + b1))
        a2 = _OW5[2] * (1.0 + tau / (_EPSL_Z + b0))
    inv = 1.0 / (a0 + a1 + a2)
    return inv * a0, inv * a1, inv * a2


# ---------------------------------------------------------------------------
# Low-dissipation reconstruction
# ---------------------------------------------------------------------------

def _lowdiss_LR(u_im2, u_im1, u_i, u_ip1, u_ip2, c_tau):
    """Low-dissipation paired L+R reconstruction."""
    # FV smoothness for the three 3-point sub-stencils
    b0 = _js_smoothness((u_im2, u_im1, u_i), 0)
    b1 = _js_smoothness((u_im1, u_i, u_ip1), 1)
    b2 = _js_smoothness((u_i, u_ip1, u_ip2), 2)

    sigma = _compute_sigma(b0, b1, b2, u_im2, u_im1, u_i, u_ip1, u_ip2, c_tau)

    # Left face
    ulin_L = _linear_fv(u_im2, u_im1, u_i, u_ip1, u_ip2)
    vL0 = _substencil_values((u_im2, u_im1, u_i), 0)
    vL1 = _substencil_values((u_im1, u_i, u_ip1), 1)
    vL2 = _substencil_values((u_i, u_ip1, u_ip2), 2)
    wL0, wL1, wL2 = _wenoz_weights(b0, b1, b2, left_face=True)
    uweno_L = wL0 * vL0 + wL1 * vL1 + wL2 * vL2
    uL = (1.0 - sigma) * ulin_L + sigma * uweno_L

    # Right face -- compute smoothness from reversed data
    bR0 = _js_smoothness((u_ip2, u_ip1, u_i), 0)
    bR1 = _js_smoothness((u_ip1, u_i, u_im1), 1)
    bR2 = _js_smoothness((u_i, u_im1, u_im2), 2)
    ulin_R = _linear_fv(u_ip2, u_ip1, u_i, u_im1, u_im2)
    vR0 = _substencil_values((u_ip2, u_ip1, u_i), 0)
    vR1 = _substencil_values((u_ip1, u_i, u_im1), 1)
    vR2 = _substencil_values((u_i, u_im1, u_im2), 2)
    wR0, wR1, wR2 = _wenoz_weights(bR0, bR1, bR2, left_face=True)
    uweno_R = wR0 * vR0 + wR1 * vR1 + wR2 * vR2
    uR = (1.0 - sigma) * ulin_R + sigma * uweno_R

    return uL, uR


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def hybrid_linear_weno_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """Hybrid linear-WENO-Z reconstruction (default C_tau=1.0). (FV)"""
    return _lowdiss_LR(u_im2, u_im1, u_i, u_ip1, u_ip2, _DEFAULT_C_TAU)


def hybrid_linear_weno_mild_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """Hybrid linear-WENO-Z, mild (C_tau=0.5). (FV)"""
    return _lowdiss_LR(u_im2, u_im1, u_i, u_ip1, u_ip2, 0.5)


def hybrid_linear_weno_strong_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """Hybrid linear-WENO-Z, strong (C_tau=2.0). (FV)"""
    return _lowdiss_LR(u_im2, u_im1, u_i, u_ip1, u_ip2, 2.0)

# ---------------------------------------------------------------------------
# JIT compilation (pynalgo convention)
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    reveal_locals()  # noqa: F821
else:
    _js_smoothness = JIT(_js_smoothness)
    _linear_fv = JIT(_linear_fv)
    _substencil_values = JIT(_substencil_values)
    _compute_sigma = JIT(_compute_sigma)
    _wenoz_weights = JIT(_wenoz_weights)
    _lowdiss_LR = JIT(_lowdiss_LR)
    hybrid_linear_weno_fv = JIT(hybrid_linear_weno_fv)
    hybrid_linear_weno_mild_fv = JIT(hybrid_linear_weno_mild_fv)
    hybrid_linear_weno_strong_fv = JIT(hybrid_linear_weno_strong_fv)
#
# :D
#
