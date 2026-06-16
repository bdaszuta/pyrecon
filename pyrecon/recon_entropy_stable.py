"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Entropy-stable scalar reconstruction (adapted from Duan & Tang 2020)
"""
import math
from pyrecon._jit_utils import JIT, TYPE_CHECKING

from pyrecon.recon_weno5 import (
    _weno5z_LR_pw, _stencils_pw, _OPTIMW_PW, _js_smoothness,
)

# ---------------------------------------------------------------------------
# 5-point polynomial de-averaging: cell averages -> point values
#
# A 4th-degree polynomial P(x) is uniquely determined by its 5 cell
# averages.  The matrix M maps polynomial coefficients
#   b = [b0, b1*h, b2*h^2, b3*h^3, b4*h^4]
# to cell averages at k = -2, -1, 0, 1, 2.  M_INV rows are precomputed.
#
# De-averaging is only valid for smooth data.  At discontinuities the
# polynomial oscillates and creates spurious overshoots, so we skip
# de-averaging when a jump is detected via JS smoothness indicators.
# ---------------------------------------------------------------------------

_M_INV = (
    (3.0/640.0,    -29.0/480.0,   1067.0/960.0,  -29.0/480.0,    3.0/640.0),
    (5.0/48.0,     -17.0/24.0,    0.0,            17.0/24.0,     -5.0/48.0),
    (-1.0/16.0,     3.0/4.0,     -11.0/8.0,       3.0/4.0,      -1.0/16.0),
    (-1.0/12.0,     1.0/6.0,      0.0,           -1.0/6.0,       1.0/12.0),
    (1.0/24.0,     -1.0/6.0,      1.0/4.0,       -1.0/6.0,       1.0/24.0),
)

# Threshold: max(beta)/min(beta) above this -> skip de-averaging
_DEAVG_SMOOTH_THRESH = 1e4


def _is_smooth(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """Return True if the 5-point stencil is smooth enough to de-average.

    Uses JS smoothness indicator ratio: if max(beta)/min(beta) is
    large, a discontinuity is present and de-averaging would create
    spurious polynomial oscillations.
    """
    b0, b1, b2 = _js_smoothness(u_im2, u_im1, u_i, u_ip1, u_ip2)
    b_max = max(b0, b1, b2)
    b_min = min(b0, b1, b2)
    if b_max < 1e-80:
        return True  # effectively constant
    return b_max / max(b_min, 1e-40) < _DEAVG_SMOOTH_THRESH


def _deaverage_5(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """De-average 5 cell averages to 5 point values at cell centres.

    Returns (u_pt_im2, u_pt_im1, u_pt_i, u_pt_ip1, u_pt_ip2)
    accurate to O(h^5) for smooth functions.  Falls back to the raw
    cell averages if a discontinuity is detected (avoids polynomial
    overshoot at jumps).
    """
    if not _is_smooth(u_im2, u_im1, u_i, u_ip1, u_ip2):
        return (u_im2, u_im1, u_i, u_ip1, u_ip2)

    u = (u_im2, u_im1, u_i, u_ip1, u_ip2)

    b0 = (_M_INV[0][0]*u[0] + _M_INV[0][1]*u[1] + _M_INV[0][2]*u[2]
          + _M_INV[0][3]*u[3] + _M_INV[0][4]*u[4])
    b1 = (_M_INV[1][0]*u[0] + _M_INV[1][1]*u[1] + _M_INV[1][2]*u[2]
          + _M_INV[1][3]*u[3] + _M_INV[1][4]*u[4])
    b2 = (_M_INV[2][0]*u[0] + _M_INV[2][1]*u[1] + _M_INV[2][2]*u[2]
          + _M_INV[2][3]*u[3] + _M_INV[2][4]*u[4])
    b3 = (_M_INV[3][0]*u[0] + _M_INV[3][1]*u[1] + _M_INV[3][2]*u[2]
          + _M_INV[3][3]*u[3] + _M_INV[3][4]*u[4])
    b4 = (_M_INV[4][0]*u[0] + _M_INV[4][1]*u[1] + _M_INV[4][2]*u[2]
          + _M_INV[4][3]*u[3] + _M_INV[4][4]*u[4])

    return (
        b0 - 2.0*b1 + 4.0*b2 - 8.0*b3 + 16.0*b4,
        b0 - b1 + b2 - b3 + b4,
        b0,
        b0 + b1 + b2 + b3 + b4,
        b0 + 2.0*b1 + 4.0*b2 + 8.0*b3 + 16.0*b4,
    )


# ---------------------------------------------------------------------------
# Entropy definitions
# ---------------------------------------------------------------------------

def _v_quad(u):
    """Entropy variable for eta(u)=u^2/2: v = u."""
    return u


def _u_quad(v):
    """Inverse for eta(u)=u^2/2: u = v."""
    return v


def _v_log(u):
    """Entropy variable for eta(u)=u*log(u): v = 1 + log(u).

    Pass-through (v = u) when u <= 0 to avoid domain error;
    this breaks entropy consistency for non-positive inputs.
    """
    if u <= 0.0:
        return u
    return 1.0 + math.log(u)


def _u_log(v):
    """Inverse for eta(u)=u*log(u): u = exp(v-1)."""
    return math.exp(v - 1.0)


def _v_cubic(u):
    """Entropy variable for eta(u)=u^4/4: v = u^3."""
    return u * u * u


def _u_cubic(v):
    r"""Inverse for :math:`\eta(u)=u^4/4`: :math:`u = \mathrm{sign}(v)\,|v|^{1/3}`."""
    if v >= 0.0:
        return v ** (1.0 / 3.0)
    return -((-v) ** (1.0 / 3.0))


# ---------------------------------------------------------------------------
# Per-entropy WENO5-Z PW reconstruction cores
# ---------------------------------------------------------------------------


def _es_reconstruct_quad(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """Entropy-stable WENO5-Z PW reconstruction with eta=u^2/2 (v=u).

    Since v(u)=u, this is equivalent to weno5z_pw.
    """
    return _weno5z_LR_pw(u_im2, u_im1, u_i, u_ip1, u_ip2)


def _es_reconstruct_log(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """Entropy-stable WENO5-Z PW reconstruction with eta=u*log(u)."""
    v_im2 = _v_log(u_im2)
    v_im1 = _v_log(u_im1)
    v_i   = _v_log(u_i)
    v_ip1 = _v_log(u_ip1)
    v_ip2 = _v_log(u_ip2)

    vL, vR = _weno5z_LR_pw(v_im2, v_im1, v_i, v_ip1, v_ip2)
    return _u_log(vL), _u_log(vR)


def _es_reconstruct_cubic(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """Entropy-stable WENO5-Z PW reconstruction with eta=u^4/4."""
    v_im2 = _v_cubic(u_im2)
    v_im1 = _v_cubic(u_im1)
    v_i   = _v_cubic(u_i)
    v_ip1 = _v_cubic(u_ip1)
    v_ip2 = _v_cubic(u_ip2)

    vL, vR = _weno5z_LR_pw(v_im2, v_im1, v_i, v_ip1, v_ip2)
    return _u_cubic(vL), _u_cubic(vR)


# ---------------------------------------------------------------------------
# Public API: FV variants (cell-averaged input -> de-average -> PW pipeline)
# ---------------------------------------------------------------------------

def es_scalar_quad_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """Entropy-stable reconstruction with quadratic entropy. (FV)

    De-averages cell-averaged u to point values, then runs the PW
    entropy-stable pipeline.  Since v(u)=u, the entropy transform is
    the identity.  De-averaging is skipped at discontinuities.
    """
    u_pt = _deaverage_5(u_im2, u_im1, u_i, u_ip1, u_ip2)
    return _es_reconstruct_quad(*u_pt)


def es_scalar_log_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    r"""Entropy-stable reconstruction with logarithmic entropy. (FV)

    De-averages cell-averaged u to point values, then runs the PW
    entropy-stable pipeline with v = 1+log(u).

    Suitable for positive-definite fields (density-like).  The entropy
    variable naturally enforces positivity.  De-averaging is skipped
    at discontinuities.

    WARNING: requires strictly positive input values.  For zero or
    negative values, the log entropy transform is bypassed (:math:`v = u`
    instead of :math:`v = 1+\log(u)`), but the exponential back-transform
    :math:`u = \exp(v-1)` is still applied.  The result is
    :math:`\exp(\text{WENO5Z}(\text{raw}) - 1)`, not WENO5Z on the
    original variable space.
    """
    u_pt = _deaverage_5(u_im2, u_im1, u_i, u_ip1, u_ip2)
    return _es_reconstruct_log(*u_pt)


def es_scalar_cubic_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """Entropy-stable reconstruction with cubic-potential entropy. (FV)

    De-averages cell-averaged u to point values, then runs the PW
    entropy-stable pipeline with v = u^3.  De-averaging is skipped
    at discontinuities.

    The entropy variable v=u^3 provides a nonlinear transformation
    that can improve reconstruction for problems with strong gradients.
    This entropy structure is natural for Burgers-like equations where
    u^4 terms appear in energy estimates.
    """
    u_pt = _deaverage_5(u_im2, u_im1, u_i, u_ip1, u_ip2)
    return _es_reconstruct_cubic(*u_pt)


# ---------------------------------------------------------------------------
# Public API: PW variants (point-value input -> direct PW pipeline)
# ---------------------------------------------------------------------------

def es_scalar_quad_pw(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """Entropy-stable (PW) with quadratic entropy eta=u^2/2.

    Since v(u)=u, this is equivalent to weno5z_pw on the input
    point values.  Included for interface completeness.
    """
    return _es_reconstruct_quad(u_im2, u_im1, u_i, u_ip1, u_ip2)


def es_scalar_log_pw(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """Entropy-stable (PW) with logarithmic entropy eta=u*log(u).

    Accepts point values (NOT cell averages).  Transforms to
    v = 1+log(u), reconstructs via WENO5-Z PW, transforms back.

    WARNING: requires strictly positive input values.
    """
    return _es_reconstruct_log(u_im2, u_im1, u_i, u_ip1, u_ip2)


def es_scalar_cubic_pw(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """Entropy-stable (PW) with cubic-potential entropy eta=u^4/4.

    Accepts point values (NOT cell averages).  Transforms to
    v = u^3, reconstructs via WENO5-Z PW, transforms back.
    """
    return _es_reconstruct_cubic(u_im2, u_im1, u_i, u_ip1, u_ip2)
# ---------------------------------------------------------------------------
# JIT compilation (pynalgo convention)
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    reveal_locals()  # noqa: F821
else:
    _is_smooth = JIT(_is_smooth)
    _deaverage_5 = JIT(_deaverage_5)
    _es_reconstruct_quad = JIT(_es_reconstruct_quad)
    _es_reconstruct_log = JIT(_es_reconstruct_log)
    _es_reconstruct_cubic = JIT(_es_reconstruct_cubic)
    _v_quad = JIT(_v_quad)
    _u_quad = JIT(_u_quad)
    _v_log = JIT(_v_log)
    _u_log = JIT(_u_log)
    _v_cubic = JIT(_v_cubic)
    _u_cubic = JIT(_u_cubic)
    es_scalar_quad_fv = JIT(es_scalar_quad_fv)
    es_scalar_log_fv = JIT(es_scalar_log_fv)
    es_scalar_cubic_fv = JIT(es_scalar_cubic_fv)
    es_scalar_quad_pw = JIT(es_scalar_quad_pw)
    es_scalar_log_pw = JIT(es_scalar_log_pw)
    es_scalar_cubic_pw = JIT(es_scalar_cubic_pw)
#
# :D
#
