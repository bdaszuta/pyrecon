"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: WENO7 and WENO7-Z reconstruction
"""

from pyrecon._jit_utils import JIT, TYPE_CHECKING

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OW7_FV = (1/35, 12/35, 18/35, 4/35)
_OW7_PW = (1/64, 21/64, 35/64, 7/64)

_EPSL = 1e-40
_EPSL_Z = 1e-12


# ---------------------------------------------------------------------------
# FV: closed-form JS smoothness matrices for 4-point stencils, denom 240
# ---------------------------------------------------------------------------

_JS7_FV_DENOM = 240.0
_JS7_FV_M = (
    # Stencil 0: (-3,-2,-1,0)
    ((  547, -1941,  2321,  -927),
     (-1941,  7043, -8623,  3521),
     ( 2321, -8623, 11003, -4701),
     ( -927,  3521, -4701,  2107)),
    # Stencil 1: (-2,-1,0,1)
    ((  267,  -821,   801,  -247),
     ( -821,  2843, -2983,   961),
     (  801, -2983,  3443, -1261),
     ( -247,   961, -1261,   547)),
    # Stencil 2: (-1,0,1,2)
    ((  547, -1261,   961,  -247),
     (-1261,  3443, -2983,   801),
     (  961, -2983,  2843,  -821),
     ( -247,   801,  -821,   267)),
    # Stencil 3: (0,1,2,3)
    (( 2107, -4701,  3521,  -927),
     (-4701, 11003, -8623,  2321),
     ( 3521, -8623,  7043, -1941),
     ( -927,  2321, -1941,   547)),
)


def _js_smoothness_fv(u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3):
    """FV JS smoothness for 4-point sub-stencils. Returns (b0,b1,b2,b3)."""
    def _is4(vals, k):
        m = _JS7_FV_M[k]
        beta = 0.0
        for i in range(4):
            row = m[i]
            vi = vals[i]
            for j in range(4):
                beta += row[j] * vi * vals[j]
        return beta / _JS7_FV_DENOM
    b0 = _is4((u_im3, u_im2, u_im1, u_i), 0)
    b1 = _is4((u_im2, u_im1, u_i, u_ip1), 1)
    b2 = _is4((u_im1, u_i, u_ip1, u_ip2), 2)
    b3 = _is4((u_i, u_ip1, u_ip2, u_ip3), 3)
    return b0, b1, b2, b3


# ---------------------------------------------------------------------------
# PW: use the same closed-form FV JS matrices for smoothness ordering.
# Standard JS smoothness indicator matrices for 4-point sub-stencils.
# ---------------------------------------------------------------------------

_js_smoothness_pw = _js_smoothness_fv  # alias to FV smoothness (re-bound to JIT version below)


# ---------------------------------------------------------------------------
# FV sub-stencil polynomials (4-point, denominator 12)
# ---------------------------------------------------------------------------

def _stencils_fv_weno7(fm3, fm2, fm1, f0, fp1, fp2, fp3):
    """WENO7 FV sub-stencil polynomials (4-point, 7-cell, left-biased). Returns (u0,u1,u2,u3)."""
    u0 = (-3*fm3 + 13*fm2 - 23*fm1 + 25*f0) / 12.0
    u1 = (1*fm2 - 5*fm1 + 13*f0 + 3*fp1) / 12.0
    u2 = (-1*fm1 + 7*f0 + 7*fp1 - 1*fp2) / 12.0
    u3 = (3*f0 + 13*fp1 - 5*fp2 + 1*fp3) / 12.0
    return u0, u1, u2, u3


# ---------------------------------------------------------------------------
# PW sub-stencil polynomials (4-point, denominator 16)
# ---------------------------------------------------------------------------

def _stencils_pw_weno7(fm3, fm2, fm1, f0, fp1, fp2, fp3):
    """WENO7 PW sub-stencil polynomials (4-point, 7-cell, left-biased). Returns (u0,u1,u2,u3)."""
    u0 = (-5*fm3 + 21*fm2 - 35*fm1 + 35*f0) / 16.0
    u1 = (fm2 - 5*fm1 + 15*f0 + 5*fp1) / 16.0
    u2 = (-fm1 + 9*f0 + 9*fp1 - fp2) / 16.0
    u3 = (5*f0 + 15*fp1 - 5*fp2 + fp3) / 16.0
    return u0, u1, u2, u3


# ---------------------------------------------------------------------------
# WENO7-JS cores (FV + PW)
# ---------------------------------------------------------------------------

def _weno7_js_LR_fv(u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3):
    """WENO7-JS paired L+R reconstruction (FV weights)."""
    b0, b1, b2, b3 = _js_smoothness_fv(
        u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3)

    d0, d1, d2, d3 = _OW7_FV
    aL0 = d0 / ((_EPSL + b0) ** 2)
    aL1 = d1 / ((_EPSL + b1) ** 2)
    aL2 = d2 / ((_EPSL + b2) ** 2)
    aL3 = d3 / ((_EPSL + b3) ** 2)
    invL = 1.0 / (aL0 + aL1 + aL2 + aL3)
    ukL0, ukL1, ukL2, ukL3 = _stencils_fv_weno7(
        u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3)
    uL = invL * (aL0 * ukL0 + aL1 * ukL1 + aL2 * ukL2 + aL3 * ukL3)

    aR0 = d0 / ((_EPSL + b3) ** 2)
    aR1 = d1 / ((_EPSL + b2) ** 2)
    aR2 = d2 / ((_EPSL + b1) ** 2)
    aR3 = d3 / ((_EPSL + b0) ** 2)
    invR = 1.0 / (aR0 + aR1 + aR2 + aR3)
    ukR0, ukR1, ukR2, ukR3 = _stencils_fv_weno7(
        u_ip3, u_ip2, u_ip1, u_i, u_im1, u_im2, u_im3)
    uR = invR * (aR0 * ukR0 + aR1 * ukR1 + aR2 * ukR2 + aR3 * ukR3)

    return uL, uR


def _weno7_js_LR_pw(u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3):
    """WENO7-JS paired L+R reconstruction (PW weights)."""
    b0, b1, b2, b3 = _js_smoothness_pw(
        u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3)

    d0, d1, d2, d3 = _OW7_PW
    aL0 = d0 / ((_EPSL + b0) ** 2)
    aL1 = d1 / ((_EPSL + b1) ** 2)
    aL2 = d2 / ((_EPSL + b2) ** 2)
    aL3 = d3 / ((_EPSL + b3) ** 2)
    invL = 1.0 / (aL0 + aL1 + aL2 + aL3)
    ukL0, ukL1, ukL2, ukL3 = _stencils_pw_weno7(
        u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3)
    uL = invL * (aL0 * ukL0 + aL1 * ukL1 + aL2 * ukL2 + aL3 * ukL3)

    aR0 = d0 / ((_EPSL + b3) ** 2)
    aR1 = d1 / ((_EPSL + b2) ** 2)
    aR2 = d2 / ((_EPSL + b1) ** 2)
    aR3 = d3 / ((_EPSL + b0) ** 2)
    invR = 1.0 / (aR0 + aR1 + aR2 + aR3)
    ukR0, ukR1, ukR2, ukR3 = _stencils_pw_weno7(
        u_ip3, u_ip2, u_ip1, u_i, u_im1, u_im2, u_im3)
    uR = invR * (aR0 * ukR0 + aR1 * ukR1 + aR2 * ukR2 + aR3 * ukR3)

    return uL, uR


# ---------------------------------------------------------------------------
# WENO7-Z cores (FV + PW)
# ---------------------------------------------------------------------------

def _weno7z_LR_fv(u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3):
    """WENO7-Z paired L+R reconstruction (FV weights)."""
    b0, b1, b2, b3 = _js_smoothness_fv(
        u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3)

    tau = abs(b0 - b3)
    d0, d1, d2, d3 = _OW7_FV
    aL0 = d0 * (1.0 + tau / (_EPSL_Z + b0))
    aL1 = d1 * (1.0 + tau / (_EPSL_Z + b1))
    aL2 = d2 * (1.0 + tau / (_EPSL_Z + b2))
    aL3 = d3 * (1.0 + tau / (_EPSL_Z + b3))
    invL = 1.0 / (aL0 + aL1 + aL2 + aL3)
    ukL0, ukL1, ukL2, ukL3 = _stencils_fv_weno7(
        u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3)
    uL = invL * (aL0 * ukL0 + aL1 * ukL1 + aL2 * ukL2 + aL3 * ukL3)

    aR0 = d0 * (1.0 + tau / (_EPSL_Z + b3))
    aR1 = d1 * (1.0 + tau / (_EPSL_Z + b2))
    aR2 = d2 * (1.0 + tau / (_EPSL_Z + b1))
    aR3 = d3 * (1.0 + tau / (_EPSL_Z + b0))
    invR = 1.0 / (aR0 + aR1 + aR2 + aR3)
    ukR0, ukR1, ukR2, ukR3 = _stencils_fv_weno7(
        u_ip3, u_ip2, u_ip1, u_i, u_im1, u_im2, u_im3)
    uR = invR * (aR0 * ukR0 + aR1 * ukR1 + aR2 * ukR2 + aR3 * ukR3)

    return uL, uR


def _weno7z_LR_pw(u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3):
    """WENO7-Z paired L+R reconstruction (PW weights)."""
    b0, b1, b2, b3 = _js_smoothness_pw(
        u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3)

    tau = abs(b0 - b3)
    d0, d1, d2, d3 = _OW7_PW
    aL0 = d0 * (1.0 + tau / (_EPSL_Z + b0))
    aL1 = d1 * (1.0 + tau / (_EPSL_Z + b1))
    aL2 = d2 * (1.0 + tau / (_EPSL_Z + b2))
    aL3 = d3 * (1.0 + tau / (_EPSL_Z + b3))
    invL = 1.0 / (aL0 + aL1 + aL2 + aL3)
    ukL0, ukL1, ukL2, ukL3 = _stencils_pw_weno7(
        u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3)
    uL = invL * (aL0 * ukL0 + aL1 * ukL1 + aL2 * ukL2 + aL3 * ukL3)

    aR0 = d0 * (1.0 + tau / (_EPSL_Z + b3))
    aR1 = d1 * (1.0 + tau / (_EPSL_Z + b2))
    aR2 = d2 * (1.0 + tau / (_EPSL_Z + b1))
    aR3 = d3 * (1.0 + tau / (_EPSL_Z + b0))
    invR = 1.0 / (aR0 + aR1 + aR2 + aR3)
    ukR0, ukR1, ukR2, ukR3 = _stencils_pw_weno7(
        u_ip3, u_ip2, u_ip1, u_i, u_im1, u_im2, u_im3)
    uR = invR * (aR0 * ukR0 + aR1 * ukR1 + aR2 * ukR2 + aR3 * ukR3)

    return uL, uR


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def weno7_fv(u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3):
    r"""WENO7-JS reconstruction (FV weights) at :math:`i+1/2`."""
    return _weno7_js_LR_fv(u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3)


def weno7_pw(u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3):
    r"""WENO7-JS reconstruction (PW weights) at :math:`i+1/2`."""
    return _weno7_js_LR_pw(u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3)


def weno7z_fv(u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3):
    r"""WENO7-Z reconstruction (FV weights) at :math:`i+1/2`."""
    return _weno7z_LR_fv(u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3)


def weno7z_pw(u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3):
    r"""WENO7-Z reconstruction (PW weights) at :math:`i+1/2`."""
    return _weno7z_LR_pw(u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3)
# ---------------------------------------------------------------------------
# JIT compilation (pynalgo convention)
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    reveal_locals()  # noqa: F821
else:
    _js_smoothness_fv = JIT(_js_smoothness_fv)
    _js_smoothness_pw = _js_smoothness_fv  # re-bound to JIT version
    _stencils_fv_weno7 = JIT(_stencils_fv_weno7)
    _stencils_pw_weno7 = JIT(_stencils_pw_weno7)
    _weno7_js_LR_fv = JIT(_weno7_js_LR_fv)
    _weno7_js_LR_pw = JIT(_weno7_js_LR_pw)
    _weno7z_LR_fv = JIT(_weno7z_LR_fv)
    _weno7z_LR_pw = JIT(_weno7z_LR_pw)
    weno7_fv = JIT(weno7_fv)
    weno7_pw = JIT(weno7_pw)
    weno7z_fv = JIT(weno7z_fv)
    weno7z_pw = JIT(weno7z_pw)
#
# :D
#
