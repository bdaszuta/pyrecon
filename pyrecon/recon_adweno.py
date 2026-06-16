"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Curvature-sharpened WENO5-Z FV reconstruction
 (not Xu & Shu 2005; uses minmod on curvatures with smoothness gate)
"""
from pyrecon.utils import minmod
from pyrecon.recon_weno5 import _js_smoothness
from pyrecon._jit_utils import JIT, TYPE_CHECKING

_OPTIMW = (1/10, 3/5, 3/10)
_EPSL = 1e-40
_EPSL_Z = 1e-12
_KAPPA = 0.5


def _stencils(u0, u1, u2, u3, u4):
    """WENO5 candidate stencil polynomials (left-biased)."""
    return (
        (2/6) * u0 + (-7/6) * u1 + (11/6) * u2,
        (-1/6) * u1 + (5/6) * u2 + (2/6) * u3,
        (2/6) * u2 + (5/6) * u3 + (-1/6) * u4,
    )


def _weno5z_L(u_im2, u_im1, u_i, u_ip1, u_ip2):
    r"""WENO5-Z left-face reconstruction at :math:`i+1/2`."""
    b0, b1, b2 = _js_smoothness(u_im2, u_im1, u_i, u_ip1, u_ip2)
    tau = abs(b0 - b2)
    a0 = _OPTIMW[0] * (1.0 + tau / (_EPSL_Z + b0))
    a1 = _OPTIMW[1] * (1.0 + tau / (_EPSL_Z + b1))
    a2 = _OPTIMW[2] * (1.0 + tau / (_EPSL_Z + b2))
    inv = 1.0 / (a0 + a1 + a2)
    uk0, uk1, uk2 = _stencils(u_im2, u_im1, u_i, u_ip1, u_ip2)
    return inv * (a0 * uk0 + a1 * uk1 + a2 * uk2)


def _weno5z_R(u_im2, u_im1, u_i, u_ip1, u_ip2):
    r"""WENO5-Z right-face reconstruction at :math:`i-1/2` (reversed stencil)."""
    b0, b1, b2 = _js_smoothness(u_im2, u_im1, u_i, u_ip1, u_ip2)
    tau = abs(b0 - b2)
    a0 = _OPTIMW[0] * (1.0 + tau / (_EPSL_Z + b2))
    a1 = _OPTIMW[1] * (1.0 + tau / (_EPSL_Z + b1))
    a2 = _OPTIMW[2] * (1.0 + tau / (_EPSL_Z + b0))
    inv = 1.0 / (a0 + a1 + a2)
    uk0, uk1, uk2 = _stencils(u_ip2, u_ip1, u_i, u_im1, u_im2)
    return inv * (a0 * uk0 + a1 * uk1 + a2 * uk2)


def adweno5_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    r"""Curvature-sharpened WENO5-Z reconstruction (FV).

    Computes WENO5-Z face values, then adds a curvature-based sharpening
    correction gated by smoothness so it vanishes in smooth regions.

    uL = :math:`u_{i+1/2}^-` + :math:`\kappa` * gate * minmod(curv)
    uR = :math:`u_{i-1/2}^+` + :math:`\kappa` * gate * minmod(curv)

    where gate = :math:`(\tau/(\epsilon + \bar{\beta}))^3` is a
    discontinuity detector (~0 in smooth regions, ~O(1) near shocks)
    so the correction vanishes smoothly.
    """
    wL = _weno5z_L(u_im2, u_im1, u_i, u_ip1, u_ip2)
    wR = _weno5z_R(u_im2, u_im1, u_i, u_ip1, u_ip2)

    curv_im1 = u_im2 - 2.0 * u_im1 + u_i
    curv_i   = u_im1 - 2.0 * u_i   + u_ip1
    curv_ip1 = u_i   - 2.0 * u_ip1 + u_ip2

    cf_L = minmod(curv_im1, curv_i, curv_ip1, curv_ip1)
    cf_R = minmod(curv_im1, curv_i, curv_ip1, curv_ip1)

    # Smoothness gate: suppress correction in smooth regions.
    # Use WENO-Z tau measure: tau = |b0 - b2| vanishes O(h^8) smoothly.
    b0, b1, b2 = _js_smoothness(u_im2, u_im1, u_i, u_ip1, u_ip2)
    tau = abs(b0 - b2)
    bbar = (b0 + b1 + b2) / 3.0
    # ~0 in smooth regions, ~O(1) near discontinuities
    gate = (tau / (_EPSL_Z + bbar)) ** 3

    uL = wL + _KAPPA * cf_L * gate
    uR = wR + _KAPPA * cf_R * gate

    return uL, uR
# ---------------------------------------------------------------------------
# JIT compilation (pynalgo convention)
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    reveal_locals()  # noqa: F821
else:
    _stencils = JIT(_stencils)
    _weno5z_L = JIT(_weno5z_L)
    _weno5z_R = JIT(_weno5z_R)
    adweno5_fv = JIT(adweno5_fv)
#
# :D
#
