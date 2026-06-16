"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: WENO3 reconstruction methods
"""
from pyrecon._jit_utils import JIT, TYPE_CHECKING
# Constants
_DW3_FV = (1.0 / 3.0, 2.0 / 3.0)   # finite volume optimal weights
_DW3_PW = (1.0 / 4.0, 3.0 / 4.0)   # pointwise optimal weights
_EPSL = 1e-40
_EPSL_Z = 1e-12


def _weno3_LR(u_im1, u_i, u_ip1, dw, use_z):
    r"""Shared paired L+R computation for WENO3.

    uL = :math:`u_{i+1/2}^-`, uR = :math:`u_{i-1/2}^+`
    """
    # Smoothness indicators
    b0 = (u_i - u_im1) ** 2
    b1 = (u_ip1 - u_i) ** 2

    if use_z:
        tau = abs(b0 - b1)
        aL0 = dw[0] * (1.0 + tau / (_EPSL_Z + b0))
        aL1 = dw[1] * (1.0 + tau / (_EPSL_Z + b1))
        # Right face: centred stencil (S0:R) gets higher optimal weight
        aR0 = dw[1] * (1.0 + tau / (_EPSL_Z + b0))
        aR1 = dw[0] * (1.0 + tau / (_EPSL_Z + b1))
    else:
        aL0 = dw[0] / ((_EPSL + b0) ** 2)
        aL1 = dw[1] / ((_EPSL + b1) ** 2)
        # Right face: centred stencil (S0:R) gets higher optimal weight
        aR0 = dw[1] / ((_EPSL + b0) ** 2)
        aR1 = dw[0] / ((_EPSL + b1) ** 2)

    inv_sum_L = 1.0 / (aL0 + aL1)
    inv_sum_R = 1.0 / (aR0 + aR1)

    # Left face stencils (u_{i+1/2}^-)
    uk0_L = (-u_im1 + 3.0 * u_i) * 0.5
    uk1_L = (u_i + u_ip1) * 0.5
    uL = inv_sum_L * (aL0 * uk0_L + aL1 * uk1_L)

    # Right face stencils (u_{i-1/2}^+)
    # S0:R (cells i-1,i) straddles the face -> gets dw[1] (higher)
    # S1:R (cells i,i+1) is upwind          -> gets dw[0] (lower)
    uk0_R = (u_im1 + u_i) * 0.5
    uk1_R = (3.0 * u_i - u_ip1) * 0.5
    uR = inv_sum_R * (aR0 * uk0_R + aR1 * uk1_R)

    return uL, uR


def weno3_fv(u_im1, u_i, u_ip1):
    """WENO3-JS (cell-centered, FV weights)."""
    return _weno3_LR(u_im1, u_i, u_ip1, _DW3_FV, False)


def weno3_pw(u_im1, u_i, u_ip1):
    """WENO3-JS (pointwise, PW weights)."""
    return _weno3_LR(u_im1, u_i, u_ip1, _DW3_PW, False)


def weno3z_fv(u_im1, u_i, u_ip1):
    """WENO3-Z (cell-centered, FV weights, Borges-type tau)."""
    return _weno3_LR(u_im1, u_i, u_ip1, _DW3_FV, True)


def weno3z_pw(u_im1, u_i, u_ip1):
    """WENO3-Z (pointwise, PW weights, Borges-type tau)."""
    return _weno3_LR(u_im1, u_i, u_ip1, _DW3_PW, True)
# ---------------------------------------------------------------------------
# JIT compilation (pynalgo convention)
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    reveal_locals()  # noqa: F821
else:
    _weno3_LR = JIT(_weno3_LR)
    weno3_fv = JIT(weno3_fv)
    weno3_pw = JIT(weno3_pw)
    weno3z_fv = JIT(weno3z_fv)
    weno3z_pw = JIT(weno3z_pw)
#
# :D
#
