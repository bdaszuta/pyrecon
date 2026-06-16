"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: ES-WENO3 reconstruction methods (Yamaleev & Carpenter 2009)
"""
# Constants
from pyrecon._jit_utils import JIT, TYPE_CHECKING
_DW3_FV = (1.0 / 3.0, 2.0 / 3.0)   # finite volume optimal weights
_DW3_PW = (1.0 / 4.0, 3.0 / 4.0)   # pointwise optimal weights
_EPSL = 1e-40
_EPSL_Z = 1e-12
_P = 1  # Yamaleev & Carpenter (2009, Appendix 8.2): p = 1 for ES-WENO3


def _esweno3_LR(u_im1, u_i, u_ip1, dw):
    r"""ES-WENO3 paired L+R reconstruction.

    uL = :math:`u_{i+1/2}^-`, uR = :math:`u_{i-1/2}^+`
    """
    # Jiang-Shu smoothness indicators
    b0 = (u_i - u_im1) ** 2
    b1 = (u_ip1 - u_i) ** 2

    # Global smoothness indicator tau_p (Y&C 2009, Appendix 8.2, phi=0)
    tau = (u_im1 - 2.0 * u_i + u_ip1) ** 2

    # ES weights for left face
    aL0 = dw[0] * (1.0 + (tau / (_EPSL_Z + b0)) ** _P)
    aL1 = dw[1] * (1.0 + (tau / (_EPSL_Z + b1)) ** _P)

    # Right face: centred stencil (S0:R, cells i-1,i) gets higher
    # optimal weight, so swap dw[0] <-> dw[1] vs the left face.
    aR0 = dw[1] * (1.0 + (tau / (_EPSL_Z + b0)) ** _P)
    aR1 = dw[0] * (1.0 + (tau / (_EPSL_Z + b1)) ** _P)

    inv_sum_L = 1.0 / (aL0 + aL1)
    inv_sum_R = 1.0 / (aR0 + aR1)

    # Left face stencils (u_{i+1/2}^-)
    uk0_L = (-u_im1 + 3.0 * u_i) * 0.5
    uk1_L = (u_i + u_ip1) * 0.5
    uL = inv_sum_L * (aL0 * uk0_L + aL1 * uk1_L)

    # Right face stencils (u_{i-1/2}^+)
    uk0_R = (u_im1 + u_i) * 0.5
    uk1_R = (3.0 * u_i - u_ip1) * 0.5
    uR = inv_sum_R * (aR0 * uk0_R + aR1 * uk1_R)

    return uL, uR


def esweno3_fv(u_im1, u_i, u_ip1):
    """ES-WENO3 reconstruction (FV weights, p=1)."""
    return _esweno3_LR(u_im1, u_i, u_ip1, _DW3_FV)


def esweno3_pw(u_im1, u_i, u_ip1):
    """ES-WENO3 reconstruction (PW weights, p=1)."""
    return _esweno3_LR(u_im1, u_i, u_ip1, _DW3_PW)
# ---------------------------------------------------------------------------
# JIT compilation (pynalgo convention)
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    reveal_locals()  # noqa: F821
else:
    _esweno3_LR = JIT(_esweno3_LR)
    esweno3_fv = JIT(esweno3_fv)
    esweno3_pw = JIT(esweno3_pw)
#
# :D
#
