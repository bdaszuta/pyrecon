"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Lag6 -- 6th-order Lagrange interpolation reconstruction
"""
from pyrecon._jit_utils import JIT, TYPE_CHECKING
def lag6_fv(u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3):
    r"""Lagrangian 6th-order interpolation at :math:`i+1/2` (FV).

    Returns (uL, uR) both = :math:`(1/60)(u_{i-2}+u_{i+3} - 8(u_{i-1}+u_{i+2}) + 37(u_i+u_{i+1}))`.
    """
    oo60 = 1.0 / 60.0
    uface = oo60 * (
        (u_im2 + u_ip3)
        - 8.0 * (u_im1 + u_ip2)
        + 37.0 * (u_i + u_ip1)
    )
    return uface, uface


def lag6_pw(u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3):
    r"""Lagrangian 6th-order interpolation at :math:`i+1/2` (PW).

    Returns (uL, uR) both = :math:`(1/256)(3(u_{i-2}+u_{i+3}) - 25(u_{i-1}+u_{i+2}) + 150(u_i+u_{i+1}))`.
    """
    oo256 = 1.0 / 256.0
    uface = oo256 * (
        3.0 * (u_im2 + u_ip3)
        - 25.0 * (u_im1 + u_ip2)
        + 150.0 * (u_i + u_ip1)
    )
    return uface, uface
# ---------------------------------------------------------------------------
# JIT compilation (pynalgo convention)
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    reveal_locals()  # noqa: F821
else:
    lag6_fv = JIT(lag6_fv)
    lag6_pw = JIT(lag6_pw)
#
# :D
#
