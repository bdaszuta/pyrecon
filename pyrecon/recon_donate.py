"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Donor cell (first-order) reconstruction
"""
from pyrecon._jit_utils import JIT, TYPE_CHECKING
def donate_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """First-order donor cell reconstruction (FV).

    Returns u_i for both left and right face values (piecewise constant).
    All 5 stencil values accepted for interface uniformity; only u_i is used.
    """
    return u_i, u_i
# ---------------------------------------------------------------------------
# JIT compilation (pynalgo convention)
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    reveal_locals()  # noqa: F821
else:
    donate_fv = JIT(donate_fv)
#
# :D
#
