"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: CENO (Central Essentially Non-Oscillatory) reconstruction
"""
from pyrecon._jit_utils import JIT, TYPE_CHECKING
from pyrecon.utils import MC2

_ALPHA = 0.7


def _ceno3lim(d0, d1, d2):
    """Select minimum abs-magnitude correction among same-sign candidates.

    Applies an alpha=0.7 bias toward the central stencil correction
    d1 in the comparison (scales d1 down by 30% in the min-abs
    selection), favouring the centred candidate. This centering
    preference may reduce resolution at sharp features.
    """
    if not ((d0 >= 0.0 and d1 >= 0.0 and d2 >= 0.0) or
            (d0 < 0.0 and d1 < 0.0 and d2 < 0.0)):
        return 0.0

    absd0 = abs(d0)
    absd1 = abs(_ALPHA * d1)
    absd2 = abs(d2)

    if absd1 < absd0:
        if absd2 < absd1:
            return d2
        return d1
    if absd2 < absd0:
        return d2
    return d0


def ceno3_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """CENO3 at i+1/2 (FV)."""
    oo2 = 0.5
    oo6 = 1.0 / 6.0

    # Left face
    slope_c = oo2 * MC2(u_i - u_im1, u_ip1 - u_i)
    baseL = u_i + slope_c

    dL0 = (2.0 * u_im2 - 7.0 * u_im1 + 11.0 * u_i) * oo6 - baseL
    dL1 = (-u_im1 + 5.0 * u_i + 2.0 * u_ip1) * oo6 - baseL
    dL2 = (2.0 * u_i + 5.0 * u_ip1 - u_ip2) * oo6 - baseL

    uL = baseL + _ceno3lim(dL0, dL1, dL2)

    # Right face
    slopeR = oo2 * MC2(u_i - u_ip1, u_im1 - u_i)
    baseR = u_i + slopeR

    dR0 = (2.0 * u_ip2 - 7.0 * u_ip1 + 11.0 * u_i) * oo6 - baseR
    dR1 = (-u_ip1 + 5.0 * u_i + 2.0 * u_im1) * oo6 - baseR
    dR2 = (2.0 * u_i + 5.0 * u_im1 - u_im2) * oo6 - baseR

    uR = baseR + _ceno3lim(dR0, dR1, dR2)

    return uL, uR


def ceno5_fv(u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3):
    """CENO5 at i+1/2 (FV).  7-point stencil, 3 quartic sub-stencils."""
    oo2 = 0.5
    oo60 = 1.0 / 60.0

    slope_c = oo2 * MC2(u_i - u_im1, u_ip1 - u_i)
    baseL = u_i + slope_c

    dL0 = (-3.0 * u_im3 + 17.0 * u_im2 - 43.0 * u_im1 +
           77.0 * u_i + 12.0 * u_ip1) * oo60 - baseL
    dL1 = (2.0 * u_im2 - 13.0 * u_im1 + 47.0 * u_i +
           27.0 * u_ip1 - 3.0 * u_ip2) * oo60 - baseL
    dL2 = (-3.0 * u_im1 + 27.0 * u_i + 47.0 * u_ip1 -
           13.0 * u_ip2 + 2.0 * u_ip3) * oo60 - baseL
    uL = baseL + _ceno3lim(dL0, dL1, dL2)

    baseR = u_i - slope_c
    dR0 = (-3.0 * u_ip3 + 17.0 * u_ip2 - 43.0 * u_ip1 +
           77.0 * u_i + 12.0 * u_im1) * oo60 - baseR
    dR1 = (2.0 * u_ip2 - 13.0 * u_ip1 + 47.0 * u_i +
           27.0 * u_im1 - 3.0 * u_im2) * oo60 - baseR
    dR2 = (-3.0 * u_ip1 + 27.0 * u_i + 47.0 * u_im1 -
           13.0 * u_im2 + 2.0 * u_im3) * oo60 - baseR
    uR = baseR + _ceno3lim(dR0, dR1, dR2)

    return uL, uR
# ---------------------------------------------------------------------------
# JIT compilation (pynalgo convention)
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    reveal_locals()  # noqa: F821
else:
    _ceno3lim = JIT(_ceno3lim)
    ceno3_fv = JIT(ceno3_fv)
    ceno5_fv = JIT(ceno5_fv)
#
# :D
#
