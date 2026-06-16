"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Linear (2nd-order) reconstruction methods
"""
from pyrecon.utils import MC2
from pyrecon._jit_utils import JIT, TYPE_CHECKING


def _van_leer_slope(u_left, u_center, u_right):
    """Van Leer slope limiter: compute Van Leer harmonic mean and return
    u_center + du_l * du_r / (du_l + du_r).

    Slope is computed between u_left and u_right relative to u_center.
    Returns the upwind-biased face value (u_center going toward u_right).
    """
    dul = u_right - u_center
    dur = u_center - u_left
    du2 = dul * dur
    if du2 <= 0.0:
        dum = 0.0
    else:
        dum = 2.0 * du2 / (dul + dur)
    return u_center + 0.5 * dum


def lin_vl_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    r"""Linear reconstruction with van Leer slope limiter (FV).

    uL = :math:`u_{i+1/2}^-` from (u_im1, u_i, u_ip1)
    uR = :math:`u_{i-1/2}^+` from (u_ip1, u_i, u_im1)  [reversed args]
    """
    uL = _van_leer_slope(u_im1, u_i, u_ip1)
    uR = _van_leer_slope(u_ip1, u_i, u_im1)
    return uL, uR


def _mc2_face(u_left, u_center, u_right):
    """MC2 limiter: u_center + 0.5 * MC2(slope_left, slope_right)."""
    slope = 0.5 * MC2(u_center - u_left, u_right - u_center)
    return u_center + slope


def lin_mc2_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    r"""Linear reconstruction with MC2 limiter (FV).

    uL = :math:`u_{i+1/2}^-` from (u_im1, u_i, u_ip1)
    uR = :math:`u_{i-1/2}^+` from (u_ip1, u_i, u_im1)  [reversed args]
    """
    uL = _mc2_face(u_im1, u_i, u_ip1)
    uR = _mc2_face(u_ip1, u_i, u_im1)
    return uL, uR
# ---------------------------------------------------------------------------
# JIT compilation (pynalgo convention)
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    reveal_locals()  # noqa: F821
else:
    _van_leer_slope = JIT(_van_leer_slope)
    lin_vl_fv = JIT(lin_vl_fv)
    _mc2_face = JIT(_mc2_face)
    lin_mc2_fv = JIT(lin_mc2_fv)
#
# :D
#
