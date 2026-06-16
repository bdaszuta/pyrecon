"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: MP (Monotonicity-Preserving) family reconstruction methods
"""
import math

from pyrecon._jit_utils import JIT, TYPE_CHECKING
from pyrecon.utils import minmod, sign

# MP epsilon (controls strictness of monotonicity bounds)
_MP_EPS = 1e-12


# ---------------------------------------------------------------------------
# Internal: MP limiters
# ---------------------------------------------------------------------------

def _mpnlimiter_5p(eps, u, uimt, uimo, ui, uipo, uipt):
    """5-point MP limiter (used by mp3, mp5).

    When eps > 0: returns u unchanged if u lies within the interval
    [ui, u_MP] (product (u - ui)*(u - u_MP) <= eps * |U|_2).
    """
    oo2 = 0.5
    fot = 4.0 / 3.0
    alphatil = 4.0

    if eps > 0:
        U_L2 = math.sqrt(uimt**2 + uimo**2 + ui**2 + uipo**2 + uipt**2)
        u_MP = ui + minmod(uipo - ui, alphatil * (ui - uimo))
        if (u - ui) * (u - u_MP) <= eps * U_L2:
            return u

    dm = uimt - 2.0 * uimo + ui
    d0 = uimo - 2.0 * ui + uipo
    dp = ui - 2.0 * uipo + uipt

    dm4p = minmod(4.0 * d0 - dp, 4.0 * dp - d0, d0, dp)
    dm4m = minmod(4.0 * d0 - dm, 4.0 * dm - d0, d0, dm)

    u_ul = ui + alphatil * (ui - uimo)
    u_av = oo2 * (ui + uipo)
    u_md = u_av - oo2 * dm4p
    u_lc = ui + oo2 * (ui - uimo) + fot * dm4m

    u_min = max(min(ui, uipo, u_md), min(ui, u_ul, u_lc))
    u_max = min(max(ui, uipo, u_md), max(ui, u_ul, u_lc))

    return u + minmod(u_min - u, u_max - u)


def _mpnlimiter_7p(eps, u, uim3, uimt, uimo, ui, uipo, uipt, uip3):
    """7-point MP limiter (used by mp7).

    Includes epsilon-based early exit (same as 5p variant).
    Uses 6-argument minmod for dm4p/dm4m curvature terms.
    """
    oo2 = 0.5
    fot = 4.0 / 3.0
    alphatil = 4.0

    if eps > 0:
        U_L2 = math.sqrt(uimt**2 + uimo**2 + ui**2 + uipo**2 + uipt**2)
        u_MP = ui + minmod(uipo - ui, alphatil * (ui - uimo))
        if (u - ui) * (u - u_MP) <= eps * U_L2:
            return u

    dm2 = uim3 - 2.0 * uimt + uimo
    dm = uimt - 2.0 * uimo + ui
    d0 = uimo - 2.0 * ui + uipo
    dp = ui - 2.0 * uipo + uipt
    dp2 = uipo - 2.0 * uipt + uip3

    dm4p = minmod(4.0 * d0 - dp, 4.0 * dp - d0, d0, dp, dm, dp2)
    dm4m = minmod(4.0 * d0 - dm, 4.0 * dm - d0, d0, dm, dp, dm2)

    u_ul = ui + alphatil * (ui - uimo)
    u_av = oo2 * (ui + uipo)
    u_md = u_av - oo2 * dm4p
    u_lc = ui + oo2 * (ui - uimo) + fot * dm4m

    u_min = max(min(ui, uipo, u_md), min(ui, u_ul, u_lc))
    u_max = min(max(ui, uipo, u_md), max(ui, u_ul, u_lc))

    return u + minmod(u_min - u, u_max - u)


def _mpnlimiter_R(eps, u, uimt, uimo, ui, uipo, uipt):
    """Refined MP limiter (He et al. 2016) for mp5_R."""
    oo2 = 0.5
    fot = 4.0 / 3.0
    alphatil = 4.0

    dm = uimt - 2.0 * uimo + ui
    d0 = uimo - 2.0 * ui + uipo
    dp = ui - 2.0 * uipo + uipt

    dm4p = minmod(4.0 * d0 - dp, 4.0 * dp - d0, d0, dp)
    dm4m = minmod(4.0 * d0 - dm, 4.0 * dm - d0, d0, dm)

    u_ul = ui + alphatil * (ui - uimo)
    u_av = oo2 * (ui + uipo)
    u_md = u_av - oo2 * dm4p
    u_lc = ui + oo2 * (ui - uimo) + fot * dm4m

    u_min = max(min(ui, uipo, u_md), min(ui, u_ul, u_lc))
    u_max = min(max(ui, uipo, u_md), max(ui, u_ul, u_lc))

    # TVD limit
    du_mh = ui - uimo
    du_ph = uipo - ui
    u_mp = ui + minmod(uipo - ui, alphatil * (ui - uimo))

    if ((u_max - u_min) > (max(ui, u_mp) - min(ui, u_mp))
            and (u < u_min or u_max < u)):
        phi_c = 2.0
        r_ph = du_mh / du_ph if du_ph > 0 else 0.0
        phi_ph = (r_ph + abs(r_ph)) / (1.0 + abs(r_ph))
        u_tvd = ui + 1.0 / phi_c * phi_ph * (uipo - ui)
        return u_tvd

    # Inside [u_min, u_max], use smooth limiter
    u_re_ph = (
        ui + 0.5 * (sign(du_mh) + sign(du_ph)) *
        abs(du_mh * du_ph) / (abs(du_mh) + abs(du_ph) + eps)
    )

    u_ph = (
        0.5 * (u + u_re_ph) -
        sign((u - u_min) * (u - u_max)) *
        0.5 * (u - u_re_ph)
    )

    return u_ph


# ---------------------------------------------------------------------------
# Internal: single-face left-biased reconstructions
# ---------------------------------------------------------------------------

def _rec_mp3(uimt, uimo, ui, uipo, uipt):
    r"""Compute :math:`u_{i+1/2}^-` using MP3 interpolation + 5-point limiter."""
    ulim = (-1.0 / 6.0) * uimo + (5.0 / 6.0) * ui + (2.0 / 6.0) * uipo
    return _mpnlimiter_5p(_MP_EPS, ulim, uimt, uimo, ui, uipo, uipt)


def _rec_mp5(uimt, uimo, ui, uipo, uipt):
    r"""Compute :math:`u_{i+1/2}^-` using MP5 interpolation + 5-point limiter."""
    ulim = ((2.0 / 60.0) * uimt +
            (-13.0 / 60.0) * uimo +
            (47.0 / 60.0) * ui +
            (27.0 / 60.0) * uipo +
            (-3.0 / 60.0) * uipt)
    return _mpnlimiter_5p(_MP_EPS, ulim, uimt, uimo, ui, uipo, uipt)


def _rec_mp7(uim3, uim2, uim1, ui, uip1, uip2, uip3):
    r"""Compute :math:`u_{i+1/2}^-` using MP7 interpolation + 7-point limiter."""
    ulim = ((-3.0 / 420.0) * uim3 +
            (25.0 / 420.0) * uim2 +
            (-101.0 / 420.0) * uim1 +
            (319.0 / 420.0) * ui +
            (214.0 / 420.0) * uip1 +
            (-38.0 / 420.0) * uip2 +
            (4.0 / 420.0) * uip3)
    return _mpnlimiter_7p(_MP_EPS, ulim, uim3, uim2, uim1, ui, uip1, uip2, uip3)


def _rec_mp5_R(uimt, uimo, ui, uipo, uipt):
    r"""Compute :math:`u_{i+1/2}^-` using MP5 interpolation + refined limiter."""
    ulim = ((2.0 / 60.0) * uimt +
            (-13.0 / 60.0) * uimo +
            (47.0 / 60.0) * ui +
            (27.0 / 60.0) * uipo +
            (-3.0 / 60.0) * uipt)
    return _mpnlimiter_R(_MP_EPS, ulim, uimt, uimo, ui, uipo, uipt)


# ---------------------------------------------------------------------------
# Public: cell-centered pointwise functions returning (uL, uR) pair
# ---------------------------------------------------------------------------


def mp3_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    r"""MP3 reconstruction at :math:`i+1/2` (3rd order, 5-point stencil, FV)."""
    uL = _rec_mp3(u_im2, u_im1, u_i, u_ip1, u_ip2)
    uR = _rec_mp3(u_ip2, u_ip1, u_i, u_im1, u_im2)
    return uL, uR


def mp5_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """MP5 reconstruction (5th order, 5-point stencil, FV)."""
    uL = _rec_mp5(u_im2, u_im1, u_i, u_ip1, u_ip2)
    uR = _rec_mp5(u_ip2, u_ip1, u_i, u_im1, u_im2)
    return uL, uR


def mp7_fv(u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3):
    """MP7 reconstruction (7th order, 7-point stencil, FV)."""
    uL = _rec_mp7(u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3)
    uR = _rec_mp7(u_ip3, u_ip2, u_ip1, u_i, u_im1, u_im2, u_im3)
    return uL, uR


def mp5_r_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """MP5_R reconstruction (5th-order, refined limiter, 5-pt stencil, FV)."""
    uL = _rec_mp5_R(u_im2, u_im1, u_i, u_ip1, u_ip2)
    uR = _rec_mp5_R(u_ip2, u_ip1, u_i, u_im1, u_im2)
    return uL, uR
# ---------------------------------------------------------------------------
# JIT compilation (pynalgo convention)
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    reveal_locals()  # noqa: F821
else:
    _mpnlimiter_5p = JIT(_mpnlimiter_5p)
    _mpnlimiter_7p = JIT(_mpnlimiter_7p)
    _mpnlimiter_R = JIT(_mpnlimiter_R)
    _rec_mp3 = JIT(_rec_mp3)
    _rec_mp5 = JIT(_rec_mp5)
    _rec_mp7 = JIT(_rec_mp7)
    _rec_mp5_R = JIT(_rec_mp5_R)
    mp3_fv = JIT(mp3_fv)
    mp5_fv = JIT(mp5_fv)
    mp7_fv = JIT(mp7_fv)
    mp5_r_fv = JIT(mp5_r_fv)
#
# :D
#
