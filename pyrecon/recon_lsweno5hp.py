"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: LS-WENO5-HP: WENO5-Z with physics-informed log-space fallbacks
"""
import math
from pyrecon._jit_utils import JIT, TYPE_CHECKING
from pyrecon.recon_weno5 import weno5z_fv as _weno5_default, _JS3_M
from pyrecon.utils import exp_face_log

_LOG_EPS = 1e-40
_DROP_THRESH = 3.0
_SHARP_THRESH = 10.0

# ---------------------------------------------------------------------------
# Two-region plateau+exponential fit (physics-informed)
# ---------------------------------------------------------------------------
def _two_region_fit(u_im2, u_im1, u_i, u_ip1, u_ip2, left_face=True):
    """Two-region fit: constant interior + exponential exterior.

    For the right face (left_face=False), the face always lies inside
    the plateau region so the exponential tail fit is never applied --
    the interior density is always returned.

    The surface position parameter alpha is clamped to [0, 1].

    Returns face value, or None if the pattern is not detected.
    """
    # Plateau detection: im2 and im1 must be on the same side as u_i
    if not (u_im2 > _LOG_EPS and u_im1 > _LOG_EPS and u_i > _LOG_EPS):
        return None
    if u_im2 / u_i > 5.0:
        return None
    ratio_plateau = max(u_im2, u_im1) / max(min(u_im2, u_im1), _LOG_EPS)
    if ratio_plateau > 2.0:
        return None

    # Tail detection: need two cells for H fitting
    if not (u_ip1 > _LOG_EPS and u_ip2 > _LOG_EPS and u_ip1 > u_ip2):
        return None

    # Interior density from 2-point linear at adjacent face
    # (captures perturbation trend better than simple average)
    rho_int = (-u_im2 + 3.0 * u_im1) / 2.0
    H_fit = -1.0 / math.log(max(u_ip2, _LOG_EPS) / max(u_ip1, _LOG_EPS))

    # Surface position: for steep drops (dx/H >> 1),
    # exp(-(1-alpha)*dx/H) is negligible, giving the closed form:
    alpha = u_i / rho_int - H_fit
    alpha = max(0.0, min(1.0, alpha))

    if left_face:
        dist = 1.0 - alpha
    else:
        dist = -alpha

    if dist <= 0:
        return rho_int
    return rho_int * math.exp(-dist / H_fit)


# ---------------------------------------------------------------------------
# Beta-ratio sharp-feature fallback: 3-point ENO from smoothest sub-stencil
# Note: different algorithm from the same-named _sharp_fallback in lsweno5h.py
# (this is 3-point ENO; lsweno5h uses 2-point linear).
# ---------------------------------------------------------------------------

def _sharp_fallback(u_im2, u_im1, u_i, u_ip1, u_ip2, left_face=True):
    """3-point ENO from the smoothest log-space sub-stencil.

    Returns face value, or None if no sharp feature detected.
    """
    v = [math.log(max(x, _LOG_EPS)) for x in (u_im2, u_im1, u_i, u_ip1, u_ip2)]

    if left_face:
        b_vals = [(v[0], v[1], v[2]), (v[1], v[2], v[3]), (v[2], v[3], v[4])]
    else:
        b_vals = [(v[4], v[3], v[2]), (v[3], v[2], v[1]), (v[2], v[1], v[0])]

    def _js(vals, m):
        b = 0.0
        for i in range(3):
            r = m[i]
            for j in range(3):
                b += r[j] * vals[i] * vals[j]
        return b / 6.0

    b0 = _js(b_vals[0], _JS3_M[0])
    b1 = _js(b_vals[1], _JS3_M[1])
    b2 = _js(b_vals[2], _JS3_M[2])

    b_abs = [abs(b0), abs(b1), abs(b2)]
    b_max_abs = max(b_abs)
    b_min_abs = min(b_abs)
    if (b_min_abs == 0 or b_max_abs < 1.0
            or b_max_abs / b_min_abs <= _SHARP_THRESH):
        return None

    k_smooth = b_abs.index(b_min_abs)
    if left_face:
        if k_smooth == 0:
            return (2.0 * u_im2 - 7.0 * u_im1 + 11.0 * u_i) / 6.0
        elif k_smooth == 1:
            return (-u_im1 + 5.0 * u_i + 2.0 * u_ip1) / 6.0
        else:
            return (2.0 * u_i + 5.0 * u_ip1 - u_ip2) / 6.0
    else:
        if k_smooth == 0:
            return (2.0 * u_ip2 - 7.0 * u_ip1 + 11.0 * u_i) / 6.0
        elif k_smooth == 1:
            return (-u_ip1 + 5.0 * u_i + 2.0 * u_im1) / 6.0
        else:
            return (2.0 * u_i + 5.0 * u_im1 - u_im2) / 6.0


# ---------------------------------------------------------------------------
# Main pointwise function
# ---------------------------------------------------------------------------

def lsweno5hp_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    r"""LS-WENO5-HP: physics-informed hybrid reconstruction.

    :math:`u_L = u_{i+1/2}^-`, :math:`u_R = u_{i-1/2}^+`.
    """
    # --- Left face ---
    if u_i > _LOG_EPS and u_ip1 > _LOG_EPS and u_i > u_ip1:
        log_drop = math.log(u_i / u_ip1)
        if log_drop > _DROP_THRESH:
            tr = _two_region_fit(u_im2, u_im1, u_i, u_ip1, u_ip2, True)
            uL = tr if tr is not None else exp_face_log(u_i, u_ip1)
        else:
            sf = _sharp_fallback(u_im2, u_im1, u_i, u_ip1, u_ip2, True)
            uL = (sf if sf is not None
                  else _weno5_default(u_im2, u_im1, u_i, u_ip1, u_ip2)[0])
    else:
        sf = _sharp_fallback(u_im2, u_im1, u_i, u_ip1, u_ip2, True)
        uL = (sf if sf is not None
              else _weno5_default(u_im2, u_im1, u_i, u_ip1, u_ip2)[0])

    # --- Right face ---
    if u_im1 > _LOG_EPS and u_i > _LOG_EPS and u_im1 > u_i:
        log_drop = math.log(u_im1 / u_i)
        if log_drop > _DROP_THRESH:
            tr = _two_region_fit(u_im2, u_im1, u_i, u_ip1, u_ip2, False)
            uR = tr if tr is not None else exp_face_log(u_im1, u_i)
        else:
            sf = _sharp_fallback(u_im2, u_im1, u_i, u_ip1, u_ip2, False)
            uR = (sf if sf is not None
                  else _weno5_default(u_im2, u_im1, u_i, u_ip1, u_ip2)[1])
    else:
        sf = _sharp_fallback(u_im2, u_im1, u_i, u_ip1, u_ip2, False)
        uR = (sf if sf is not None
              else _weno5_default(u_im2, u_im1, u_i, u_ip1, u_ip2)[1])

    return uL, uR
# ---------------------------------------------------------------------------
# JIT compilation (pynalgo convention)
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    reveal_locals()  # noqa: F821
else:
    _two_region_fit = JIT(_two_region_fit)
    _sharp_fallback = JIT(_sharp_fallback)
    lsweno5hp_fv = JIT(lsweno5hp_fv)
#
# :D
#
