"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: LS-WENO5-H: WENO5-Z with log-space fallbacks at sharp features
"""
import math
from pyrecon._jit_utils import JIT, TYPE_CHECKING
from pyrecon.recon_weno5 import _weno5z_LR_fv, _JS3_M
from pyrecon.utils import exp_face_log

_LOG_EPS = 1e-40
_DROP_THRESH = 3.0
_SHARP_THRESH = 10.0


# ---------------------------------------------------------------------------
# WENO5-Z in original space
# ---------------------------------------------------------------------------

def _weno5z_orig(u_im2, u_im1, u_i, u_ip1, u_ip2, left_face=True):
    """WENO5-Z in original space (delegates to pyrecon.recon_weno5)."""
    uL, uR = _weno5z_LR_fv(u_im2, u_im1, u_i, u_ip1, u_ip2)
    return uL if left_face else uR

# ---------------------------------------------------------------------------
# Beta-ratio sharp-feature fallback: 2-point linear from smooth side
# ---------------------------------------------------------------------------

def _sharp_fallback(u_im2, u_im1, u_i, u_ip1, u_ip2, left_face=True):
    """2-point reconstruction when stencil spans a sharp feature.

    Uses beta ratios in LOG-SPACE, where beta ranges from ~1e-4 (smooth
    perturbation) to ~50 (sharp drop), giving clean separation.

    Returns a 2-point extrapolation toward the face from the smooth side,
    or a central average when the smooth side points away from the face.
    """
    v = [math.log(max(x, _LOG_EPS)) for x in (u_im2, u_im1, u_i, u_ip1, u_ip2)]

    if left_face:
        b_vals = [(v[0], v[1], v[2]),
                  (v[1], v[2], v[3]),
                  (v[2], v[3], v[4])]
    else:
        b_vals = [(v[4], v[3], v[2]),
                  (v[3], v[2], v[1]),
                  (v[2], v[1], v[0])]

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
    b_min_abs = min(b_abs)
    b_max_abs = max(b_abs)
    if (b_min_abs == 0 or b_max_abs < 0.1
            or b_max_abs / b_min_abs <= _SHARP_THRESH):
        return None  # no sharp feature

    k_smooth = b_abs.index(b_min_abs)
    if left_face:
        if k_smooth <= 1:
            return (-u_im1 + 3.0 * u_i) / 2.0
        else:
            return (u_i + u_ip1) / 2.0
    else:
        if k_smooth <= 1:
            return (u_im1 + u_i) / 2.0
        else:
            return (-u_ip1 + 3.0 * u_i) / 2.0


# ---------------------------------------------------------------------------
# Main pointwise function
# ---------------------------------------------------------------------------

def lsweno5h_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """LS-WENO5-H: WENO5-Z default, log-space fallbacks at sharp features."""
    # --- Left face ---
    if u_i > _LOG_EPS and u_ip1 > _LOG_EPS and u_i > u_ip1:
        log_drop = math.log(u_i / u_ip1)
        if log_drop > _DROP_THRESH:
            uL = exp_face_log(u_i, u_ip1)
        else:
            sf = _sharp_fallback(u_im2, u_im1, u_i, u_ip1, u_ip2, True)
            uL = (sf if sf is not None
                  else _weno5z_orig(u_im2, u_im1, u_i, u_ip1, u_ip2, True))
    else:
        sf = _sharp_fallback(u_im2, u_im1, u_i, u_ip1, u_ip2, True)
        uL = (sf if sf is not None
              else _weno5z_orig(u_im2, u_im1, u_i, u_ip1, u_ip2, True))

    # --- Right face ---
    if u_im1 > _LOG_EPS and u_i > _LOG_EPS and u_im1 > u_i:
        log_drop = math.log(u_im1 / u_i)
        if log_drop > _DROP_THRESH:
            uR = exp_face_log(u_im1, u_i)
        else:
            sf = _sharp_fallback(u_im2, u_im1, u_i, u_ip1, u_ip2, False)
            uR = (sf if sf is not None
                  else _weno5z_orig(u_im2, u_im1, u_i, u_ip1, u_ip2, False))
    else:
        sf = _sharp_fallback(u_im2, u_im1, u_i, u_ip1, u_ip2, False)
        uR = (sf if sf is not None
              else _weno5z_orig(u_im2, u_im1, u_i, u_ip1, u_ip2, False))

    return uL, uR
# ---------------------------------------------------------------------------
# JIT compilation (pynalgo convention)
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    reveal_locals()  # noqa: F821
else:
    _weno5z_orig = JIT(_weno5z_orig)
    _sharp_fallback = JIT(_sharp_fallback)
    lsweno5h_fv = JIT(lsweno5h_fv)
#
# :D
#
