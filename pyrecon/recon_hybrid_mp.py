"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Hybrid reconstruction-level MP5-MC2
    (curvature-ratio switch). Not Ahn & Lee 2019
    flux-level MP5-Hyb (Roe vs LF).
"""
from pyrecon.utils import MC2, minmod
from pyrecon._jit_utils import JIT, TYPE_CHECKING

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPSL = 1e-40

# MP5 interpolation coefficients (for u_{i+1/2}^-)
_MP5_COEFFS = (2.0 / 60.0, -13.0 / 60.0, 47.0 / 60.0, 27.0 / 60.0, -3.0 / 60.0)

# Hybrid switching threshold: if the local curvature ratio exceeds this,
# switch from MP5 to MUSCL for robustness.
_HYBRID_THRESH = 10.0


# ---------------------------------------------------------------------------
# MP5 limiter (simplified from recon_mpn.py for standalone use)
# ---------------------------------------------------------------------------

def _mpnlimiter(u, uimt, uimo, ui, uipo, uipt):
    """MP5-type curvature limiter (simplified from recon_mpn._mpnlimiter_5p; drops epsilon early-exit)."""
    oo2 = 0.5
    fot = 4.0 / 3.0
    alphatil = 4.0

    dm = uimt - 2.0 * uimo + ui
    d0 = uimo - 2.0 * ui + uipo
    dp = ui - 2.0 * uipo + uipt

    # minmod of second derivative estimates
    dm4p = minmod(4.0 * d0 - dp, 4.0 * dp - d0, d0, dp)
    dm4m = minmod(4.0 * d0 - dm, 4.0 * dm - d0, d0, dm)

    u_ul = ui + alphatil * (ui - uimo)
    u_av = oo2 * (ui + uipo)
    u_md = u_av - oo2 * dm4p
    u_lc = ui + oo2 * (ui - uimo) + fot * dm4m

    u_min = max(min(ui, uipo, u_md), min(ui, u_ul, u_lc))
    u_max = min(max(ui, uipo, u_md), max(ui, u_ul, u_lc))

    return u + minmod(u_min - u, u_max - u)


# ---------------------------------------------------------------------------
# MUSCL-MC2 fallback (3-point, paired L+R)
# ---------------------------------------------------------------------------

def _muscl_mc2_LR(u_im1, u_i, u_ip1):
    """MUSCL-MC2 reconstruction on {i-1, i, i+1}.

    """
    slope = MC2(u_i - u_im1, u_ip1 - u_i)
    uL = u_i + 0.5 * slope
    uR = u_i - 0.5 * slope
    return uL, uR


# ---------------------------------------------------------------------------
# Hybrid criterion: detect extreme local variation
# ---------------------------------------------------------------------------

def _needs_fallback(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """Determine whether MP5 should fall back to MUSCL.

    Criterion: maximum absolute curvature ratio exceeds threshold.
    Large values indicate regions where MP5 may produce oscillations.
    """
    d2m = abs(u_im2 - 2.0 * u_im1 + u_i)
    d2c = abs(u_im1 - 2.0 * u_i + u_ip1)
    d2p = abs(u_i - 2.0 * u_ip1 + u_ip2)

    d2_max = max(d2m, d2c, d2p)
    d2_min = min(d2m, d2c, d2p) + _EPSL

    return (d2_max / d2_min) > _HYBRID_THRESH


# ---------------------------------------------------------------------------
# Hybrid MP5-MUSCL reconstruction core
# ---------------------------------------------------------------------------

def _hybrid_LR(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """Hybrid paired L+R reconstruction.

    Uses MP5 in smooth regions, fallback (MUSCL-MC2) near extreme gradients.

    """
    if _needs_fallback(u_im2, u_im1, u_i, u_ip1, u_ip2):
        return _muscl_mc2_LR(u_im1, u_i, u_ip1)

    # MP5 left face
    uL_raw = (_MP5_COEFFS[0] * u_im2 +
              _MP5_COEFFS[1] * u_im1 +
              _MP5_COEFFS[2] * u_i +
              _MP5_COEFFS[3] * u_ip1 +
              _MP5_COEFFS[4] * u_ip2)
    uL = _mpnlimiter(uL_raw, u_im2, u_im1, u_i, u_ip1, u_ip2)

    # MP5 right face (reversed stencil)
    uR_raw = (_MP5_COEFFS[0] * u_ip2 +
              _MP5_COEFFS[1] * u_ip1 +
              _MP5_COEFFS[2] * u_i +
              _MP5_COEFFS[3] * u_im1 +
              _MP5_COEFFS[4] * u_im2)
    uR = _mpnlimiter(uR_raw, u_ip2, u_ip1, u_i, u_im1, u_im2)

    return uL, uR


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def hybrid_mp_mc2_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """Hybrid MP5-MUSCL(MC2) reconstruction.

    Uses 5th-order MP5 in smooth regions, falls back to MUSCL-MC2
    in regions of extreme gradients for robustness.

    """
    return _hybrid_LR(u_im2, u_im1, u_i, u_ip1, u_ip2)
# ---------------------------------------------------------------------------
# JIT compilation (pynalgo convention)
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    reveal_locals()  # noqa: F821
else:
    _mpnlimiter = JIT(_mpnlimiter)
    _muscl_mc2_LR = JIT(_muscl_mc2_LR)
    _needs_fallback = JIT(_needs_fallback)
    _hybrid_LR = JIT(_hybrid_LR)
    hybrid_mp_mc2_fv = JIT(hybrid_mp_mc2_fv)
#
# :D
#
