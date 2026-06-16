"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Modified MP reconstruction for multi-phase flows (Ha, Lee 2020)
"""
from pyrecon.utils import minmod
from pyrecon._jit_utils import JIT, TYPE_CHECKING

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MP_EPS = 1e-12
_ALPHA = 4.0


# ---------------------------------------------------------------------------
# Internal: smooth extrema detector
# ---------------------------------------------------------------------------


def _is_smooth_extremum(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """Detect if cell i is at a smooth local extremum.

    A smooth extremum is characterized by:
    - Sign change in the first derivative estimate
    - Consistent second derivative signs (all same sign -> smooth,
      mixed signs -> possible discontinuity)
    - Curvature-magnitude fallback: if second derivative signs are mixed
      but their magnitudes are similar (|min|/|max| > 0.5), the stencil
      is still considered smooth.

    Returns True if the stencil suggests a smooth extremum that
    should bypass limiting.
    """
    # First derivative estimates
    dL = u_i - u_im1
    dR = u_ip1 - u_i

    # Not an extremum if both slopes have same sign
    if dL * dR >= 0:
        return False

    # Second derivatives
    d2_m = u_im2 - 2.0 * u_im1 + u_i
    d2_0 = u_im1 - 2.0 * u_i + u_ip1
    d2_p = u_i - 2.0 * u_ip1 + u_ip2

    # Check if all second derivatives have the same sign
    # (indicating smooth curvature, not a discontinuity)
    if d2_m * d2_0 > 0 and d2_0 * d2_p > 0 and d2_m * d2_p > 0:
        return True

    # Alternative check: the magnitude of curvature variation
    d2_max = max(abs(d2_m), abs(d2_0), abs(d2_p))
    d2_min = min(abs(d2_m), abs(d2_0), abs(d2_p))

    if d2_max > _MP_EPS and d2_min / d2_max > 0.5:
        # Similar curvature magnitudes -> smooth
        return True

    return False


# ---------------------------------------------------------------------------
# Internal: modified MP limiter for multi-phase
# ---------------------------------------------------------------------------


def _mp_mp_limiter(ulim, u_im2, u_im1, u_i, u_ip1, u_ip2):
    """Modified MP limiter for multi-phase flows.

    Bypasses limiting at smooth extrema to preserve phase interface
    resolution.  At discontinuities, applies MP limiting with a 5%
    overshoot relaxation to avoid excessive clipping at phase boundaries.
    """
    # Check for smooth extremum bypass
    if _is_smooth_extremum(u_im2, u_im1, u_i, u_ip1, u_ip2):
        # At smooth extremum, use unlimited interpolation
        # (the MP scheme would clip the extremum otherwise)
        return ulim

    # Standard MP limiter follows
    oo2 = 0.5
    fot = 4.0 / 3.0

    u_mp = u_i + minmod(u_ip1 - u_i, _ALPHA * (u_i - u_im1))
    if (ulim - u_i) * (ulim - u_mp) <= _MP_EPS:
        return ulim

    d_m = u_im2 - 2.0 * u_im1 + u_i
    d_0 = u_im1 - 2.0 * u_i + u_ip1
    d_p = u_i - 2.0 * u_ip1 + u_ip2

    dm4p = minmod(4.0 * d_0 - d_p, 4.0 * d_p - d_0, d_0, d_p)
    dm4m = minmod(4.0 * d_0 - d_m, 4.0 * d_m - d_0, d_0, d_m)

    u_ul = u_i + _ALPHA * (u_i - u_im1)
    u_av = oo2 * (u_i + u_ip1)
    u_md = u_av - oo2 * dm4p
    u_lc = u_i + oo2 * (u_i - u_im1) + fot * dm4m

    u_min = max(min(u_i, u_ip1, u_md), min(u_i, u_ul, u_lc))
    u_max = min(max(u_i, u_ip1, u_md), max(u_i, u_ul, u_lc))

    # Relaxed limiting for multi-phase: use softer clipping
    # Allow a small overshoot (5%) to avoid excessive clipping
    # at phase boundaries
    u_range = u_max - u_min
    relax = 0.05 * u_range
    u_min = u_min - relax
    u_max = u_max + relax

    return ulim + minmod(u_min - ulim, u_max - ulim)


# ---------------------------------------------------------------------------
# Internal: left/right face interpolations
# ---------------------------------------------------------------------------


def _rec_mp5_mp_L(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """Modified MP5 left-biased interpolation for multi-phase."""
    ulim = (
        (2.0 / 60.0) * u_im2
        + (-13.0 / 60.0) * u_im1
        + (47.0 / 60.0) * u_i
        + (27.0 / 60.0) * u_ip1
        + (-3.0 / 60.0) * u_ip2
    )
    return _mp_mp_limiter(ulim, u_im2, u_im1, u_i, u_ip1, u_ip2)


def _rec_mp5_mp_R(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """Modified MP5 right-biased interpolation for multi-phase."""
    ulim = (
        (2.0 / 60.0) * u_ip2
        + (-13.0 / 60.0) * u_ip1
        + (47.0 / 60.0) * u_i
        + (27.0 / 60.0) * u_im1
        + (-3.0 / 60.0) * u_im2
    )
    return _mp_mp_limiter(ulim, u_ip2, u_ip1, u_i, u_im1, u_im2)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def mp5_mp_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """Modified MP5 for multi-phase flows (FV).

    Uses smooth extremum bypass and relaxed limiting to preserve
    phase interface resolution in multi-phase flow simulations.

    Reference: Ha & Lee (2020).
    """
    uL = _rec_mp5_mp_L(u_im2, u_im1, u_i, u_ip1, u_ip2)
    uR = _rec_mp5_mp_R(u_im2, u_im1, u_i, u_ip1, u_ip2)
    return uL, uR
# ---------------------------------------------------------------------------
# JIT compilation (pynalgo convention)
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    reveal_locals()  # noqa: F821
else:
    _is_smooth_extremum = JIT(_is_smooth_extremum)
    _mp_mp_limiter = JIT(_mp_mp_limiter)
    _rec_mp5_mp_L = JIT(_rec_mp5_mp_L)
    _rec_mp5_mp_R = JIT(_rec_mp5_mp_R)
    mp5_mp_fv = JIT(mp5_mp_fv)
#
# :D
#
