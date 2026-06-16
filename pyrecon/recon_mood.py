"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: MOOD (Multi-dimensional Optimal Order Detection) reconstruction
"""
from pyrecon._jit_utils import JIT, TYPE_CHECKING

# ---------------------------------------------------------------------------
# Threshold for oscillation detection
# ---------------------------------------------------------------------------

_MOOD_EPS = 1e-12  # small parameter for flat regions


# ---------------------------------------------------------------------------
# Internal: interpolation polynomials
# ---------------------------------------------------------------------------


def _p5_L(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """5th-order upwind interpolation at face i+1/2 (cf. MP5 linear part)."""
    return (
        (2.0 / 60.0) * u_im2
        + (-13.0 / 60.0) * u_im1
        + (47.0 / 60.0) * u_i
        + (27.0 / 60.0) * u_ip1
        + (-3.0 / 60.0) * u_ip2
    )


def _p5_R(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """5th-order upwind interpolation at face i-1/2 (symmetric)."""
    return (
        (2.0 / 60.0) * u_ip2
        + (-13.0 / 60.0) * u_ip1
        + (47.0 / 60.0) * u_i
        + (27.0 / 60.0) * u_im1
        + (-3.0 / 60.0) * u_im2
    )


def _p3_L(u_im1, u_i, u_ip1):
    """3rd-order upwind interpolation at face i+1/2."""
    return (
        (-1.0 / 6.0) * u_im1
        + (5.0 / 6.0) * u_i
        + (2.0 / 6.0) * u_ip1
    )


def _p3_R(u_im1, u_i, u_ip1):
    """3rd-order upwind interpolation at face i-1/2 (symmetric)."""
    return (
        (-1.0 / 6.0) * u_ip1
        + (5.0 / 6.0) * u_i
        + (2.0 / 6.0) * u_im1
    )


# ---------------------------------------------------------------------------
# Detection criteria
# ---------------------------------------------------------------------------


def _detect_oscillation(uL, uR, u_im2, u_im1, u_i, u_ip1, u_ip2):
    """Detect oscillations in the reconstructed face values.

    MOOD detection: check if the reconstructed values create new extrema
    or exceed the local bounds.

    Returns True if oscillation detected (need to drop order).
    """
    u_min = min(u_im2, u_im1, u_i, u_ip1, u_ip2)
    u_max = max(u_im2, u_im1, u_i, u_ip1, u_ip2)

    if uL < u_min or uL > u_max:
        return True
    if uR < u_min or uR > u_max:
        return True

    beta = _compute_smoothness(u_im2, u_im1, u_i, u_ip1, u_ip2)
    u_range = u_max - u_min
    if u_range > _MOOD_EPS and beta > 10.0 * u_range * u_range:
        return True

    return False


def _compute_smoothness(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """Compute a smoothness indicator for the 5-point stencil."""
    d2_m = u_im2 - 2.0 * u_im1 + u_i
    d2_0 = u_im1 - 2.0 * u_i + u_ip1
    d2_p = u_i - 2.0 * u_ip1 + u_ip2
    d3_m = -u_im2 + 3.0 * u_im1 - 3.0 * u_i + u_ip1
    d3_p = -u_im1 + 3.0 * u_i - 3.0 * u_ip1 + u_ip2
    return d2_m * d2_m + d2_0 * d2_0 + d2_p * d2_p + d3_m * d3_m + d3_p * d3_p


# ---------------------------------------------------------------------------
# MOOD cascade: 3rd -> 1st order fallback
# ---------------------------------------------------------------------------


def _mood_L_fallback(u_im1, u_i, u_ip1):
    """MOOD left face 3rd-to-1st cascade (5th-order check done at mood_fv)."""
    uL_p3 = _p3_L(u_im1, u_i, u_ip1)
    u_min3 = min(u_im1, u_i, u_ip1)
    u_max3 = max(u_im1, u_i, u_ip1)
    if uL_p3 >= u_min3 and uL_p3 <= u_max3:
        return uL_p3
    return u_i


def _mood_R_fallback(u_im1, u_i, u_ip1):
    """MOOD right face 3rd-to-1st cascade (5th-order check done at mood_fv)."""
    uR_p3 = _p3_R(u_im1, u_i, u_ip1)
    u_min3 = min(u_im1, u_i, u_ip1)
    u_max3 = max(u_im1, u_i, u_ip1)
    if uR_p3 >= u_min3 and uR_p3 <= u_max3:
        return uR_p3
    return u_i


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def mood_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """MOOD reconstruction (FV).

    A posteriori limiting cascade:
      5th-order -> 3rd-order -> 1st-order

    The 5th-order candidates and oscillation detection are computed once
    and shared between left and right faces.

    Reference: Clain, Diot & Loubere (2011).
    """
    p5L = _p5_L(u_im2, u_im1, u_i, u_ip1, u_ip2)
    p5R = _p5_R(u_im2, u_im1, u_i, u_ip1, u_ip2)

    if not _detect_oscillation(p5L, p5R, u_im2, u_im1, u_i, u_ip1, u_ip2):
        return p5L, p5R

    # Oscillation detected: drop to 3rd->1st cascade
    uL = _mood_L_fallback(u_im1, u_i, u_ip1)
    uR = _mood_R_fallback(u_im1, u_i, u_ip1)
    return uL, uR
# ---------------------------------------------------------------------------
# JIT compilation (pynalgo convention)
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    reveal_locals()  # noqa: F821
else:
    _p5_L = JIT(_p5_L)
    _p5_R = JIT(_p5_R)
    _p3_L = JIT(_p3_L)
    _p3_R = JIT(_p3_R)
    _detect_oscillation = JIT(_detect_oscillation)
    _compute_smoothness = JIT(_compute_smoothness)
    _mood_L_fallback = JIT(_mood_L_fallback)
    _mood_R_fallback = JIT(_mood_R_fallback)
    mood_fv = JIT(mood_fv)
#
# :D
#
