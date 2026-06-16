"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: BVD central-upwind reconstruction (Chamarthi, Frankel 2021)

HOCUS6 with compact C5 tridiagonal solver + alpha=7.
Also includes explicit 6th-order FV + MP5 (alpha=4) variant, bvd_e6_mp5_fv.
"""
from pyrecon.utils import minmod

from pyrecon._jit_utils import JIT, TYPE_CHECKING
_MP_EPS = 1e-12

# ---------------------------------------------------------------------------
# HOCUS6: alpha = 7
# ---------------------------------------------------------------------------
_ALPHA = 7.0
_ALPHA_LEGACY = 4.0

# ---------------------------------------------------------------------------
# C5 compact interpolation RHS (Chamarthi & Frankel 2021)
# ---------------------------------------------------------------------------

def _c5_rhs_L(u_im1, u_i, u_ip1):
    r"""C5 right-hand side for left-biased interpolation at :math:`i+1/2`.

    :math:`\frac{1}{18} u_{i-1} + \frac{19}{18} u_i + \frac{5}{9} u_{i+1}`
    """
    return (1.0 / 18.0) * u_im1 + (19.0 / 18.0) * u_i + (5.0 / 9.0) * u_ip1


def _c5_rhs_R(u_i, u_ip1, u_ip2):
    r"""C5 right-hand side for right-biased interpolation at :math:`i+1/2`.

    :math:`\frac{5}{9} u_i + \frac{19}{18} u_{i+1} + \frac{1}{18} u_{i+2}`
    """
    return (5.0 / 9.0) * u_i + (19.0 / 18.0) * u_ip1 + (1.0 / 18.0) * u_ip2


# ---------------------------------------------------------------------------
# C5 tridiagonal solver (Thomas algorithm, local stencil)
# ---------------------------------------------------------------------------

def _solve_c5_local_L(rhs):
    r"""Solve 4x4 tridiagonal system for left-biased C5 interpolation.

    Solves for faces :math:`u_{i-3/2}, u_{i-1/2}, u_{i+1/2}, u_{i+3/2}`.
    Matrix: a=1/2 (sub), b=1 (diag), c=1/6 (super).
    The caller MUST subtract boundary terms from *rhs*:
      rhs[0] -= a * u_{i-5/2}  (donor cell i-3)
      rhs[3] -= c * u_{i+5/2}  (donor cell i+2)

    Returns face values (u_{i-3/2}, u_{i-1/2}, u_{i+1/2}, u_{i+3/2}).
    """
    # Thomas algorithm for the 4x4 system
    c = 1.0 / 6.0
    a = 1.0 / 2.0
    b = 1.0

    # Forward elimination
    cp = [0.0] * 4
    dp = [0.0] * 4

    cp[0] = c / b
    dp[0] = rhs[0] / b
    for i in range(1, 4):
        denom = b - a * cp[i - 1]
        cp[i] = c / denom
        dp[i] = (rhs[i] - a * dp[i - 1]) / denom

    # Back substitution
    u = [0.0] * 4
    u[3] = dp[3]
    for i in range(2, -1, -1):
        u[i] = dp[i] - cp[i] * u[i + 1]

    return u[0], u[1], u[2], u[3]


def _solve_c5_local_R(rhs):
    r"""Solve 3x3 tridiagonal system for right-biased C5 interpolation.

    Solves for faces :math:`u_{i-1/2}, u_{i+1/2}, u_{i+3/2}`.
    Matrix: a=1/6 (sub), b=1 (diag), c=1/2 (super).
    The caller MUST subtract boundary terms from *rhs*:
      rhs[0] -= a * u_{i-3/2}  (donor cell i-2)
      rhs[2] -= c * u_{i+5/2}  (donor cell i+2)

    Returns face values (u_{i-1/2}, u_{i+1/2}, u_{i+3/2}).
    """
    # Thomas algorithm for the 3x3 system
    c = 1.0 / 2.0
    a = 1.0 / 6.0
    b = 1.0

    # Forward elimination
    cp = [0.0] * 3
    dp = [0.0] * 3

    cp[0] = c / b
    dp[0] = rhs[0] / b
    for i in range(1, 3):
        denom = b - a * cp[i - 1]
        cp[i] = c / denom
        dp[i] = (rhs[i] - a * dp[i - 1]) / denom

    # Back substitution
    u = [0.0] * 3
    u[2] = dp[2]
    for i in range(1, -1, -1):
        u[i] = dp[i] - cp[i] * u[i + 1]

    return u[0], u[1], u[2]


# ---------------------------------------------------------------------------
# MP5 5th-order interpolation + limiter
# ---------------------------------------------------------------------------

def _mp5_L(u_im2, u_im1, u_i, u_ip1, u_ip2, alpha=_ALPHA):
    """5th-order MP5 interpolation at left face (i+1/2).

    Coefficients: (2, -13, 47, 27, -3)/60.
    """
    u_MP = (2.0 * u_im2 - 13.0 * u_im1 + 47.0 * u_i
            + 27.0 * u_ip1 - 3.0 * u_ip2) / 60.0
    return _mp5_limiter(u_MP, u_im2, u_im1, u_i, u_ip1, u_ip2, alpha)


def _mp5_R(u_im2, u_im1, u_i, u_ip1, u_ip2, alpha=_ALPHA):
    """5th-order MP5 interpolation at right face (i-1/2) with reversed stencil.

    Delegates to _mp5_L on the reversed stencil so the limiter
    (which is left-face-oriented) receives the correct orientation.

    Coefficients: (-3, 27, 47, -13, 2)/60.
    """
    return _mp5_L(u_ip2, u_ip1, u_i, u_im1, u_im2, alpha=alpha)


def _mp5_limiter(u, u_im2, u_im1, u_i, u_ip1, u_ip2, alpha):
    """MP5 limiter with configurable alpha parameter.

    Alpha controls the allowable overshoot.
    """
    # Median check
    u_MP = u_i + minmod(u_ip1 - u_i, alpha * (u_i - u_im1))
    if (u - u_i) * (u - u_MP) <= _MP_EPS:
        return u

    # Curvature terms
    d_m = u_im2 - 2.0 * u_im1 + u_i
    d_0 = u_im1 - 2.0 * u_i + u_ip1
    d_p = u_i - 2.0 * u_ip1 + u_ip2

    dm4p = minmod(4.0 * d_0 - d_p, 4.0 * d_p - d_0, d_0, d_p)
    dm4m = minmod(4.0 * d_0 - d_m, 4.0 * d_m - d_0, d_0, d_m)

    u_ul = u_i + alpha * (u_i - u_im1)
    u_av = 0.5 * (u_i + u_ip1)
    u_md = u_av - 0.5 * dm4p
    u_lc = u_i + 0.5 * (u_i - u_im1) + (4.0 / 3.0) * dm4m

    u_min = max(min(u_i, u_ip1, u_md), min(u_i, u_ul, u_lc))
    u_max = min(max(u_i, u_ip1, u_md), max(u_i, u_ul, u_lc))

    return u + minmod(u_min - u, u_max - u)


# ---------------------------------------------------------------------------
# BVD selection: HOCUS6 (C5) vs MP5 (alpha=7)
# ---------------------------------------------------------------------------

def _bvd_hocus6_select(u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3):
    """BVD selection between HOCUS6 (C5) and MP5 (alpha=7).

    Uses cell-to-cell TBV (Total Boundary Variation) for both schemes,
    selecting the one with smaller TBV.

    Returns (uL, uR) at faces i+1/2 and i-1/2.
    """
    # --- HOCUS6 (C5) candidate ---
    # Left-biased interpolation at faces i-3/2, i-1/2, i+1/2, i+3/2.
    # C5 equation: a*u_{j-1/2} + u_{j+1/2} + c*u_{j+3/2} = RHS,
    # with a=1/2, c=1/6 (left-biased).
    rhs_L = [_c5_rhs_L(u_im3, u_im2, u_im1),
             _c5_rhs_L(u_im2, u_im1, u_i),
             _c5_rhs_L(u_im1, u_i, u_ip1),
             _c5_rhs_L(u_i, u_ip1, u_ip2)]
    # Boundary faces via adjacent-cell arithmetic mean where possible.
    # u_{i-5/2}: only cell i-3 available -> donor cell u_im3.
    # u_{i+5/2}: cells i+2, i+3 -> (u_ip2 + u_ip3)/2.
    rhs_L[0] -= 0.5 * u_im3                     # a * u_{i-5/2}
    rhs_L[3] -= (1.0 / 6.0) * (u_ip2 + u_ip3) * 0.5  # c * u_{i+5/2}
    h6_L_im1, h6_L_i, h6_L_ip1, h6_L_ip2 = _solve_c5_local_L(rhs_L)

    # Right-biased interpolation at faces i-1/2, i+1/2, i+3/2.
    # C5 equation: a*u_{j-1/2} + u_{j+1/2} + c*u_{j+3/2} = RHS,
    # with a=1/6, c=1/2 (right-biased).
    rhs_R = [_c5_rhs_R(u_im2, u_im1, u_i),
             _c5_rhs_R(u_im1, u_i, u_ip1),
             _c5_rhs_R(u_i, u_ip1, u_ip2)]
    # u_{i-3/2}: cells i-2, i-1 -> (u_im2 + u_im1)/2.
    # u_{i+5/2}: cells i+2, i+3 -> (u_ip2 + u_ip3)/2.
    rhs_R[0] -= (1.0 / 6.0) * (u_im2 + u_im1) * 0.5  # a * u_{i-3/2}
    rhs_R[2] -= 0.5 * (u_ip2 + u_ip3) * 0.5           # c * u_{i+5/2}
    h6_R_im1, h6_R_i, h6_R_ip1 = _solve_c5_local_R(rhs_R)

    # TBV for C5: |L_biased - R_biased| at each face
    tbv_h6 = abs(h6_L_im1 - h6_R_im1) + abs(h6_L_i - h6_R_i)

    # --- MP5 (alpha=7) candidate ---
    mL_im1 = _mp5_L(u_im3, u_im2, u_im1, u_i, u_ip1, alpha=_ALPHA)
    mR_i   = _mp5_R(u_im2, u_im1, u_i, u_ip1, u_ip2, alpha=_ALPHA)

    mL_i   = _mp5_L(u_im2, u_im1, u_i, u_ip1, u_ip2, alpha=_ALPHA)
    mR_ip1 = _mp5_R(u_im1, u_i, u_ip1, u_ip2, u_ip3, alpha=_ALPHA)

    # TBV at face i-1/2: |mp5R_i - mp5L_{i-1}|
    tbv_mp5 = abs(mR_i - mL_im1)
    # TBV at face i+1/2: |mp5R_{i+1} - mp5L_i|
    tbv_mp5 += abs(mR_ip1 - mL_i)

    # Select candidate with smaller TBV
    if tbv_h6 < tbv_mp5:
        return h6_L_i, h6_R_i   # (u at i+1/2, u at i-1/2)
    else:
        return mL_i, mR_i


# ---------------------------------------------------------------------------
# 6th-order explicit central FV interpolation
# ---------------------------------------------------------------------------

def _central6_fv_L(u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3):
    """6th-order central interpolation at left face (i+1/2).

    Coefficients: (-1, 9, 37, 37, -8, 1) * u / 60.
    """
    return (-1.0 * u_im2 + 9.0 * u_im1 + 37.0 * u_i
            + 37.0 * u_ip1 - 8.0 * u_ip2 + 1.0 * u_ip3) / 60.0


def _central6_fv_R(u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3):
    """6th-order central interpolation at right face (i-1/2).

    Coefficients: (1, -8, 37, 37, 9, -1) * u / 60.
    """
    return (1.0 * u_im2 - 8.0 * u_im1 + 37.0 * u_i
            + 37.0 * u_ip1 + 9.0 * u_ip2 - 1.0 * u_ip3) / 60.0


# ---------------------------------------------------------------------------
# MP5 legacy (alpha=4)
# ---------------------------------------------------------------------------

def _mp5_legacy_L(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """Legacy 5th-order MP5 interpolation at left face (alpha=4)."""
    return _mp5_L(u_im2, u_im1, u_i, u_ip1, u_ip2, alpha=_ALPHA_LEGACY)


def _mp5_legacy_R(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """Legacy 5th-order MP5 interpolation at right face (alpha=4)."""
    return _mp5_R(u_im2, u_im1, u_i, u_ip1, u_ip2, alpha=_ALPHA_LEGACY)


# ---------------------------------------------------------------------------
# BVD selection: explicit 6th-order central vs MP5 (alpha=4)
# ---------------------------------------------------------------------------

def _bvd_e6_mp5_select(u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3):
    """BVD selection between explicit 6th-order central and MP5 (alpha=4).

    TBV for MP5: cell-to-cell interface jumps at faces i-1/2 and i+1/2.
    TBV for central6: same jumps using central interpolation.

    Returns (uL, uR) at faces i+1/2 and i-1/2.
    """
    # Central6 candidate
    cL_i = _central6_fv_L(u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3)
    cR_i = _central6_fv_R(u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3)

    # MP5 (alpha=4) candidate
    mL_i = _mp5_legacy_L(u_im2, u_im1, u_i, u_ip1, u_ip2)
    mR_i = _mp5_legacy_R(u_im2, u_im1, u_i, u_ip1, u_ip2)

    mL_i1 = _mp5_legacy_L(u_im3, u_im2, u_im1, u_i, u_ip1)
    mR_ip1 = _mp5_legacy_R(u_im1, u_i, u_ip1, u_ip2, u_ip3)

    # TBV for central6
    tbv_c6 = abs(cL_i - cR_i)

    # TBV for MP5: |mp5R_i - mp5L_{i-1}| + |mp5R_{i+1} - mp5L_i|
    tbv_mp5 = abs(mR_i - mL_i1) + abs(mR_ip1 - mL_i)

    if tbv_c6 < tbv_mp5:
        return cL_i, cR_i
    else:
        return mL_i, mR_i


# ---------------------------------------------------------------------------
# Public BVD-CU API
# ---------------------------------------------------------------------------

def bvd_cu_fv(u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3):
    r"""BVD central-upwind HOCUS7 (FV) at :math:`i+1/2`, :math:`i-1/2`.

    Reference: Chamarthi & Frankel (2021).
    """
    return _bvd_hocus6_select(u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3)


def bvd_e6_mp5_fv(u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3):
    r"""BVD explicit 6th-order + MP5 (alpha=4) (FV) at :math:`i+1/2`, :math:`i-1/2`.

    Legacy variant of the BVD-CU scheme.
    """
    return _bvd_e6_mp5_select(u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3)

# ---------------------------------------------------------------------------
# JIT compilation (pynalgo convention)
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    reveal_locals()  # noqa: F821
else:
    _c5_rhs_L = JIT(_c5_rhs_L)
    _c5_rhs_R = JIT(_c5_rhs_R)
    _solve_c5_local_L = JIT(_solve_c5_local_L)
    _solve_c5_local_R = JIT(_solve_c5_local_R)
    _mp5_L = JIT(_mp5_L)
    _mp5_R = JIT(_mp5_R)
    _mp5_limiter = JIT(_mp5_limiter)
    _bvd_hocus6_select = JIT(_bvd_hocus6_select)
    _central6_fv_L = JIT(_central6_fv_L)
    _central6_fv_R = JIT(_central6_fv_R)
    _mp5_legacy_L = JIT(_mp5_legacy_L)
    _mp5_legacy_R = JIT(_mp5_legacy_R)
    _bvd_e6_mp5_select = JIT(_bvd_e6_mp5_select)
    bvd_cu_fv = JIT(bvd_cu_fv)
    bvd_e6_mp5_fv = JIT(bvd_e6_mp5_fv)
#
# :D
#
