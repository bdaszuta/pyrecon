"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: ENO-MR -- Parameter-Free Multi-Resolution ENO
"""
# ---------------------------------------------------------------------------
# FV stencil coefficients at x_{i+1/2} (left-biased)
# ---------------------------------------------------------------------------

from pyrecon._jit_utils import JIT, TYPE_CHECKING
from pyrecon.utils import minmod

# 2-point stencils
_COEFF_S_0_1 = (0.5, 0.5)                        # [i, i+1]

# 3-point stencils
_COEFF_S_M1_1 = (-1.0/6.0, 5.0/6.0, 1.0/3.0)    # [i-1, i, i+1]
_COEFF_S_M2_0 = (1.0/3.0, -7.0/6.0, 11.0/6.0)    # [i-2, i-1, i]
_COEFF_S_0_2  = (1.0/3.0, 5.0/6.0, -1.0/6.0)     # [i, i+1, i+2]

# 4-point stencils
_COEFF_S_M2_1 = (1.0/12.0, -5.0/12.0, 13.0/12.0, 1.0/4.0)   # [i-2,i-1,i,i+1]
_COEFF_S_M1_2 = (-1.0/12.0, 7.0/12.0, 7.0/12.0, -1.0/12.0)   # [i-1,i,i+1,i+2]

# 5-point stencils
# [i-2,i-1,i,i+1,i+2]
_COEFF_S_M2_2 = (1.0/30.0, -13.0/60.0, 47.0/60.0,
                 9.0/20.0, -1.0/20.0)
# [i-3,i-2,i-1,i,i+1]
_COEFF_S_M3_1 = (-1.0/20.0, 17.0/60.0, -43.0/60.0,
                 77.0/60.0, 1.0/5.0)
# [i-1,i,i+1,i+2,i+3]
_COEFF_S_M1_3 = (-1.0/20.0, 9.0/20.0, 47.0/60.0,
                 -13.0/60.0, 1.0/30.0)

# 6-point stencils
# [i-3,i-2,i-1,i,i+1,i+2]
_COEFF_S_M3_2 = (-1.0/60.0, 7.0/60.0, -23.0/60.0,
                 19.0/20.0, 11.0/30.0, -1.0/30.0)
# [i-2,i-1,i,i+1,i+2,i+3]
_COEFF_S_M2_3 = (1.0/60.0, -2.0/15.0, 37.0/60.0,
                 37.0/60.0, -2.0/15.0, 1.0/60.0)

# 7-point stencil (global)
# [i-3,i-2,i-1,i,i+1,i+2,i+3]
_COEFF_S_M3_3 = (-1.0/140.0, 5.0/84.0, -101.0/420.0,
                 319.0/420.0, 107.0/210.0, -19.0/210.0, 1.0/105.0)


# ---------------------------------------------------------------------------
# IS coefficients: b_l where IS = |sum(b_l * u_{i+l})|
# ---------------------------------------------------------------------------

def _is3(u_im2, u_im1, u_i):
    """IS for 3-point stencil [i-2,i-1,i]."""
    return abs(u_im2 - 2.0 * u_im1 + u_i)


def _is3_mid(u_im1, u_i, u_ip1):
    """IS for 3-point stencil [i-1,i,i+1]."""
    return abs(u_im1 - 2.0 * u_i + u_ip1)


def _is3_r(u_i, u_ip1, u_ip2):
    """IS for 3-point stencil [i,i+1,i+2]."""
    return abs(u_i - 2.0 * u_ip1 + u_ip2)


def _is4_left(u_im2, u_im1, u_i, u_ip1):
    """IS for 4-point stencil [i-2,i-1,i,i+1]."""
    return abs(u_im2 - 3.0 * u_im1 + 3.0 * u_i - u_ip1)


def _is4_right(u_im1, u_i, u_ip1, u_ip2):
    """IS for 4-point stencil [i-1,i,i+1,i+2]."""
    return abs(u_im1 - 3.0 * u_i + 3.0 * u_ip1 - u_ip2)


def _is5(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """IS for 5-point stencil."""
    return abs(u_im2 - 4.0 * u_im1 + 6.0 * u_i - 4.0 * u_ip1 + u_ip2)


def _is5_left(u_im3, u_im2, u_im1, u_i, u_ip1):
    """IS for 5-point stencil [i-3,i-2,i-1,i,i+1] (4th derivative)."""
    return abs(u_im3 - 4.0 * u_im2 + 6.0 * u_im1 - 4.0 * u_i + u_ip1)


def _is5_right(u_im1, u_i, u_ip1, u_ip2, u_ip3):
    """IS for 5-point stencil [i-1,i,i+1,i+2,i+3] (4th derivative)."""
    return abs(u_im1 - 4.0 * u_i + 6.0 * u_ip1 - 4.0 * u_ip2 + u_ip3)


def _is6_left(u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2):
    """IS for 6-point stencil [i-3,i-2,i-1,i,i+1,i+2] (5th derivative)."""
    return abs(u_im3 - 5.0 * u_im2 + 10.0 * u_im1
               - 10.0 * u_i + 5.0 * u_ip1 - u_ip2)


def _is6_right(u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3):
    """IS for 6-point stencil [i-2,i-1,i,i+1,i+2,i+3] (5th derivative)."""
    return abs(u_im2 - 5.0 * u_im1 + 10.0 * u_i
               - 10.0 * u_ip1 + 5.0 * u_ip2 - u_ip3)


def _is7(u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3):
    """IS for 7-point stencil (6th derivative)."""
    return abs(u_im3 - 6.0 * u_im2 + 15.0 * u_im1
               - 20.0 * u_i + 15.0 * u_ip1 - 6.0 * u_ip2 + u_ip3)


def _is0_baseline(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """Baseline smoothness IS_0.

    IS_0 = min( max(|d^1|, |d^2|)_left, max(|d^1|, |d^2|)_right )
    where left = {i-1,i}, right = {i,i+1}.

    Reference: Hua & Shen (2023), Eqs. 25-26.
    """
    # Left side: first and second differences at cells i-1, i
    d1_left = abs(u_i - u_im1)
    d2_left = abs(u_i - 2.0 * u_im1 + u_im2)
    left_max = max(d1_left, d2_left)

    # Right side: first and second differences at cells i, i+1
    d1_right = abs(u_ip1 - u_i)
    d2_right = abs(u_i - 2.0 * u_ip1 + u_ip2)
    right_max = max(d1_right, d2_right)

    return min(left_max, right_max)

# ---------------------------------------------------------------------------
# Per-face left-biased reconstruction for one face at i+1/2
# ---------------------------------------------------------------------------

def _eno_mr3_face(u_im2, u_im1, u_i, u_ip1, u_ip2):
    r"""ENO-MR 3rd-order, left-biased reconstruction at :math:`i+1/2`.

    Candidates: :math:`S_{i-1}^{i+1}` (3-pt), then fallback to minmod.
    """
    is0 = _is0_baseline(u_im2, u_im1, u_i, u_ip1, u_ip2)

    # Global stencil S_{i-1}^{i+1}
    is_global = _is3_mid(u_im1, u_i, u_ip1)
    if is_global < is0:
        c = _COEFF_S_M1_1
        return c[0] * u_im1 + c[1] * u_i + c[2] * u_ip1

    # Fallback: minmod
    return u_i + 0.5 * minmod(u_ip1 - u_i, u_i - u_im1)


def _eno_mr5_face(u_im2, u_im1, u_i, u_ip1, u_ip2):
    r"""ENO-MR 5th-order, left-biased reconstruction at :math:`i+1/2`.

    Candidates (decreasing order):
      :math:`S_{i-2}^{i+2}` (5-pt) -> 4-point stencils -> 3-point stencil -> fallback
    """
    is0 = _is0_baseline(u_im2, u_im1, u_i, u_ip1, u_ip2)

    # 1. Global 5-point stencil S_{i-2}^{i+2}
    if _is5(u_im2, u_im1, u_i, u_ip1, u_ip2) < is0:
        c = _COEFF_S_M2_2
        return (c[0] * u_im2 + c[1] * u_im1 + c[2] * u_i
                + c[3] * u_ip1 + c[4] * u_ip2)

    # 2. 4-point stencil S_{i-2}^{i+1}
    if _is4_left(u_im2, u_im1, u_i, u_ip1) < is0:
        c = _COEFF_S_M2_1
        return c[0] * u_im2 + c[1] * u_im1 + c[2] * u_i + c[3] * u_ip1

    # 3. 4-point stencil S_{i-1}^{i+2}
    if _is4_right(u_im1, u_i, u_ip1, u_ip2) < is0:
        c = _COEFF_S_M1_2
        return c[0] * u_im1 + c[1] * u_i + c[2] * u_ip1 + c[3] * u_ip2

    # 4. 3-point stencil S_{i-1}^{i+1}
    if _is3_mid(u_im1, u_i, u_ip1) < is0:
        c = _COEFF_S_M1_1
        return c[0] * u_im1 + c[1] * u_i + c[2] * u_ip1

    # Fallback: minmod
    return u_i + 0.5 * minmod(u_ip1 - u_i, u_i - u_im1)


def _eno_mr7_face(u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3):
    r"""ENO-MR 7th-order, left-biased reconstruction at :math:`i+1/2`.

    Input: u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3.

    Candidates (decreasing order):
      :math:`S_{i-3}^{i+3}` (7-pt) -> 6-pt -> 5-pt -> 4-pt -> 3-pt -> fallback
    """
    # Baseline IS_0 (Hua Shen 2023, Eq. 25-26) over 5-point window
    is0 = _is0_baseline(u_im2, u_im1, u_i, u_ip1, u_ip2)

    # 1. Global 7-point stencil S_{i-3}^{i+3}
    if _is7(u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3) < is0:
        c = _COEFF_S_M3_3
        return (c[0] * u_im3 + c[1] * u_im2 + c[2] * u_im1
                + c[3] * u_i + c[4] * u_ip1 + c[5] * u_ip2 + c[6] * u_ip3)

    # 2. 6-point stencil S_{i-3}^{i+2}
    if _is6_left(u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2) < is0:
        c = _COEFF_S_M3_2
        return (c[0] * u_im3 + c[1] * u_im2 + c[2] * u_im1
                + c[3] * u_i + c[4] * u_ip1 + c[5] * u_ip2)

    # 3. 6-point stencil S_{i-2}^{i+3}
    if _is6_right(u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3) < is0:
        c = _COEFF_S_M2_3
        return (c[0] * u_im2 + c[1] * u_im1 + c[2] * u_i
                + c[3] * u_ip1 + c[4] * u_ip2 + c[5] * u_ip3)

    # 4. 5-point stencil S_{i-2}^{i+2}
    if _is5(u_im2, u_im1, u_i, u_ip1, u_ip2) < is0:
        c = _COEFF_S_M2_2
        return (c[0] * u_im2 + c[1] * u_im1 + c[2] * u_i
                + c[3] * u_ip1 + c[4] * u_ip2)

    # 5. 4-point stencil S_{i-2}^{i+1}
    if _is4_left(u_im2, u_im1, u_i, u_ip1) < is0:
        c = _COEFF_S_M2_1
        return c[0] * u_im2 + c[1] * u_im1 + c[2] * u_i + c[3] * u_ip1

    # 6. 4-point stencil S_{i-1}^{i+2}
    if _is4_right(u_im1, u_i, u_ip1, u_ip2) < is0:
        c = _COEFF_S_M1_2
        return c[0] * u_im1 + c[1] * u_i + c[2] * u_ip1 + c[3] * u_ip2

    # 7. 3-point stencil S_{i-1}^{i+1}
    if _is3_mid(u_im1, u_i, u_ip1) < is0:
        c = _COEFF_S_M1_1
        return c[0] * u_im1 + c[1] * u_i + c[2] * u_ip1

    # Fallback: minmod
    return u_i + 0.5 * minmod(u_ip1 - u_i, u_i - u_im1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def eno_mr3_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    r"""ENO-MR 3rd-order reconstruction at :math:`i+1/2`.
    """
    uL = _eno_mr3_face(u_im2, u_im1, u_i, u_ip1, u_ip2)
    uR = _eno_mr3_face(u_ip2, u_ip1, u_i, u_im1, u_im2)
    return uL, uR


def eno_mr5_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    r"""ENO-MR 5th-order reconstruction at :math:`i+1/2`.
    """
    uL = _eno_mr5_face(u_im2, u_im1, u_i, u_ip1, u_ip2)
    uR = _eno_mr5_face(u_ip2, u_ip1, u_i, u_im1, u_im2)
    return uL, uR


def eno_mr7_fv(u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3):
    r"""ENO-MR 7th-order reconstruction at :math:`i+1/2`.
    """
    uL = _eno_mr7_face(u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3)
    uR = _eno_mr7_face(u_ip3, u_ip2, u_ip1, u_i, u_im1, u_im2, u_im3)
    return uL, uR
# ---------------------------------------------------------------------------
# JIT compilation (pynalgo convention)
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    reveal_locals()  # noqa: F821
else:
    _is3 = JIT(_is3)
    _is3_mid = JIT(_is3_mid)
    _is3_r = JIT(_is3_r)
    _is4_left = JIT(_is4_left)
    _is4_right = JIT(_is4_right)
    _is5 = JIT(_is5)
    _is5_left = JIT(_is5_left)
    _is5_right = JIT(_is5_right)
    _is6_left = JIT(_is6_left)
    _is6_right = JIT(_is6_right)
    _is7 = JIT(_is7)
    _is0_baseline = JIT(_is0_baseline)
    _eno_mr3_face = JIT(_eno_mr3_face)
    _eno_mr5_face = JIT(_eno_mr5_face)
    _eno_mr7_face = JIT(_eno_mr7_face)
    eno_mr3_fv = JIT(eno_mr3_fv)
    eno_mr5_fv = JIT(eno_mr5_fv)
    eno_mr7_fv = JIT(eno_mr7_fv)
#
# :D
#
