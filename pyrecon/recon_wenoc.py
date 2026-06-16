"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: WENO-C / WENO-ZC -- Combined WENO (Two-Layer Hierarchy)
"""
from pyrecon._jit_utils import JIT, TYPE_CHECKING

# ---------------------------------------------------------------------------
# FV stencil coefficients at x_{i+1/2} (left-biased)
# ---------------------------------------------------------------------------

# WENO5-C sub-stencils (4-point, order 4):
_COEFF_S_M2_1 = (1.0/12.0, -5.0/12.0, 13.0/12.0, 1.0/4.0)     # S_{i-2}^{i+1}
_COEFF_S_M1_2 = (-1.0/12.0, 7.0/12.0, 7.0/12.0, -1.0/12.0)    # S_{i-1}^{i+2}

# WENO7-C sub-stencils (5-point, order 5):
_COEFF_S_M3_1 = (-1.0/20.0, 17.0/60.0, -43.0/60.0,              # S_{i-3}^{i+1}
                 77.0/60.0, 1.0/5.0)
_COEFF_S_M2_2 = (1.0/30.0, -13.0/60.0, 47.0/60.0,               # S_{i-2}^{i+2}
                 9.0/20.0, -1.0/20.0)
_COEFF_S_M1_3 = (-1.0/20.0, 9.0/20.0, 47.0/60.0,                # S_{i-1}^{i+3}
                 -13.0/60.0, 1.0/30.0)

# WENO7-C sub-stencils (6-point, order 6):
_COEFF_S_M3_2 = (-1.0/60.0, 7.0/60.0, -23.0/60.0,               # S_{i-3}^{i+2}
                 19.0/20.0, 11.0/30.0, -1.0/30.0)
_COEFF_S_M2_3 = (1.0/60.0, -2.0/15.0, 37.0/60.0,                # S_{i-2}^{i+3}
                 37.0/60.0, -2.0/15.0, 1.0/60.0)

# WENO7-C global (7-point):
_COEFF_S_M3_3 = (-1.0/140.0, 5.0/84.0, -101.0/420.0,            # S_{i-3}^{i+3}
                 319.0/420.0, 107.0/210.0, -19.0/210.0, 1.0/105.0)


# ---------------------------------------------------------------------------
# Generic Jiang-Shu smoothness indicator for any stencil
# ---------------------------------------------------------------------------

def _fit_poly_and_is(stencil_vals, stencil_offsets):
    r"""Fit polynomial to cell averages and compute Jiang-Shu smoothness.

    Fits a polynomial :math:`p(x) = \\sum a_k x^k` to cell-averaged values
    by solving the constraint system :math:`A a = u`, then computes the
    Jiang-Shu smoothness indicator

    .. math::

        IS = \\sum_{l=1}^{N-1} \\int_{-1/2}^{1/2} \\left(\\frac{d^l p}{dx^l}\\right)^2 dx

    where :math:`\\frac{d^l p}{dx^l} = \\sum_{k=l}^{N-1} a_k \\frac{k!}{(k-l)!} x^{k-l}`
    and :math:`\\int_{-1/2}^{1/2} x^m dx = 0` if :math:`m` odd,
    :math:`1/((m+1)\\,2^m)` if :math:`m` even.

    Parameters
    ----------
    stencil_vals : sequence
        Cell-averaged values in stencil order.
    stencil_offsets : sequence
        Cell index offsets (e.g., [-2, -1, 0, 1, 2]).

    Returns
    -------
    float
        Smoothness indicator (scalar, non-negative).
    """
    N = len(stencil_vals)
    if N <= 1:
        return 0.0

    # Build constraint matrix A: avg of x^k over each cell
    A = [[0.0] * N for _ in range(N)]
    for row, j in enumerate(stencil_offsets):
        for k in range(N):
            A[row][k] = ((j + 0.5) ** (k + 1) - (j - 0.5) ** (k + 1)) / (k + 1)

    # Solve A * a = stencil_vals for polynomial coefficients a_k
    # Use Gaussian elimination with partial pivoting
    M = [A[i][:] + [stencil_vals[i]] for i in range(N)]

    for col in range(N):
        # Partial pivot: find row with largest abs value in this column
        max_row = col
        max_val = abs(M[col][col])
        for r in range(col + 1, N):
            v = abs(M[r][col])
            if v > max_val:
                max_val = v
                max_row = r
        if max_row != col:
            M[col], M[max_row] = M[max_row], M[col]

        pivot = M[col][col]
        if abs(pivot) < 1e-14:
            continue

        for j in range(col, N + 1):
            M[col][j] /= pivot

        for row in range(N):
            if row != col and abs(M[row][col]) > 1e-14:
                factor = M[row][col]
                for j in range(col, N + 1):
                    M[row][j] -= factor * M[col][j]

    a = [M[i][N] for i in range(N)]

    # Compute IS via analytical integrals of polynomial derivatives
    # (see docstring for the full formula)

    is_val = 0.0
    for deg in range(1, N):
        # Build derivative coefficients for this deg
        deriv = [0.0] * N
        for k in range(deg, N):
            # k! / (k-deg)!
            fact = 1.0
            for m in range(k - deg + 1, k + 1):
                fact *= m
            deriv[k - deg] = a[k] * fact
        # Compute integral of deriv^2
        for i in range(N - deg):
            for j in range(N - deg):
                pow_sum = i + j
                if pow_sum % 2 == 0:
                    is_val += (deriv[i] * deriv[j]
                               / ((pow_sum + 1) * (2 ** pow_sum)))

    return abs(is_val)  # should be non-negative


# Precomputed stencil offsets for all sizes
_OFFSETS_3 = (
    (-2, -1, 0),
    (-1, 0, 1),
    (0, 1, 2),
)
_OFFSETS_4 = (
    (-2, -1, 0, 1),
    (-1, 0, 1, 2),
)
_OFFSETS_4_WENO7 = (
    (-3, -2, -1, 0),
    (-2, -1, 0, 1),
    (-1, 0, 1, 2),
    (0, 1, 2, 3),
)
_OFFSETS_5 = (
    (-3, -2, -1, 0, 1),
    (-2, -1, 0, 1, 2),
    (-1, 0, 1, 2, 3),
)
_OFFSETS_6 = (
    (-3, -2, -1, 0, 1, 2),
    (-2, -1, 0, 1, 2, 3),
)


# ---------------------------------------------------------------------------
# WENO5-C single-face helper
# ---------------------------------------------------------------------------

def _wenoc5_face(u_im2, u_im1, u_i, u_ip1, u_ip2, use_z_weights, p_total):
    """WENO5-C reconstruction at i+1/2 (left-biased).

    use_z_weights: True = WENO-ZC (eps=1e-40), False = WENO-C (JS weights, eps=1e-12).
    Note: the Z-weights epsilon (1e-40) is deliberately much smaller than the
    standard WENO convention (1e-6 to 1e-12) to avoid biasing smoothness
    indicators toward zero in near-smooth regions.
    p_total: exponent for total ideal weights d_tilde_s = (1+s)^p_total.
    """
    # Layer 1, s=0: 3 sub-stencils of order 3
    f_3 = [
        (2.0 * u_im2 - 7.0 * u_im1 + 11.0 * u_i) / 6.0,
        (-u_im1 + 5.0 * u_i + 2.0 * u_ip1) / 6.0,
        (2.0 * u_i + 5.0 * u_ip1 - u_ip2) / 6.0,
    ]

    if use_z_weights:
        eps = 1e-40
    else:
        eps = 1e-12

    beta_3 = [_fit_poly_and_is([u_im2, u_im1, u_i], _OFFSETS_3[0]),
              _fit_poly_and_is([u_im1, u_i, u_ip1], _OFFSETS_3[1]),
              _fit_poly_and_is([u_i, u_ip1, u_ip2], _OFFSETS_3[2])]

    if use_z_weights:
        tau = abs(beta_3[0] - beta_3[2])  # k=3 is odd
        d_3w = (1.0/10.0, 6.0/10.0, 3.0/10.0)
        alpha_3 = [d_3w[r] * (1.0 + (tau / (beta_3[r] + eps)) ** 2)
                   for r in range(3)]
    else:
        d_3w = (1.0/10.0, 6.0/10.0, 3.0/10.0)
        alpha_3 = [d_3w[r] / ((beta_3[r] + eps) ** 2) for r in range(3)]

    sum_a3 = alpha_3[0] + alpha_3[1] + alpha_3[2]
    w_3 = [a / sum_a3 for a in alpha_3]
    f_tilde_3 = w_3[0] * f_3[0] + w_3[1] * f_3[1] + w_3[2] * f_3[2]

    # Total smoothness for s=0
    beta_tilde_3 = (w_3[0] * beta_3[0] + w_3[1] * beta_3[1]
                    + w_3[2] * beta_3[2])

    # Layer 1, s=1: 2 sub-stencils of order 4
    f_4 = [
        (_COEFF_S_M2_1[0] * u_im2 + _COEFF_S_M2_1[1] * u_im1 +
         _COEFF_S_M2_1[2] * u_i + _COEFF_S_M2_1[3] * u_ip1),
        (_COEFF_S_M1_2[0] * u_im1 + _COEFF_S_M1_2[1] * u_i +
         _COEFF_S_M1_2[2] * u_ip1 + _COEFF_S_M1_2[3] * u_ip2),
    ]

    beta_4 = [_fit_poly_and_is([u_im2, u_im1, u_i, u_ip1], _OFFSETS_4[0]),
              _fit_poly_and_is([u_im1, u_i, u_ip1, u_ip2], _OFFSETS_4[1])]

    if use_z_weights:
        d_4 = (2.0/5.0, 3.0/5.0)
        alpha_4 = [d_4[r] * (1.0 + (tau / (beta_4[r] + eps)) ** 2)
                   for r in range(2)]
    else:
        d_4 = (2.0/5.0, 3.0/5.0)
        alpha_4 = [d_4[r] / ((beta_4[r] + eps) ** 2) for r in range(2)]

    sum_a4 = alpha_4[0] + alpha_4[1]
    w_4 = [a / sum_a4 for a in alpha_4]
    f_tilde_4 = w_4[0] * f_4[0] + w_4[1] * f_4[1]

    # Total smoothness for s=1
    beta_tilde_4 = w_4[0] * beta_4[0] + w_4[1] * beta_4[1]

    # Layer 2: combine f_tilde_3 and f_tilde_4
    d_tilde = (1.0 ** p_total, 2.0 ** p_total)
    alpha_tilde = [d_tilde[0] / ((beta_tilde_3 + eps) ** 2),
                   d_tilde[1] / ((beta_tilde_4 + eps) ** 2)]
    sum_at = alpha_tilde[0] + alpha_tilde[1]
    gamma_3 = alpha_tilde[0] / sum_at
    gamma_4 = alpha_tilde[1] / sum_at

    return gamma_3 * f_tilde_3 + gamma_4 * f_tilde_4


# ---------------------------------------------------------------------------
# WENO7-C single-face helper
# ---------------------------------------------------------------------------

def _wenoc7_face(u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3,
                 use_z_weights, p_total):
    """WENO7-C reconstruction at i+1/2 (left-biased).

    use_z_weights: True = WENO-ZC (eps=1e-40), False = WENO-C (JS weights, eps=1e-12).
    Note: the Z-weights epsilon (1e-40) is deliberately much smaller than the
    standard WENO convention (1e-6 to 1e-12) to avoid biasing smoothness
    indicators toward zero in near-smooth regions.
    p_total: exponent for total ideal weights d_tilde_s = (1+s)^p_total.
    """
    if use_z_weights:
        eps = 1e-40
    else:
        eps = 1e-12

    # Layer 1, s=0: 4 sub-stencils of order 4
    f_4w = [
        (-3.0 * u_im3 + 13.0 * u_im2 - 23.0 * u_im1 + 25.0 * u_i) / 12.0,
        (u_im2 - 5.0 * u_im1 + 13.0 * u_i + 3.0 * u_ip1) / 12.0,
        (-u_im1 + 7.0 * u_i + 7.0 * u_ip1 - u_ip2) / 12.0,
        (3.0 * u_i + 13.0 * u_ip1 - 5.0 * u_ip2 + u_ip3) / 12.0,
    ]

    beta_4w = [_fit_poly_and_is(
                   [u_im3, u_im2, u_im1, u_i],
                   _OFFSETS_4_WENO7[0]),
               _fit_poly_and_is(
                   [u_im2, u_im1, u_i, u_ip1],
                   _OFFSETS_4_WENO7[1]),
               _fit_poly_and_is(
                   [u_im1, u_i, u_ip1, u_ip2],
                   _OFFSETS_4_WENO7[2]),
               _fit_poly_and_is(
                   [u_i, u_ip1, u_ip2, u_ip3],
                   _OFFSETS_4_WENO7[3])]

    if use_z_weights:
        # 4-stencil tau (WENO-C variant for 4 sub-stencils)
        tau = abs(beta_4w[0] - beta_4w[1] - beta_4w[2] + beta_4w[3])
        d_4w = (1.0/35.0, 12.0/35.0, 18.0/35.0, 4.0/35.0)
        alpha_4w = [d_4w[r] * (1.0 + (tau / (beta_4w[r] + eps)) ** 2)
                    for r in range(4)]
    else:
        d_4w = (1.0/35.0, 12.0/35.0, 18.0/35.0, 4.0/35.0)
        alpha_4w = [d_4w[r] / ((beta_4w[r] + eps) ** 2) for r in range(4)]

    sum_a4w = alpha_4w[0] + alpha_4w[1] + alpha_4w[2] + alpha_4w[3]
    w_4w = [a / sum_a4w for a in alpha_4w]
    f_tilde_4 = (w_4w[0] * f_4w[0] + w_4w[1] * f_4w[1] +
                 w_4w[2] * f_4w[2] + w_4w[3] * f_4w[3])
    beta_tilde_4 = (w_4w[0] * beta_4w[0] + w_4w[1] * beta_4w[1] +
                    w_4w[2] * beta_4w[2] + w_4w[3] * beta_4w[3])

    # Layer 1, s=1: 3 sub-stencils of order 5
    f_5 = [
        (_COEFF_S_M3_1[0] * u_im3 + _COEFF_S_M3_1[1] * u_im2 +
         _COEFF_S_M3_1[2] * u_im1 + _COEFF_S_M3_1[3] * u_i +
         _COEFF_S_M3_1[4] * u_ip1),
        (_COEFF_S_M2_2[0] * u_im2 + _COEFF_S_M2_2[1] * u_im1 +
         _COEFF_S_M2_2[2] * u_i + _COEFF_S_M2_2[3] * u_ip1 +
         _COEFF_S_M2_2[4] * u_ip2),
        (_COEFF_S_M1_3[0] * u_im1 + _COEFF_S_M1_3[1] * u_i +
         _COEFF_S_M1_3[2] * u_ip1 + _COEFF_S_M1_3[3] * u_ip2 +
         _COEFF_S_M1_3[4] * u_ip3),
    ]

    beta_5 = [_fit_poly_and_is(
                  [u_im3, u_im2, u_im1, u_i, u_ip1],
                  _OFFSETS_5[0]),
              _fit_poly_and_is(
                  [u_im2, u_im1, u_i, u_ip1, u_ip2],
                  _OFFSETS_5[1]),
              _fit_poly_and_is(
                  [u_im1, u_i, u_ip1, u_ip2, u_ip3],
                  _OFFSETS_5[2])]

    if use_z_weights:
        d_5 = (1.0/7.0, 4.0/7.0, 2.0/7.0)
        alpha_5 = [d_5[r] * (1.0 + (tau / (beta_5[r] + eps)) ** 2)
                   for r in range(3)]
    else:
        d_5 = (1.0/7.0, 4.0/7.0, 2.0/7.0)
        alpha_5 = [d_5[r] / ((beta_5[r] + eps) ** 2) for r in range(3)]

    sum_a5 = alpha_5[0] + alpha_5[1] + alpha_5[2]
    w_5 = [a / sum_a5 for a in alpha_5]
    f_tilde_5 = (w_5[0] * f_5[0] + w_5[1] * f_5[1] +
                 w_5[2] * f_5[2])
    beta_tilde_5 = (w_5[0] * beta_5[0] + w_5[1] * beta_5[1] +
                    w_5[2] * beta_5[2])

    # Layer 1, s=2: 2 sub-stencils of order 6
    f_6 = [
        (_COEFF_S_M3_2[0] * u_im3 + _COEFF_S_M3_2[1] * u_im2 +
         _COEFF_S_M3_2[2] * u_im1 + _COEFF_S_M3_2[3] * u_i +
         _COEFF_S_M3_2[4] * u_ip1 + _COEFF_S_M3_2[5] * u_ip2),
        (_COEFF_S_M2_3[0] * u_im2 + _COEFF_S_M2_3[1] * u_im1 +
         _COEFF_S_M2_3[2] * u_i + _COEFF_S_M2_3[3] * u_ip1 +
         _COEFF_S_M2_3[4] * u_ip2 + _COEFF_S_M2_3[5] * u_ip3),
    ]

    beta_6 = [_fit_poly_and_is(
                  [u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2],
                  _OFFSETS_6[0]),
              _fit_poly_and_is(
                  [u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3],
                  _OFFSETS_6[1])]

    if use_z_weights:
        d_6 = (3.0/7.0, 4.0/7.0)
        alpha_6 = [d_6[r] * (1.0 + (tau / (beta_6[r] + eps)) ** 2)
                   for r in range(2)]
    else:
        d_6 = (3.0/7.0, 4.0/7.0)
        alpha_6 = [d_6[r] / ((beta_6[r] + eps) ** 2) for r in range(2)]

    sum_a6 = alpha_6[0] + alpha_6[1]
    w_6 = [a / sum_a6 for a in alpha_6]
    f_tilde_6 = w_6[0] * f_6[0] + w_6[1] * f_6[1]
    beta_tilde_6 = w_6[0] * beta_6[0] + w_6[1] * beta_6[1]

    # Layer 2: combine 3 intermediate fluxes
    d_tilde = (1.0 ** p_total, 2.0 ** p_total, 3.0 ** p_total)
    alpha_tilde = [d_tilde[0] / ((beta_tilde_4 + eps) ** 2),
                   d_tilde[1] / ((beta_tilde_5 + eps) ** 2),
                   d_tilde[2] / ((beta_tilde_6 + eps) ** 2)]
    sum_at = alpha_tilde[0] + alpha_tilde[1] + alpha_tilde[2]
    gamma_4 = alpha_tilde[0] / sum_at
    gamma_5 = alpha_tilde[1] / sum_at
    gamma_6 = alpha_tilde[2] / sum_at

    return gamma_4 * f_tilde_4 + gamma_5 * f_tilde_5 + gamma_6 * f_tilde_6


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def wenoc5_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """WENO5-C (JS weights, p=2)."""
    uL = _wenoc5_face(u_im2, u_im1, u_i, u_ip1, u_ip2, False, 2)
    uR = _wenoc5_face(u_ip2, u_ip1, u_i, u_im1, u_im2, False, 2)
    return uL, uR


def wenoc5z_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """WENO5-ZC (Z weights, p=2)."""
    uL = _wenoc5_face(u_im2, u_im1, u_i, u_ip1, u_ip2, True, 2)
    uR = _wenoc5_face(u_ip2, u_ip1, u_i, u_im1, u_im2, True, 2)
    return uL, uR


def wenoc7_fv(u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3):
    """WENO7-C (JS weights, p=2)."""
    uL = _wenoc7_face(u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3, False, 2)
    uR = _wenoc7_face(u_ip3, u_ip2, u_ip1, u_i, u_im1, u_im2, u_im3, False, 2)
    return uL, uR


def wenoc7z_fv(u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3):
    """WENO7-ZC (Z weights, p=2)."""
    uL = _wenoc7_face(u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3, True, 2)
    uR = _wenoc7_face(u_ip3, u_ip2, u_ip1, u_i, u_im1, u_im2, u_im3, True, 2)
    return uL, uR

# ---------------------------------------------------------------------------
# JIT compilation (pynalgo convention)
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    reveal_locals()  # noqa: F821
else:
    _fit_poly_and_is = JIT(_fit_poly_and_is)
    _wenoc5_face = JIT(_wenoc5_face)
    _wenoc7_face = JIT(_wenoc7_face)
    wenoc5_fv = JIT(wenoc5_fv)
    wenoc5z_fv = JIT(wenoc5z_fv)
    wenoc7_fv = JIT(wenoc7_fv)
    wenoc7z_fv = JIT(wenoc7z_fv)
#
# :D
#
