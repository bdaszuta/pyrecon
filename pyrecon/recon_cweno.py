"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: CWENO (Central WENO) reconstruction
"""
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

from pyrecon._jit_utils import JIT, TYPE_CHECKING
_EPSL = 1e-40
_P = 2

# FV stencil denominators
_K_1_2 = 0.5
_K_1_6 = 1.0 / 6.0
_K_1_60 = 1.0 / 60.0

# CWENO linear weights: optimal polynomial = 0.5, sub-stencils share 0.5
_DOPT = 0.5
_DK = 1.0 / 6.0

# Central WENO linear weights
_CW_OPT = 0.75
_CW_SIDE = 0.125


# ===========================================================================
# CWENO3-Z (WENO-Z-like variant, kept as cweno_z3_fv)
# ===========================================================================

def _cweno3_sub_L(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """FV 3-pt sub-stencil polynomials at left face."""
    u0 = _K_1_6 * (2.0 * u_im2 - 7.0 * u_im1 + 11.0 * u_i)
    u1 = _K_1_6 * (-u_im1 + 5.0 * u_i + 2.0 * u_ip1)
    u2 = _K_1_6 * (2.0 * u_i + 5.0 * u_ip1 - u_ip2)
    return u0, u1, u2


def _cweno3_opt_L(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """FV 5-pt optimal polynomial at left face."""
    return _K_1_60 * (2.0 * u_im2 - 13.0 * u_im1 + 47.0 * u_i +
                      27.0 * u_ip1 - 3.0 * u_ip2)


def _cweno3_sub_R(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """FV 3-pt sub-stencil polynomials at right face (reversed)."""
    u0 = _K_1_6 * (2.0 * u_ip2 - 7.0 * u_ip1 + 11.0 * u_i)
    u1 = _K_1_6 * (-u_ip1 + 5.0 * u_i + 2.0 * u_im1)
    u2 = _K_1_6 * (2.0 * u_i + 5.0 * u_im1 - u_im2)
    return u0, u1, u2


def _cweno3_opt_R(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """FV 5-pt optimal polynomial at right face."""
    return _K_1_60 * (-3.0 * u_im2 + 27.0 * u_im1 + 47.0 * u_i -
                      13.0 * u_ip1 + 2.0 * u_ip2)


def _cweno3_smoothness(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """Squared undivided second differences."""
    b0 = (u_im2 - 2.0 * u_im1 + u_i) ** 2
    b1 = (u_im1 - 2.0 * u_i + u_ip1) ** 2
    b2 = (u_i - 2.0 * u_ip1 + u_ip2) ** 2
    return b0, b1, b2


def cweno_z3_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    r"""CWENO3-Z (WENO-Z-like variant) at :math:`i+1/2`."""
    # Left face
    bL0, bL1, bL2 = _cweno3_smoothness(u_im2, u_im1, u_i, u_ip1, u_ip2)
    tau_L = abs(bL0 - bL2)
    aL0, aL1, aL2 = (
        _DK / (_EPSL + bL0) ** _P,
        _DK / (_EPSL + bL1) ** _P,
        _DK / (_EPSL + bL2) ** _P,
    )
    aL_opt = _DOPT / (_EPSL + tau_L) ** _P
    sL = 1.0 / (aL0 + aL1 + aL2 + aL_opt)
    wL0, wL1, wL2, wL_opt = aL0 * sL, aL1 * sL, aL2 * sL, aL_opt * sL
    u_opt = _cweno3_opt_L(u_im2, u_im1, u_i, u_ip1, u_ip2)
    uk0, uk1, uk2 = _cweno3_sub_L(u_im2, u_im1, u_i, u_ip1, u_ip2)
    uL = wL_opt * u_opt + wL0 * uk0 + wL1 * uk1 + wL2 * uk2

    # Right face
    bR0, bR1, bR2 = _cweno3_smoothness(u_ip2, u_ip1, u_i, u_im1, u_im2)
    tau_R = abs(bR0 - bR2)
    aR0, aR1, aR2 = (
        _DK / (_EPSL + bR0) ** _P,
        _DK / (_EPSL + bR1) ** _P,
        _DK / (_EPSL + bR2) ** _P,
    )
    aR_opt = _DOPT / (_EPSL + tau_R) ** _P
    sR = 1.0 / (aR0 + aR1 + aR2 + aR_opt)
    wR0, wR1, wR2, wR_opt = aR0 * sR, aR1 * sR, aR2 * sR, aR_opt * sR
    u_opt_R = _cweno3_opt_R(u_im2, u_im1, u_i, u_ip1, u_ip2)
    ukR0, ukR1, ukR2 = _cweno3_sub_R(u_im2, u_im1, u_i, u_ip1, u_ip2)
    uR = wR_opt * u_opt_R + wR0 * ukR0 + wR1 * ukR1 + wR2 * ukR2

    return uL, uR


# ===========================================================================
# CWENO5-Z (WENO-Z-like variant, kept as cweno_z5_fv)
# ===========================================================================
# 5x5 Jiang-Shu smoothness matrices for 5-point quartic sub-stencils.
# Derived from u^T M_k u where M_k = R_k^T @ Q4_JS @ R_k,
# R_k maps cell averages to monomial coefficients on [-1/2, 1/2].
#
# Sub-stencil 0: offsets [-3, -2, -1, 0, 1]
_JS5_SUB0 = (
    (1727/1260,  -60871/10080,  33071/3360,  -70237/10080,  18079/10080),
    (-60871/10080, 138563/5040,  -3229/70,    168509/5040,  -88297/10080),
    (33071/3360,  -3229/70,      135431/1680,  -25499/420,    55051/3360),
    (-70237/10080, 168509/5040,  -25499/420,   242723/5040,  -140251/10080),
    (18079/10080, -88297/10080,   55051/3360,  -140251/10080,  11329/2520),
)
# Sub-stencil 1: offsets [-2, -1, 0, 1, 2]
_JS5_SUB1 = (
    (1727/1260,  -51001/10080,  7547/1120,   -38947/10080,   8209/10080),
    (-51001/10080, 104963/5040, -24923/840,    89549/5040,  -38947/10080),
    (7547/1120,   -24923/840,    77051/1680,  -24923/840,     7547/1120),
    (-38947/10080,  89549/5040, -24923/840,   104963/5040,  -51001/10080),
    (8209/10080,  -38947/10080,  7547/1120,   -51001/10080,   1727/1260),
)
# Sub-stencil 2: offsets [-1, 0, 1, 2, 3]
_JS5_SUB2 = (
    (11329/2520, -140251/10080,  55051/3360,  -88297/10080,   18079/10080),
    (-140251/10080, 242723/5040, -25499/420,   168509/5040,  -70237/10080),
    (55051/3360,  -25499/420,    135431/1680,  -3229/70,      33071/3360),
    (-88297/10080, 168509/5040,  -3229/70,     138563/5040,  -60871/10080),
    (18079/10080, -70237/10080,  33071/3360,   -60871/10080,   1727/1260),
)


def _js5_sub0(u_im3, u_im2, u_im1, u_i, u_ip1):
    """5-point JS smoothness for sub-stencil 0 (offsets [-3,-2,-1,0,1])."""
    m = _JS5_SUB0
    return (m[0][0] * u_im3 * u_im3 + 2.0 * m[0][1] * u_im3 * u_im2 +
            2.0 * m[0][2] * u_im3 * u_im1 + 2.0 * m[0][3] * u_im3 * u_i +
            2.0 * m[0][4] * u_im3 * u_ip1 +
            m[1][1] * u_im2 * u_im2 + 2.0 * m[1][2] * u_im2 * u_im1 +
            2.0 * m[1][3] * u_im2 * u_i + 2.0 * m[1][4] * u_im2 * u_ip1 +
            m[2][2] * u_im1 * u_im1 + 2.0 * m[2][3] * u_im1 * u_i +
            2.0 * m[2][4] * u_im1 * u_ip1 +
            m[3][3] * u_i * u_i + 2.0 * m[3][4] * u_i * u_ip1 +
            m[4][4] * u_ip1 * u_ip1)


def _js5_sub1(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """5-point JS smoothness for sub-stencil 1 (offsets [-2,-1,0,1,2])."""
    m = _JS5_SUB1
    return (m[0][0] * u_im2 * u_im2 + 2.0 * m[0][1] * u_im2 * u_im1 +
            2.0 * m[0][2] * u_im2 * u_i + 2.0 * m[0][3] * u_im2 * u_ip1 +
            2.0 * m[0][4] * u_im2 * u_ip2 +
            m[1][1] * u_im1 * u_im1 + 2.0 * m[1][2] * u_im1 * u_i +
            2.0 * m[1][3] * u_im1 * u_ip1 + 2.0 * m[1][4] * u_im1 * u_ip2 +
            m[2][2] * u_i * u_i + 2.0 * m[2][3] * u_i * u_ip1 +
            2.0 * m[2][4] * u_i * u_ip2 +
            m[3][3] * u_ip1 * u_ip1 + 2.0 * m[3][4] * u_ip1 * u_ip2 +
            m[4][4] * u_ip2 * u_ip2)


def _js5_sub2(u_im1, u_i, u_ip1, u_ip2, u_ip3):
    """5-point JS smoothness for sub-stencil 2 (offsets [-1,0,1,2,3])."""
    m = _JS5_SUB2
    return (m[0][0] * u_im1 * u_im1 + 2.0 * m[0][1] * u_im1 * u_i +
            2.0 * m[0][2] * u_im1 * u_ip1 + 2.0 * m[0][3] * u_im1 * u_ip2 +
            2.0 * m[0][4] * u_im1 * u_ip3 +
            m[1][1] * u_i * u_i + 2.0 * m[1][2] * u_i * u_ip1 +
            2.0 * m[1][3] * u_i * u_ip2 + 2.0 * m[1][4] * u_i * u_ip3 +
            m[2][2] * u_ip1 * u_ip1 + 2.0 * m[2][3] * u_ip1 * u_ip2 +
            2.0 * m[2][4] * u_ip1 * u_ip3 +
            m[3][3] * u_ip2 * u_ip2 + 2.0 * m[3][4] * u_ip2 * u_ip3 +
            m[4][4] * u_ip3 * u_ip3)
def _cweno5_sub_L(u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3):
    """CWENO5 FV sub-stencil polynomials at left face. Returns (u0, u1, u2)."""
    u0 = _K_1_60 * (-3.0 * u_im3 + 17.0 * u_im2 - 43.0 * u_im1
                    + 77.0 * u_i + 12.0 * u_ip1)
    u1 = _K_1_60 * (2.0 * u_im2 - 13.0 * u_im1 + 47.0 * u_i
                    + 27.0 * u_ip1 - 3.0 * u_ip2)
    u2 = _K_1_60 * (-3.0 * u_im1 + 27.0 * u_i + 47.0 * u_ip1
                    - 13.0 * u_ip2 + 2.0 * u_ip3)
    return u0, u1, u2


def _cweno5_opt_L(u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3):
    """CWENO5 FV optimal polynomial at left face."""
    return (-1.0/140.0 * u_im3 + 5.0/84.0 * u_im2 - 101.0/420.0 * u_im1
            + 319.0/420.0 * u_i + 107.0/210.0 * u_ip1
            - 19.0/210.0 * u_ip2 + 1.0/105.0 * u_ip3)


def _cweno5_sub_R(u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3):
    """CWENO5 FV sub-stencil polynomials at right face (reversed). Returns (u0, u1, u2)."""
    u0 = _K_1_60 * (-3.0 * u_ip3 + 17.0 * u_ip2 - 43.0 * u_ip1
                    + 77.0 * u_i + 12.0 * u_im1)
    u1 = _K_1_60 * (2.0 * u_ip2 - 13.0 * u_ip1 + 47.0 * u_i
                    + 27.0 * u_im1 - 3.0 * u_im2)
    u2 = _K_1_60 * (-3.0 * u_ip1 + 27.0 * u_i + 47.0 * u_im1
                    - 13.0 * u_im2 + 2.0 * u_im3)
    return u0, u1, u2


def _cweno5_opt_R(u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3):
    """CWENO5 FV optimal polynomial at right face."""
    return (1.0/105.0 * u_im3 - 19.0/210.0 * u_im2 + 107.0/210.0 * u_im1
            + 319.0/420.0 * u_i - 101.0/420.0 * u_ip1
            + 5.0/84.0 * u_ip2 - 1.0/140.0 * u_ip3)


def _cweno5_smoothness(u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3):
    """5-point JS smoothness for the 3 quartic sub-stencils.

    b0: sub-stencil 0 on offsets [-3,-2,-1,0,1]
    b1: sub-stencil 1 on offsets [-2,-1,0,1,2]
    b2: sub-stencil 2 on offsets [-1,0,1,2,3]
    """
    b0 = _js5_sub0(u_im3, u_im2, u_im1, u_i, u_ip1)
    b1 = _js5_sub1(u_im2, u_im1, u_i, u_ip1, u_ip2)
    b2 = _js5_sub2(u_im1, u_i, u_ip1, u_ip2, u_ip3)
    return b0, b1, b2


def cweno_z5_fv(u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3):
    r"""CWENO5-Z (WENO-Z-like variant) at :math:`i+1/2`."""
    # Left face
    bL0, bL1, bL2 = _cweno5_smoothness(
        u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3,
    )
    tau_L = abs(bL0 - bL2)
    aL0, aL1, aL2 = (
        _DK / (_EPSL + bL0) ** _P,
        _DK / (_EPSL + bL1) ** _P,
        _DK / (_EPSL + bL2) ** _P,
    )
    aL_opt = _DOPT / (_EPSL + tau_L) ** _P
    sL = 1.0 / (aL0 + aL1 + aL2 + aL_opt)
    wL0, wL1, wL2, wL_opt = aL0 * sL, aL1 * sL, aL2 * sL, aL_opt * sL
    u_opt = _cweno5_opt_L(u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3)
    uk0, uk1, uk2 = _cweno5_sub_L(u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3)
    uL = wL_opt * u_opt + wL0 * uk0 + wL1 * uk1 + wL2 * uk2

    # Right face
    bR0, bR1, bR2 = _cweno5_smoothness(
        u_ip3, u_ip2, u_ip1, u_i, u_im1, u_im2, u_im3,
    )
    tau_R = abs(bR0 - bR2)
    aR0, aR1, aR2 = (
        _DK / (_EPSL + bR0) ** _P,
        _DK / (_EPSL + bR1) ** _P,
        _DK / (_EPSL + bR2) ** _P,
    )
    aR_opt = _DOPT / (_EPSL + tau_R) ** _P
    sR = 1.0 / (aR0 + aR1 + aR2 + aR_opt)
    wR0, wR1, wR2, wR_opt = aR0 * sR, aR1 * sR, aR2 * sR, aR_opt * sR
    u_opt_R = _cweno5_opt_R(u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3)
    ukR0, ukR1, ukR2 = _cweno5_sub_R(
        u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3,
    )
    uR = wR_opt * u_opt_R + wR0 * ukR0 + wR1 * ukR1 + wR2 * ukR2

    return uL, uR


# ===========================================================================
# Full Cravero 2017 CWENO -- Reconstruction matrices (polynomial coeffs
# from cell averages, local coordinate x in [-1/2, 1/2] for cell i)
# ===========================================================================
# Each matrix R satisfies: a = R @ u, where a = [a_0, ..., a_d] are the
# polynomial coefficients of degree d, and u are the cell averages.
#
# Matrix entry R[j, k] gives a_j from u_{i+offset[k]}.

# Quadratic on offsets [-2, -1, 0]  (left sub-stencil, CWENO5 paper)
_Q2_L = (
    (-0.04166666666666666, 0.08333333333333333, 0.9583333333333334),
    (0.5, -2.0, 1.5),
    (0.5, -1.0, 0.5),
)

# Quadratic on offsets [-1, 0, 1]  (center sub-stencil)
_Q2_C = (
    (-0.04166666666666666, 1.083333333333333, -0.04166666666666666),
    (-0.5, 0.0, 0.5),
    (0.5, -1.0, 0.5),
)

# Quadratic on offsets [0, 1, 2]  (right sub-stencil)
_Q2_R = (
    (0.9583333333333334, 0.08333333333333333, -0.04166666666666666),
    (-1.5, 2.0, -0.5),
    (0.5, -1.0, 0.5),
)

# Quartic on offsets [-2, -1, 0, 1, 2] (optimal, CWENO5 paper)
_Q4 = (
    (0.0046875, -0.06041666666666667, 1.111458333333333,
        -0.06041666666666667, 0.0046875),
    (0.1041666666666667, -0.7083333333333334, 0.0, 0.7083333333333334,
        -0.1041666666666667),
    (-0.0625, 0.75, -1.375, 0.75, -0.0625),
    (-0.08333333333333333, 0.1666666666666667, 0.0, -0.1666666666666667,
        0.08333333333333333),
    (0.04166666666666666, -0.1666666666666667, 0.25, -0.1666666666666667,
        0.04166666666666666),
)

# Cubic on offsets [-3, -2, -1, 0] (LL sub-stencil, CWENO7 paper)
_C3_LL = (
    (0.04166666666666666, -0.1666666666666667, 0.2083333333333333,
        0.9166666666666666),
    (-0.2916666666666667, 1.375, -2.875, 1.791666666666667),
    (-0.5, 2.0, -2.5, 1.0),
    (-0.1666666666666667, 0.5, -0.5, 0.1666666666666667),
)

# Cubic on offsets [-2, -1, 0, 1] (L sub-stencil, CWENO7 paper)
_C3_L = (
    (0.0, -0.04166666666666666, 1.083333333333333, -0.04166666666666666),
    (0.2083333333333333, -1.125, 0.625, 0.2916666666666667),
    (0.0, 0.5, -1.0, 0.5),
    (-0.1666666666666667, 0.5, -0.5, 0.1666666666666667),
)

# Cubic on offsets [-1, 0, 1, 2] (R sub-stencil, CWENO7 paper)
_C3_R = (
    (-0.04166666666666666, 1.083333333333333, -0.04166666666666666, 0.0),
    (-0.2916666666666667, -0.625, 1.125, -0.2083333333333333),
    (0.5, -1.0, 0.5, 0.0),
    (-0.1666666666666667, 0.5, -0.5, 0.1666666666666667),
)

# Cubic on offsets [0, 1, 2, 3] (RR sub-stencil, CWENO7 paper)
_C3_RR = (
    (0.9166666666666666, 0.2083333333333333, -0.1666666666666667,
        0.04166666666666666),
    (-1.791666666666667, 2.875, -1.375, 0.2916666666666667),
    (1.0, -2.5, 2.0, -0.5),
    (-0.1666666666666667, 0.5, -0.5, 0.1666666666666667),
)

# 6th-degree on offsets [-3, -2, -1, 0, 1, 2, 3] (optimal, CWENO7 paper)
_P6 = (
    (-0.0006975446428571429, 0.008872767857142857, -0.07087983630952381,
        1.125409226190476, -0.07087983630952381, 0.008872767857142857,
        -0.0006975446428571429),
    (-0.02248263888888889, 0.1940972222222222, -0.8207465277777778, 0.0,
        0.8207465277777778, -0.1940972222222222, 0.02248263888888889),
    (0.009635416666666667, -0.1203125, 0.89453125, -1.567708333333333,
        0.89453125, -0.1203125, 0.009635416666666667),
    (0.02430555555555556, -0.1805555555555556, 0.2881944444444444, 0.0,
        -0.2881944444444444, 0.1805555555555556, -0.02430555555555556),
    (-0.008680555555555556, 0.09375, -0.296875, 0.4236111111111111,
        -0.296875, 0.09375, -0.008680555555555556),
    (-0.004166666666666667, 0.01666666666666667, -0.02083333333333333, 0.0,
        0.02083333333333333, -0.01666666666666667, 0.004166666666666667),
    (0.001388888888888889, -0.008333333333333333, 0.02083333333333333,
        -0.02777777777777778, 0.02083333333333333, -0.008333333333333333,
        0.001388888888888889),
)

# ===========================================================================
# Jiang-Shu smoothness indicator matrices Q_d
# I[p] = a^T Q_d a  for degree-d polynomial with coefficients a = [a_0,...,a_d]
# ===========================================================================

# Q2: for quadratic polynomials (degree 2)
_Q2_JS = (
    (0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 4.333333333333333),  # 13/3
)

# Q3: for cubic polynomials (degree 3)
_Q3_JS = (
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.25),
    (0.0, 0.0, 4.333333333333333, 0.0),
    (0.0, 0.25, 0.0, 39.1125),  # 3129/80
)

# Q4: for quartic polynomials (degree 4)
_Q4_JS = (
    (0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.25, 0.0),
    (0.0, 0.0, 4.333333333333333, 0.0, 2.1),  # 21/10
    (0.0, 0.25, 0.0, 39.1125, 0.0),            # 3129/80
    (0.0, 0.0, 2.1, 0.0, 625.8357142857143),    # 87617/140
)

# Q6: for 6th-degree polynomials (degree 6)
_Q6_JS = (
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.25, 0.0, 0.0625, 0.0),
    (0.0, 0.0, 4.333333333333333, 0.0, 2.1, 0.0, 0.7767857142857143),
    (0.0, 0.25, 0.0, 39.1125, 0.0, 31.53348214285714, 0.0),
    (0.0, 0.0, 2.1, 0.0, 625.8357142857143, 0.0, 756.8139880952381),
    (0.0, 0.0625, 0.0, 31.53348214285714, 0.0, 15645.9037078373, 0.0),
    (0.0, 0.0, 0.7767857142857143, 0.0, 756.8139880952381, 0.0,
        563252.5366781655),
)

# ===========================================================================
# Helper functions for polynomial reconstruction and JS indicators
# ===========================================================================

def _poly_coeffs(R, u):
    """Compute polynomial coefficients a = R @ u for given reconstruction
    matrix R and cell averages u (as sequence).
    Returns list of coefficients [a_0, ..., a_d].
    """
    d = len(R) - 1
    a = [0.0] * (d + 1)
    for j in range(d + 1):
        s = 0.0
        row = R[j]
        for k, uk in enumerate(u):
            s += row[k] * uk
        a[j] = s
    return a


def _poly_eval(a, x):
    r"""Evaluate polynomial with coefficients a at point x.

    :math:`a = [a_0, a_1, \dots, a_d] \rightarrow a_0 + a_1 x + a_2 x^2 + \dots`
    """
    result = 0.0
    xp = 1.0
    for coeff in a:
        result += coeff * xp
        xp *= x
    return result


def _js_indicator(a, Q):
    r"""Compute Jiang-Shu smoothness indicator :math:`I[p] = a^T Q a`."""
    result = 0.0
    d = len(a) - 1
    for i in range(d + 1):
        row = Q[i]
        ai = a[i]
        if ai == 0.0:
            continue
        s = 0.0
        for j in range(d + 1):
            s += row[j] * a[j]
        result += ai * s
    return result


# CWENO weights and exponential parameters
_CWENO_T = 2  # exponent t in alpha_k = d_k / (epsilon + I[P_k])^t
_CWENO_EPS = 1e-40


# ===========================================================================
# CWENO3 (Cravero 2017 -- paper CWENO5)
#   5-cell stencil {i-2..i+2}, P_opt quartic, 3 parabolic sub-stencils
# ===========================================================================

# Linear weights for CWENO3 (paper CWENO5)
_CW3_D0 = 0.75   # d_0 = 3/4
_CW3_D1 = 0.0625 # d_1 = 1/16
_CW3_D2 = 0.125  # d_2 = 1/8
_CW3_D3 = 0.0625 # d_3 = 1/16

# Face evaluation point (cell face at i+1/2 corresponds to x = 0.5)
_FACE_X = 0.5


def cweno3_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    r"""CWENO3 at :math:`i+1/2`.

    Full CWENO framework with P_0 construction from P_opt and sub-stencils
    P_1, P_2, P_3, Jiang-Shu smoothness indicators, and nonlinear weighting.

    Parameters are cell averages: :math:`u_{i-2}, u_{i-1}, u_i, u_{i+1}, u_{i+2}`.
    Returns (:math:`u_{i+1/2}^-`, :math:`u_{i-1/2}^+`).

    Reference: Cravero et al. (2017).
    """
    # --- Left face (i+1/2) ---
    # Sub-stencil polynomial coefficients
    a1 = _poly_coeffs(_Q2_L, [u_im2, u_im1, u_i])
    a2 = _poly_coeffs(_Q2_C, [u_im1, u_i, u_ip1])
    a3 = _poly_coeffs(_Q2_R, [u_i, u_ip1, u_ip2])

    # Optimal polynomial coefficients
    a_opt = _poly_coeffs(_Q4, [u_im2, u_im1, u_i, u_ip1, u_ip2])

    # P_0 = (1/d_0)(P_opt - d_1*P_1 - d_2*P_2 - d_3*P_3)
    # Pad sub-stencil polynomials to degree 4 (same as P_opt/P_0)
    inv_d0 = 1.0 / _CW3_D0
    a0 = [0.0] * 5
    for j in range(3):
        a0[j] = inv_d0 * (a_opt[j] - _CW3_D1 * a1[j]
                          - _CW3_D2 * a2[j]
                          - _CW3_D3 * a3[j])
    for j in range(3, 5):
        a0[j] = inv_d0 * a_opt[j]

    # Jiang-Shu smoothness indicators
    IS0 = _js_indicator(a0, _Q4_JS)  # P_0 is quartic
    IS1 = _js_indicator(a1, _Q2_JS)  # P_1 is quadratic
    IS2 = _js_indicator(a2, _Q2_JS)  # P_2 is quadratic
    IS3 = _js_indicator(a3, _Q2_JS)  # P_3 is quadratic

    # Nonlinear weights: alpha_k = d_k / (epsilon + I[P_k])^t
    eps = _CWENO_EPS
    t = _CWENO_T
    alpha0 = _CW3_D0 / (eps + IS0) ** t
    alpha1 = _CW3_D1 / (eps + IS1) ** t
    alpha2 = _CW3_D2 / (eps + IS2) ** t
    alpha3 = _CW3_D3 / (eps + IS3) ** t
    alpha_sum = alpha0 + alpha1 + alpha2 + alpha3
    omega0 = alpha0 / alpha_sum
    omega1 = alpha1 / alpha_sum
    omega2 = alpha2 / alpha_sum
    omega3 = alpha3 / alpha_sum

    # Reconstructed face value (at x = 1/2)
    uL = (omega0 * _poly_eval(a0, _FACE_X) +
          omega1 * _poly_eval(a1, _FACE_X) +
          omega2 * _poly_eval(a2, _FACE_X) +
          omega3 * _poly_eval(a3, _FACE_X))

    # --- Right face (i-1/2) ---
    # The left face of cell i (x = -1/2) gives u_{i-1/2}^+, which is
    # the reconstruction from cell i at interface i-1/2.
    # In CWENO, weights are computed once per cell and the reconstruction
    # polynomial P_rec is defined for the whole cell.
    # So: u_{i-1/2}^+ = P_rec(-1/2).
    uR = (omega0 * _poly_eval(a0, -_FACE_X) +
          omega1 * _poly_eval(a1, -_FACE_X) +
          omega2 * _poly_eval(a2, -_FACE_X) +
          omega3 * _poly_eval(a3, -_FACE_X))

    return uL, uR


# ===========================================================================
# CWENO5 (Cravero 2017 -- paper CWENO7)
#   7-cell stencil {i-3..i+3}, P_opt 6th-degree, 4 cubic sub-stencils
# ===========================================================================

# Linear weights for CWENO5 (paper CWENO7)
_CW5_D0 = 0.75           # d_0 = 3/4
_CW5_D1 = 1.0 / 24.0     # d_1 = 1/24
_CW5_D2 = 1.0 / 12.0     # d_2 = 1/12
_CW5_D3 = 1.0 / 12.0     # d_3 = 1/12
_CW5_D4 = 1.0 / 24.0     # d_4 = 1/24


def cweno5_fv(u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3):
    r"""CWENO5 at :math:`i+1/2`.

    Full CWENO framework with P_0 construction from P_opt and sub-stencils
    P_1..P_4 (cubics on 4 cells each), Jiang-Shu indicators, nonlinear weights.

    Parameters are cell averages: :math:`u_{i-3} \dots u_{i+3}`.
    Returns (:math:`u_{i+1/2}^-`, :math:`u_{i-1/2}^+`).

    Reference: Cravero et al. (2017).
    """
    # --- Left face (i+1/2) ---
    # Sub-stencil polynomial coefficients (cubics)
    a1 = _poly_coeffs(_C3_LL, [u_im3, u_im2, u_im1, u_i])
    a2 = _poly_coeffs(_C3_L,  [u_im2, u_im1, u_i, u_ip1])
    a3 = _poly_coeffs(_C3_R,  [u_im1, u_i, u_ip1, u_ip2])
    a4 = _poly_coeffs(_C3_RR, [u_i, u_ip1, u_ip2, u_ip3])

    # Optimal polynomial coefficients (6th-degree)
    a_opt = _poly_coeffs(_P6, [u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3])

    # P_0 = (1/d_0)(P_opt - d_1*P_1 - d_2*P_2 - d_3*P_3 - d_4*P_4)
    # Pad sub-stencil polynomials (degree 3) to degree 6 (same as P_opt)
    inv_d0 = 1.0 / _CW5_D0
    a0 = [0.0] * 7
    for j in range(4):
        a0[j] = inv_d0 * (a_opt[j] - _CW5_D1 * a1[j] - _CW5_D2 * a2[j]
                          - _CW5_D3 * a3[j] - _CW5_D4 * a4[j])
    for j in range(4, 7):
        a0[j] = inv_d0 * a_opt[j]

    # Jiang-Shu smoothness indicators
    IS0 = _js_indicator(a0, _Q6_JS)  # P_0 is 6th-degree
    IS1 = _js_indicator(a1, _Q3_JS)  # P_1 is cubic
    IS2 = _js_indicator(a2, _Q3_JS)  # P_2 is cubic
    IS3 = _js_indicator(a3, _Q3_JS)  # P_3 is cubic
    IS4 = _js_indicator(a4, _Q3_JS)  # P_4 is cubic

    # Nonlinear weights
    eps = _CWENO_EPS
    t = _CWENO_T
    alpha0 = _CW5_D0 / (eps + IS0) ** t
    alpha1 = _CW5_D1 / (eps + IS1) ** t
    alpha2 = _CW5_D2 / (eps + IS2) ** t
    alpha3 = _CW5_D3 / (eps + IS3) ** t
    alpha4 = _CW5_D4 / (eps + IS4) ** t
    alpha_sum = alpha0 + alpha1 + alpha2 + alpha3 + alpha4
    omega0 = alpha0 / alpha_sum
    omega1 = alpha1 / alpha_sum
    omega2 = alpha2 / alpha_sum
    omega3 = alpha3 / alpha_sum
    omega4 = alpha4 / alpha_sum

    # Reconstructed face values
    uL = (omega0 * _poly_eval(a0, _FACE_X) +
          omega1 * _poly_eval(a1, _FACE_X) +
          omega2 * _poly_eval(a2, _FACE_X) +
          omega3 * _poly_eval(a3, _FACE_X) +
          omega4 * _poly_eval(a4, _FACE_X))

    uR = (omega0 * _poly_eval(a0, -_FACE_X) +
          omega1 * _poly_eval(a1, -_FACE_X) +
          omega2 * _poly_eval(a2, -_FACE_X) +
          omega3 * _poly_eval(a3, -_FACE_X) +
          omega4 * _poly_eval(a4, -_FACE_X))

    return uL, uR


# ===========================================================================
# Central WENO (Levy+1999)
# ===========================================================================

def _cw_smoothness(u_im1, u_i, u_ip1):
    r"""FV smoothness for Central WENO.

    Uses squared-undivided-differences for all stencils (consistent :math:`O(h^4)`
    scaling for the 3-pt optimal and :math:`O(h^2)` for 2-pt sub-stencils).
    The optimal stencil naturally gets smaller smoothness -> higher weight.
    """
    b_opt = (u_im1 - 2.0 * u_i + u_ip1) ** 2
    b_L = (u_i - u_im1) ** 2
    b_R = (u_ip1 - u_i) ** 2
    return b_opt, b_L, b_R


def central_weno_fv(u_im1, u_i, u_ip1):
    r"""Central WENO (Levy+1999) at :math:`i+1/2`. 3-pt."""
    b_opt, b_L, b_R = _cw_smoothness(u_im1, u_i, u_ip1)

    a_opt = _CW_OPT / (_EPSL + b_opt) ** _P
    a_L = _CW_SIDE / (_EPSL + b_L) ** _P
    a_R = _CW_SIDE / (_EPSL + b_R) ** _P
    s = 1.0 / (a_opt + a_L + a_R)
    w_opt, w_L, w_R = a_opt * s, a_L * s, a_R * s

    # Left face
    Q_opt_L = _K_1_6 * (-u_im1 + 5.0 * u_i + 2.0 * u_ip1)
    P_L_L = -0.5 * u_im1 + 1.5 * u_i
    P_R_L = 0.5 * (u_i + u_ip1)
    uL = w_opt * Q_opt_L + w_L * P_L_L + w_R * P_R_L

    # Right face
    Q_opt_R = _K_1_6 * (2.0 * u_im1 + 5.0 * u_i - u_ip1)
    P_L_R = 0.5 * (u_im1 + u_i)
    P_R_R = 1.5 * u_i - 0.5 * u_ip1
    uR = w_opt * Q_opt_R + w_L * P_L_R + w_R * P_R_R

    return uL, uR


# ===========================================================================
# Capdeville CWENO5 (Capdeville 2008)
# ===========================================================================

def _capdeville5_opt_L(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """Capdeville CWENO5 optimal polynomial at left face (i+1/2)."""
    return _K_1_60 * (2.0 * u_im2 - 13.0 * u_im1 + 47.0 * u_i +
                      27.0 * u_ip1 - 3.0 * u_ip2)


def _capdeville5_opt_R(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """Capdeville CWENO5 optimal polynomial at right face (i-1/2)."""
    return _K_1_60 * (-3.0 * u_im2 + 27.0 * u_im1 + 47.0 * u_i -
                      13.0 * u_ip1 + 2.0 * u_ip2)


def _capdeville5_sub_L(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """Capdeville CWENO5 sub-stencil polynomials at left face (i+1/2).

    Returns (pL, pC, pR) for the three 3rd-order sub-stencils.
    """
    pL = _K_1_6 * (2.0 * u_im2 - 7.0 * u_im1 + 11.0 * u_i)
    pC = _K_1_6 * (-u_im1 + 5.0 * u_i + 2.0 * u_ip1)
    pR = _K_1_6 * (2.0 * u_i + 5.0 * u_ip1 - u_ip2)
    return pL, pC, pR


def _capdeville5_sub_R(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """Capdeville CWENO5 sub-stencil polynomials at right face (i-1/2, reversed).

    Returns (pL, pC, pR) for the three 3rd-order sub-stencils.
    """
    pL = _K_1_6 * (2.0 * u_ip2 - 7.0 * u_ip1 + 11.0 * u_i)
    pC = _K_1_6 * (-u_ip1 + 5.0 * u_i + 2.0 * u_im1)
    pR = _K_1_6 * (2.0 * u_i + 5.0 * u_im1 - u_im2)
    return pL, pC, pR


def _capdeville5_smoothness(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """FV JS smoothness for 3-point sub-stencils, denominator 6."""
    def _is3(vals, m):
        # FV JS 3x3 matrices: stencil 0=[-2,-1,0]:
        # /6 [[8,-19,11],[-19,50,-31],[11,-31,20]]
        # stencil 1=[-1,0,1]: /6 [[8,-13,5],[-13,26,-13],[5,-13,8]]
        # stencil 2=[0,1,2]:  /6 [[20,-31,11],[-31,50,-19],[11,-19,8]]
        beta = 0.0
        for i in range(3):
            row = m[i]
            vi = vals[i]
            for j in range(3):
                beta += row[j] * vi * vals[j]
        return beta / 6.0

    bL = _is3((u_im2, u_im1, u_i), [[8,-19,11],[-19,50,-31],[11,-31,20]])
    bC = _is3((u_im1, u_i, u_ip1), [[8,-13,5],[-13,26,-13],[5,-13,8]])
    bR = _is3((u_i, u_ip1, u_ip2), [[20,-31,11],[-31,50,-19],[11,-19,8]])
    return bL, bC, bR


def cweno5_capdeville_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    r"""Capdeville CWENO5 (2008) at :math:`i+1/2`. 5-pt."""
    # Left face
    bL, bC, bR = _capdeville5_smoothness(u_im2, u_im1, u_i, u_ip1, u_ip2)
    tau = abs(bL - bR)
    a_opt = _DOPT / (_EPSL + tau) ** _P
    aL = _DK / (_EPSL + bL) ** _P
    aC = _DK / (_EPSL + bC) ** _P
    aR = _DK / (_EPSL + bR) ** _P
    s = 1.0 / (a_opt + aL + aC + aR)
    w_opt, wL, wC, wR = a_opt * s, aL * s, aC * s, aR * s
    P_opt = _capdeville5_opt_L(u_im2, u_im1, u_i, u_ip1, u_ip2)
    PL, PC, PR = _capdeville5_sub_L(u_im2, u_im1, u_i, u_ip1, u_ip2)
    uL = w_opt * P_opt + wL * PL + wC * PC + wR * PR

    # Right face
    bL_r, bC_r, bR_r = _capdeville5_smoothness(u_ip2, u_ip1, u_i, u_im1, u_im2)
    tau_r = abs(bL_r - bR_r)
    a_opt_r = _DOPT / (_EPSL + tau_r) ** _P
    aL_r = _DK / (_EPSL + bL_r) ** _P
    aC_r = _DK / (_EPSL + bC_r) ** _P
    aR_r = _DK / (_EPSL + bR_r) ** _P
    s_r = 1.0 / (a_opt_r + aL_r + aC_r + aR_r)
    w_opt_r, wL_r, wC_r, wR_r = (
        a_opt_r * s_r,
        aL_r * s_r,
        aC_r * s_r,
        aR_r * s_r,
    )
    P_opt_r = _capdeville5_opt_R(u_im2, u_im1, u_i, u_ip1, u_ip2)
    PL_r, PC_r, PR_r = _capdeville5_sub_R(u_im2, u_im1, u_i, u_ip1, u_ip2)
    uR = w_opt_r * P_opt_r + wL_r * PL_r + wC_r * PC_r + wR_r * PR_r

    return uL, uR
# ---------------------------------------------------------------------------
# JIT compilation (pynalgo convention)
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    reveal_locals()  # noqa: F821
else:
    _cweno3_sub_L = JIT(_cweno3_sub_L)
    _cweno3_opt_L = JIT(_cweno3_opt_L)
    _cweno3_sub_R = JIT(_cweno3_sub_R)
    _cweno3_opt_R = JIT(_cweno3_opt_R)
    _cweno3_smoothness = JIT(_cweno3_smoothness)
    cweno_z3_fv = JIT(cweno_z3_fv)
    _cweno5_sub_L = JIT(_cweno5_sub_L)
    _cweno5_opt_L = JIT(_cweno5_opt_L)
    _cweno5_sub_R = JIT(_cweno5_sub_R)
    _cweno5_opt_R = JIT(_cweno5_opt_R)
    _cweno5_smoothness = JIT(_cweno5_smoothness)
    _js5_sub0 = JIT(_js5_sub0)
    _js5_sub1 = JIT(_js5_sub1)
    _js5_sub2 = JIT(_js5_sub2)
    cweno_z5_fv = JIT(cweno_z5_fv)
    _poly_coeffs = JIT(_poly_coeffs)
    _poly_eval = JIT(_poly_eval)
    _js_indicator = JIT(_js_indicator)
    cweno3_fv = JIT(cweno3_fv)
    cweno5_fv = JIT(cweno5_fv)
    _cw_smoothness = JIT(_cw_smoothness)
    central_weno_fv = JIT(central_weno_fv)
    _capdeville5_opt_L = JIT(_capdeville5_opt_L)
    _capdeville5_opt_R = JIT(_capdeville5_opt_R)
    _capdeville5_sub_L = JIT(_capdeville5_sub_L)
    _capdeville5_sub_R = JIT(_capdeville5_sub_R)
    _capdeville5_smoothness = JIT(_capdeville5_smoothness)
    cweno5_capdeville_fv = JIT(cweno5_capdeville_fv)
#
# :D
#
