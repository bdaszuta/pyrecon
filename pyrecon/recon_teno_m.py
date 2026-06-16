"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: TENO-M reconstruction methods (Li, Fu, Adams 2021)

TENO-M extends the standard TENO framework by filtering (rather than
discarding) nonsmooth candidate stencils.  Three nonlinear limiter
variants are provided:

  teno_m_va_fv   -- Van Albada limiter (most dissipative, 2nd/3rd-order TVD)
  teno_m_tvd5_fv -- 5th-order TVD limiter with curvature information
  teno_m_mp_fv   -- Monotonicity-Preserving limiter (least dissipative)

Key difference from standard TENO: NO renormalization.
Assembly:  u_face = sum_k d_k * f^M_k  where d_k are optimal linear weights.
"""
import math

from pyrecon._jit_utils import JIT, TYPE_CHECKING
from pyrecon.recon_teno5 import (
    _teno5_cutoff,
    _teno5_stencils_L_fv,
    _teno5_stencils_R_fv,
    _teno_B0,
    _teno_B1,
    _teno_B2,
)
from pyrecon.utils import minmod

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# TENO-M optimal linear weights (paper Eq. 29).
# Paper indexing: d0=1/10 (S0=backward), d1=3/5 (S1=central), d2=3/10 (S2=forward).
# Tuple order: (central=3/5, forward=3/10, backward=1/10), matching stencil ordering.
_DTENO_M_FV = (6.0 / 10.0, 3.0 / 10.0, 1.0 / 10.0)

# Cutoff threshold (same as standard TENO5)
_C_T = 1e-5

# Small epsilon for safe division in limiters
_LIM_EPS = 1e-40

# MP limiter epsilon (controls strictness of monotonicity bounds)
_MP_EPS = 1e-12

# ---------------------------------------------------------------------------
# VA (Van Albada) Limiter  --  Eqs. 23--25
# ---------------------------------------------------------------------------


def _va_limited(u_im2, u_im1, u_i, u_ip1, u_ip2):
    r"""Van Albada limiter on 3-cell stencil :math:`\{i-1, i, i+1\}`.

    Computes :math:`u_{i+1/2}^-` using 3rd-order VA-limited reconstruction.
    Returns u_i at extrema (:math:`d_+ \cdot d_- \leq 0`).
    Extra args u_im2, u_ip2 accepted for uniform calling convention but ignored.

    Eq. 25 with :math:`\kappa = 1/3`.
    """
    d_plus = u_ip1 - u_i
    d_minus = u_i - u_im1
    if d_plus * d_minus <= 0.0:
        return u_i
    r = d_plus / d_minus
    phi = 2.0 * r / (r * r + 1.0)
    kappa = 1.0 / 3.0
    return (u_i + 0.25 * phi *
            ((1.0 - kappa * phi) * d_minus + (1.0 + kappa * phi) * d_plus))


def _va_limited_R(u_ip2, u_ip1, u_i, u_im1, u_im2):
    r"""Van Albada limiter for right face :math:`u_{i-1/2}^+`.

    Mirrored call: reverses the 5-cell stencil.
    """
    return _va_limited(u_ip2, u_ip1, u_i, u_im1, u_im2)


# ---------------------------------------------------------------------------
# TVD5 Limiter  --  Eqs. 26--28
# ---------------------------------------------------------------------------


def _tvd5_limited(u_im2, u_im1, u_i, u_ip1, u_ip2):
    r"""5th-order TVD limiter on 5-cell stencil :math:`\{i-2,i-1,i,i+1,i+2\}`.

    Computes :math:`u_{i+1/2}^-` using curvature-informed TVD slope.

    Eq. 27 slope ratios (5-cell stencil centered at i):

    .. math::
       r_j   = (f_{i+1} - f_i) / (f_i - f_{i-1})

       r_{j+1} = (f_{i+2} - f_{i+1}) / (f_{i+1} - f_i)

       r_{j-1} = (f_i - f_{i-1}) / (f_{i-1} - f_{i-2})

    Eq. 27 curvature :math:`\beta`:

    .. math::
       \beta = (-2/r_{j-1} + 11 + 24 r_j - 3 r_j r_{j+1}) / 30

    Eq. 26 slope function:

    .. math::
       \phi = \max(0, \min(\alpha, \alpha r_j, \beta)) \quad \text{with } \alpha = 2

    Eq. 28 reconstruction:

    .. math::
       u = u_i + \frac{1}{2} \phi (u_i - u_{i-1})
    """
    alpha = 2.0

    d0 = u_i - u_im1      # f_i - f_{i-1}
    d1 = u_ip1 - u_i      # f_{i+1} - f_i
    d2 = u_ip2 - u_ip1    # f_{i+2} - f_{i+1}
    dm1 = u_im1 - u_im2   # f_{i-1} - f_{i-2}

    # Handle flat regions gracefully
    if abs(d0) < _LIM_EPS:
        return u_i

    r_j = d1 / d0

    # Compute r_{j-1} and r_{j+1} safely
    r_jm1 = d0 / (dm1 + _LIM_EPS) if abs(dm1) > _LIM_EPS else 1.0
    r_jp1 = d2 / (d1 + _LIM_EPS) if abs(d1) > _LIM_EPS else 1.0

    # Curvature term beta (Eq. 27)
    beta = (-2.0 / max(r_jm1, _LIM_EPS) + 11.0 + 24.0 * r_j -
            3.0 * r_j * r_jp1) / 30.0

    # TVD5 slope function (Eq. 26)
    phi = max(0.0, min(alpha, min(alpha * r_j, beta)))

    # TVD5 reconstruction (Eq. 28)
    return u_i + 0.5 * phi * d0


def _tvd5_limited_R(u_ip2, u_ip1, u_i, u_im1, u_im2):
    r"""TVD5 limiter for right face :math:`u_{i-1/2}^+`.

    Reverses the 5-cell stencil: :math:`\{i+2,i+1,i,i-1,i-2\}`.
    """
    return _tvd5_limited(u_ip2, u_ip1, u_i, u_im1, u_im2)


# ---------------------------------------------------------------------------
# MP (Monotonicity-Preserving) Limiter  --  Eqs. 30--39
# ---------------------------------------------------------------------------


def _mp_m4_curvature(d0, dm, dp):
    r"""M4 curvature measurement.

    .. math::
       d_{i+1/2}^{M4} = \operatorname{minmod}(4 d_0 - d_p, 4 d_p - d_0, d_0, d_p)

    More restrictive than MM variant: clamps when :math:`d_{i+1}/d_i < 1/4` or :math:`> 4`.

    Reference: Eq. 34.
    """
    return minmod(4.0 * d0 - dp, 4.0 * dp - d0, d0, dp)


def _mp_limited_L(u_im2, u_im1, u_i, u_ip1, u_ip2):
    r"""MP limiter for left face :math:`u_{i+1/2}^-`.

    Adapted from _mpnlimiter_5p in recon_mpn.py.
    Uses 5th-order linear reconstruction as candidate.
    """
    oo2 = 0.5
    fot = 4.0 / 3.0
    alphatil = 4.0

    # 5th-order linear reconstruction as candidate
    u_candidate = ((2.0 / 60.0) * u_im2 +
                   (-13.0 / 60.0) * u_im1 +
                   (47.0 / 60.0) * u_i +
                   (27.0 / 60.0) * u_ip1 +
                   (-3.0 / 60.0) * u_ip2)

    # Early exit: if candidate is already within MP bounds
    U_L2 = math.sqrt(u_im2**2 + u_im1**2 + u_i**2 + u_ip1**2 + u_ip2**2)
    u_MP_simple = u_i + minmod(u_ip1 - u_i, alphatil * (u_i - u_im1))
    if (u_candidate - u_i) * (u_candidate - u_MP_simple) <= _MP_EPS * U_L2:
        return u_candidate

    # Curvature measurements (Eq. 33)
    dm = u_im2 - 2.0 * u_im1 + u_i     # d_{i-1}
    d0 = u_im1 - 2.0 * u_i + u_ip1     # d_i
    dp = u_i - 2.0 * u_ip1 + u_ip2     # d_{i+1}

    # M4 curvature at face i+1/2 (Eq. 34)
    dm4p = _mp_m4_curvature(d0, dm, dp)
    dm4m = _mp_m4_curvature(d0, dp, dm)  # backward direction (swap dp/dm)

    # Bounds (Eqs. 35--38)
    u_ul = u_i + alphatil * (u_i - u_im1)              # UL (Eq. 35)
    u_av = oo2 * (u_i + u_ip1)                          # average
    u_md = u_av - oo2 * dm4p                             # MD (Eq. 36)
    u_lc = u_i + oo2 * (u_i - u_im1) + fot * dm4m       # LC (Eq. 37)

    u_min = max(min(u_i, u_ip1, u_md), min(u_i, u_ul, u_lc))
    u_max = min(max(u_i, u_ip1, u_md), max(u_i, u_ul, u_lc))

    # Final MP-limited value (Eq. 39): median clamp
    return u_candidate + minmod(u_min - u_candidate, u_max - u_candidate)


def _mp_limited_R(u_ip2, u_ip1, u_i, u_im1, u_im2):
    r"""MP limiter for right face :math:`u_{i-1/2}^+`.

    Reverses the 5-cell stencil.
    """
    return _mp_limited_L(u_ip2, u_ip1, u_i, u_im1, u_im2)


# ---------------------------------------------------------------------------
# TENO-M-VA core (FV)
# ---------------------------------------------------------------------------


def _teno_m_va_LR(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """TENO-M-VA paired L+R reconstruction.

    Van Albada limiter on nonsmooth stencils.
    """
    dt = _DTENO_M_FV

    # --- Left face ---
    b0_L = _teno_B0(u_im1, u_i, u_ip1)
    b1_L = _teno_B1(u_i, u_ip1, u_ip2)
    b2_L = _teno_B2(u_im2, u_im1, u_i)
    dL0, dL1, dL2 = _teno5_cutoff(b0_L, b1_L, b2_L)

    ukL0, ukL1, ukL2 = _teno5_stencils_L_fv(u_im2, u_im1, u_i, u_ip1, u_ip2)

    fM_L0 = ukL0 if dL0 == 1.0 else _va_limited(u_im2, u_im1, u_i, u_ip1, u_ip2)
    fM_L1 = ukL1 if dL1 == 1.0 else _va_limited(u_im2, u_im1, u_i, u_ip1, u_ip2)
    fM_L2 = ukL2 if dL2 == 1.0 else _va_limited(u_im2, u_im1, u_i, u_ip1, u_ip2)

    # --- Right face ---
    b1_R = _teno_B1(u_i, u_im1, u_im2)
    b2_R = _teno_B2(u_ip2, u_ip1, u_i)
    dR0, dR1, dR2 = _teno5_cutoff(b0_L, b1_R, b2_R)

    ukR0, ukR1, ukR2 = _teno5_stencils_R_fv(u_im2, u_im1, u_i, u_ip1, u_ip2)

    fM_R0 = ukR0 if dR0 == 1.0 else _va_limited_R(u_ip2, u_ip1, u_i, u_im1, u_im2)
    fM_R1 = ukR1 if dR1 == 1.0 else _va_limited_R(u_ip2, u_ip1, u_i, u_im1, u_im2)
    fM_R2 = ukR2 if dR2 == 1.0 else _va_limited_R(u_ip2, u_ip1, u_i, u_im1, u_im2)

    # Assemble WITHOUT renormalization (Eq. 29)
    uL = dt[0] * fM_L0 + dt[1] * fM_L1 + dt[2] * fM_L2
    uR = dt[0] * fM_R0 + dt[1] * fM_R1 + dt[2] * fM_R2

    return uL, uR


# ---------------------------------------------------------------------------
# TENO-M-TVD5 core (FV)
# ---------------------------------------------------------------------------


def _teno_m_tvd5_LR(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """TENO-M-TVD5 paired L+R reconstruction.

    5th-order TVD limiter with curvature information on nonsmooth stencils.
    """
    dt = _DTENO_M_FV

    # --- Left face ---
    b0_L = _teno_B0(u_im1, u_i, u_ip1)
    b1_L = _teno_B1(u_i, u_ip1, u_ip2)
    b2_L = _teno_B2(u_im2, u_im1, u_i)
    dL0, dL1, dL2 = _teno5_cutoff(b0_L, b1_L, b2_L)

    ukL0, ukL1, ukL2 = _teno5_stencils_L_fv(u_im2, u_im1, u_i, u_ip1, u_ip2)

    fM_L0 = ukL0 if dL0 == 1.0 else _tvd5_limited(u_im2, u_im1, u_i, u_ip1, u_ip2)
    fM_L1 = ukL1 if dL1 == 1.0 else _tvd5_limited(u_im2, u_im1, u_i, u_ip1, u_ip2)
    fM_L2 = ukL2 if dL2 == 1.0 else _tvd5_limited(u_im2, u_im1, u_i, u_ip1, u_ip2)

    # --- Right face ---
    b1_R = _teno_B1(u_i, u_im1, u_im2)
    b2_R = _teno_B2(u_ip2, u_ip1, u_i)
    dR0, dR1, dR2 = _teno5_cutoff(b0_L, b1_R, b2_R)

    ukR0, ukR1, ukR2 = _teno5_stencils_R_fv(u_im2, u_im1, u_i, u_ip1, u_ip2)

    fM_R0 = ukR0 if dR0 == 1.0 else _tvd5_limited_R(u_ip2, u_ip1, u_i, u_im1, u_im2)
    fM_R1 = ukR1 if dR1 == 1.0 else _tvd5_limited_R(u_ip2, u_ip1, u_i, u_im1, u_im2)
    fM_R2 = ukR2 if dR2 == 1.0 else _tvd5_limited_R(u_ip2, u_ip1, u_i, u_im1, u_im2)

    uL = dt[0] * fM_L0 + dt[1] * fM_L1 + dt[2] * fM_L2
    uR = dt[0] * fM_R0 + dt[1] * fM_R1 + dt[2] * fM_R2

    return uL, uR


# ---------------------------------------------------------------------------
# TENO-M-MP core (FV)
# ---------------------------------------------------------------------------


def _teno_m_mp_LR(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """TENO-M-MP paired L+R reconstruction.

    Monotonicity-Preserving limiter on nonsmooth stencils.
    """
    dt = _DTENO_M_FV

    # --- Left face ---
    b0_L = _teno_B0(u_im1, u_i, u_ip1)
    b1_L = _teno_B1(u_i, u_ip1, u_ip2)
    b2_L = _teno_B2(u_im2, u_im1, u_i)
    dL0, dL1, dL2 = _teno5_cutoff(b0_L, b1_L, b2_L)

    ukL0, ukL1, ukL2 = _teno5_stencils_L_fv(u_im2, u_im1, u_i, u_ip1, u_ip2)

    fM_L0 = ukL0 if dL0 == 1.0 else _mp_limited_L(u_im2, u_im1, u_i, u_ip1, u_ip2)
    fM_L1 = ukL1 if dL1 == 1.0 else _mp_limited_L(u_im2, u_im1, u_i, u_ip1, u_ip2)
    fM_L2 = ukL2 if dL2 == 1.0 else _mp_limited_L(u_im2, u_im1, u_i, u_ip1, u_ip2)

    # --- Right face ---
    b1_R = _teno_B1(u_i, u_im1, u_im2)
    b2_R = _teno_B2(u_ip2, u_ip1, u_i)
    dR0, dR1, dR2 = _teno5_cutoff(b0_L, b1_R, b2_R)

    ukR0, ukR1, ukR2 = _teno5_stencils_R_fv(u_im2, u_im1, u_i, u_ip1, u_ip2)

    fM_R0 = ukR0 if dR0 == 1.0 else _mp_limited_R(u_ip2, u_ip1, u_i, u_im1, u_im2)
    fM_R1 = ukR1 if dR1 == 1.0 else _mp_limited_R(u_ip2, u_ip1, u_i, u_im1, u_im2)
    fM_R2 = ukR2 if dR2 == 1.0 else _mp_limited_R(u_ip2, u_ip1, u_i, u_im1, u_im2)

    uL = dt[0] * fM_L0 + dt[1] * fM_L1 + dt[2] * fM_L2
    uR = dt[0] * fM_R0 + dt[1] * fM_R1 + dt[2] * fM_R2

    return uL, uR


# ---------------------------------------------------------------------------
# Public API: TENO-M-VA (FV)
# ---------------------------------------------------------------------------


def teno_m_va_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    r"""TENO-M-VA reconstruction (FV weights) at i+1/2.

    Van Albada limiter on nonsmooth stencils.
    Most dissipative TENO-M variant, best symmetry preservation.

    Parameters
    ----------
    u_im2, u_im1, u_i, u_ip1, u_ip2 : float
        5-cell stencil values (FV).

    Returns
    -------
    (uL, uR) : uL = :math:`u_{i+1/2}^-`, uR = :math:`u_{i-1/2}^+`.
    """
    return _teno_m_va_LR(u_im2, u_im1, u_i, u_ip1, u_ip2)


# ---------------------------------------------------------------------------
# Public API: TENO-M-TVD5 (FV)
# ---------------------------------------------------------------------------


def teno_m_tvd5_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    r"""TENO-M-TVD5 reconstruction (FV weights) at i+1/2.

    5th-order TVD limiter with curvature information on nonsmooth stencils.
    Moderate dissipation, good for under-resolved turbulence.

    Parameters
    ----------
    u_im2, u_im1, u_i, u_ip1, u_ip2 : float
        5-cell stencil values (FV).

    Returns
    -------
    (uL, uR) : uL = :math:`u_{i+1/2}^-`, uR = :math:`u_{i-1/2}^+`.
    """
    return _teno_m_tvd5_LR(u_im2, u_im1, u_i, u_ip1, u_ip2)


# ---------------------------------------------------------------------------
# Public API: TENO-M-MP (FV)
# ---------------------------------------------------------------------------


def teno_m_mp_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    r"""TENO-M-MP reconstruction (FV weights) at i+1/2.

    Monotonicity-Preserving limiter on nonsmooth stencils.
    Least dissipative TENO-M variant, allows smooth local extrema.

    Parameters
    ----------
    u_im2, u_im1, u_i, u_ip1, u_ip2 : float
        5-cell stencil values (FV).

    Returns
    -------
    (uL, uR) : uL = :math:`u_{i+1/2}^-`, uR = :math:`u_{i-1/2}^+`.
    """
    return _teno_m_mp_LR(u_im2, u_im1, u_i, u_ip1, u_ip2)
# ---------------------------------------------------------------------------
# JIT compilation (pynalgo convention)
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    reveal_locals()  # noqa: F821
else:
    _va_limited = JIT(_va_limited)
    _va_limited_R = JIT(_va_limited_R)
    _tvd5_limited = JIT(_tvd5_limited)
    _tvd5_limited_R = JIT(_tvd5_limited_R)
    _mp_m4_curvature = JIT(_mp_m4_curvature)
    _mp_limited_L = JIT(_mp_limited_L)
    _mp_limited_R = JIT(_mp_limited_R)
    _teno_m_va_LR = JIT(_teno_m_va_LR)
    _teno_m_tvd5_LR = JIT(_teno_m_tvd5_LR)
    _teno_m_mp_LR = JIT(_teno_m_mp_LR)
    teno_m_va_fv = JIT(teno_m_va_fv)
    teno_m_tvd5_fv = JIT(teno_m_tvd5_fv)
    teno_m_mp_fv = JIT(teno_m_mp_fv)
#
# :D
#
