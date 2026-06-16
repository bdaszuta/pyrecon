"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: BVD reconstruction (Sun, Inaba, Xiao 2016)
"""
import math

from pyrecon._jit_utils import JIT, TYPE_CHECKING
from pyrecon.recon_weno5 import weno5z_fv

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_KBETA = 1.6       # THINC steepness parameter (paper: beta = 1.6)
_EPSL = 1e-40
_THINC_EPS = 1e-20  # epsilon for closed-form THINC (Eq 23)
_INF = float('inf')  # module-level: Numba nopython rejects float('inf') inline


# ---------------------------------------------------------------------------
# Closed-form THINC face values (Sun/Inaba/Xiao 2016, Eqs 18-23)
# ---------------------------------------------------------------------------

def _thinc_pair_closed(u_im1, u_i, u_ip1):
    r"""Closed-form THINC reconstruction for cell i.

    Returns (uL, uR) where:
      uL = :math:`\Phi_i^{<2>}(x_{i+1/2})`  -- right face (pyrecon uL convention)
      uR = :math:`\Phi_i^{<2>}(x_{i-1/2})`  -- left face  (pyrecon uR convention)

    Reference: Sun, Inaba & Xiao, J. Comput. Phys. 314, 305-325 (2016), Eqs. 18-23.
    """
    u_min = min(u_im1, u_ip1)
    u_max_val = max(u_im1, u_ip1) - u_min

    if u_max_val < _THINC_EPS:
        return u_i, u_i

    gamma = 1.0 if u_ip1 > u_im1 else -1.0

    # Eq 23: auxiliary variables A, B
    # B = exp(gamma * beta * (2 * (u_i - u_min + eps) / (u_max_val + eps) - 1))
    arg = 2.0 * (u_i - u_min + _THINC_EPS) / (u_max_val + _THINC_EPS) - 1.0
    B = math.exp(gamma * _KBETA * arg)

    tanh_beta = math.tanh(_KBETA)
    cosh_beta = math.cosh(_KBETA)
    A = (B / cosh_beta - 1.0) / tanh_beta

    # Eq 21: left face (i-1/2) -> pyrecon uR
    # Phi_i(x_{i-1/2}) = u_min + u_max_val/2
    #   * (1 + gamma*(tanh(beta)+A)/(1+A*tanh(beta)))
    uR = u_min + 0.5 * u_max_val * (
        1.0 + gamma * (tanh_beta + A) / (1.0 + A * tanh_beta)
    )

    # Eq 22: right face (i+1/2) -> pyrecon uL
    # Phi_i(x_{i+1/2}) = u_min + u_max_val/2 * (1 + gamma * A)
    uL = u_min + 0.5 * u_max_val * (1.0 + gamma * A)

    return uL, uR


# ---------------------------------------------------------------------------
# THINC face value
# ---------------------------------------------------------------------------

def _thinc_face(u_im1, u_i, u_ip1, xtilde):
    """THINC reconstruction at face position xtilde (0=left, 1=right).

    For left-biased reconstruction:
      u(x_tilde) = u_min + 0.5 * (u_max - u_min)
                   * (1 + gamma * tanh(beta * (x_tilde - 0.5)))

    where gamma = sign(u_ip1 - u_im1).
    """
    u_min = min(u_im1, u_ip1)
    u_max = max(u_im1, u_ip1)

    if u_max - u_min < _EPSL:
        return u_i

    gamma = 1.0 if u_ip1 > u_im1 else -1.0
    return u_min + 0.5 * (u_max - u_min) * (
        1.0 + gamma * math.tanh(_KBETA * (xtilde - 0.5))
    )


# ---------------------------------------------------------------------------
# BVD selection based on Total Boundary Variation (TBV)
# ---------------------------------------------------------------------------

def _bvd_select_tbv(u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3):
    """Select between WENO5-Z and THINC based on TBV minimization.

    Phi^<1> = WENO5-Z (5th-order)
    Phi^<2> = THINC
    Selection via Total Boundary Variation (TBV) minimization.

    Reference: Sun, Inaba & Xiao (2016), Remark 6.
    """

    # === WENO5-Z candidate (Phi^<1>, 5th-order) ===
    # Cell i-1: left face at i-1/2
    wL_i1, _ = weno5z_fv(u_im3, u_im2, u_im1, u_i, u_ip1)
    # Cell i: right face at i-1/2, left face at i+1/2
    wL_i, wR_i = weno5z_fv(u_im2, u_im1, u_i, u_ip1, u_ip2)
    # Cell i+1: right face at i+1/2
    _, wR_ip1 = weno5z_fv(u_im1, u_i, u_ip1, u_ip2, u_ip3)

    # TBV for WENO5-Z: |uR - uL| at i-1/2 + |uR - uL| at i+1/2
    tbv_weno = abs(wR_i - wL_i1) + abs(wR_ip1 - wL_i)

    # === THINC candidate values (Phi^<2>) ===
    # Face i-1/2 from cell i-1 (left): THINC at x=1 for cell i-1
    tL_i1 = _thinc_face(u_im2, u_im1, u_i, 1.0)
    # Face i-1/2 from cell i (right): THINC at x=0 for cell i
    tR_i = _thinc_face(u_im1, u_i, u_ip1, 0.0)
    # Face i+1/2 from cell i (left): THINC at x=1 for cell i
    tL_i = _thinc_face(u_im1, u_i, u_ip1, 1.0)
    # Face i+1/2 from cell i+1 (right): THINC at x=0 for cell i+1
    tR_ip1 = _thinc_face(u_i, u_ip1, u_ip2, 0.0)

    # TBV for THINC
    tbv_thinc = abs(tR_i - tL_i1) + abs(tR_ip1 - tL_i)

    # Select the candidate with smaller TBV
    if tbv_thinc < tbv_weno:
        return tL_i, tR_i
    else:
        return wL_i, wR_i


def bvd_tbv_fv(u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3):
    """BVD reconstruction: WENO5-Z/THINC switching via TBV (FV).

    Candidate Phi^<1> = WENO5-Z (5th-order).
    Candidate Phi^<2> = THINC.
    Selection via Total Boundary Variation minimization (Remark 6).

    Reference: Sun, Inaba & Xiao (2016), arXiv:1602.00814.
    """
    return _bvd_select_tbv(u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3)


# ---------------------------------------------------------------------------
# 4-way per-interface BVD (Sun/Inaba/Xiao 2016, Eqs 5-7)
# ---------------------------------------------------------------------------

def bvd_fv(u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3):
    """BVD reconstruction: 4-way per-interface BV minimization (FV).

    Implements the full paper algorithm:
    - Candidate Phi^<1> = WENO5-Z (5th-order)
    - Candidate Phi^<2> = THINC (closed-form)
    - 4-way BV minimization at each interface
    - Cell conflict resolution
    - Extrema guard for THINC applicability

    Returns (uL, uR) for cell i, where:
      uL = reconstructed value at cell i's right face (i+1/2)
      uR = reconstructed value at cell i's left face (i-1/2)

    Reference: Sun, Inaba & Xiao (2016), arXiv:1602.00814.
    """
    # ------------------------------------------------------------------
    # Step 1: Prepare candidates for cells i-1, i, i+1
    # ------------------------------------------------------------------

    # --- WENO-Z candidates (Phi^<1>) ---
    # Cell i-1
    wL_im1, wR_im1 = weno5z_fv(u_im3, u_im2, u_im1, u_i, u_ip1)
    # Cell i
    wL_i, wR_i = weno5z_fv(u_im2, u_im1, u_i, u_ip1, u_ip2)
    # Cell i+1
    wL_ip1, wR_ip1 = weno5z_fv(u_im1, u_i, u_ip1, u_ip2, u_ip3)

    # --- THINC candidates (Phi^<2>) ---
    # Cell i-1
    tL_im1, tR_im1 = _thinc_pair_closed(u_im2, u_im1, u_i)
    # Cell i
    tL_i, tR_i = _thinc_pair_closed(u_im1, u_i, u_ip1)
    # Cell i+1
    tL_ip1, tR_ip1 = _thinc_pair_closed(u_i, u_ip1, u_ip2)

    # --- Extrema guard (Eq 24) ---
    # THINC only applies when (u_{i+1} - u_i) * (u_i - u_{i-1}) > 0
    ext_ok_im1 = (u_i - u_im1) * (u_im1 - u_im2) > 0.0
    ext_ok_i   = (u_ip1 - u_i) * (u_i - u_im1) > 0.0
    ext_ok_ip1 = (u_ip2 - u_ip1) * (u_ip1 - u_i) > 0.0

    # ------------------------------------------------------------------
    # Step 2: 4-way BV minimization at interface i-1/2
    #   BV = |Phi^{<p>}_{i-1}(x_{i-1/2}) - Phi^{<q>}_i(x_{i-1/2})|
    #
    # At interface i-1/2:
    #   Cell i-1 provides its RIGHT face  (wL_im1 or tL_im1 in pyrecon)
    #   Cell i   provides its LEFT face   (wR_i    or tR_i    in pyrecon)
    # ------------------------------------------------------------------

    # (WENO, WENO)
    bv00_L = abs(wL_im1 - wR_i)
    # (THINC, WENO) -- only if THINC valid for cell i-1
    bv01_L = abs(tL_im1 - wR_i) if ext_ok_im1 else _INF
    # (WENO, THINC) -- only if THINC valid for cell i
    bv10_L = abs(wL_im1 - tR_i) if ext_ok_i else _INF
    # (THINC, THINC) -- only if THINC valid for both
    bv11_L = abs(tL_im1 - tR_i) if (ext_ok_im1 and ext_ok_i) else _INF

    bvs_L = [bv00_L, bv01_L, bv10_L, bv11_L]
    min_bv_L = min(bvs_L)

    # Select winning pair for interface i-1/2
    # Tiebreak: prefer fewer THINC components
    # priority (W,W) > (T,W) = (W,T) > (T,T)
    if bv00_L == min_bv_L:
        sel_L_im1, sel_L_i = 1, 1   # (WENO, WENO)
    elif bv01_L == min_bv_L:
        sel_L_im1, sel_L_i = 2, 1   # (THINC, WENO)
    elif bv10_L == min_bv_L:
        sel_L_im1, sel_L_i = 1, 2   # (WENO, THINC)
    else:
        sel_L_im1, sel_L_i = 2, 2   # (THINC, THINC)

    # Actual BV at i-1/2 with the winning pair
    phi_im1_R = tL_im1 if sel_L_im1 == 2 else wL_im1
    phi_i_L   = tR_i    if sel_L_i == 2    else wR_i
    bv_L_signed = phi_im1_R - phi_i_L

    # ------------------------------------------------------------------
    # Step 3: 4-way BV minimization at interface i+1/2
    #   BV = |Phi^{<p>}_i(x_{i+1/2}) - Phi^{<q>}_{i+1}(x_{i+1/2})|
    #
    # At interface i+1/2:
    #   Cell i   provides its RIGHT face  (wL_i or tL_i in pyrecon)
    #   Cell i+1 provides its LEFT face   (wR_ip1 or tR_ip1 in pyrecon)
    # ------------------------------------------------------------------

    # (WENO, WENO)
    bv00_R = abs(wL_i - wR_ip1)
    # (THINC, WENO) -- only if THINC valid for cell i
    bv01_R = abs(tL_i - wR_ip1) if ext_ok_i else _INF
    # (WENO, THINC) -- only if THINC valid for cell i+1
    bv10_R = abs(wL_i - tR_ip1) if ext_ok_ip1 else _INF
    # (THINC, THINC) -- only if THINC valid for both
    bv11_R = abs(tL_i - tR_ip1) if (ext_ok_i and ext_ok_ip1) else _INF

    bvs_R = [bv00_R, bv01_R, bv10_R, bv11_R]
    min_bv_R = min(bvs_R)

    # Select winning pair for interface i+1/2
    if bv00_R == min_bv_R:
        sel_R_i, sel_R_ip1 = 1, 1
    elif bv01_R == min_bv_R:
        sel_R_i, sel_R_ip1 = 2, 1
    elif bv10_R == min_bv_R:
        sel_R_i, sel_R_ip1 = 1, 2
    else:
        sel_R_i, sel_R_ip1 = 2, 2

    # Actual BV at i+1/2 with the winning pair
    phi_i_R    = tL_i    if sel_R_i == 2    else wL_i
    phi_ip1_L  = tR_ip1  if sel_R_ip1 == 2  else wR_ip1
    bv_R_signed = phi_i_R - phi_ip1_L

    # ------------------------------------------------------------------
    # Step 4: Conflict resolution for cell i (Eq 7)
    #
    # If cell i gets different assignments from the two interfaces:
    #   If (BV_{i-1/2}) * (BV_{i+1/2}) < 0  ->  use WENO (Phi^<1>)
    #   Otherwise                            ->  use THINC (Phi^<2>)
    #
    # Extrema guard overrides: if extrema guard fails, WENO unconditionally.
    # ------------------------------------------------------------------
    if not ext_ok_i:
        sel_i = 1  # WENO forced by extrema guard
    elif sel_L_i != sel_R_i:
        if bv_L_signed * bv_R_signed < 0.0:
            sel_i = 1  # WENO
        else:
            sel_i = 2  # THINC
    else:
        sel_i = sel_L_i  # both interfaces agree

    # ------------------------------------------------------------------
    # Step 5: Extract face values for cell i (Eq 8)
    # ------------------------------------------------------------------
    if sel_i == 1:
        return wL_i, wR_i
    else:
        return tL_i, tR_i
# ---------------------------------------------------------------------------
# JIT compilation (pynalgo convention)
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    reveal_locals()  # noqa: F821
else:
    _thinc_pair_closed = JIT(_thinc_pair_closed)
    _thinc_face = JIT(_thinc_face)
    _bvd_select_tbv = JIT(_bvd_select_tbv)
    bvd_tbv_fv = JIT(bvd_tbv_fv)
    bvd_fv = JIT(bvd_fv)
#
# :D
#
