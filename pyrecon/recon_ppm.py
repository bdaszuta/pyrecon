"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: PPM (Piecewise Parabolic Method) and PPMX reconstruction
"""
from pyrecon.utils import sign
from pyrecon._jit_utils import JIT, TYPE_CHECKING

# ============================================================================
# Standard PPM (Colella-Woodward)
# ============================================================================

def ppm_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """PPM reconstruction at i+1/2 (FV).

    Piecewise parabolic method on a 5-point stencil with limiting and
    monotonicity enforcement.

    Reference: Colella & Woodward, J. Comput. Phys. 54, 174-201 (1984).
    """
    # Uniform mesh coefficients
    c1 = 0.5
    c2 = 0.5
    c3 = 0.5
    c4 = 0.5
    c5 = 1.0 / 6.0
    c6 = -1.0 / 6.0
    C2 = 1.25

    # --- Step 1: average slopes and interface averages ---
    qa = u_i - u_im1
    qb = u_ip1 - u_i

    dd_im1 = c1 * qa + c2 * (u_im1 - u_im2)
    dd = c1 * qb + c2 * qa
    dd_ip1 = c1 * (u_ip2 - u_ip1) + c2 * qb

    dph = (c3 * u_im1 + c4 * u_i) + (c5 * dd_im1 + c6 * dd)
    dph_ip1 = (c3 * u_i + c4 * u_ip1) + (c5 * dd + c6 * dd_ip1)

    # --- Step 2a: second derivative limiting at interfaces ---
    d2qc_im1 = u_im2 + u_i - 2.0 * u_im1
    d2qc = u_im1 + u_ip1 - 2.0 * u_i
    d2qc_ip1 = u_i + u_ip2 - 2.0 * u_ip1

    # i-1/2 face limiting
    qa_tmp_l = dph - u_im1
    qb_tmp_l = u_i - dph
    qa_l = 3.0 * (u_im1 + u_i - 2.0 * dph)
    qb_l = d2qc_im1
    qc_l = d2qc
    qd_l = 0.0
    if sign(qa_l) == sign(qb_l) and sign(qa_l) == sign(qc_l):
        qd_l = sign(qa_l) * min(C2 * abs(qb_l),
                                 min(C2 * abs(qc_l), abs(qa_l)))
    if qa_tmp_l * qb_tmp_l < 0.0:
        dph = 0.5 * (u_im1 + u_i) - qd_l / 6.0

    # i+1/2 face limiting
    qa_tmp_r = dph_ip1 - u_i
    qb_tmp_r = u_ip1 - dph_ip1
    qa_r = 3.0 * (u_i + u_ip1 - 2.0 * dph_ip1)
    qb_r = d2qc
    qc_r = d2qc_ip1
    qd_r = 0.0
    if sign(qa_r) == sign(qb_r) and sign(qa_r) == sign(qc_r):
        qd_r = sign(qa_r) * min(C2 * abs(qb_r),
                                 min(C2 * abs(qc_r), abs(qa_r)))
    if qa_tmp_r * qb_tmp_r < 0.0:
        dph_ip1 = 0.5 * (u_i + u_ip1) - qd_r / 6.0

    # Second derivative of parabolic interpolant
    d2qf = 6.0 * (dph + dph_ip1 - 2.0 * u_i)

    # --- Step 2b: cache Riemann states ---
    qminus = dph
    qplus = dph_ip1

    # --- Step 3: cell-centered differences ---
    dqf_minus = u_i - qminus
    dqf_plus = qplus - u_i

    # --- Step 4a: CS limiters ---
    qa_tmp = dqf_minus * dqf_plus
    qb_tmp = (u_ip1 - u_i) * (u_i - u_im1)

    # Second derivative smoothness check
    qe = 0.0
    if (sign(d2qc_im1) == sign(d2qc) and
        sign(d2qc_im1) == sign(d2qc_ip1) and
        sign(d2qc_im1) == sign(d2qf)):
        qe = sign(d2qf) * min(
            min(C2 * abs(d2qc_im1), C2 * abs(d2qc)),
            min(C2 * abs(d2qc_ip1), abs(d2qf)))

    # Roundoff guard
    qa_ro = max(abs(u_im1), abs(u_im2))
    qb_ro = max(max(abs(u_i), abs(u_ip1)), abs(u_ip2))

    rho = 0.0
    if abs(d2qf) > 1e-12 * max(qa_ro, qb_ro):
        rho = qe / d2qf

    tmp_m = u_i - rho * dqf_minus
    tmp_p = u_i + rho * dqf_plus
    tmp2_m = u_i - 2.0 * dqf_plus
    tmp2_p = u_i + 2.0 * dqf_minus

    if qa_tmp <= 0.0 or qb_tmp <= 0.0:
        if rho <= (1.0 - 1e-12):
            qminus = tmp_m
            qplus = tmp_p
    else:
        if abs(dqf_minus) >= 2.0 * abs(dqf_plus):
            qminus = tmp2_m
        if abs(dqf_plus) >= 2.0 * abs(dqf_minus):
            qplus = tmp2_p

    # --- Step 5: convert to L/R Riemann states ---
    uL = qplus   # u_{i+1/2}^-
    uR = qminus  # u_{i-1/2}^+

    return uL, uR


# ============================================================================
# PPMX (Colella-Sekora extremum-preserving)
# ============================================================================

def ppmx_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """PPMX reconstruction at i+1/2 (FV).

    Implements the Colella-Sekora extremum-preserving parabolic
    reconstruction on a 5-point stencil.
    """
    # --- initial 4th-order interface interpolation (CS eqn 16) ---
    qlv = (7.0 * (u_i + u_im1) - (u_im2 + u_ip1)) / 12.0
    qrv = (7.0 * (u_i + u_ip1) - (u_im1 + u_ip2)) / 12.0

    # --- CS limiters at left face (i-1/2) ---
    d2qc = 3.0 * ((u_im1 + u_i) - 2.0 * qlv)
    d2ql = (u_im2 + u_i) - 2.0 * u_im1
    d2qr = (u_im1 + u_ip1) - 2.0 * u_i

    d2qlim = 0.0
    lim_slope = min(abs(d2ql), abs(d2qr))
    if d2qc > 0.0 and d2ql > 0.0 and d2qr > 0.0:
        d2qlim = sign(d2qc) * min(1.25 * lim_slope, abs(d2qc))
    if d2qc < 0.0 and d2ql < 0.0 and d2qr < 0.0:
        d2qlim = sign(d2qc) * min(1.25 * lim_slope, abs(d2qc))
    if ((u_im1 - qlv) * (u_i - qlv)) > 0.0:
        qlv = 0.5 * (u_i + u_im1) - d2qlim / 6.0

    # --- CS limiters at right face (i+1/2) ---
    d2qc = 3.0 * ((u_i + u_ip1) - 2.0 * qrv)
    d2ql = d2qr  # reuse previous d2qr = (u_im1 + u_ip1) - 2*u_i
    d2qr = (u_i + u_ip2) - 2.0 * u_ip1

    d2qlim = 0.0
    lim_slope = min(abs(d2ql), abs(d2qr))
    if d2qc > 0.0 and d2ql > 0.0 and d2qr > 0.0:
        d2qlim = sign(d2qc) * min(1.25 * lim_slope, abs(d2qc))
    if d2qc < 0.0 and d2ql < 0.0 and d2qr < 0.0:
        d2qlim = sign(d2qc) * min(1.25 * lim_slope, abs(d2qc))
    if ((u_i - qrv) * (u_ip1 - qrv)) > 0.0:
        qrv = 0.5 * (u_i + u_ip1) - d2qlim / 6.0

    # --- extrema detection and handling ---
    qa = (qrv - u_i) * (u_i - qlv)
    qb = (u_im1 - u_i) * (u_i - u_ip1)
    if qa <= 0.0 or qb <= 0.0:
        d2q = 6.0 * (qlv + qrv - 2.0 * u_i)
        d2qc2 = (u_im1 + u_ip1) - 2.0 * u_i
        d2ql2 = (u_im2 + u_i) - 2.0 * u_im1
        d2qr2 = (u_i + u_ip2) - 2.0 * u_ip1

        d2qlim = 0.0
        lim_slope = min(min(abs(d2ql2), abs(d2qr2)), abs(d2qc2))
        if (d2qc2 > 0.0 and d2ql2 > 0.0 and d2qr2 > 0.0 and d2q > 0.0):
            d2qlim = sign(d2q) * \
                min(1.25 * lim_slope, abs(d2q))
        if (d2qc2 < 0.0 and d2ql2 < 0.0 and d2qr2 < 0.0 and d2q < 0.0):
            d2qlim = sign(d2q) * \
                min(1.25 * lim_slope, abs(d2q))

        rho = 0.0
        if abs(d2q) > 1.0e-12 * max(abs(u_im1),
                                     max(abs(u_i), abs(u_ip1))):
            rho = d2qlim / d2q
        qlv = u_i + (qlv - u_i) * rho
        qrv = u_i + (qrv - u_i) * rho
    else:
        qc = qrv - u_i
        qd = qlv - u_i
        if abs(qc) >= 2.0 * abs(qd):
            qrv = u_i - 2.0 * qd
        if abs(qd) >= 2.0 * abs(qc):
            qlv = u_i - 2.0 * qc

    # uL = u_{i+1/2}^- (right face of cell i) = qrv
    # uR = u_{i-1/2}^+ (left face of cell i) = qlv
    uL = qrv
    uR = qlv

    return uL, uR
# ---------------------------------------------------------------------------
# JIT compilation (pynalgo convention)
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    reveal_locals()  # noqa: F821
else:
    ppm_fv = JIT(ppm_fv)
    ppmx_fv = JIT(ppmx_fv)
#
# :D
#
