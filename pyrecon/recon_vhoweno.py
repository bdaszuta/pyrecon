"""\n ,-*\n(_)\n\n@author: Boris Daszuta\n@SPDX-License-Identifier: BSD-3-Clause\n@function: VHO-WENO9 and VHO-WENO11 reconstruction methods.\n"""
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

from pyrecon._jit_utils import JIT, TYPE_CHECKING
_P = 2
_EPSL = 1e-40

# FV sub-stencil coefficients at x_{i+1/2}
#
# VHO-WENO9: 5-point sub-stencils
_COEFFS9 = (
    (1/5, -21/20, 137/60, -163/60, 137/60),
    (-1/20, 17/60, -43/60, 77/60, 1/5),
    (1/30, -13/60, 47/60, 9/20, -1/20),
    (-1/20, 9/20, 47/60, -13/60, 1/30),
    (1/5, 77/60, -43/60, 17/60, -1/20),
)

# FV optimal linear weights for VHO-WENO9 cascade
_OW9 = (1/126, 10/63, 10/21, 20/63, 5/126)

# JS smoothness matrices for 5-point stencils, denominator 10080
_JS9_DENOM = 10080.0
_JS9_M = (
    (( 45316, -208501,  364863, -288007,   86329),
     (-208501,  965926, -1704396, 1358458, -411487),
     ( 364863, -1704396, 3042786, -2462076,  758823),
     (-288007,  1358458, -2462076, 2041126, -649501),
     (  86329,  -411487,   758823, -649501,  215836)),
    (( 13816,  -60871,   99213,  -70237,   18079),
     ( -60871,  277126, -464976,  337018,  -88297),
     (  99213, -464976,  812586, -611976,  165153),
     ( -70237,  337018, -611976,  485446, -140251),
     (  18079,  -88297,  165153, -140251,   45316)),
    (( 13816,  -51001,   67923,  -38947,    8209),
     ( -51001,  209926, -299076,  179098,  -38947),
     (  67923, -299076,  462306, -299076,   67923),
     ( -38947,  179098, -299076,  209926,  -51001),
     (   8209,  -38947,   67923,  -51001,   13816)),
    (( 45316, -140251,  165153,  -88297,   18079),
     (-140251,  485446, -611976,  337018,  -70237),
     ( 165153, -611976,  812586, -464976,   99213),
     ( -88297,  337018, -464976,  277126,  -60871),
     (  18079,  -70237,   99213,  -60871,   13816)),
    ((215836, -649501,  758823, -411487,   86329),
     (-649501, 2041126, -2462076, 1358458, -288007),
     ( 758823, -2462076, 3042786, -1704396,  364863),
     (-411487,  1358458, -1704396,  965926, -208501),
     (  86329,  -288007,   364863, -208501,   45316)),
)

# ---------------------------------------------------------------------------
# VHO-WENO11 constants
# ---------------------------------------------------------------------------

# Global 9-point FV polynomial coefficients at x_{i+1/2}
# Not used; retained for reference/completeness.
_COEFF9_GLOBAL = (
    1/630, -41/2520, 199/2520, -641/2520, 1879/2520,
    275/504, -61/504, 11/504, -1/504,
)

# Global 11-point FV polynomial coefficients at x_{i+1/2}
_COEFF11_GLOBAL = (
    -1/2772, 61/13860, -703/27720, 371/3960, -7303/27720,
    20417/27720, 15797/27720, -4003/27720, 947/27720,
    -17/3080, 1/2310,
)

# 6-point FV sub-stencil coefficients at x_{i+1/2}
_COEFFS11 = (
    (-1/6, 31/30, -163/60, 79/20, -71/20, 49/20),
    (1/30, -13/60, 37/60, -21/20, 29/20, 1/6),
    (-1/60, 7/60, -23/60, 19/20, 11/30, -1/30),
    (1/60, -2/15, 37/60, 37/60, -2/15, 1/60),
    (-1/30, 11/30, 19/20, -23/60, 7/60, -1/60),
    (1/6, 29/20, -21/20, 37/60, -13/60, 1/30),
)

# JS smoothness matrices for 6-point stencils, denominator 120960
_JS11_DENOM = 120960.0
_JS11_M = (
    ((1152561, -6475092, 14721128, -16959402,  9917175, -2356370),
     (-6475092, 36480687, -83230522, 96298236, -56603394, 13530085),
     (14721128, -83230522, 190757572, -222001952, 131450836, -31697062),
     (-16959402, 96298236, -222001952, 260445372, -155885622, 38103368),
     (9917175, -56603394, 131450836, -155885622, 94851237, -23730232),
     (-2356370, 13530085, -31697062, 38103368, -23730232, 6150211)),
    ((271779, -1507864, 3347304, -3704454,  2033509,  -440274),
     (-1507864, 8449957, -18956662, 21202516, -11755234, 2567287),
     (3347304, -18956662, 43093692, -48919392, 27526876, -6091818),
     (-3704454, 21202516, -48919392, 56662212, -32612122, 7371240),
     (2033509, -11755234, 27526876, -32612122, 19365967, -4558996),
     (-440274, 2567287, -6091818, 7371240, -4558996, 1152561)),
    ((139633, -714988, 1431992, -1396330,   662503,  -122810),
     (-714988, 3824847, -7940202, 7964956, -3863994,  729381),
     (1431992, -7940202, 17195652, -17908832, 8952516, -1731126),
     (-1396330, 7964956, -17908832, 19510972, -10213942, 2043176),
     (662503, -3863994, 8952516, -10213942, 5653317, -1190400),
     (-122810, 729381, -1731126, 2043176, -1190400,  271779)),
    ((271779, -1190400, 2043176, -1731126,   729381,  -122810),
     (-1190400, 5653317, -10213942, 8952516, -3863994,  662503),
     (2043176, -10213942, 19510972, -17908832, 7964956, -1396330),
     (-1731126, 8952516, -17908832, 17195652, -7940202, 1431992),
     (729381, -3863994, 7964956, -7940202, 3824847, -714988),
     (-122810, 662503, -1396330, 1431992, -714988,  139633)),
    ((1152561, -4558996, 7371240, -6091818,  2567287,  -440274),
     (-4558996, 19365967, -32612122, 27526876, -11755234, 2033509),
     (7371240, -32612122, 56662212, -48919392, 21202516, -3704454),
     (-6091818, 27526876, -48919392, 43093692, -18956662, 3347304),
     (2567287, -11755234, 21202516, -18956662, 8449957, -1507864),
     (-440274, 2033509, -3704454, 3347304, -1507864,  271779)),
    ((6150211, -23730232, 38103368, -31697062,  13530085, -2356370),
     (-23730232, 94851237, -155885622, 131450836, -56603394, 9917175),
     (38103368, -155885622, 260445372, -222001952, 96298236, -16959402),
     (-31697062, 131450836, -222001952, 190757572, -83230522, 14721128),
     (13530085, -56603394, 96298236, -83230522, 36480687, -6475092),
     (-2356370, 9917175, -16959402, 14721128, -6475092,  1152561)),
)


# ---------------------------------------------------------------------------
# 2-stencil WENO
# ---------------------------------------------------------------------------

def _weno2(u_a, u_b, beta_a, beta_b, d_a, d_b):
    r"""2-stencil WENO blend: :math:`\tilde{\omega}_a u_a + \tilde{\omega}_b u_b`."""
    alpha_a = d_a / ((_EPSL + beta_a) ** _P)
    alpha_b = d_b / ((_EPSL + beta_b) ** _P)
    inv_sum = 1.0 / (alpha_a + alpha_b)
    return (alpha_a * inv_sum) * u_a + (alpha_b * inv_sum) * u_b


# ---------------------------------------------------------------------------
# VHO-WENO9 cascade
# ---------------------------------------------------------------------------

def _vhoweno9_cascade(vals):
    """Cascade through 5 sub-stencils. Returns face value."""
    vL = []
    bL = []
    for k in range(5):
        sk = [vals[k + j] for j in range(5)]
        s_val = 0.0
        for j in range(5):
            s_val += _COEFFS9[k][j] * sk[j]
        vL.append(s_val)
        m = _JS9_M[k]
        beta = 0.0
        for i in range(5):
            row = m[i]
            vi = sk[i]
            for j in range(5):
                beta += row[j] * vi * sk[j]
        bL.append(beta / _JS9_DENOM)

    q = vL[0]
    beta_q = bL[0]
    denom_ow = _OW9[0]
    for k in range(1, 5):
        ow_prev = denom_ow
        denom_ow += _OW9[k]
        d_prev = ow_prev / denom_ow
        d_new = _OW9[k] / denom_ow
        q = _weno2(q, vL[k], beta_q, bL[k], d_prev, d_new)
        beta_q = d_prev * beta_q + d_new * bL[k]
    return q


def _vhoweno9_LR(u_im4, u_im3, u_im2, u_im1, u_i,
                 u_ip1, u_ip2, u_ip3, u_ip4):
    """VHOWENO9 paired L+R reconstruction via 5-sub-stencil cascade."""
    all_vals = [u_im4, u_im3, u_im2, u_im1, u_i,
                u_ip1, u_ip2, u_ip3, u_ip4]
    uL = _vhoweno9_cascade(all_vals)
    uR = _vhoweno9_cascade(all_vals[::-1])
    return uL, uR


# ---------------------------------------------------------------------------
# VHO-WENO11: global polynomial + sub-stencil safety
# ---------------------------------------------------------------------------

def _vhoweno11_cascade(vals):
    """Returns face value using global 11-pt polynomial in smooth regions,
    falling back to sub-stencil WENO blend near discontinuities."""
    g11 = 0.0
    for j in range(11):
        g11 += _COEFF11_GLOBAL[j] * vals[j]

    vL = []
    bL = []
    for k in range(6):
        sk = [vals[k + j] for j in range(6)]
        s_val = 0.0
        for j in range(6):
            s_val += _COEFFS11[k][j] * sk[j]
        vL.append(s_val)
        m = _JS11_M[k]
        beta = 0.0
        for i in range(6):
            row = m[i]
            vi = sk[i]
            for j in range(6):
                beta += row[j] * vi * sk[j]
        bL.append(beta / _JS11_DENOM)

    # Smooth region: all sub-stencils have similar smoothness
    # (ratio < 10) -> global polynomial is safe, use it directly.
    bg = max(bL)
    bmin = min(bL)
    if bmin > 0 and bg / bmin < 10.0:
        return g11

    # Discontinuity: WENO blend of global + smooth sub-stencils.
    d_g = 1/2
    d_k = (1/2) / 6
    a_all = [d_g / ((_EPSL + bg) ** _P)]
    for b in bL:
        a_all.append(d_k / ((_EPSL + b) ** _P))
    inv_sum = 1.0 / sum(a_all)
    result = a_all[0] * inv_sum * g11
    for k in range(6):
        result += a_all[k + 1] * inv_sum * vL[k]
    return result


def _vhoweno11_LR(u_im5, u_im4, u_im3, u_im2, u_im1, u_i,
                  u_ip1, u_ip2, u_ip3, u_ip4, u_ip5):
    """VHOWENO11 paired L+R reconstruction via 6-sub-stencil cascade."""
    all_vals = [u_im5, u_im4, u_im3, u_im2, u_im1, u_i,
                u_ip1, u_ip2, u_ip3, u_ip4, u_ip5]
    uL = _vhoweno11_cascade(all_vals)
    uR = _vhoweno11_cascade(all_vals[::-1])
    return uL, uR


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def vhoweno9_fv(u_im4, u_im3, u_im2, u_im1, u_i,
                u_ip1, u_ip2, u_ip3, u_ip4):
    r"""VHO-WENO9 reconstruction (FV) at :math:`i+1/2`."""
    return _vhoweno9_LR(u_im4, u_im3, u_im2, u_im1, u_i,
                        u_ip1, u_ip2, u_ip3, u_ip4)


def vhoweno11_fv(u_im5, u_im4, u_im3, u_im2, u_im1, u_i,
                 u_ip1, u_ip2, u_ip3, u_ip4, u_ip5):
    r"""VHO-WENO11 reconstruction (FV) at :math:`i+1/2`."""
    return _vhoweno11_LR(u_im5, u_im4, u_im3, u_im2, u_im1, u_i,
                         u_ip1, u_ip2, u_ip3, u_ip4, u_ip5)
# ---------------------------------------------------------------------------
# JIT compilation (pynalgo convention)
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    reveal_locals()  # noqa: F821
else:
    _weno2 = JIT(_weno2)
    _vhoweno9_cascade = JIT(_vhoweno9_cascade)
    _vhoweno9_LR = JIT(_vhoweno9_LR)
    _vhoweno11_cascade = JIT(_vhoweno11_cascade)
    _vhoweno11_LR = JIT(_vhoweno11_LR)
    vhoweno9_fv = JIT(vhoweno9_fv)
    vhoweno11_fv = JIT(vhoweno11_fv)
#
# :D
#
