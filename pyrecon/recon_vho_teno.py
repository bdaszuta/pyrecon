"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: VHO-TENO-AA reconstruction (Fu 2021, arXiv:2109.14340)
"""
from pyrecon._jit_utils import JIT, TYPE_CHECKING

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPSL = 1e-40
_Q = 7

# Small-stencil optimal weights [Sec 4.2]
_D_SMALL = (0.5065006634, 0.3699651429, 0.1235341937)

# Adaptive C_T parameters [Eq 3.7-3.9]
_ALPHA1 = 14.0
_ALPHA2 = 6.4
_CR = 0.265

# Large-stencil PW flux coefficients via Lagrange interpolation at x_{i+1/2}
# Evaluated analytically (exact rationals evaluated at runtime as float).
_FLUX_S3 = (3/256, -25/256, 75/128, 75/128, -25/256, 3/256)
_FLUX_S4 = (-5/2048, 49/2048, -245/2048, 1225/2048,
            1225/2048, -245/2048, 49/2048, -5/2048)
_FLUX_S5 = (35/65536, -405/65536, 2268/65536, -8820/65536, 39690/65536,
            39690/65536, -8820/65536, 2268/65536, -405/65536, 35/65536)

# Smoothness indicator matrices [Eq 4.12]

_JS_S3_DENOM = 120960.0
_JS_S3_M = (
    (271779.0, -1190400.0, 2043176.0, -1731126.0, 729381.0, -122810.0),
    (-1190400.0, 5653317.0, -10213942.0, 8952516.0, -3863994.0, 662503.0),
    (2043176.0, -10213942.0, 19510972.0, -17908832.0, 7964956.0, -1396330.0),
    (-1731126.0, 8952516.0, -17908832.0, 17195652.0, -7940202.0, 1431992.0),
    (729381.0, -3863994.0, 7964956.0, -7940202.0, 3824847.0, -714988.0),
    (-122810.0, 662503.0, -1396330.0, 1431992.0, -714988.0, 139633.0),
)

_JS_S4_DENOM = 62270208000.0
_JS_S4_M = (
    (139164877641.0, -899924254832.0, 2458417783421.0,
     -3683162871400.0, 3276540273915.0, -1735578339536.0,
     508082860927.0, -63540330136.0),
    (-899924254832.0, 6047605530599.0, -17032830822632.0,
     26139834406835.0, -23703767206320.0, 12752830987157.0,
     -3781934290104.0, 478185649297.0),
    (2458417783421.0, -17032830822632.0, 49429163447121.0,
     -77830234197760.0, 72081969734455.0, -39457900025976.0,
     11870432980667.0, -1519018899296.0),
    (-3683162871400.0, 26139834406835.0, -77830234197760.0,
     125743620342175.0, -119183144250200.0, 66508955457625.0,
     -20333087333760.0, 2637218446485.0),
    (3276540273915.0, -23703767206320.0, 72081969734455.0,
     -119183144250200.0, 115593819531025.0, -65869291684240.0,
     20504404216445.0, -2700530615080.0),
    (-1735578339536.0, 12752830987157.0, -39457900025976.0,
     66508955457625.0, -65869291684240.0, 38342902371231.0,
     -12173507874152.0, 1631589107891.0),
    (508082860927.0, -3781934290104.0, 11870432980667.0,
     -20333087333760.0, 20504404216445.0, -12173507874152.0,
     3944861897609.0, -539252457632.0),
    (-63540330136.0, 478185649297.0, -1519018899296.0,
     2637218446485.0, -2700530615080.0, 1631589107891.0,
     -539252457632.0, 75349098471.0),
)

_JS_S5_DENOM = 8002967132160000.0
_JS_S5_M = (
    (18029727887840089.0, -154191170123981946.0, 579881530239731994.0,
     -1259189860681305546.0, 1741175680141321884.0, -1591710014995264014.0,
     963183779965705446.0, -372517050611008494.0, 83658804107311371.0,
     -8321425930350784.0),
    (-154191170123981946.0, 1348293007163766699.0, -5169093401026572426.0,
     11411895694681499274.0, -16006619439244371726.0, 14814121402146046896.0,
     -9061444215837859374.0, 3538085493481756326.0, -801380092073068224.0,
     80332720832784501.0),
    (579881530239731994.0, -5169093401026572426.0, 20182234533315641784.0,
     -45304379441983648656.0, 64492774694022421404.0, -60470122540856933844.0,
     37413272471051763576.0, -14756096212381589184.0, 3372296778652026906.0,
     -340768411032841554.0),
    (-1259189860681305546.0, 11411895694681499274.0, -45304379441983648656.0,
     103357453544031038184.0, -149366368092063276756.0, 141967424853963770796.0,
     -88906070411999596224.0, 35443242847776072456.0, -8177320085291548434.0,
     833310951566994906.0),
    (1741175680141321884.0, -16006619439244371726.0, 64492774694022421404.0,
     -149366368092063276756.0, 219076505295455164644.0,
     -211148670815826053184.0,
     133920085124400917076.0, -54000630043378140204.0, 12585922179914138256.0,
     -1294174583422121394.0),
    (-1591710014995264014.0, 14814121402146046896.0, -60470122540856933844.0,
     141967424853963770796.0, -211148670815826053184.0, 206327958083635665564.0,
     -132580615621471812396.0, 54106125255941087844.0, -12748157290006381746.0,
     1323646687469874084.0),
    (963183779965705446.0, -9061444215837859374.0, 37413272471051763576.0,
     -88906070411999596224.0, 133920085124400917076.0, -132580615621471812396.0,
     86303482562247542424.0, -35659213408419764976.0, 8499098531337045174.0,
     -891778811273940726.0),
    (-372517050611008494.0, 3538085493481756326.0, -14756096212381589184.0,
     35443242847776072456.0, -54000630043378140204.0, 54106125255941087844.0,
     -35659213408419764976.0, 14917582508640112584.0, -3598451878580785206.0,
     381872487532258854.0),
    (83658804107311371.0, -801380092073068224.0, 3372296778652026906.0,
     -8177320085291548434.0, 12585922179914138256.0, -12748157290006381746.0,
     8499098531337045174.0, -3598451878580785206.0, 878723894536476309.0,
     -94390842595214406.0),
    (-8321425930350784.0, 80332720832784501.0, -340768411032841554.0,
     833310951566994906.0, -1294174583422121394.0, 1323646687469874084.0,
     -891778811273940726.0, 381872487532258854.0, -94390842595214406.0,
     10271226852556519.0),
)

# Map stencil size to (M, denom)
_JS_LARGE_6 = (_JS_S3_M, _JS_S3_DENOM)
_JS_LARGE_8 = (_JS_S4_M, _JS_S4_DENOM)
_JS_LARGE_10 = (_JS_S5_M, _JS_S5_DENOM)

# ---------------------------------------------------------------------------
# Small-stencil JS smoothness [Eq 4.12]
# ---------------------------------------------------------------------------

def _js_small(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """JS smoothness indicators for 3-point small stencils (PW).
    Returns (b0, b1, b2)."""
    b0 = (0.25 * (u_im1 - u_ip1) ** 2
          + (13.0 / 12.0) * (u_im1 - 2.0 * u_i + u_ip1) ** 2)
    b1 = (0.25 * (3.0 * u_i - 4.0 * u_ip1 + u_ip2) ** 2
          + (13.0 / 12.0) * (u_i - 2.0 * u_ip1 + u_ip2) ** 2)
    b2 = (0.25 * (u_im2 - 4.0 * u_im1 + 3.0 * u_i) ** 2
          + (13.0 / 12.0) * (u_im2 - 2.0 * u_im1 + u_i) ** 2)
    return b0, b1, b2


# ---------------------------------------------------------------------------
# Large-stencil JS smoothness (via precomputed quadratic form)
# ---------------------------------------------------------------------------

def _js_large(M, denom, vals):
    r"""Evaluate :math:`\beta = v^T M v / \mathrm{denom}` where vals = stencil values."""
    s = 0.0
    for i in range(len(M)):
        mi = M[i]
        vi = vals[i]
        for j in range(len(mi)):
            s += mi[j] * vi * vals[j]
    return s / float(denom)


# ---------------------------------------------------------------------------
# Stencil flux polynomials
# ---------------------------------------------------------------------------

def _stencils_small_L(u_im2, u_im1, u_i, u_ip1, u_ip2):
    r"""Small-stencil fluxes at :math:`x_{i+1/2}` [Eq 4.13]. Returns (f0,f1,f2)."""
    f0 = (-u_im1 + 5.0 * u_i + 2.0 * u_ip1) / 6.0
    f1 = (2.0 * u_i + 5.0 * u_ip1 - u_ip2) / 6.0
    f2 = (2.0 * u_im2 - 7.0 * u_im1 + 11.0 * u_i) / 6.0
    return f0, f1, f2


def _stencils_small_R(u_im2, u_im1, u_i, u_ip1, u_ip2):
    r"""Small-stencil fluxes at :math:`x_{i-1/2}` (right-biased).

    Returns (f0_R, f1_R, f2_R) in order matching (b0,b1,b2) from _js_small
    called with reversed args (u_ip2, u_ip1, u_i, u_im1, u_im2):
      f0_R: CENTER stencil :math:`[i-1,i,i+1]`
      f1_R: LEFT stencil :math:`[i-2,i-1,i]`
      f2_R: RIGHT stencil :math:`[i,i+1,i+2]`
    """
    # Equivalent to L-face stencils on reversed arguments.
    f0 = (2.0 * u_im1 + 5.0 * u_i - u_ip1) / 6.0
    f1 = (-u_i + 5.0 * u_im1 + 2.0 * u_im2) / 6.0
    f2 = (2.0 * u_ip2 - 7.0 * u_ip1 + 11.0 * u_i) / 6.0
    return f0, f1, f2


def _stencil_large(coeffs, vals):
    """Large-stencil flux: sum coeffs[k] * vals[k]."""
    s = 0.0
    for c, v in zip(coeffs, vals):
        s += c * v
    return s


# ---------------------------------------------------------------------------
# Scale separation and adaptive C_T [Eq 3.1, 3.3-3.9]
# ---------------------------------------------------------------------------

def _scale_separation(betas):
    r""":math:`\gamma_k = 1/(\beta_k + \varepsilon)^7`, normalize to :math:`\chi_k`."""
    gamma = [1.0 / (b + _EPSL) ** _Q for b in betas]
    inv_sum = 1.0 / sum(gamma)
    return [g * inv_sum for g in gamma]


def _adaptive_CT_full(f_jm2, f_jm1, f_j, f_jp1, f_jp2, f_jp3):
    r"""Adaptive C_T with all six values for full eta computation.
    :math:`\eta_{j+1/2} = \min(\eta_{j-1}, \eta_j, \eta_{j+1}, \eta_{j+2})`"""
    def _eta_single(fa, fb, fc):
        df_m = fb - fa
        df_p = fc - fb
        num = abs(2.0 * df_m * df_p) + _EPSL
        den = df_m * df_m + df_p * df_p + _EPSL
        return num / den
    e_m1 = _eta_single(f_jm2, f_jm1, f_j)
    e_0 = _eta_single(f_jm1, f_j, f_jp1)
    e_1 = _eta_single(f_j, f_jp1, f_jp2)
    e_2 = _eta_single(f_jp1, f_jp2, f_jp3)
    eta_face = min(e_m1, e_0, e_1, e_2)
    m = 1.0 - min(1.0, eta_face / _CR)
    g = (1.0 - m) ** 4 * (1.0 + 4.0 * m)
    beta = _ALPHA1 - _ALPHA2 * (1.0 - g)
    exponent = int(beta)
    return 10.0 ** (-exponent)


# ---------------------------------------------------------------------------
# Recursive stencil selection core [Sec 3.3]
# ---------------------------------------------------------------------------

def _select_face(betas_small, betas_large, f_small, f_large,
                 optimw_small, C_T):
    """Recursive hierarchical stencil selection for one face.

    Tries large stencils from largest to smallest.
    Falls back to weighted combination of small stencils.
    If all small stencils rejected, returns f_small[1] (centered).
    """
    # Try large stencils, largest first
    for idx in range(len(betas_large) - 1, -1, -1):
        combined = betas_small + (betas_large[idx],)
        chi = _scale_separation(combined)
        if chi[-1] >= C_T:
            return f_large[idx]

    # Fallback: weighted combination of small stencils
    chi = _scale_separation(betas_small)
    delta = [1.0 if c >= C_T else 0.0 for c in chi]
    denom = (optimw_small[0] * delta[0]
             + optimw_small[1] * delta[1]
             + optimw_small[2] * delta[2])
    if denom < _EPSL:
        return f_small[1]
    inv = 1.0 / denom
    return inv * (optimw_small[0] * delta[0] * f_small[0]
                  + optimw_small[1] * delta[1] * f_small[1]
                  + optimw_small[2] * delta[2] * f_small[2])


# ---------------------------------------------------------------------------
# VHO-TENO8-AA: 9-point symmetric window [i-4 .. i+4]
# L-face: S4(8pt) + S3(6pt).  R-face: S4(8pt) + S3(6pt).
# Symmetric: both faces get both large stencils.
# ---------------------------------------------------------------------------

def vho_teno8_aa_pw(u_im4, u_im3, u_im2, u_im1, u_i,
                    u_ip1, u_ip2, u_ip3, u_ip4):
    """VHO-TENO8-AA reconstruction (PW, 9-point symmetric stencil).

    Reference: Fu 2021, arXiv:2109.14340.
    """
    # --- L-face (x_{i+1/2}^-) ---
    b0, b1, b2 = _js_small(u_im2, u_im1, u_i, u_ip1, u_ip2)
    f0, f1, f2 = _stencils_small_L(u_im2, u_im1, u_i, u_ip1, u_ip2)

    vals_S3 = (u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3)
    vals_S4 = (u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3, u_ip4)
    M_S3, d_S3 = _JS_LARGE_6
    M_S4, d_S4 = _JS_LARGE_8
    b3 = _js_large(M_S3, d_S3, vals_S3)
    b4 = _js_large(M_S4, d_S4, vals_S4)
    f3 = _stencil_large(_FLUX_S3, vals_S3)
    f4 = _stencil_large(_FLUX_S4, vals_S4)

    C_T = _adaptive_CT_full(u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3)
    uL = _select_face((b0, b1, b2), [b3, b4], (f0, f1, f2), [f3, f4],
                      _D_SMALL, C_T)

    # --- R-face (x_{i-1/2}^+) ---
    b0_R, b1_R, b2_R = _js_small(u_ip2, u_ip1, u_i, u_im1, u_im2)
    betas_small_R = (b0_R, b1_R, b2_R)
    f0_R, f1_R, f2_R = _stencils_small_R(u_im2, u_im1, u_i, u_ip1, u_ip2)

    # R-face large stencils (shifted left by 1)
    vals_S3_R = (u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2)
    vals_S4_R = (u_im4, u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3)
    b3_R = _js_large(M_S3, d_S3, vals_S3_R)
    b4_R = _js_large(M_S4, d_S4, vals_S4_R)
    f3_R = _stencil_large(_FLUX_S3, vals_S3_R)
    f4_R = _stencil_large(_FLUX_S4, vals_S4_R)

    C_T_R = _adaptive_CT_full(u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2)
    uR = _select_face(betas_small_R, [b3_R, b4_R],
                      (f0_R, f1_R, f2_R), [f3_R, f4_R],
                      _D_SMALL, C_T_R)

    return uL, uR


# ---------------------------------------------------------------------------
# VHO-TENO10-AA: 11-point symmetric window [i-5 .. i+5]
# L-face: S5(10pt) + S4(8pt) + S3(6pt).
# R-face: S5(10pt) + S4(8pt) + S3(6pt).
# Symmetric: both faces get all three large stencils.
# ---------------------------------------------------------------------------

def vho_teno10_aa_pw(u_im5, u_im4, u_im3, u_im2, u_im1, u_i,
                     u_ip1, u_ip2, u_ip3, u_ip4, u_ip5):
    """VHO-TENO10-AA reconstruction (PW, 11-point symmetric stencil).

    Reference: Fu 2021, arXiv:2109.14340.
    """
    # --- L-face (x_{i+1/2}^-) ---
    b0, b1, b2 = _js_small(u_im2, u_im1, u_i, u_ip1, u_ip2)
    f0, f1, f2 = _stencils_small_L(u_im2, u_im1, u_i, u_ip1, u_ip2)

    vals_S3 = (u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3)
    vals_S4 = (u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3, u_ip4)
    vals_S5 = (u_im4, u_im3, u_im2, u_im1, u_i,
               u_ip1, u_ip2, u_ip3, u_ip4, u_ip5)

    M_S3, d_S3 = _JS_LARGE_6
    M_S4, d_S4 = _JS_LARGE_8
    M_S5, d_S5 = _JS_LARGE_10

    b3 = _js_large(M_S3, d_S3, vals_S3)
    b4 = _js_large(M_S4, d_S4, vals_S4)
    b5 = _js_large(M_S5, d_S5, vals_S5)

    f3 = _stencil_large(_FLUX_S3, vals_S3)
    f4 = _stencil_large(_FLUX_S4, vals_S4)
    f5 = _stencil_large(_FLUX_S5, vals_S5)

    C_T = _adaptive_CT_full(u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3)
    uL = _select_face((b0, b1, b2), [b3, b4, b5],
                      (f0, f1, f2), [f3, f4, f5],
                      _D_SMALL, C_T)

    # --- R-face (x_{i-1/2}^+) ---
    b0_R, b1_R, b2_R = _js_small(u_ip2, u_ip1, u_i, u_im1, u_im2)
    betas_small_R = (b0_R, b1_R, b2_R)
    f0_R, f1_R, f2_R = _stencils_small_R(u_im2, u_im1, u_i, u_ip1, u_ip2)

    # R-face large stencils (shifted left by 1)
    vals_S3_R = (u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2)
    vals_S4_R = (u_im4, u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3)
    vals_S5_R = (u_im5, u_im4, u_im3, u_im2, u_im1, u_i,
                 u_ip1, u_ip2, u_ip3, u_ip4)
    b3_R = _js_large(M_S3, d_S3, vals_S3_R)
    b4_R = _js_large(M_S4, d_S4, vals_S4_R)
    b5_R = _js_large(M_S5, d_S5, vals_S5_R)
    f3_R = _stencil_large(_FLUX_S3, vals_S3_R)
    f4_R = _stencil_large(_FLUX_S4, vals_S4_R)
    f5_R = _stencil_large(_FLUX_S5, vals_S5_R)

    C_T_R = _adaptive_CT_full(u_im3, u_im2, u_im1, u_i, u_ip1, u_ip2)
    uR = _select_face(betas_small_R, [b3_R, b4_R, b5_R],
                      (f0_R, f1_R, f2_R), [f3_R, f4_R, f5_R],
                      _D_SMALL, C_T_R)

    return uL, uR
# ---------------------------------------------------------------------------
# JIT compilation (pynalgo convention)
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    reveal_locals()  # noqa: F821
else:
    _js_small = JIT(_js_small)
    _js_large = JIT(_js_large)
    _stencils_small_L = JIT(_stencils_small_L)
    _stencils_small_R = JIT(_stencils_small_R)
    _stencil_large = JIT(_stencil_large)
    _scale_separation = JIT(_scale_separation)
    _adaptive_CT_full = JIT(_adaptive_CT_full)
    _select_face = JIT(_select_face)
    vho_teno8_aa_pw = JIT(vho_teno8_aa_pw)
    vho_teno10_aa_pw = JIT(vho_teno10_aa_pw)
#
# :D
#

