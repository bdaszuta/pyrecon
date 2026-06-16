"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Reconstruction interface: method registry, dispatcher, array wrapper
"""
import inspect
from pyrecon.recon_donate import donate_fv
from pyrecon.recon_linear import lin_vl_fv, lin_mc2_fv
from pyrecon.recon_weno5 import (
    weno5_fv, weno5z_fv, weno5d_si_fv, weno5_pw, weno5z_pw, weno5d_si_pw,
)
from pyrecon.recon_weno3 import weno3_fv, weno3_pw, weno3z_fv, weno3z_pw
from pyrecon.recon_weno7 import weno7_fv, weno7z_fv, weno7_pw, weno7z_pw
from pyrecon.recon_weno_ao53 import weno_ao53_fv, weno_ao53_pw
from pyrecon.recon_weno5_ext import (
    weno5zcplus_fv, weno5z_ns_fv, weno5zp_fv,
    weno5_ha_js_fv, weno5cz_fv, weno5_bc_fv,
)
from pyrecon.recon_mpn import mp3_fv, mp5_fv, mp7_fv, mp5_r_fv
from pyrecon.recon_ceno import ceno3_fv, ceno5_fv
from pyrecon.recon_cteno import cteno5_fv, cteno5z_fv
from pyrecon.recon_cweno import (
    cweno3_fv, cweno5_fv, central_weno_fv, cweno5_capdeville_fv,
    cweno_z3_fv, cweno_z5_fv,
)
from pyrecon.recon_ppm import ppm_fv, ppmx_fv
from pyrecon.recon_lag6 import lag6_fv, lag6_pw
from pyrecon.recon_teno5 import (
    teno5_fv, teno5_pw, teno5_mc2_fv, teno5_mc2_pw,
    teno5_koren_fv, teno5_koren_pw,
)
from pyrecon.recon_teno_a import teno_a_fv, teno_a_pw
from pyrecon.recon_teno_hybrid import teno_hybrid_fv, teno_hybrid_pw
from pyrecon.recon_vho_teno import vho_teno8_aa_pw, vho_teno10_aa_pw
from pyrecon.recon_teno_thinc import teno_thinc_fv, teno_thinc_pw
from pyrecon.recon_thinc_bvd import thinc_bvd_fv, thinc_bvd_pw
from pyrecon.recon_teno_m import teno_m_va_fv, teno_m_tvd5_fv, teno_m_mp_fv
from pyrecon.recon_esweno3 import esweno3_fv, esweno3_pw
from pyrecon.recon_aweno import aweno5_fv
from pyrecon.recon_bvd import bvd_fv, bvd_tbv_fv
from pyrecon.recon_bvd_cu import bvd_cu_fv, bvd_e6_mp5_fv
from pyrecon.recon_adweno import adweno5_fv
from pyrecon.recon_mood import mood_fv
from pyrecon.recon_mp_mp import mp5_mp_fv
from pyrecon.recon_esweno5 import weno5z_p2_fv
from pyrecon.recon_lowdiss import (
    hybrid_linear_weno_fv, hybrid_linear_weno_mild_fv,
    hybrid_linear_weno_strong_fv,
)
from pyrecon.recon_hybrid_mp import hybrid_mp_mc2_fv
from pyrecon.recon_entropy_stable import (
    es_scalar_quad_fv, es_scalar_log_fv, es_scalar_cubic_fv,
    es_scalar_quad_pw, es_scalar_log_pw, es_scalar_cubic_pw,
)
from pyrecon.recon_geno import geno5_fv
from pyrecon.recon_round import round_a_fv, round_b_fv, round_c_fv
from pyrecon.recon_vhoweno import vhoweno9_fv, vhoweno11_fv
from pyrecon.recon_lsweno5h import lsweno5h_fv
from pyrecon.recon_weno5z_sp import weno5z_sp_fv, weno5z_sp_pw
from pyrecon.recon_weno5m import weno5m_fv, weno5m_pw
from pyrecon.recon_lsweno5hp import lsweno5hp_fv
from pyrecon.recon_enomr import eno_mr3_fv, eno_mr5_fv, eno_mr7_fv
from pyrecon.recon_wenoc import wenoc5_fv, wenoc5z_fv, wenoc7_fv, wenoc7z_fv

# ---------------------------------------------------------------------------
# Method registry: name -> (function, stencil_width, description)
# All canonical names use _fv (finite volume) or _pw (pointwise) suffix.
# ---------------------------------------------------------------------------

_METHODS = {
    "donate_fv":      (donate_fv, 5, "First-order donor cell (FV)"),
    "lin_vl_fv":      (lin_vl_fv, 5, "Linear van Leer minmod limiter (FV)"),
    "lin_mc2_fv":     (lin_mc2_fv, 5, "Linear MC2 monotonized central (FV)"),
    "weno5_fv":       (weno5_fv, 5, "WENO5-JS (FV)"),
    "weno5z_fv":      (weno5z_fv, 5, "WENO5-Z (FV)"),
    "weno5d_si_fv":   (weno5d_si_fv, 5, "WENO5-D-SI (FV)"),
    "weno5m_fv":      (weno5m_fv, 5, "WENO5-M mapped (FV)"),
    "weno5m_pw":      (weno5m_pw, 5, "WENO5-M mapped (PW)"),
    "weno5_pw":       (weno5_pw, 5, "WENO5-JS (PW)"),
    "weno5z_pw":      (weno5z_pw, 5, "WENO5-Z (PW)"),
    "weno5d_si_pw":   (weno5d_si_pw, 5, "WENO5-D-SI (PW)"),
    "weno5zcplus_fv": (weno5zcplus_fv, 5, "WENO5-ZC+ (FV)"),
    "weno5z_ns_fv":   (weno5z_ns_fv, 5, "WENO5-Z-NS (FV)"),
    "weno5zp_fv":     (weno5zp_fv, 5, "WENO5-ZP p=2 with sign checks (FV)"),
    "weno5_ha_js_fv": (weno5_ha_js_fv, 5, "WENO5-Ha-JS (FV)"),
    "weno5cz_fv":     (weno5cz_fv, 5, "WENO5-CZ (FV)"),
    "weno5_bc_fv":    (weno5_bc_fv, 5, "WENO5-BC (FV)"),
    "weno3_fv":       (weno3_fv, 3, "WENO3-JS (FV)"),
    "weno3_pw":       (weno3_pw, 3, "WENO3-JS (PW)"),
    "weno3z_fv":      (weno3z_fv, 3, "WENO3-Z (FV)"),
    "weno3z_pw":      (weno3z_pw, 3, "WENO3-Z (PW)"),
    "weno7_fv":       (weno7_fv, 7, "WENO7-JS (FV)"),
    "weno7z_fv":      (weno7z_fv, 7, "WENO7-Z (FV)"),
    "weno7_pw":       (weno7_pw, 7, "WENO7-JS (PW)"),
    "weno7z_pw":      (weno7z_pw, 7, "WENO7-Z (PW)"),
    "weno_ao53_fv":   (weno_ao53_fv, 5, "WENO-AO(5,3) (FV)"),
    "weno_ao53_pw":   (weno_ao53_pw, 5, "WENO-AO(5,3) (PW)"),
    "mp3_fv":         (mp3_fv, 5, "MP3 (FV)"),
    "mp5_fv":         (mp5_fv, 5, "MP5 (FV)"),
    "mp7_fv":         (mp7_fv, 7, "MP7 (FV)"),
    "mp5_r_fv":       (mp5_r_fv, 5, "MP5-R (FV)"),
    # --- BVD / MOOD / MP-variant literature methods ---
    "bvd_cu_fv":      (bvd_cu_fv, 7, "BVD central-upwind HOCUS7 (FV)"),
    "bvd_e6_mp5_fv":  (bvd_e6_mp5_fv, 7,
                       "BVD explicit 6th-order + MP5 alpha=4 (FV)"),
    "mood_fv":        (mood_fv, 5,
                       "MOOD multi-dimensional optimal order detection (FV)"),
    "mp5_mp_fv":      (mp5_mp_fv, 5, "MP5 modified multi-phase (FV)"),
    "ceno3_fv":       (ceno3_fv, 5, "CENO3 (FV)"),
    "ceno5_fv":       (ceno5_fv, 7, "CENO5 (FV)"),
    "cweno3_fv":      (cweno3_fv, 5, "CWENO3 Cravero 2017 (FV)"),
    "cweno5_fv":      (cweno5_fv, 7, "CWENO5 Cravero 2017 (FV)"),
    "cweno_z3_fv":    (cweno_z3_fv, 5, "CWENO3-Z WENO-Z-like variant (FV)"),
    "cweno_z5_fv":    (cweno_z5_fv, 7, "CWENO5-Z WENO-Z-like variant (FV)"),
    "cweno5_capdeville_fv": (cweno5_capdeville_fv, 5,
                       "CWENO5 Capdeville (2008) (FV)"),
    "central_weno_fv":(central_weno_fv, 3, "Central WENO (FV)"),
    "ppm_fv":         (ppm_fv, 5, "PPM (FV)"),
    "ppmx_fv":        (ppmx_fv, 5, "PPMX (FV)"),
    "lag6_fv":        (lag6_fv, 6, "6th-order Lagrange polynomial (FV)"),
    "lag6_pw":        (lag6_pw, 6, "6th-order Lagrange polynomial (PW)"),
    "teno5_fv":       (teno5_fv, 5, "TENO5 (FV)"),
    "teno5_pw":       (teno5_pw, 5, "TENO5 (PW)"),
    "teno5_mc2_fv":   (teno5_mc2_fv, 5, "TENO5-MC2 (FV)"),
    "teno5_mc2_pw":   (teno5_mc2_pw, 5, "TENO5-MC2 (PW)"),
    "teno5_koren_fv": (teno5_koren_fv, 5, "TENO5-Koren (FV)"),
    "teno5_koren_pw": (teno5_koren_pw, 5, "TENO5-Koren (PW)"),
    "teno_a_fv":      (teno_a_fv, 5, "TENO-A adaptive CT (FV)"),
    "teno_a_pw":      (teno_a_pw, 5, "TENO-A adaptive CT (PW)"),
    "teno_hybrid_fv":  (teno_hybrid_fv, 5,
                       "TENO Hybrid w/ discontinuity indicator (FV)"),
    "teno_hybrid_pw":  (teno_hybrid_pw, 5,
                       "TENO Hybrid w/ discontinuity indicator (PW)"),
    "vho_teno8_aa_pw":  (vho_teno8_aa_pw, 9,
                       "VHO-TENO8-AA adaptive order (PW)"),
    "vho_teno10_aa_pw": (vho_teno10_aa_pw, 11,
                       "VHO-TENO10-AA adaptive order (PW)"),
    "teno_thinc_fv":  (teno_thinc_fv, 5, "TENO-THINC hybrid (FV)"),
    "teno_thinc_pw":  (teno_thinc_pw, 5, "TENO-THINC hybrid (PW)"),
    "thinc_bvd_fv":   (thinc_bvd_fv, 5, "THINC-BVD (FV)"),
    "thinc_bvd_pw":   (thinc_bvd_pw, 5, "THINC-BVD (PW)"),
    # --- TENO-M ---
    "teno_m_va_fv":   (teno_m_va_fv, 5, "TENO-M-VA Van Albada limiter (FV)"),
    "teno_m_tvd5_fv": (teno_m_tvd5_fv, 5,
                       "TENO-M-TVD5 5th-order TVD limiter (FV)"),
    "teno_m_mp_fv":   (teno_m_mp_fv, 5,
                       "TENO-M-MP monotonicity-preserving (FV)"),
    "esweno3_fv":     (esweno3_fv, 3, "ES-WENO3 (FV)"),
    "esweno3_pw":     (esweno3_pw, 3, "ES-WENO3 (PW)"),
    "aweno5_fv":      (aweno5_fv, 5, "AWENO5 (FV)"),
    "bvd_fv":         (bvd_fv, 7, "BVD WENO5Z-THINC (FV)"),
    "bvd_tbv_fv":     (bvd_tbv_fv, 7, "BVD per-cell TBV variant Remark 6 (FV)"),
    "adweno5_fv":     (adweno5_fv, 5, "Curvature-sharpened WENO5-Z (FV)"),
    # --- ES-WENO methodology ---
    "weno5z_p2_fv":   (weno5z_p2_fv, 5,
                        "WENO5-Z p=2 (FV)"),
    # --- Hybrid linear-WENO shock-capturing ---
    "hybrid_linear_weno_fv":     (hybrid_linear_weno_fv, 5,
                       "Hybrid linear-WENO-Z, C_tau=1.0 (FV)"),
    "hybrid_linear_weno_mild_fv": (hybrid_linear_weno_mild_fv, 5,
                       "Hybrid linear-WENO-Z, C_tau=0.5 mild (FV)"),
    "hybrid_linear_weno_strong_fv":(hybrid_linear_weno_strong_fv, 5,
                       "Hybrid linear-WENO-Z, C_tau=2.0 strong (FV)"),
    # --- Hybrid flux MP ---
    "hybrid_mp_mc2_fv":(hybrid_mp_mc2_fv, 5, "Hybrid MP5-MUSCL(MC2) (FV)"),
    # --- Entropy-stable scalar ---
    "es_scalar_quad_fv":(es_scalar_quad_fv, 5, "Entropy-stable eta=u^2/2 (FV)"),
    "es_scalar_log_fv": (es_scalar_log_fv, 5,
                       "Entropy-stable eta=u*log(u) (FV)"),
    "es_scalar_cubic_fv":(es_scalar_cubic_fv, 5,
                       "Entropy-stable eta=u^4/4 (FV)"),
    # --- Entropy-stable scalar (PW) ---
    "es_scalar_quad_pw": (es_scalar_quad_pw, 5,
                          "Entropy-stable eta=u^2/2 (PW)"),
    "es_scalar_log_pw":  (es_scalar_log_pw, 5,
                          "Entropy-stable eta=u*log(u) (PW)"),
    "es_scalar_cubic_pw":(es_scalar_cubic_pw, 5,
                          "Entropy-stable eta=u^4/4 (PW)"),
    # --- Unified framework ROUND ---
    "round_a_fv":     (round_a_fv, 3, "ROUND-A aggressive (FV)"),
    "round_b_fv":     (round_b_fv, 3, "ROUND-B balanced (FV)"),
    "round_c_fv":     (round_c_fv, 3, "ROUND-C smooth (FV)"),
    # --- VHO-WENO ---
    "vhoweno9_fv":    (vhoweno9_fv, 9, "VHO-WENO9, 9th-order (FV)"),
    "vhoweno11_fv":   (vhoweno11_fv, 11, "VHO-WENO11, 11th-order (FV)"),
    "lsweno5hp_fv":   (lsweno5hp_fv, 5,
                       "LS-WENO5-HP physics-informed hybrid (FV)"),
    "weno5z_sp_fv":   (weno5z_sp_fv, 5,
                        "WENO5-Z sign-patch (FV, custom)"),
    "weno5z_sp_pw":   (weno5z_sp_pw, 5,
                        "WENO5-Z sign-patch (PW, custom)"),
    "lsweno5h_fv":    (lsweno5h_fv, 5, "LS-WENO5-H log-space hybrid (FV)"),
    # --- GENO ---
    "geno5_fv":       (geno5_fv, 5, "GENO5 gradient-based ENO 5th-order (FV)"),
    # --- CTENO ---
    "cteno5_fv":      (cteno5_fv, 5, "CTENO5 central-TENO hard cutoff (FV)"),
    "cteno5z_fv":     (cteno5z_fv, 5,
                       "CTENO5Z central-TENO WENOZ-inspired tau (FV)"),
    # --- ENO-MR ---
    "eno_mr3_fv":     (eno_mr3_fv, 5,
                       "ENO-MR3 multi-resolution 3rd-order (FV)"),
    "eno_mr5_fv":     (eno_mr5_fv, 5,
                       "ENO-MR5 multi-resolution 5th-order (FV)"),
    "eno_mr7_fv":     (eno_mr7_fv, 7,
                       "ENO-MR7 multi-resolution 7th-order (FV)"),
    # --- WENO-C ---
    "wenoc5_fv":       (wenoc5_fv, 5, "WENO5-C combined two-layer (FV)"),
    "wenoc5z_fv":      (wenoc5z_fv, 5, "WENO5-ZC combined two-layer (FV)"),
    "wenoc7_fv":       (wenoc7_fv, 7, "WENO7-C combined two-layer (FV)"),
    "wenoc7z_fv":      (wenoc7z_fv, 7, "WENO7-ZC combined two-layer (FV)"),
}


def list_methods():
    """Return (name, stencil_width, description) for all canonical methods."""
    return [(name, fn[1], fn[2]) for name, fn in _METHODS.items()]


def get_method(name):
    """Return the reconstruction function for a registered method name.

    Looks up *name* in the method registry and returns the callable
    (FV or PW).  Raises ValueError for unknown names.
    """
    if name in _METHODS:
        return _METHODS[name][0]
    raise ValueError(
        f"Unknown method '{name}'. Available: {sorted(_METHODS.keys())}"
    )


def _safe_get(z, idx, n):
    """Get z[idx] with optional periodic wrapping.

    If n > 0: periodic boundary (idx % n).
    If n == 0: clamped boundary (clamp to [0, len(z)-1]).
    """
    if n > 0:
        return z[idx % n]
    if idx < 0:
        return z[0]
    if idx >= len(z):
        return z[-1]
    return z[idx]


def reconstruct_array(method, z, il=0, iu=None, periodic=False):
    r"""Apply a reconstruction method to a 1D array.

    Parameters
    ----------
    method : str or callable
        Method name or callable reconstruction function.
    z : array-like
        1D array of cell-centered values.
    il, iu : int
        Index bounds (inclusive).
    periodic : bool
        If True, use periodic boundary conditions (indices wrap modulo len(z)).
        Otherwise, clamp out-of-bounds indices to nearest edge.

    Returns
    -------
    zl, zr : arrays (length len(z)+1)
        zl[i+1] = :math:`u_{i+1/2}^-`, zr[i] = :math:`u_{i-1/2}^+`.
    """
    if isinstance(method, str):
        fn = get_method(method)
    else:
        fn = method

    n = len(z)
    if iu is None:
        iu = n - 1

    # n_wrap > 0 enables periodic wrapping
    n_wrap = n if periodic else 0

    zl = [0.0] * (n + 1)
    zr = [0.0] * (n + 1)

    # Determine stencil width from registry
    stencil_width = None
    if isinstance(method, str):
        entry = _METHODS.get(method)
        if entry is not None:
            stencil_width = entry[1]
    if stencil_width is None:
        sig = inspect.signature(fn)
        stencil_width = len(sig.parameters)

    # Stencil index offsets for each supported width.
    # offset=0 always maps to z[i] (the current cell).
    _STENCIL_OFFSETS = {
        3:  [-1, 0, 1],
        5:  [-2, -1, 0, 1, 2],
        6:  [-2, -1, 0, 1, 2, 3],
        7:  [-3, -2, -1, 0, 1, 2, 3],
        9:  [-4, -3, -2, -1, 0, 1, 2, 3, 4],
        11: [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    }
    offsets = _STENCIL_OFFSETS.get(stencil_width)
    if offsets is None:
        raise ValueError(f"Unsupported stencil width: {stencil_width} args")

    for i in range(il, iu + 1):
        args = [_safe_get(z, i + off, n_wrap) for off in offsets]
        uL, uR = fn(*args)
        zl[i + 1] = uL
        zr[i] = uR

    return zl, zr
#
# :D
#
