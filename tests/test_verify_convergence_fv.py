"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Convergence verification for all FV (_fv) reconstruction methods
"""
import math
from pyrecon.interface import reconstruct_array, _METHODS

_ZR_FACE_SHIFT = {"lag6_fv"}

# ---------------------------------------------------------------------------
# Test function: f(x) = exp(2 x) on [0, 1]
#
# u'(x) = 2 exp(2 x) > 0 everywhere -- no critical points, so WENO
# methods achieve their design order without degradation.  Strictly
# positive, so entropy-stable methods (log, cubic) work correctly.
# Not periodic -- use periodic=False with boundary exclusion.
# ---------------------------------------------------------------------------

_A_EXP = 2.0


def cell_average(x_left, x_right):
    return (math.exp(_A_EXP * x_right) - math.exp(_A_EXP * x_left)) / (
        _A_EXP * (x_right - x_left))


def exact_face(x):
    return math.exp(_A_EXP * x)


def build_field(N):
    dx = 1.0 / N
    return [cell_average(i * dx, (i + 1) * dx) for i in range(N)]


# ---------------------------------------------------------------------------
# Error norms
# ---------------------------------------------------------------------------

def L2_error_zl(zl, N, exclude=3):
    dx = 1.0 / N
    e2 = 0.0
    n_valid = 0
    for i in range(exclude, N - exclude):
        e2 += (zl[i + 1] - exact_face((i + 1) * dx)) ** 2
        n_valid += 1
    return math.sqrt(e2 / n_valid) if n_valid > 0 else 0.0


def L2_error_zr(zr, N, method_name, exclude=3):
    dx = 1.0 / N
    e2 = 0.0
    n_valid = 0
    for i in range(exclude, N - exclude):
        x_face = (i + 1) * dx if method_name in _ZR_FACE_SHIFT else i * dx
        e2 += (zr[i] - exact_face(x_face)) ** 2
        n_valid += 1
    return math.sqrt(e2 / n_valid) if n_valid > 0 else 0.0


def measure_order(method_name, Ns=(32, 64, 128), exclude=3):
    errors_zl, errors_zr = [], []
    for N in Ns:
        z = build_field(N)
        zl, zr = reconstruct_array(method_name, z, periodic=False)
        errors_zl.append(L2_error_zl(zl, N, exclude=exclude))
        errors_zr.append(L2_error_zr(zr, N, method_name, exclude=exclude))

    def fit(errs):
        # If all errors are zero, the method is exact
        if all(e < 1e-30 for e in errs):
            return 99.0
        # If finest error is at machine precision, method achieves design
        if errs[-1] < 1e-13:
            return 99.0
        orders = []
        for k in range(len(Ns) - 1):
            if errs[k] > 1e-16 and errs[k + 1] > 1e-16:
                orders.append(
                    math.log(errs[k] / errs[k + 1]) /
                    math.log(Ns[k + 1] / Ns[k]))
        return sum(orders) / len(orders) if orders else 0.0

    return fit(errors_zl), fit(errors_zr)


# ---------------------------------------------------------------------------
# Expected convergence orders: (design, minimum_acceptable)
# Both MUST equal design.
# ---------------------------------------------------------------------------

EXPECTED = {
    # First-order --------------------------------------------------------
    "donate_fv":                 (1, 1),

    # Linear (2nd-order) -------------------------------------------------
    "lin_vl_fv":                 (2, 2),
    "lin_mc2_fv":                (2, 2),

    # WENO3 family -------------------------------------------------------
    "weno3_fv":                  (3, 3),
    "weno3z_fv":                 (3, 3),

    # WENO5 family -------------------------------------------------------
    "weno5_fv":                  (5, 5),
    "weno5z_fv":                 (5, 5),
    "weno5d_si_fv":              (5, 5),
    "weno5m_fv":                 (5, 5),
    "weno5zcplus_fv":            (5, 5),
    "weno5z_ns_fv":              (5, 5),
    "weno5zp_fv":                (5, 5),
    "weno5_ha_js_fv":            (5, 5),
    "weno5cz_fv":                (5, 5),
    "weno5_bc_fv":               (5, 5),
    "weno5z_sp_fv":              (5, 5),

    # WENO7 family -------------------------------------------------------
    "weno7_fv":                  (7, 7),
    "weno7z_fv":                 (7, 7),

    # WENO-AO ------------------------------------------------------------
    "weno_ao53_fv":              (5, 5),

    # WENO-C ----------------------------------------------------------
    "wenoc5_fv":                 (5, 5),
    "wenoc5z_fv":                (5, 5),
    "wenoc7_fv":                 (7, 7),
    "wenoc7z_fv":                (7, 7),

    # CTENO -----------------------------------------------------------
    "cteno5_fv":                 (5, 5),
    "cteno5z_fv":                (5, 5),

    # ENO-MR ----------------------------------------------------------
    "eno_mr3_fv":                (3, 3),
    "eno_mr5_fv":                (5, 5),
    "eno_mr7_fv":                (7, 7),

    # GENO ------------------------------------------------------------
    "geno5_fv":                  (5, 5),

    # Entropy-stable scalar ----------------------------------------------
    "es_scalar_quad_fv":          (5, 5),
    "es_scalar_log_fv":           (5, 5),
    "es_scalar_cubic_fv":         (5, 5),

    # LS-WENO ------------------------------------------------------------
    "lsweno5h_fv":                (5, 5),
    "lsweno5hp_fv":               (5, 5),

    # MOOD ---------------------------------------------------------------
    "mood_fv":                    (5, 5),

    # ROUND / NVD -----------------------------------------------------
    # ROUND-B satisfies F(1/2)=3/4 but F'(1/2)=1.5 (not 1),
    # limiting it to 2nd order.  ROUND-A and ROUND-C fail F(1/2)=3/4
    # and are 1st order on smooth monotonic data.
    "round_a_fv":                 (1, 1),
    "round_b_fv":                 (2, 2),
    "round_c_fv":                 (1, 1),

    # TENO family --------------------------------------------------------
    "teno5_fv":                  (5, 5),
    "teno5_mc2_fv":              (5, 5),
    "teno5_koren_fv":            (5, 5),
    "teno_a_fv":                 (5, 5),
    "teno_hybrid_fv":            (5, 5),
    "teno_thinc_fv":             (5, 5),
    # TENO-M (Li, Fu, Adams 2021)
    "teno_m_va_fv":              (5, 5),
    "teno_m_tvd5_fv":            (5, 5),
    "teno_m_mp_fv":              (5, 5),

    # THINC-BVD ----------------------------------------------------------
    "thinc_bvd_fv":              (5, 5),

    # MP family ----------------------------------------------------------
    "mp3_fv":                    (3, 3),
    "mp5_fv":                    (5, 5),
    "mp7_fv":                    (7, 7),
    "mp5_r_fv":                  (5, 5),
    "mp5_mp_fv":                 (5, 5),

    # CENO / CWENO / Central WENO ----------------------------------------
    "ceno3_fv":                  (3, 3),
    "ceno5_fv":                  (5, 5),
    "cweno3_fv":                 (3, 3),
    "cweno5_fv":                 (5, 5),
    "cweno_z3_fv":               (5, 5),
    "cweno_z5_fv":               (7, 7),
    "cweno5_capdeville_fv":      (5, 5),
    "central_weno_fv":           (3, 3),

    # PPM / Lagrange -----------------------------------------------------
    "ppm_fv":                    (4, 4),
    "ppmx_fv":                   (4, 4),
    "lag6_fv":                   (6, 6),

    # BVD -----------------------------------------------------------------
    "bvd_fv":                    (2, 2),
    "bvd_tbv_fv":                (5, 5),
    "bvd_cu_fv":                 (3, 3),
    "bvd_e6_mp5_fv":             (3, 3),

    # Energy-stable / entropy-stable -------------------------------------
    "esweno3_fv":                (3, 3),
    "esweno5_fv":                (5, 5),
    "weno5z_p2_fv":              (5, 5),

    # Adaptive / sharpening ----------------------------------------------
    "aweno5_fv":                 (5, 5),
    "adweno5_fv":                (5, 5),

    # Low-dissipation shock-capturing ------------------------------------
    "hybrid_linear_weno_fv":     (5, 5),
    "hybrid_linear_weno_mild_fv":(5, 5),
    "hybrid_linear_weno_strong_fv":(5, 5),

    # Hybrid --------------------------------------------------------------
    "hybrid_mp_mc2_fv":          (5, 5),

    # VHO-WENO ------------------------------------------------------------
    "vhoweno9_fv":               (9, 9),
    "vhoweno11_fv":              (11, 11),
}


# ---------------------------------------------------------------------------
# Test entry point
# ---------------------------------------------------------------------------

def test_fv_convergence():
    """Verify all FV methods achieve at least design order."""
    all_fv = [n for n in _METHODS if n.endswith("_fv")]
    failures = []
    results = []

    for name in sorted(all_fv):
        spec = EXPECTED.get(name)
        if spec is None:
            continue
        expected, min_ok = spec

        entry = _METHODS.get(name)
        stencil_width = entry[1] if entry else 7
        exclude = stencil_width // 2

        if expected >= 9:
            ns = (16, 32)
        elif expected >= 7:
            ns = (16, 32, 64)
        else:
            ns = (32, 64, 128)
        order_zl, order_zr = measure_order(name, Ns=ns, exclude=exclude)
        zl_ok = round(order_zl) >= min_ok
        zr_ok = round(order_zr) >= min_ok
        ok = zl_ok and zr_ok

        results.append((name, expected, order_zl, order_zr, ok))

        if not (zl_ok and zr_ok):
            failures.append(
                f"  {name:22s} design={expected} zl={order_zl:.2f} "
                f"zr={order_zr:.2f} (min={min_ok})")

    print(f"\n{'Method':22s} {'Exp':>4s} {'zl':>6s} {'zr':>6s} {'OK':>4s}")
    print(f"{'-'*22} {'-'*4} {'-'*6} {'-'*6} {'-'*4}")
    for name, exp, zl, zr, ok in sorted(results, key=lambda x: -x[2]):
        print(f"{name:22s} {exp:4d} {zl:6.2f} {zr:6.2f} "
              f"{'OK' if ok else 'FAIL'}")

    n_ok = sum(1 for _, _, _, _, ok in results if ok)
    print(f"\n{n_ok}/{len(results)} FV methods OK")

    if failures:
        raise AssertionError(
            "Convergence failures (investigate each):\n" + "\n".join(failures))
#
# :D
#
