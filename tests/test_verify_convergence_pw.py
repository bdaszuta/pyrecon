"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Convergence verification for all PW (_pw) reconstruction methods
"""
import math
from pyrecon.interface import reconstruct_array, _METHODS

_ZR_FACE_SHIFT = {"lag6_pw"}

# ---------------------------------------------------------------------------
# Test function: f(x) = exp(2 x) on [0, 1]
#
# u'(x) = 2 exp(2 x) > 0 everywhere -- no critical points.  Strictly
# positive, so entropy-stable methods work correctly.  Not periodic.
# Point values are sampled at cell centres.
# ---------------------------------------------------------------------------

_A_EXP = 2.0


def point_value(x):
    return math.exp(_A_EXP * x)


def exact_face(x):
    return math.exp(_A_EXP * x)


def build_field(N):
    dx = 1.0 / N
    return [point_value((i + 0.5) * dx) for i in range(N)]


# ---------------------------------------------------------------------------
# Error norms
# ---------------------------------------------------------------------------

def L2_error_zl(zl, N, exclude=5):
    dx = 1.0 / N
    e2 = 0.0
    n_valid = 0
    for i in range(exclude, N - exclude):
        e2 += (zl[i + 1] - exact_face((i + 1) * dx)) ** 2
        n_valid += 1
    return math.sqrt(e2 / n_valid) if n_valid > 0 else 0.0


def L2_error_zr(zr, N, method_name, exclude=5):
    dx = 1.0 / N
    e2 = 0.0
    n_valid = 0
    for i in range(exclude, N - exclude):
        x_face = (i + 1) * dx if method_name in _ZR_FACE_SHIFT else i * dx
        e2 += (zr[i] - exact_face(x_face)) ** 2
        n_valid += 1
    return math.sqrt(e2 / n_valid) if n_valid > 0 else 0.0


def measure_order(method_name, Ns=(32, 64, 128), exclude=5):
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
        orders = []
        for k in range(len(Ns) - 1):
            if errs[k] > 1e-16 and errs[k + 1] > 1e-16:
                orders.append(
                    math.log(errs[k] / errs[k + 1]) /
                    math.log(Ns[k + 1] / Ns[k]))
        return sum(orders) / len(orders) if orders else 0.0

    return fit(errors_zl), fit(errors_zr)


# ---------------------------------------------------------------------------
# Expected convergence orders: (design, min) -- both equal design.
# ---------------------------------------------------------------------------

EXPECTED = {
    "weno3_pw":                  (3, 3),
    "weno3z_pw":                 (3, 3),

    "weno5m_pw":                 (5, 5),
    "weno5_pw":                  (5, 5),
    "weno5z_pw":                 (5, 5),
    "weno5d_si_pw":              (5, 5),
    "weno_ao53_pw":              (5, 5),

    "weno7_pw":                  (7, 7),
    "weno7z_pw":                 (7, 7),

    "teno5_pw":                  (5, 5),
    "teno5_mc2_pw":              (5, 5),
    "teno5_koren_pw":            (5, 5),
    "teno_a_pw":                 (5, 5),
    "teno_hybrid_pw":            (5, 5),
    "teno_thinc_pw":             (5, 5),

    "thinc_bvd_pw":              (5, 5),

    "esweno3_pw":                (3, 3),

    # Entropy-stable scalar (PW)
    "es_scalar_quad_pw":          (5, 5),
    "es_scalar_log_pw":           (5, 5),
    "es_scalar_cubic_pw":         (5, 5),

    "lag6_pw":                   (6, 6),

    "vho_teno8_aa_pw":            (8, 8),
    "vho_teno10_aa_pw":           (10, 10),
    "weno5z_sp_pw":              (5, 5),
}


_HO_NS = {  # (Ns_tuple, exclude) -- methods needing smaller grids
    "vho_teno8_aa_pw":  ((12, 16, 24), 5),
    "vho_teno10_aa_pw": ((10, 14, 20), 5),
}


# ---------------------------------------------------------------------------
# Test entry point
# ---------------------------------------------------------------------------

def test_pw_convergence():
    """Verify all PW methods achieve at least design order."""
    all_pw = [n for n in _METHODS if n.endswith("_pw")]
    failures = []
    results = []

    for name in sorted(all_pw):
        spec = EXPECTED.get(name)
        if spec is None:
            continue
        expected, min_ok = spec

        ns_spec = _HO_NS.get(name, ((32, 64, 128), 5))
        ns, excl = ns_spec
        order_zl, order_zr = measure_order(name, Ns=ns, exclude=excl)
        zl_ok = round(order_zl) >= min_ok
        zr_ok = round(order_zr) >= min_ok
        ok = zl_ok and zr_ok

        results.append((name, expected, order_zl, order_zr, ok))

        if not (zl_ok and zr_ok):
            failures.append(
                f"  {name:18s} design={expected} zl={order_zl:.2f} "
                f"zr={order_zr:.2f} (min={min_ok})")

    print(f"\n{'Method':18s} {'Exp':>4s} {'zl':>6s} {'zr':>6s} {'OK':>4s}")
    print(f"{'-'*18} {'-'*4} {'-'*6} {'-'*6} {'-'*4}")
    for name, exp, zl, zr, ok in sorted(results, key=lambda x: -x[2]):
        print(f"{name:18s} {exp:4d} {zl:6.2f} {zr:6.2f} "
              f"{'OK' if ok else 'FAIL'}")

    n_ok = sum(1 for _, _, _, _, ok in results if ok)
    print(f"\n{n_ok}/{len(results)} PW methods OK")

    if failures:
        raise AssertionError(
            "Convergence failures (investigate each):\n" + "\n".join(failures))
#
# :D
#
