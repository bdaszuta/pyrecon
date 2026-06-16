"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Property-based tests: constant preservation, boundedness,
 basic sanity
"""
from pyrecon.interface import list_methods, reconstruct_array

_LIMITED_METHODS = {
    "lin_vl_fv", "lin_mc2_fv", "mp3_fv", "mp5_fv", "mp7_fv", "mp5_r_fv",
    "ppm_fv", "ppmx_fv", "bvd_fv", "mood_fv",
    "mp5_mp_fv",
    "teno5_fv", "teno5_pw", "teno5_mc2_fv", "teno5_mc2_pw",
    "teno5_koren_fv", "teno5_koren_pw",
    "teno_a_fv", "teno_a_pw", "teno_hybrid_fv", "teno_hybrid_pw",
    "teno_thinc_fv", "teno_thinc_pw",
}


def _make_constant_field(N=64, val=3.0):
    return [val] * N


def _make_step_field(N=64):
    mid = N // 2
    return [0.0] * mid + [1.0] * (N - mid)


def test_constant_preservation():
    """All methods preserve constant fields exactly on interior faces."""
    N = 32
    z = _make_constant_field(N)
    failures = []
    for name, _sw, _desc in list_methods():
        zl, zr = reconstruct_array(name, z, periodic=True)
        for i in range(2, N - 2):
            if abs(zl[i] - 3.0) > 1e-13 or abs(zr[i] - 3.0) > 1e-13:
                failures.append(name)
                break
    if failures:
        raise AssertionError(f"Constant preservation failures: {failures}")


def test_boundedness_step():
    """Limited methods stay within data range on step function."""
    N = 64
    z = _make_step_field(N)
    failures = []
    for name, _sw, _desc in list_methods():
        if name not in _LIMITED_METHODS:
            continue
        zl, zr = reconstruct_array(name, z, periodic=False)
        for i in range(3, N - 3):
            if zl[i] < -0.15 or zl[i] > 1.15:
                failures.append(f"{name}: zl[{i}]={zl[i]:.4f}")
                break
            if zr[i] < -0.15 or zr[i] > 1.15:
                failures.append(f"{name}: zr[{i}]={zr[i]:.4f}")
                break
    if failures:
        msg = "Boundedness failures on step:\n" + "\n".join(failures)
        raise AssertionError(msg)
#
# :D
#
