"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Tests for interface.
"""
from pyrecon.interface import list_methods, get_method, reconstruct_array


def test_list_methods():
    methods = list_methods()
    names = [m[0] for m in methods]
    assert "weno5_fv" in names
    assert "donate_fv" in names


def test_get_method_valid():
    fn = get_method("weno5_fv")
    uL, uR = fn(1.0, 2.0, 3.0, 4.0, 5.0)
    assert abs(uL - 3.5) < 1e-14


def test_get_method_invalid():
    try:
        get_method("nonexistent")
        assert False
    except ValueError:
        pass


def test_reconstruct_array_linear():
    """Linear field: zl[i+1]=u_{i+1/2}^-, zr[i]=u_{i-1/2}^+."""
    z = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    zl, zr = reconstruct_array("weno5_fv", z)
    n = len(z)
    for i in range(2, n - 2):
        assert abs(zl[i + 1] - (z[i] + 0.5)) < 1e-13
        assert abs(zr[i] - (z[i] - 0.5)) < 1e-13


def test_reconstruct_array_constant():
    """Constant field: all face values equal the constant."""
    z = [7.0] * 10
    for method in ["donate_fv", "weno5_fv", "ppm_fv", "mp5_fv"]:
        zl, zr = reconstruct_array(method, z)
        for v in zl[2:-2]:
            assert abs(v - 7.0) < 1e-13
        for v in zr[2:-2]:
            assert abs(v - 7.0) < 1e-13


def test_reconstruct_array_with_fn():
    from pyrecon.recon_weno5 import weno5z_fv as weno5z
    z = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    zl, zr = reconstruct_array(weno5z, z)
    for i in range(2, len(z) - 2):
        assert abs(zl[i + 1] - (z[i] + 0.5)) < 1e-13
        assert abs(zr[i] - (z[i] - 0.5)) < 1e-13
#
# :D
#
