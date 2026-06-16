"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Math utilities for reconstruction methods
"""
import math
from pyrecon._jit_utils import JIT, TYPE_CHECKING
def sign(x, y=None):
    """sign(x): return 1.0 if x >= 0 else -1.0
    sign(x, y): return abs(x) if y >= 0 else -abs(x)
    """
    if y is None:
        return 1.0 if x >= 0.0 else -1.0
    return abs(x) if y >= 0.0 else -abs(x)


def MC2(x, y):
    """Monotonized central-difference limiter."""
    s1 = sign(1.0, x)
    s2 = sign(1.0, y)
    mn = 2.0 * min(abs(x), abs(y))
    return 0.5 * (s1 + s2) * min(mn, 0.5 * abs(x + y))


def minmod(x, y, z=None, w=None, v=None, u=None):
    r"""minmod limiter with 2, 4, or 6 arguments.

    2-arg: :math:`0.5 * (sign(x) + sign(y)) * min(abs(x), abs(y))`
    4-arg: :math:`0.125 * (sign(x)+sign(y)) * abs((sign(x)+sign(z))*(sign(x)+sign(w))) * min(abs(x),abs(y),abs(z),abs(w))`
    6-arg: :math:`0.03125 * (sign(x)+sign(y)) * abs(prod_{k in {z,w,v,u}} (sign(x)+sign(k))) * min(abs(x),abs(y),abs(z),abs(w),abs(v),abs(u))`
    """
    if z is None:
        return 0.5 * (sign(x) + sign(y)) * min(abs(x), abs(y))
    if v is None:
        oo8 = 1.0 / 8.0
        return oo8 * (sign(x) + sign(y)) * abs(
            (sign(x) + sign(z)) * (sign(x) + sign(w))
        ) * min_abs(x, y, z, w)
    oo32 = 1.0 / 32.0
    return oo32 * (sign(x) + sign(y)) * abs(
        (sign(x) + sign(z)) *
        (sign(x) + sign(w)) *
        (sign(x) + sign(v)) *
        (sign(x) + sign(u))
    ) * min_abs(x, y, z, w, v, u)


def maxmod(x, y):
    """maxmod limiter."""
    return 0.5 * (sign(x) + sign(y)) * max(abs(x), abs(y))


def min_abs(a, b, c=None, d=None, e=None, f=None):
    """min of absolute values: 2, 4, or 6 arguments."""
    if c is None:
        return min(abs(a), abs(b))
    if e is None:
        return min(min(abs(a), abs(b)), min(abs(c), abs(d)))
    min_4 = min(min(abs(a), abs(b)), min(abs(c), abs(d)))
    return min(min_4, min(abs(e), abs(f)))


def max_abs(a, b):
    """max of absolute values."""
    return max(abs(a), abs(b))


def thinc_value_L(u_im1, u_i, u_ip1, kbeta=6.0):
    r"""THINC left face value :math:`u_{i+1/2}` from 3-point stencil."""
    u_min = min(u_im1, u_ip1)
    u_max = max(u_im1, u_ip1)
    if u_max - u_min < 1e-40:
        return u_i
    gamma = 1.0 if u_ip1 > u_im1 else -1.0
    return u_min + 0.5 * (u_max - u_min) * (
        1.0 + gamma * math.tanh(kbeta * 0.5))


def thinc_value_R(u_im1, u_i, u_ip1, kbeta=6.0):
    r"""THINC right face value :math:`u_{i-1/2}` from 3-point stencil."""
    u_min = min(u_im1, u_ip1)
    u_max = max(u_im1, u_ip1)
    if u_max - u_min < 1e-40:
        return u_i
    gamma = 1.0 if u_ip1 > u_im1 else -1.0
    return u_min + 0.5 * (u_max - u_min) * (
        1.0 + gamma * math.tanh(-kbeta * 0.5))


def exp_face_log(u_left, u_right, log_eps=1e-40):
    """Log-space 2-point FV reconstruction (geometric mean corrected).

    Used by LS-WENO methods for sharp-drop fallback.
    """
    if u_left <= log_eps or u_right <= log_eps:
        return math.sqrt(max(u_left, log_eps) * max(u_right, log_eps))
    ratio = u_left / u_right
    if ratio <= 1.0:
        return math.sqrt(u_left * u_right)
    kh = math.log(ratio)
    half_kh = 0.5 * kh
    if half_kh > 50.0:
        return u_left * kh * math.exp(-kh)
    sinh_half = math.sinh(half_kh)
    factor = (half_kh / sinh_half) * math.exp(-half_kh)
    return u_left * factor
# ---------------------------------------------------------------------------
# JIT compilation (pynalgo convention)
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    reveal_locals()  # noqa: F821
else:
    sign = JIT(sign)
    MC2 = JIT(MC2)
    minmod = JIT(minmod)
    maxmod = JIT(maxmod)
    min_abs = JIT(min_abs)
    max_abs = JIT(max_abs)
    thinc_value_L = JIT(thinc_value_L)
    thinc_value_R = JIT(thinc_value_R)
    exp_face_log = JIT(exp_face_log)
#
# :D
#
