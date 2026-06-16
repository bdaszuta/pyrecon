"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: THINC-BVD reconstruction method.

Reference: Takagi et al., J. Comput. Phys. (2022).
"""
from pyrecon.utils import thinc_value_L, thinc_value_R
from pyrecon.recon_weno5 import _js_smoothness, _OPTIMW_FV, _OPTIMW_PW
from pyrecon._jit_utils import JIT, TYPE_CHECKING

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_K_1_6 = 1.0 / 6.0
_K_1_8 = 1.0 / 8.0

_EPSL = 1e-40
_EPSL_Z = 1e-12
_KBETA = 6.0       # THINC steepness parameter
_KRATIO_CUT = 1e2  # smoothness ratio cutoff

# ---------------------------------------------------------------------------
# WENO5z stencil polynomials (FV)
# ---------------------------------------------------------------------------

def _weno5_stencils_fv(u_0, u_1, u_2, u_3, u_4):
    """WENO5 FV stencil polynomials for left-biased reconstruction.

    Given 5 consecutive cell values, returns 3 sub-stencil values:
      uk0 = 1/6 * (2*u_0 - 7*u_1 + 11*u_2)
      uk1 = 1/6 * (-u_1 + 5*u_2 + 2*u_3)
      uk2 = 1/6 * (2*u_2 + 5*u_3 - u_4)
    """
    uk0 = _K_1_6 * (2.0 * u_0 - 7.0 * u_1 + 11.0 * u_2)
    uk1 = _K_1_6 * (-u_1 + 5.0 * u_2 + 2.0 * u_3)
    uk2 = _K_1_6 * (2.0 * u_2 + 5.0 * u_3 - u_4)
    return uk0, uk1, uk2


def _weno5_stencils_pw(u_0, u_1, u_2, u_3, u_4):
    """WENO5 PW stencil polynomials for left-biased reconstruction.

    Given 5 consecutive cell values, returns 3 sub-stencil values:
      uk0 = 1/8 * (3*u_0 - 10*u_1 + 15*u_2)
      uk1 = 1/8 * (-u_1 + 6*u_2 + 3*u_3)
      uk2 = 1/8 * (3*u_2 + 6*u_3 - u_4)
    """
    uk0 = _K_1_8 * (3.0 * u_0 - 10.0 * u_1 + 15.0 * u_2)
    uk1 = _K_1_8 * (-u_1 + 6.0 * u_2 + 3.0 * u_3)
    uk2 = _K_1_8 * (3.0 * u_2 + 6.0 * u_3 - u_4)
    return uk0, uk1, uk2


# ---------------------------------------------------------------------------
# WENO5z paired L+R reconstruction (inline in THINC-BVD)
# ---------------------------------------------------------------------------

def _weno5z_LR_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """WENO5-Z paired left+right face reconstruction (FV).

    Returns
    -------
    (uL, uR)
    """
    b0, b1, b2 = _js_smoothness(u_im2, u_im1, u_i, u_ip1, u_ip2)
    db = abs(b0 - b2)

    # Left face weights
    aL0 = _OPTIMW_FV[0] * (1.0 + db / (_EPSL_Z + b0))
    aL1 = _OPTIMW_FV[1] * (1.0 + db / (_EPSL_Z + b1))
    aL2 = _OPTIMW_FV[2] * (1.0 + db / (_EPSL_Z + b2))
    inv_sum_L = 1.0 / (aL0 + aL1 + aL2)

    ukL0, ukL1, ukL2 = _weno5_stencils_fv(u_im2, u_im1, u_i, u_ip1, u_ip2)
    uL = inv_sum_L * (aL0 * ukL0 + aL1 * ukL1 + aL2 * ukL2)

    # Right face weights (b swapped)
    aR0 = _OPTIMW_FV[0] * (1.0 + db / (_EPSL_Z + b2))
    aR1 = _OPTIMW_FV[1] * (1.0 + db / (_EPSL_Z + b1))
    aR2 = _OPTIMW_FV[2] * (1.0 + db / (_EPSL_Z + b0))
    inv_sum_R = 1.0 / (aR0 + aR1 + aR2)

    ukR0, ukR1, ukR2 = _weno5_stencils_fv(u_ip2, u_ip1, u_i, u_im1, u_im2)
    uR = inv_sum_R * (aR0 * ukR0 + aR1 * ukR1 + aR2 * ukR2)

    return uL, uR


def _weno5z_LR_pw(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """WENO5-Z paired left+right face reconstruction (PW).

    Returns
    -------
    (uL, uR)
    """
    b0, b1, b2 = _js_smoothness(u_im2, u_im1, u_i, u_ip1, u_ip2)
    db = abs(b0 - b2)

    # Left face weights
    aL0 = _OPTIMW_PW[0] * (1.0 + db / (_EPSL_Z + b0))
    aL1 = _OPTIMW_PW[1] * (1.0 + db / (_EPSL_Z + b1))
    aL2 = _OPTIMW_PW[2] * (1.0 + db / (_EPSL_Z + b2))
    inv_sum_L = 1.0 / (aL0 + aL1 + aL2)

    ukL0, ukL1, ukL2 = _weno5_stencils_pw(u_im2, u_im1, u_i, u_ip1, u_ip2)
    uL = inv_sum_L * (aL0 * ukL0 + aL1 * ukL1 + aL2 * ukL2)

    # Right face weights (b swapped)
    aR0 = _OPTIMW_PW[0] * (1.0 + db / (_EPSL_Z + b2))
    aR1 = _OPTIMW_PW[1] * (1.0 + db / (_EPSL_Z + b1))
    aR2 = _OPTIMW_PW[2] * (1.0 + db / (_EPSL_Z + b0))
    inv_sum_R = 1.0 / (aR0 + aR1 + aR2)

    ukR0, ukR1, ukR2 = _weno5_stencils_pw(u_ip2, u_ip1, u_i, u_im1, u_im2)
    uR = inv_sum_R * (aR0 * ukR0 + aR1 * ukR1 + aR2 * ukR2)

    return uL, uR


# ---------------------------------------------------------------------------
# THINC-BVD paired L+R reconstruction core
# ---------------------------------------------------------------------------

def _thinc_bvd_LR_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """THINC-BVD paired L+R reconstruction (FV).

    Computes JS smoothness on full 5-point stencil.
    If max(b) / min(b) > 100: use THINC at both faces.
    Otherwise: use WENO5z.

    Returns
    -------
    (uL, uR)
    """
    b0, b1, b2 = _js_smoothness(u_im2, u_im1, u_i, u_ip1, u_ip2)
    b_min = min(b0, min(b1, b2))
    b_max = max(b0, max(b1, b2))

    if b_max > _KRATIO_CUT * (b_min + _EPSL):
        # Non-smooth: THINC at both faces from 3-point stencil
        uL = thinc_value_L(u_im1, u_i, u_ip1)
        uR = thinc_value_R(u_im1, u_i, u_ip1)
    else:
        # Smooth: WENO5z
        uL, uR = _weno5z_LR_fv(u_im2, u_im1, u_i, u_ip1, u_ip2)

    return uL, uR


def _thinc_bvd_LR_pw(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """THINC-BVD paired L+R reconstruction (PW).

    Computes JS smoothness on full 5-point stencil.
    If max(b) / min(b) > 100: use THINC at both faces.
    Otherwise: use WENO5z.

    Returns
    -------
    (uL, uR)
    """
    b0, b1, b2 = _js_smoothness(u_im2, u_im1, u_i, u_ip1, u_ip2)
    b_min = min(b0, min(b1, b2))
    b_max = max(b0, max(b1, b2))

    if b_max > _KRATIO_CUT * (b_min + _EPSL):
        # Non-smooth: THINC at both faces from 3-point stencil
        uL = thinc_value_L(u_im1, u_i, u_ip1)
        uR = thinc_value_R(u_im1, u_i, u_ip1)
    else:
        # Smooth: WENO5z
        uL, uR = _weno5z_LR_pw(u_im2, u_im1, u_i, u_ip1, u_ip2)

    return uL, uR


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def thinc_bvd_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """THINC-BVD reconstruction (FV weights) at i+1/2.

    """
    return _thinc_bvd_LR_fv(u_im2, u_im1, u_i, u_ip1, u_ip2)


def thinc_bvd_pw(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """THINC-BVD reconstruction (PW weights) at i+1/2.

    """
    return _thinc_bvd_LR_pw(u_im2, u_im1, u_i, u_ip1, u_ip2)
# ---------------------------------------------------------------------------
# JIT compilation (pynalgo convention)
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    reveal_locals()  # noqa: F821
else:
    _weno5_stencils_fv = JIT(_weno5_stencils_fv)
    _weno5_stencils_pw = JIT(_weno5_stencils_pw)
    _weno5z_LR_fv = JIT(_weno5z_LR_fv)
    _weno5z_LR_pw = JIT(_weno5z_LR_pw)
    _thinc_bvd_LR_fv = JIT(_thinc_bvd_LR_fv)
    _thinc_bvd_LR_pw = JIT(_thinc_bvd_LR_pw)
    thinc_bvd_fv = JIT(thinc_bvd_fv)
    thinc_bvd_pw = JIT(thinc_bvd_pw)
#
# :D
#
