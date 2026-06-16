"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: WENO5-Z-SP: WENO5-Z with branchless sign preservation
"""
from pyrecon.recon_weno5 import _weno5z_LR_fv, _weno5z_LR_pw
from pyrecon._jit_utils import JIT, TYPE_CHECKING


# ===================================================================
# FV version
# ===================================================================

def weno5z_sp_fv(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """WENO5-Z sign-patch (FV). Post-hoc sign enforcement: if all 5 inputs
    share sign and WENO5-Z output violates it, cascade inner-upwind
    -> centered-average. Not Fjordholm & Ray 2016 SP-WENO."""
    uL, uR = _weno5z_LR_fv(u_im2, u_im1, u_i, u_ip1, u_ip2)

    # Sign-preserving flag and expected sign
    mn = min(u_im2, u_im1, u_i, u_ip1, u_ip2)
    mx = max(u_im2, u_im1, u_i, u_ip1, u_ip2)
    sign_pos = 1.0 if mn > 0.0 else 0.0
    sign_neg = 1.0 if mx < 0.0 else 0.0
    sp = sign_pos + sign_neg
    es = sign_pos - sign_neg

    # Left face SP correction
    fbL = (-u_im1 + 3.0 * u_i) / 2.0
    avgL = (u_i + u_ip1) / 2.0
    w1 = sp * (1.0 if uL * es > 0.0 else 0.0)
    w2 = sp * (1.0 if fbL * es > 0.0 else 0.0) * (1.0 - w1)
    uL_corr = w1 * uL + w2 * fbL + (sp - w1 - w2) * avgL
    uL = uL_corr + (1.0 - sp) * uL

    # Right face SP correction
    fbR = (-u_ip1 + 3.0 * u_i) / 2.0
    avgR = (u_im1 + u_i) / 2.0
    w1 = sp * (1.0 if uR * es > 0.0 else 0.0)
    w2 = sp * (1.0 if fbR * es > 0.0 else 0.0) * (1.0 - w1)
    uR_corr = w1 * uR + w2 * fbR + (sp - w1 - w2) * avgR
    uR = uR_corr + (1.0 - sp) * uR

    return uL, uR


# ===================================================================
# PW version
# ===================================================================

def weno5z_sp_pw(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """WENO5-Z sign-patch (PW). Post-hoc sign enforcement: if all 5 inputs
    share sign and WENO5-Z output violates it, cascade inner-upwind
    -> centered-average. Not Fjordholm & Ray 2016 SP-WENO."""
    uL, uR = _weno5z_LR_pw(u_im2, u_im1, u_i, u_ip1, u_ip2)

    # Sign-preserving flag and expected sign
    mn = min(u_im2, u_im1, u_i, u_ip1, u_ip2)
    mx = max(u_im2, u_im1, u_i, u_ip1, u_ip2)
    sign_pos = 1.0 if mn > 0.0 else 0.0
    sign_neg = 1.0 if mx < 0.0 else 0.0
    sp = sign_pos + sign_neg
    es = sign_pos - sign_neg

    # Left face SP correction
    fbL = (-u_im1 + 3.0 * u_i) / 2.0
    avgL = (u_i + u_ip1) / 2.0
    w1 = sp * (1.0 if uL * es > 0.0 else 0.0)
    w2 = sp * (1.0 if fbL * es > 0.0 else 0.0) * (1.0 - w1)
    uL_corr = w1 * uL + w2 * fbL + (sp - w1 - w2) * avgL
    uL = uL_corr + (1.0 - sp) * uL

    # Right face SP correction
    fbR = (-u_ip1 + 3.0 * u_i) / 2.0
    avgR = (u_im1 + u_i) / 2.0
    w1 = sp * (1.0 if uR * es > 0.0 else 0.0)
    w2 = sp * (1.0 if fbR * es > 0.0 else 0.0) * (1.0 - w1)
    uR_corr = w1 * uR + w2 * fbR + (sp - w1 - w2) * avgR
    uR = uR_corr + (1.0 - sp) * uR

    return uL, uR
# ---------------------------------------------------------------------------
# JIT compilation (pynalgo convention)
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    reveal_locals()  # noqa: F821
else:
    weno5z_sp_fv = JIT(weno5z_sp_fv)
    weno5z_sp_pw = JIT(weno5z_sp_pw)
#
# :D
#
