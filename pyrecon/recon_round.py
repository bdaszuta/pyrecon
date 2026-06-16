"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Unified framework for nonlinear reconstruction in compact stencil
"""
import math
from pyrecon._jit_utils import JIT, TYPE_CHECKING

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# NVD bounding: the region of admissible phi_f_tilde values for shape-
# preserving schemes (convection boundedness criterion)
#   phi_C_tilde <= phi_f_tilde <= 1  for 0 <= phi_C_tilde <= 1
#   phi_f_tilde = phi_C_tilde        for phi_C_tilde < 0 or phi_C_tilde > 1

_EPSL = 1e-40


# ---------------------------------------------------------------------------
# Base: NVD face value calculation
# ---------------------------------------------------------------------------

def _nvd_face(phi_tilde_C, phi_U, phi_D):
    r"""Compute face value from normalized variable.

    .. math::
       \phi_{i+1/2} = \phi_U + \tilde{\phi}_f (\phi_D - \phi_U)
    """
    return phi_U + phi_tilde_C * (phi_D - phi_U)


# ---------------------------------------------------------------------------
# Normalized variable computation
# ---------------------------------------------------------------------------

def _normalize(phi_C, phi_U, phi_D):
    """Compute normalized variable phi_C_tilde.

    phi_C_tilde = (phi_C - phi_U) / (phi_D - phi_U + eps)
    """
    return (phi_C - phi_U) / (phi_D - phi_U + _EPSL)


# ---------------------------------------------------------------------------
# ROUND mapping functions (phi_f_tilde = F(phi_C_tilde))
# ---------------------------------------------------------------------------

def _round_a_mapping(ptc):
    """ROUND-A: Aggressive nonlinear mapping.

    This is a shape-preserving cubic mapping that lies within the
    TVD region of the NVD and provides 3rd-order accuracy in smooth
    regions.

    For 0 <= ptc <= 1: phi_f = ptc * (1 + ptc - ptc^2)

    Reference: Deng et al. (2023), Eq. 42.
    """
    if ptc <= 0.0 or ptc >= 1.0:
        return ptc
    return ptc * (1.0 + ptc - ptc * ptc)


def _round_b_mapping(ptc):
    """ROUND-B: Balanced nonlinear mapping.

    Provides a balance between resolving power and shape preservation.

    phi_f = ptc + (1 - ptc) * ptc^2 / (ptc^2 + (1 - ptc)^2 + eps)

    Reference: Deng et al. (2023).
    """
    if ptc <= 0.0 or ptc >= 1.0:
        return ptc
    ptc2 = ptc * ptc
    omptc2 = (1.0 - ptc) * (1.0 - ptc)
    return ptc + (1.0 - ptc) * ptc2 / (ptc2 + omptc2 + _EPSL)


def _round_c_mapping(ptc):
    """ROUND-C: Smooth nonlinear mapping.

    A smooth mapping that transitions gradually between upwind and
    central interpolation.

    phi_f = ptc + ptc * (1 - ptc) * sin(pi * ptc) / 4

    Reference: Deng et al. (2023).
    """
    if ptc <= 0.0 or ptc >= 1.0:
        return ptc
    return ptc + ptc * (1.0 - ptc) * math.sin(math.pi * ptc) / 4.0


# ---------------------------------------------------------------------------
# ROUND-A reconstruction (single-face, left-sided at i+1/2)
# ---------------------------------------------------------------------------

def _round_L(phi_U, phi_C, phi_D, mapping_fn):
    r"""Left-face (i+1/2) reconstruction using ROUND mapping.

    Parameters
    ----------
    phi_U : upwind value (i-1)
    phi_C : center value (i)
    phi_D : downwind value (i+1)
    mapping_fn : callable, :math:`\tilde{\phi}_f = F(\tilde{\phi}_C)`

    Returns face value :math:`\phi_{i+1/2}^-`.
    """
    ptc = _normalize(phi_C, phi_U, phi_D)
    ptf = mapping_fn(ptc)
    return _nvd_face(ptf, phi_U, phi_D)


def _round_R(phi_U, phi_C, phi_D, mapping_fn):
    r"""Right-face (i-1/2) reconstruction using ROUND mapping.

    For the right face, we use phi_D=i-1, phi_C=i, phi_U=i+1
    (reversed orientation), then compute the face value.

    Parameters
    ----------
    phi_U : upwind value (i+1, convention: upwind for right face)
    phi_C : center value (i)
    phi_D : downwind value (i-1)

    Returns face value :math:`\phi_{i-1/2}^+`.
    """
    ptc = _normalize(phi_C, phi_U, phi_D)
    ptf = mapping_fn(ptc)
    return _nvd_face(ptf, phi_U, phi_D)


# ---------------------------------------------------------------------------
# Paired L+R ROUND reconstruction
# ---------------------------------------------------------------------------

def _round_LR(u_im1, u_i, u_ip1, mapping_fn):
    """Paired L+R reconstruction using ROUND mapping.

    """
    # Left face (i+1/2): U=i-1, C=i, D=i+1
    uL = _round_L(u_im1, u_i, u_ip1, mapping_fn)

    # Right face (i-1/2): U=i+1, C=i, D=i-1
    uR = _round_R(u_ip1, u_i, u_im1, mapping_fn)

    return uL, uR


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def round_a_fv(u_im1, u_i, u_ip1):
    """ROUND-A reconstruction (aggressive nonlinear, 3-point compact).

    Shape-preserving cubic mapping within the TVD region of NVD.
    Provides 3rd-order accuracy in smooth regions with strong
    discontinuity-capturing.

    """
    return _round_LR(u_im1, u_i, u_ip1, _round_a_mapping)


def round_b_fv(u_im1, u_i, u_ip1):
    """ROUND-B reconstruction (balanced nonlinear, 3-point compact).

    WENO-inspired NVD mapping that balances resolving power and
    shape preservation. Good general-purpose choice.

    """
    return _round_LR(u_im1, u_i, u_ip1, _round_b_mapping)


def round_c_fv(u_im1, u_i, u_ip1):
    """ROUND-C reconstruction (smooth nonlinear, 3-point compact).

    Smooth sinusoidal mapping for high resolving efficiency in
    smooth regions with gradual transition near discontinuities.

    """
    return _round_LR(u_im1, u_i, u_ip1, _round_c_mapping)
# ---------------------------------------------------------------------------
# JIT compilation (pynalgo convention)
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    reveal_locals()  # noqa: F821
else:
    _nvd_face = JIT(_nvd_face)
    _normalize = JIT(_normalize)
    _round_a_mapping = JIT(_round_a_mapping)
    _round_b_mapping = JIT(_round_b_mapping)
    _round_c_mapping = JIT(_round_c_mapping)
    _round_L = JIT(_round_L)
    _round_R = JIT(_round_R)
    _round_LR = JIT(_round_LR)
    round_a_fv = JIT(round_a_fv)
    round_b_fv = JIT(round_b_fv)
    round_c_fv = JIT(round_c_fv)
#
# :D
#
