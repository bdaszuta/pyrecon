"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: 1+1D SR hydro flux, eigenvalues, LLF/HLLE Riemann solvers

"""

import numpy as np
from hydro_driver.eos import (
    IRHO, IUX, IP, prim_to_cons,
)


def compute_fluxes(prim, eos):
    """Compute undensitized fluxes from primitive state (1+1D flat bg).

    Parameters
    ----------
    prim : ndarray, shape (3,)
        rho, u_x, P
    eos : IdealGasEOS

    Returns
    -------
    flux : ndarray, shape (3,)
        F_D, F_Sx, F_tau

    Two-pass: (1) convective fluxes, (2) pressure addition to normal component.
    """
    rho = prim[IRHO]
    P   = prim[IP]
    u_x = prim[IUX]

    W = np.sqrt(1.0 + u_x * u_x)
    v_x = u_x / W
    h = eos.enthalpy(rho, P)

    D   = rho * W
    S_x = rho * h * W * u_x
    tau = rho * h * W * W - P - D

    # Convective fluxes
    F_D   = D * v_x
    F_Sx  = S_x * v_x
    F_tau = tau * v_x + P * v_x

    # Pressure addition (separate pass)
    F_Sx += P

    return np.array([F_D, F_Sx, F_tau])


def compute_wave_speeds(prim, eos):
    """Compute SR acoustic eigenvalues for 1+1D flat Minkowski bg.

    Parameters
    ----------
    prim : ndarray, shape (3,)
        rho, u_x, P
    eos : IdealGasEOS

    Returns
    -------
    lambda_m, lambda_p : float

    Flat Minkowski background: alpha=1, beta=0, gamma^uu=1.
    """
    rho = prim[IRHO]
    P   = prim[IP]
    u_x = prim[IUX]

    cs2 = eos.sound_speed_sq(rho, P)
    W = np.sqrt(1.0 + u_x * u_x)
    v_x = u_x / W
    v2 = v_x * v_x                          # only x-component nonzero

    # General relativistic acoustic eigenvalue formula
    disc = cs2 * (1.0 - v2) * (
        1.0 * (1.0 - v2 * cs2) - v_x * v_x * (1.0 - cs2))
    cs_sqrt_term = np.sqrt(max(disc, 0.0))
    oo_denom = 1.0 / (1.0 - v2 * cs2)
    vi_term = v_x * (1.0 - cs2)

    lam_1 = vi_term + cs_sqrt_term
    lam_2 = vi_term - cs_sqrt_term

    if not np.isfinite(lam_1 + lam_2):
        return -1.0, 1.0

    lam_p = lam_1 * oo_denom
    lam_m = lam_2 * oo_denom

    if lam_p < lam_m:
        lam_p, lam_m = lam_m, lam_p

    return lam_m, lam_p


def llf_flux(prim_L, prim_R, eos):
    """LLF (Rusanov) Riemann solver, 1+1D.

    Parameters
    ----------
    prim_L, prim_R : ndarray, shape (3,)
    eos : IdealGasEOS

    Returns
    -------
    flux : ndarray, shape (3,)

    LLF (Rusanov) flux with extremal wavespeed.
    """
    F_L = compute_fluxes(prim_L, eos)
    F_R = compute_fluxes(prim_R, eos)
    cons_L = prim_to_cons(prim_L, eos)
    cons_R = prim_to_cons(prim_R, eos)

    lam_m_L, lam_p_L = compute_wave_speeds(prim_L, eos)
    lam_m_R, lam_p_R = compute_wave_speeds(prim_R, eos)

    # Extremal wavespeed (lambda_scale=1)
    lam_l = min(lam_m_L, lam_m_R)
    lam_r = max(lam_p_L, lam_p_R)
    lam_max = max(lam_r, -lam_l)

    return 0.5 * (F_L + F_R) - 0.5 * lam_max * (cons_R - cons_L)


def hlle_flux(prim_L, prim_R, eos,
              eps_abs=1.0e-12, eps_rel=1.0e-6):
    """HLLE Riemann solver, 1+1D.

    Parameters
    ----------
    prim_L, prim_R : ndarray, shape (3,)
    eos : IdealGasEOS
    eps_abs : float
        Absolute tolerance (default: 1.0e-12).
    eps_rel : float
        Relative tolerance (default: 1.0e-6).

    Returns
    -------
    flux : ndarray, shape (3,)

    HLLE flux with LLF fallback for degenerate eigenvalues.
    """
    F_L = compute_fluxes(prim_L, eos)
    F_R = compute_fluxes(prim_R, eos)
    cons_L = prim_to_cons(prim_L, eos)
    cons_R = prim_to_cons(prim_R, eos)

    lam_m_L, lam_p_L = compute_wave_speeds(prim_L, eos)
    lam_m_R, lam_p_R = compute_wave_speeds(prim_R, eos)

    lam_L = min(lam_m_L, lam_m_R)
    lam_R = max(lam_p_L, lam_p_R)
    dlam = lam_R - lam_L
    speed_scale = max(abs(lam_L), abs(lam_R), 1.0)
    dlam_tol = eps_abs + eps_rel * speed_scale

    # Pre-computed extremal wavespeed for LLF fallback
    lam_max = max(lam_R, -lam_L)

    if dlam <= dlam_tol:
        return 0.5 * (F_L + F_R) - 0.5 * lam_max * (cons_R - cons_L)
    elif lam_L >= 0.0:
        return F_L
    elif lam_R <= 0.0:
        return F_R
    else:
        oo_dlam = 1.0 / dlam
        return (lam_R * F_L - lam_L * F_R +
                lam_L * lam_R * (cons_R - cons_L)) * oo_dlam
#
# :D
#
