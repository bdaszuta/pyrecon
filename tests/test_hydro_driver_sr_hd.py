"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Unit tests for hydro_driver.eos and hydro_driver.sr_hd
"""
import numpy as np
import pytest
from hydro_driver.eos import (
    IdealGasEOS, prim_to_cons, cons_to_prim,
    IDN, ISX, ITAU,
)
from hydro_driver.sr_hd import (
    compute_fluxes, compute_wave_speeds, llf_flux, hlle_flux,
)


@pytest.fixture
def eos():
    return IdealGasEOS(gamma=2.0)


# -- EOS class -----------------------------------------------------------


def test_enthalpy_static(eos):
    h = eos.enthalpy(1.0, 1.0)
    assert abs(h - 3.0) < 1e-12


def test_sound_speed_sq_static(eos):
    cs2 = eos.sound_speed_sq(1.0, 1.0)
    assert abs(cs2 - 2.0/3.0) < 1e-12


def test_sound_speed_sq_vacuum(eos):
    cs2 = eos.sound_speed_sq(1e-8, 1e-12)
    assert cs2 > 0
    assert cs2 < 1.0


# -- P2C / C2P roundtrip -------------------------------------------------


def test_prim_to_cons_static(eos):
    prim = np.array([1.0, 0.0, 1.0])
    cons = prim_to_cons(prim, eos)
    assert abs(cons[IDN] - 1.0) < 1e-12
    assert abs(cons[ISX] - 0.0) < 1e-12
    assert abs(cons[ITAU] - 1.0) < 1e-12


def test_prim_to_cons_moving(eos):
    prim = np.array([1.0, 0.6, 1.0])  # u_x=0.6 -> v_x~0.514, W~1.166
    cons = prim_to_cons(prim, eos)
    W = np.sqrt(1.0 + 0.6**2)
    assert abs(cons[IDN] - W) < 1e-12
    assert cons[ISX] > 0


def test_cons_to_prim_roundtrip_static(eos):
    prim_in = np.array([1.0, 0.0, 1.0])
    cons = prim_to_cons(prim_in, eos)
    prim_out = cons_to_prim(cons, eos)
    for i in range(3):
        assert abs(prim_in[i] - prim_out[i]) < 1e-10


def test_cons_to_prim_roundtrip_mild(eos):
    prim_in = np.array([1.0, 0.5, 1.0])
    cons = prim_to_cons(prim_in, eos)
    prim_out = cons_to_prim(cons, eos)
    for i in range(3):
        assert abs(prim_in[i] - prim_out[i]) < 1e-10


def test_cons_to_prim_roundtrip_hot(eos):
    prim_in = np.array([1.0, 2.0, 10.0])
    cons = prim_to_cons(prim_in, eos)
    prim_out = cons_to_prim(cons, eos)
    for i in range(3):
        assert abs(prim_in[i] - prim_out[i]) < 1e-10


# -- Flux computation ----------------------------------------------------


def test_flux_static_fluid(eos):
    prim = np.array([1.0, 0.0, 1.0])
    flux = compute_fluxes(prim, eos)
    assert abs(flux[IDN]) < 1e-12
    assert abs(flux[ISX] - 1.0) < 1e-12   # F_Sx = P when v=0
    assert abs(flux[ITAU]) < 1e-12


def test_flux_moving_fluid(eos):
    prim = np.array([1.0, 0.5, 1.0])
    flux = compute_fluxes(prim, eos)
    assert flux[IDN] > 0
    assert flux[ISX] > 1.0      # S_x*v_x + P > P
    # F_tau = S_x - D*v_x (NOT F_Sx - F_D)
    # S_x = 1.677..., D*v_x = 0.5, so F_tau = 1.177...
    assert abs(flux[ITAU] - 1.1770509831248424) < 1e-10


# -- Wave speeds ---------------------------------------------------------


def test_wave_speeds_static(eos):
    prim = np.array([1.0, 0.0, 1.0])
    lam_m, lam_p = compute_wave_speeds(prim, eos)
    cs = np.sqrt(eos.sound_speed_sq(1.0, 1.0))
    assert abs(lam_m + cs) < 1e-12
    assert abs(lam_p - cs) < 1e-12


def test_wave_speeds_subluminal(eos):
    prim = np.array([1.0, 0.5, 1.0])
    lam_m, lam_p = compute_wave_speeds(prim, eos)
    assert abs(lam_m) < 1.0
    assert abs(lam_p) < 1.0


def test_wave_speeds_ordering(eos):
    prim = np.array([1.0, 0.5, 1.0])
    lam_m, lam_p = compute_wave_speeds(prim, eos)
    assert lam_m <= lam_p


# -- LLF solver ----------------------------------------------------------


def test_llf_identical_states(eos):
    prim = np.array([1.0, 0.0, 1.0])
    flux_llf = llf_flux(prim, prim, eos)
    flux_ref = compute_fluxes(prim, eos)
    assert np.allclose(flux_llf, flux_ref)


def test_llf_positive_flux(eos):
    prim_L = np.array([2.0, 0.5, 2.0])
    prim_R = np.array([1.0, 0.0, 1.0])
    flux = llf_flux(prim_L, prim_R, eos)
    assert flux[IDN] > 0


# -- HLLE solver ---------------------------------------------------------


def test_hlle_identical_states(eos):
    prim = np.array([1.0, 0.0, 1.0])
    flux_hlle = hlle_flux(prim, prim, eos)
    flux_ref = compute_fluxes(prim, eos)
    assert np.allclose(flux_hlle, flux_ref)


def test_hlle_supersonic_right(eos):
    prim_L = np.array([1.0, 5.0, 1.0])    # v ~ 0.981, cs=0.816, lam_m > 0
    prim_R = np.array([1.0, 2.0, 1.0])    # v ~ 0.894, cs=0.816, lam_m > 0
    lam_m_L, lam_p_L = compute_wave_speeds(prim_L, eos)
    lam_m_R, lam_p_R = compute_wave_speeds(prim_R, eos)
    lam_L = min(lam_m_L, lam_m_R)
    assert lam_L > 0, (
        f"not supersonic: lam_m_L={lam_m_L:.4f} lam_m_R={lam_m_R:.4f}")
    flux_hlle = hlle_flux(prim_L, prim_R, eos)
    flux_ref = compute_fluxes(prim_L, eos)
    assert np.allclose(flux_hlle, flux_ref)


def test_hlle_supersonic_left(eos):
    prim_L = np.array([1.0, -5.0, 1.0])
    prim_R = np.array([1.0, -2.0, 1.0])
    lam_m_L, lam_p_L = compute_wave_speeds(prim_L, eos)
    lam_m_R, lam_p_R = compute_wave_speeds(prim_R, eos)
    lam_R = max(lam_p_L, lam_p_R)
    assert lam_R < 0, (
        f"not supersonic: lam_p_L={lam_p_L:.4f} lam_p_R={lam_p_R:.4f}")
    flux_hlle = hlle_flux(prim_L, prim_R, eos)
    flux_ref = compute_fluxes(prim_R, eos)
    assert np.allclose(flux_hlle, flux_ref)


def test_hlle_degenerate_fallback(eos):
    prim = np.array([1.0, 0.0, 1e-30])    # cs ~ 4.5e-15, effectively zero
    lam_m, lam_p = compute_wave_speeds(prim, eos)
    # Eigenvalues should be very small; HLLE should fall back to LLF
    # (dlam < eps_abs triggers fallback gate)
    flux_hlle = hlle_flux(prim, prim, eos)
    flux_ref = compute_fluxes(prim, eos)
    assert np.allclose(flux_hlle, flux_ref)
#
# :D
#
