"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Integration tests for hydro_driver.shock_tube
"""
import numpy as np
import pytest
from hydro_driver.shock_tube import (
    ShockTubeConfig, setup_grid, setup_initial,
    compute_dt, evolve_step, get_problem,
)
from hydro_driver.eos import (
    IdealGasEOS, IDN, IRHO, IUX, IP,
    prim_to_cons as prim_to_cons_module,
)


@pytest.fixture
def eos():
    return IdealGasEOS(gamma=2.0)


@pytest.fixture
def sod_config():
    return get_problem("sod", ncells=100, cfl=0.3,
                       recon_method="weno5z_fv", riemann_solver="llf")


# -- Grid and ICs --------------------------------------------------------


def test_setup_grid():
    x, dx = setup_grid(10, -0.5, 0.5)
    assert len(x) == 10
    assert abs(dx - 0.1) < 1e-12
    assert abs(x[0] - (-0.45)) < 1e-12
    assert abs(x[-1] - 0.45) < 1e-12


def test_setup_initial_left_state(eos, sod_config):
    x, _ = setup_grid(sod_config.ncells,
                      sod_config.xmin, sod_config.xmax)
    prim, cons = setup_initial(x, sod_config, eos)
    # Left half (x < 0): rho=1, P=1, v=0 -> u_x=0
    assert abs(prim[IRHO, 0] - 1.0) < 1e-12
    assert abs(prim[IP, 0] - 1.0) < 1e-12
    assert abs(prim[IUX, 0]) < 1e-12


def test_setup_initial_right_state(eos, sod_config):
    x, _ = setup_grid(sod_config.ncells,
                      sod_config.xmin, sod_config.xmax)
    prim, cons = setup_initial(x, sod_config, eos)
    # Right half (x > 0): rho=0.125, P=0.1, v=0
    assert abs(prim[IRHO, -1] - 0.125) < 1e-12
    assert abs(prim[IP, -1] - 0.1) < 1e-12


def test_get_problem_unknown():
    with pytest.raises(ValueError):
        get_problem("nonexistent")


# -- CFL timestep --------------------------------------------------------


def test_compute_dt_positive(eos, sod_config):
    x, dx = setup_grid(sod_config.ncells,
                       sod_config.xmin, sod_config.xmax)
    _, cons = setup_initial(x, sod_config, eos)
    dt = compute_dt(cons, eos, dx, 0.3)
    assert dt > 0
    # dt = cfl*dx / (|v|+cs); since |v|+cs < 1 in SR, dt > cfl*dx
    assert dt > 0.3 * dx


# -- Evolution -----------------------------------------------------------


def test_evolve_step_no_nan(eos, sod_config):
    x, dx = setup_grid(sod_config.ncells,
                       sod_config.xmin, sod_config.xmax)
    _, cons = setup_initial(x, sod_config, eos)
    dt = compute_dt(cons, eos, dx, 0.3)
    cons2 = evolve_step(cons, dx, dt, eos, "weno5z_fv", "llf")
    assert not np.any(np.isnan(cons2))
    assert cons2.shape == cons.shape


def test_donor_cell_conservation(eos):
    """Donor cell (donate_fv) should conserve total D on symmetric ICs."""
    ncells = 50
    cfg = ShockTubeConfig(
        name="symmetric", rho_left=1.0, P_left=1.0, vx_left=0.0,
        rho_right=1.0, P_right=1.0, vx_right=0.0,
        ncells=ncells, cfl=0.3, tfinal=0.1,
        recon_method="donate_fv", riemann_solver="llf")
    x, dx = setup_grid(ncells, cfg.xmin, cfg.xmax)
    _, cons = setup_initial(x, cfg, eos)

    total_D_initial = np.sum(cons[IDN])
    for _ in range(10):
        dt = compute_dt(cons, eos, dx, cfg.cfl)
        cons = evolve_step(cons, dx, dt, eos, "donate_fv", "llf")
    total_D_final = np.sum(cons[IDN])
    assert abs(total_D_final - total_D_initial) < 1e-10


def test_donor_cell_symmetry(eos):
    """Symmetric ICs should stay symmetric under donor cell."""
    ncells = 100
    cfg = ShockTubeConfig(
        name="symmetric", rho_left=1.0, P_left=1.0, vx_left=0.0,
        rho_right=1.0, P_right=1.0, vx_right=0.0,
        ncells=ncells, cfl=0.3, tfinal=0.1,
        recon_method="donate_fv", riemann_solver="llf")
    x, dx = setup_grid(ncells, cfg.xmin, cfg.xmax)
    _, cons = setup_initial(x, cfg, eos)

    for _ in range(10):
        dt = compute_dt(cons, eos, dx, cfg.cfl)
        cons = evolve_step(cons, dx, dt, eos, "donate_fv", "llf")

    # All cells should have (nearly) same D
    D_vals = cons[IDN]
    assert np.max(D_vals) - np.min(D_vals) < 1e-12


def test_sod_multi_step_no_nan(eos):
    """Sod problem with WENO5Z + LLF, 50 steps, no NaN."""
    cfg = get_problem("sod", ncells=100, cfl=0.1,
                      recon_method="weno5z_fv", riemann_solver="llf")
    x, dx = setup_grid(cfg.ncells, cfg.xmin, cfg.xmax)
    _, cons = setup_initial(x, cfg, eos)

    for _ in range(50):
        dt = compute_dt(cons, eos, dx, cfg.cfl)
        cons = evolve_step(cons, dx, dt, eos, "weno5z_fv", "llf")
        if np.any(np.isnan(cons)):
            pytest.fail("NaN in conserved variables")
    assert not np.any(np.isnan(cons))


def test_blast_multi_step_no_nan(eos):
    """Blast problem, donor cell, 50 steps, no NaN."""
    cfg = get_problem("blast", ncells=100, cfl=0.05,
                      recon_method="donate_fv", riemann_solver="llf")
    x, dx = setup_grid(cfg.ncells, cfg.xmin, cfg.xmax)
    _, cons = setup_initial(x, cfg, eos)

    for _ in range(50):
        dt = compute_dt(cons, eos, dx, cfg.cfl)
        cons = evolve_step(cons, dx, dt, eos, "donate_fv", "llf")
        if np.any(np.isnan(cons)):
            pytest.fail("NaN in conserved variables")


def test_collide_multi_step_no_nan(eos):
    """Collide problem, WENO5Z + HLLE, 50 steps, no NaN."""
    cfg = get_problem("collide", ncells=200, cfl=0.05,
                      recon_method="weno5z_fv", riemann_solver="hlle")
    x, dx = setup_grid(cfg.ncells, cfg.xmin, cfg.xmax)
    _, cons = setup_initial(x, cfg, eos)

    for _ in range(50):
        dt = compute_dt(cons, eos, dx, cfg.cfl)
        cons = evolve_step(cons, dx, dt, eos, "weno5z_fv", "hlle")
        if np.any(np.isnan(cons)):
            pytest.fail("NaN in conserved variables")


def test_weno5z_converges(eos):
    """WENO5Z should converge on smooth sine ICs."""
    import numpy as np
    from hydro_driver.eos import cons_to_prim

    errors = []
    for ncells in [64, 128, 256]:
        x, dx = setup_grid(ncells, 0.0, 1.0)
        prim = np.zeros((3, ncells))
        for i in range(ncells):
            rho = 1.0 + 0.1 * np.sin(2 * np.pi * x[i] + 0.7)
            P = 1.0 + 0.05 * np.sin(2 * np.pi * (x[i] + 0.3) + 0.3)
            vx = 0.01 * np.sin(2 * np.pi * (x[i] + 0.5))
            v2 = vx * vx
            if v2 >= 1.0:
                v2 = 0.999999
            W = 1.0 / np.sqrt(1.0 - v2)
            u_x = W * vx
            prim[IRHO, i] = rho
            prim[IUX, i] = u_x
            prim[IP, i] = P

        cfg = ShockTubeConfig(
            name="conv", rho_left=1.0, P_left=1.0,
            rho_right=1.0, P_right=1.0,
            ncells=ncells, cfl=0.01, tfinal=0.02,
            recon_method="weno5z_fv", riemann_solver="llf",
            xmin=0.0, xmax=1.0)
        _, cons = setup_initial(x, cfg, eos)
        # Overwrite ICs with smooth data
        for i in range(ncells):
            cons[:, i] = prim_to_cons_module(prim[:, i], eos)
        # 1 small step
        dt = compute_dt(cons, eos, dx, cfg.cfl)
        cons = evolve_step(cons, dx, dt, eos, "weno5z_fv", "llf")
        # Recover primitives
        prim_out = np.zeros_like(prim)
        for i in range(ncells):
            prim_out[:, i] = cons_to_prim(cons[:, i].copy(), eos)
        err = np.mean(np.abs(prim_out[IRHO] - prim[IRHO]))
        errors.append(err)

    # Check convergence: error should decrease with resolution
    assert errors[0] > errors[-1], f"not converging: {errors}"


# -- RK3 integrator ------------------------------------------------------


def test_rk3_euler_agree_small_dt(eos):
    """RK3 and Euler should agree closely at very small dt."""
    from hydro_driver.shock_tube import (
        evolve_step, evolve_step_rk3,
    )
    ncells = 64
    x, dx = setup_grid(ncells, 0.0, 1.0)
    cfg = ShockTubeConfig(
        name="pulse", rho_left=1.0, P_left=1.0,
        rho_right=1.0, P_right=1.0,
        ncells=ncells, cfl=0.001, tfinal=0.01,
        xmin=0.0, xmax=1.0)
    _, cons = setup_initial(x, cfg, eos)
    dt = compute_dt(cons, eos, dx, cfg.cfl)

    cons_euler = evolve_step(cons.copy(), dx, dt, eos,
                             "weno5z_fv", "llf")
    cons_rk3 = evolve_step_rk3(cons.copy(), dx, dt, eos,
                               "weno5z_fv", "llf")
    diff = np.max(np.abs(cons_rk3 - cons_euler))
    assert diff < 1e-6, f"RK3 and Euler differ by {diff:.2e} at dt={dt:.2e}"


def test_rk3_no_nan(eos):
    """Sod problem with RK3, WENO5Z + HLLE, 50 steps, no NaN."""
    cfg = get_problem("sod", ncells=100, cfl=0.1,
                      recon_method="weno5z_fv", riemann_solver="hlle",
                      time_integrator="rk3")
    x, dx = setup_grid(cfg.ncells, cfg.xmin, cfg.xmax)
    _, cons = setup_initial(x, cfg, eos)

    for _ in range(50):
        dt = compute_dt(cons, eos, dx, cfg.cfl)
        from hydro_driver.shock_tube import evolve_step_rk3
        cons = evolve_step_rk3(cons, dx, dt, eos,
                               "weno5z_fv", "hlle")
        if np.any(np.isnan(cons)):
            pytest.fail("NaN in conserved variables")
    assert not np.any(np.isnan(cons))


def test_rk3_converges(eos):
    """RK3 + WENO5Z converges faster than first order on smooth ICs."""
    from hydro_driver.eos import cons_to_prim
    import numpy as np

    errors = []
    for ncells in [64, 128, 256]:
        x, dx = setup_grid(ncells, 0.0, 1.0)
        prim = np.zeros((3, ncells))
        for i in range(ncells):
            rho = 1.0 + 0.1 * np.sin(2 * np.pi * x[i] + 0.7)
            P = 1.0 + 0.05 * np.sin(2 * np.pi * (x[i] + 0.3) + 0.3)
            vx = 0.01 * np.sin(2 * np.pi * (x[i] + 0.5))
            v2 = vx * vx
            if v2 >= 1.0:
                v2 = 0.999999
            W = 1.0 / np.sqrt(1.0 - v2)
            u_x = W * vx
            prim[IRHO, i] = rho
            prim[IUX, i] = u_x
            prim[IP, i] = P

        cfg = ShockTubeConfig(
            name="conv", rho_left=1.0, P_left=1.0,
            rho_right=1.0, P_right=1.0,
            ncells=ncells, cfl=0.01, tfinal=0.02,
            recon_method="weno5z_fv", riemann_solver="llf",
            time_integrator="rk3", xmin=0.0, xmax=1.0)
        _, cons = setup_initial(x, cfg, eos)
        for i in range(ncells):
            cons[:, i] = prim_to_cons_module(prim[:, i], eos)

        from hydro_driver.shock_tube import evolve_step_rk3
        dt = compute_dt(cons, eos, dx, cfg.cfl)
        cons = evolve_step_rk3(cons, dx, dt, eos,
                               "weno5z_fv", "llf")
        prim_out = np.zeros_like(prim)
        for i in range(ncells):
            prim_out[:, i] = cons_to_prim(cons[:, i].copy(), eos)
        err = np.mean(np.abs(prim_out[IRHO] - prim[IRHO]))
        errors.append(err)

    assert errors[0] > errors[-1], f"not converging: {errors}"
#
# :D
#
