# ruff: noqa: E402
"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: 1+1D SR hydro shock-tube evolution driver
"""

import os
import sys
from dataclasses import dataclass

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
from hydro_driver.eos import (
    NCONS,
    IRHO, IUX, IP, NPRIM,
    IdealGasEOS, prim_to_cons, cons_to_prim,
    RHO_ATM, P_ATM,
)
from hydro_driver.sr_hd import llf_flux, hlle_flux
from pyrecon import reconstruct_array


@dataclass
class ShockTubeConfig:
    """Shock tube problem configuration (1+1D)."""
    name: str
    rho_left: float
    P_left: float
    vx_left: float = 0.0
    rho_right: float = 0.0
    P_right: float = 0.0
    vx_right: float = 0.0
    xmin: float = -0.5
    xmax: float = 0.5
    xshock: float = 0.0
    ncells: int = 400
    cfl: float = 0.3
    tfinal: float = 0.2
    gamma: float = 2.0
    recon_method: str = "weno5z_fv"
    riemann_solver: str = "llf"
    time_integrator: str = "euler"


def setup_grid(ncells, xmin, xmax):
    """Return cell-center x (shape N) and dx."""
    dx = (xmax - xmin) / ncells
    x = np.linspace(xmin + 0.5 * dx, xmax - 0.5 * dx, ncells)
    return x, dx


def setup_initial(x, config, eos):
    """Set up initial primitive and conserved arrays (1+1D).

    Input velocities are 3-velocities v_x (NOT 4-velocity u_x).
    Converts: u_x = W * v_x for storage.

    Returns
    -------
    prim_grid : ndarray, shape (3, N)
        prim[IRHO]=rho, prim[IUX]=u_x, prim[IP]=P
    cons_grid : ndarray, shape (3, N)
        D, S_x, tau
    """
    N = len(x)
    prim_grid = np.zeros((NPRIM, N))
    cons_grid = np.zeros((NCONS, N))

    for i in range(N):
        if x[i] < config.xshock:
            rho = config.rho_left
            P   = config.P_left
            vx  = config.vx_left
        else:
            rho = config.rho_right
            P   = config.P_right
            vx  = config.vx_right

        v2 = vx * vx
        if v2 >= 1.0:
            v2 = 0.999999
        W = 1.0 / np.sqrt(1.0 - v2)
        u_x = W * vx

        prim_grid[IRHO, i] = rho
        prim_grid[IUX, i]  = u_x
        prim_grid[IP, i]   = P

        cons_grid[:, i] = prim_to_cons(prim_grid[:, i], eos)

    return prim_grid, cons_grid


def compute_dt(cons_grid, eos, dx, cfl):
    """SR CFL timestep: dt = cfl * dx / max_i(|v_x_i| + cs_i).
    """
    N = cons_grid.shape[1]
    max_speed = 0.0
    for i in range(N):
        try:
            prim = cons_to_prim(cons_grid[:, i].copy(), eos)
            rho = prim[IRHO]
            P = prim[IP]
            u_x = prim[IUX]
            W = np.sqrt(1.0 + u_x * u_x)
            v_x = u_x / W
            cs = np.sqrt(eos.sound_speed_sq(rho, P))
            speed = abs(v_x) + cs
            if speed > max_speed:
                max_speed = speed
        except ValueError:
            max_speed = max(max_speed, 1.0)
    if max_speed < 1e-10:
        max_speed = 1.0
    return cfl * dx / max_speed


def compute_rhs(cons_grid, dx, eos, recon_method, riemann_solver):
    """Compute -div(F) for the conserved evolution.

    C2P -> reconstruct -> Riemann solve -> return flux divergence.
    Modifies cons_grid in-place on C2P failure (atmosphere fallback).
    """
    N = cons_grid.shape[1]
    prim_grid = np.zeros((NPRIM, N))

    # 1. C2P
    for i in range(N):
        try:
            prim_grid[:, i] = cons_to_prim(cons_grid[:, i].copy(), eos)
        except ValueError:
            prim_grid[IRHO, i] = RHO_ATM
            prim_grid[IUX, i]  = 0.0
            prim_grid[IP, i]   = P_ATM
            cons_grid[:, i] = prim_to_cons(prim_grid[:, i], eos)

    riemann = llf_flux if riemann_solver == "llf" else hlle_flux

    # 2. Reconstruct each primitive field independently
    zl_fields = []
    zr_fields = []
    for k in range(NPRIM):
        zl_k, zr_k = reconstruct_array(recon_method, prim_grid[k])
        zl_fields.append(zl_k)
        zr_fields.append(zr_k)

    # 3. Riemann solve at interior faces (fi = 1..N-1)
    flux_grid = np.zeros((NCONS, N + 1))
    for fi in range(1, N):
        prim_L = np.array([zl_fields[k][fi] for k in range(NPRIM)])
        prim_R = np.array([zr_fields[k][fi] for k in range(NPRIM)])
        flux_grid[:, fi] = riemann(prim_L, prim_R, eos)

    # 4. Boundary faces: constant continuation
    prim_edge = prim_grid[:, 0]
    flux_grid[:, 0] = riemann(prim_edge, prim_edge, eos)
    prim_edge = prim_grid[:, N - 1]
    flux_grid[:, N] = riemann(prim_edge, prim_edge, eos)

    # 5. Flux divergence: -div(F)
    rhs = np.zeros_like(cons_grid)
    oo_dx = 1.0 / dx
    for i in range(N):
        rhs[:, i] = -(flux_grid[:, i + 1] - flux_grid[:, i]) * oo_dx
    return rhs


def evolve_step(cons_grid, dx, dt, eos, recon_method, riemann_solver):
    """One Forward Euler step in 1+1D."""
    rhs = compute_rhs(cons_grid, dx, eos, recon_method, riemann_solver)
    return cons_grid + dt * rhs


def evolve_step_rk3(cons_grid, dx, dt, eos, recon_method, riemann_solver):
    """One SSP RK3 (Shu & Osher 1988) step in 1+1D.

    u^(1)  = u^n + dt * RHS(u^n)
    u^(2)  = 3/4 u^n + 1/4 u^(1) + 1/4 dt * RHS(u^(1))
    u^(n+1) = 1/3 u^n + 2/3 u^(2) + 2/3 dt * RHS(u^(2))
    """
    # Stage 1
    rhs1 = compute_rhs(cons_grid, dx, eos, recon_method, riemann_solver)
    cons1 = cons_grid + dt * rhs1

    # Stage 2
    rhs2 = compute_rhs(cons1, dx, eos, recon_method, riemann_solver)
    cons2 = (3.0/4.0 * cons_grid + 1.0/4.0 * cons1
             + 1.0/4.0 * dt * rhs2)

    # Stage 3
    rhs3 = compute_rhs(cons2, dx, eos, recon_method, riemann_solver)
    cons_new = (1.0/3.0 * cons_grid + 2.0/3.0 * cons2
                + 2.0/3.0 * dt * rhs3)
    return cons_new


def run(config, output_interval=0):
    """Run the 1+1D shock tube evolution.

    Returns: x, prim_final, cons_final, history dict.
    """
    eos = IdealGasEOS(gamma=config.gamma)
    x, dx = setup_grid(config.ncells, config.xmin, config.xmax)
    prim, cons = setup_initial(x, config, eos)

    t = 0.0
    step = 0
    history = {'t': [t], 'step': [step], 'snapshots': []}
    if output_interval > 0:
        history['snapshots'].append((t, prim.copy(), cons.copy()))

    evolve = (evolve_step if config.time_integrator == "euler"
              else evolve_step_rk3)

    while t < config.tfinal:
        dt = compute_dt(cons, eos, dx, config.cfl)
        dt = min(dt, config.tfinal - t)
        cons = evolve(cons, dx, dt, eos,
                     config.recon_method, config.riemann_solver)
        t += dt
        step += 1
        history['t'].append(t)
        history['step'].append(step)

        if output_interval > 0 and step % output_interval == 0:
            prim_snap = np.zeros_like(prim)
            for i in range(config.ncells):
                try:
                    prim_snap[:, i] = cons_to_prim(cons[:, i].copy(), eos)
                except ValueError:
                    prim_snap[IRHO, i] = RHO_ATM
                    prim_snap[IP, i] = P_ATM
            history['snapshots'].append(
                (t, prim_snap.copy(), cons.copy()))

        if step % 10 == 0:
            print(f"step={step:5d}  t={t:.6e}  dt={dt:.6e}")

    prim_final = np.zeros_like(prim)
    for i in range(config.ncells):
        try:
            prim_final[:, i] = cons_to_prim(cons[:, i].copy(), eos)
        except ValueError:
            prim_final[IRHO, i] = RHO_ATM
            prim_final[IP, i] = P_ATM

    return x, prim_final, cons, history


def get_problem(name, ncells=400, cfl=0.3, recon_method="weno5z_fv",
                riemann_solver="llf", time_integrator="euler"):
    """Standard shock tube test problems (1+1D, Gamma=2).

    All use 3-velocity v_x as input.

    Problems:
      "sod"     - Sod shock tube (mild)
      "blast"   - Strong blast wave
      "collide" - Colliding relativistic streams
    """
    problems = {
        "sod": ShockTubeConfig(
            name="sod",
            rho_left=1.0, P_left=1.0, vx_left=0.0,
            rho_right=0.125, P_right=0.1, vx_right=0.0,
            xmin=-0.5, xmax=0.5, xshock=0.0,
            ncells=ncells, cfl=cfl, tfinal=0.5,
            recon_method=recon_method,
            riemann_solver=riemann_solver,
            time_integrator=time_integrator),
        "blast": ShockTubeConfig(
            name="blast",
            rho_left=1.0, P_left=1000.0, vx_left=0.0,
            rho_right=1.0, P_right=0.01, vx_right=0.0,
            xmin=-0.5, xmax=0.5, xshock=0.0,
            ncells=ncells, cfl=cfl, tfinal=0.5,
            recon_method=recon_method,
            riemann_solver=riemann_solver,
            time_integrator=time_integrator),
        "collide": ShockTubeConfig(
            name="collide",
            rho_left=1.0, P_left=1.0, vx_left=0.9,
            rho_right=1.0, P_right=1.0, vx_right=-0.9,
            xmin=-1.0, xmax=1.0, xshock=0.0,
            ncells=ncells, cfl=cfl, tfinal=1.0,
            recon_method=recon_method,
            riemann_solver=riemann_solver,
            time_integrator=time_integrator),
    }
    if name not in problems:
        raise ValueError(
            f"Unknown problem '{name}'. "
            f"Available: {list(problems.keys())}")
    return problems[name]


def plot_profile(x, prim, title=None, savepath=None, prim_init=None):
    """Plot rho, P, v_x vs x.

    Parameters
    ----------
    x : ndarray, shape (N,)
    prim : ndarray, shape (3, N)
    title : str or None
    savepath : str or None
        If provided, save figure to this path.
    prim_init : ndarray, shape (3, N) or None
        Initial conditions overlaid as dashed lines.
    """
    import matplotlib.pyplot as plt
    v_x = prim[IUX] / np.sqrt(1.0 + prim[IUX]**2)
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    axes[0].plot(x, prim[IRHO], 'b-', lw=1.0)
    axes[0].set_ylabel('rho')
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(x, prim[IP], 'r-', lw=1.0)
    axes[1].set_ylabel('P')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(x, v_x, 'g-', lw=1.0)
    axes[2].set_ylabel('v_x')
    axes[2].set_xlabel('x')
    axes[2].grid(True, alpha=0.3)
    if prim_init is not None:
        v_x0 = prim_init[IUX] / np.sqrt(1.0 + prim_init[IUX]**2)
        axes[0].plot(x, prim_init[IRHO], 'k--', lw=0.8, alpha=0.5)
        axes[1].plot(x, prim_init[IP], 'k--', lw=0.8, alpha=0.5)
        axes[2].plot(x, v_x0, 'k--', lw=0.8, alpha=0.5)
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches='tight')
    return fig
#
# :D
#
