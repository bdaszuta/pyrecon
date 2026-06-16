# ruff: noqa: E402
"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Compute Sod shock tube with 5 reconstruction methods, overlaid
"""

import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
import matplotlib.pyplot as plt

from hydro_driver.shock_tube import (
    get_problem, run, setup_grid, setup_initial,
)
from hydro_driver.eos import IdealGasEOS, IUX, IRHO, IP

METHODS = [
    ("donate_fv",   "Donor cell"),
    ("lin_mc2_fv",  "MC2 (linear)"),
    ("ppm_fv",      "PPM"),
    ("weno5z_fv",   "WENO5-Z"),
    ("teno5_fv",    "TENO5"),
]

COLOURS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def plot_comparison(x, prims, prim_init, title, savepath):
    """Overlay multiple method results on shared 3-panel figure.

    Parameters
    ----------
    x : ndarray, shape (N,)
    prims : dict[str, ndarray]
        method_key -> prim array shape (3, N).
    prim_init : ndarray, shape (3, N)
        Initial conditions.
    title : str
        Figure suptitle.
    savepath : str
        Output path.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    fields = [
        (IRHO, "rho", False),
        (IP, "P", True),
        (IUX, "v_x", False),
    ]
    for ax, (field_idx, ylabel, log_scale) in zip(axes, fields):
        for (method_key, method_label), colour in zip(METHODS, COLOURS):
            prim = prims[method_key]
            if field_idx == IUX:
                data = prim[IUX] / np.sqrt(1.0 + prim[IUX]**2)
            else:
                data = prim[field_idx]
            ax.plot(x, data, color=colour, lw=1.1, label=method_label)
        if log_scale:
            ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    # Initial conditions overlay (black dashed)
    v_x0 = prim_init[IUX] / np.sqrt(1.0 + prim_init[IUX]**2)
    init_data = [prim_init[IRHO], prim_init[IP], v_x0]
    for ax, d in zip(axes, init_data):
        ax.plot(x, d, "k--", lw=0.8, alpha=0.5)

    axes[-1].set_xlabel("x")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               ncol=len(METHODS), bbox_to_anchor=(0.5, 0.98),
               fontsize=9)
    if title:
        fig.suptitle(title, y=1.02)
    plt.tight_layout()
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ncells = 400
    cfl = 0.3
    riemann_solver = "hlle"
    time_integrator = "rk3"

    # Grid and initial conditions (method-independent)
    cfg0 = get_problem("sod", ncells=ncells, cfl=cfl,
                       recon_method="weno5z_fv",
                       riemann_solver=riemann_solver,
                       time_integrator=time_integrator)
    eos = IdealGasEOS(gamma=cfg0.gamma)
    x, dx = setup_grid(ncells, cfg0.xmin, cfg0.xmax)
    prim_init, _ = setup_initial(x, cfg0, eos)

    prims = {}
    for method_key, method_label in METHODS:
        cfg = get_problem("sod", ncells=ncells, cfl=cfl,
                          recon_method=method_key,
                          riemann_solver=riemann_solver,
                          time_integrator=time_integrator)
        print(f"Running {method_key} ...")
        _, prim, _, _ = run(cfg)
        prims[method_key] = prim

    out = os.path.join(_REPO, "figs", "profile_sod_comparison.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plot_comparison(x, prims, prim_init,
                    title=f"Sod shock tube, t=0.5, ncells={ncells}",
                    savepath=out)
    print(f"Saved {out}")
#
# :D
#
