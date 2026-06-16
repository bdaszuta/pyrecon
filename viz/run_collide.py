# ruff: noqa: E402
"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Run Collide shock tube and save plot
"""

import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from hydro_driver.shock_tube import (
    get_problem, run, plot_profile, setup_grid, setup_initial,
)
from hydro_driver.eos import IdealGasEOS

cfg = get_problem("collide", ncells=400, cfl=0.3,
                  recon_method="weno5z_fv", riemann_solver="hlle",
                  time_integrator="rk3")
print(f"Running {cfg.name}: ncells={cfg.ncells}, tfinal={cfg.tfinal}")
eos = IdealGasEOS(gamma=cfg.gamma)
x, dx = setup_grid(cfg.ncells, cfg.xmin, cfg.xmax)
prim_init, _ = setup_initial(x, cfg, eos)

_, prim, cons, hist = run(cfg)

out = os.path.join(_REPO, "figs", "profile_collide.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
plot_profile(x, prim, prim_init=prim_init,
             title=f"Collide shock tube, t={cfg.tfinal}, ncells={cfg.ncells}",
             savepath=out)
print(f"Saved {out}")
#
# :D
#
