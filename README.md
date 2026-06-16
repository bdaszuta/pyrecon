# pyrecon

Python library of spatial reconstruction methods for hyperbolic conservation laws.

Basic idea: given cell-centered values on a 1D stencil, return left and right face values.

Documentation: https://py-recon.readthedocs.io

## Usage

```python
from pyrecon.interface import reconstruct_array, list_methods

# List available methods (canonical names use _fv / _pw suffix)
for name, sw, desc in list_methods():
    print(f"{name:20s}  ({sw}-point)  {desc}")

# Reconstruct a 1D array
z = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
zl, zr = reconstruct_array("weno5_fv", z)
# zl[i+1] = u_{i+1/2}^-  (left state at face i+1/2)
# zr[i]   = u_{i-1/2}^+  (right state at face i+1/2)

# Periodic boundary conditions
zl, zr = reconstruct_array("weno5z_fv", z, periodic=True)
```

Pointwise functions can be called directly:

```python
from pyrecon.recon_weno5 import weno5_fv
uL, uR = weno5_fv(u_im2=1.0, u_im1=2.0, u_i=3.0, u_ip1=4.0, u_ip2=5.0)
```

## Naming convention

Every method has a canonical name ending in `_fv` (finite-volume, cell-averaged input) or `_pw` (pointwise values).

Face convention (standard for all methods except `lag6_fv`):

- `uL = u_{i+1/2}^-` -- left state at face i+1/2, from cell i looking forward
- `uR = u_{i-1/2}^+` -- right state at face i-1/2, from cell i looking backward
- Exception: `lag6_fv` returns `uL = uR = u_{i+1/2}` (symmetric stencil)

## Methods (101 total)

### First-order
| Method | Stencil | Description |
|--------|---------|-------------|
| `donate_fv` | 5 | First-order donor cell |

### Linear (2nd-order)
| Method | Stencil | Description |
|--------|---------|-------------|
| `lin_vl_fv` | 5 | van Leer minmod limiter |
| `lin_mc2_fv` | 5 | MC2 monotonized central |

### WENO3
| Method | Stencil | Description |
|--------|---------|-------------|
| `weno3_fv` / `weno3_pw` | 3 | WENO3-JS (Jiang & Shu 1996) |
| `weno3z_fv` / `weno3z_pw` | 3 | WENO3-Z (Borges et al. 2008) |

### WENO5 & variants
| Method | Stencil | Description |
|--------|---------|-------------|
| `weno5_fv` | 5 | WENO5-JS (Jiang & Shu 1996) |
| `weno5z_fv` | 5 | WENO5-Z (Borges et al. 2008) |
| `weno5d_si_fv` | 5 | WENO5-D-SI scale-invariant (Don et al. 2022) |
| `weno5_ha_js_fv` | 5 | WENO5-Ha-JS (Ha et al. 2013) |
| `weno5cz_fv` | 5 | WENO5-CZ centered Z (Barreto et al. 2023) |
| `weno5_bc_fv` | 5 | WENO5-BC biased-centered (Barreto et al. 2023) |
| `weno5zcplus_fv` | 5 | WENO5-ZC+ |
| `weno5z_ns_fv` | 5 | WENO5-Z-NS |
| `weno5z_p2_fv` | 5 | WENO5-Z with p=2 (FV) |
| `weno5_pw` | 5 | WENO5-JS (PW) |
| `weno5z_pw` | 5 | WENO5-Z (PW) |
| `weno5d_si_pw` | 5 | WENO5-D-SI (PW) |
| `weno5m_fv` / `weno5m_pw` | 5 | WENO5-M Henrick mapping (Henrick et al. 2005) |

### WENO7
| Method | Stencil | Description |
|--------|---------|-------------|
| `weno7_fv` / `weno7_pw` | 7 | WENO7-JS (Balsara & Shu 2000) |
| `weno7z_fv` / `weno7z_pw` | 7 | WENO7-Z |

### WENO-AO / AWENO / ADWENO
| Method | Stencil | Description |
|--------|---------|-------------|
| `weno_ao53_fv` | 5 | WENO-AO(5,3) (Balsara et al. 2016) |
| `weno_ao53_pw` | 5 | WENO-AO(5,3) (PW) |
| `aweno5_fv` | 5 | AWENO5 adaptive (Wang et al. 2023) |
| `adweno5_fv` | 5 | Anti-diffusive WENO5 (Xu & Shu 2005) |

### WENO-C (combined two-layer)
| Method | Stencil | Description |
|--------|---------|-------------|
| `wenoc5_fv` | 5 | WENO5-C |
| `wenoc5z_fv` | 5 | WENO5-ZC |
| `wenoc7_fv` | 7 | WENO7-C |
| `wenoc7z_fv` | 7 | WENO7-ZC |

### TENO
| Method | Stencil | Description |
|--------|---------|-------------|
| `teno5_fv` / `teno5_pw` | 5 | TENO5 (Takagi et al. 2022) |
| `teno5_mc2_fv` / `teno5_mc2_pw` | 5 | TENO5-MC2 |
| `teno5_koren_fv` / `teno5_koren_pw` | 5 | TENO5-Koren |
| `teno_a_fv` / `teno_a_pw` | 5 | TENO-A adaptive CT (Fu et al. 2018) |
| `teno_hybrid_fv` / `teno_hybrid_pw` | 5 | TENO Hybrid (Fu 2019) |
| `vho_teno8_aa_pw` | 9 | VHO-TENO8-AA (Fu 2021) |
| `vho_teno10_aa_pw` | 11 | VHO-TENO10-AA (Fu 2021) |
| `teno_thinc_fv` / `teno_thinc_pw` | 5 | TENO-THINC (Takagi et al. 2022) |

### TENO-M
| Method | Stencil | Description |
|--------|---------|-------------|
| `teno_m_va_fv` | 5 | TENO-M Van Albada limiter (FV) |
| `teno_m_tvd5_fv` | 5 | TENO-M 5th-order TVD limiter (FV) |
| `teno_m_mp_fv` | 5 | TENO-M monotonicity-preserving (FV) |

### CWENO / Central WENO
| Method | Stencil | Description |
|--------|---------|-------------|
| `central_weno_fv` | 3 | Central WENO3 (Levy et al. 1999) |
| `cweno3_fv` | 5 | CWENO3 (Cravero et al. 2017) |
| `cweno5_fv` | 7 | CWENO5 (Cravero et al. 2017) |
| `cweno5_capdeville_fv` | 5 | CWENO5 quadratic (Capdeville 2008) |
| `cweno_z3_fv` | 5 | CWENO3-Z WENO-Z-like variant (FV) |
| `cweno_z5_fv` | 7 | CWENO5-Z WENO-Z-like variant (FV) |

### CENO / CTENO
| Method | Stencil | Description |
|--------|---------|-------------|
| `ceno3_fv` | 5 | CENO3 |
| `ceno5_fv` | 7 | CENO5 |
| `cteno5_fv` | 5 | CTENO5 central-TENO hard cutoff |
| `cteno5z_fv` | 5 | CTENO5Z WENOZ-inspired tau |

### GENO / ENO-MR
| Method | Stencil | Description |
|--------|---------|-------------|
| `geno5_fv` | 5 | GENO5 gradient-based ENO 5th-order |
| `eno_mr3_fv` | 5 | ENO-MR3 multi-resolution 3rd-order |
| `eno_mr5_fv` | 5 | ENO-MR5 multi-resolution 5th-order |
| `eno_mr7_fv` | 7 | ENO-MR7 multi-resolution 7th-order |

### MP (Monotonicity-Preserving)
| Method | Stencil | Description |
|--------|---------|-------------|
| `mp3_fv` | 5 | MP3 (Suresh & Huynh 1997) |
| `mp5_fv` | 5 | MP5 (Suresh & Huynh 1997) |
| `mp7_fv` | 7 | MP7 (Suresh & Huynh 1997) |
| `mp5_r_fv` | 5 | MP5-R (Rider & Margolin 2001) |
| `mp5_mp_fv` | 5 | MP5 modified multi-phase (Ha & Lee 2020) |
| `hybrid_mp_mc2_fv` | 5 | Hybrid MP5-MUSCL(MC2) (Ahn & Lee 2019) |

### PPM
| Method | Stencil | Description |
|--------|---------|-------------|
| `ppm_fv` | 5 | PPM (Colella & Woodward 1984) |
| `ppmx_fv` | 5 | PPMX (Colella & Sekora 2008) |

### Lagrange
| Method | Stencil | Description |
|--------|---------|-------------|
| `lag6_fv` / `lag6_pw` | 6 | 6th-order Lagrange polynomial |

### BVD / MOOD / THINC
| Method | Stencil | Description |
|--------|---------|-------------|
| `thinc_bvd_fv` / `thinc_bvd_pw` | 5 | THINC-BVD (Deng et al. 2019) |
| `bvd_fv` | 5 | BVD MUSCL-THINC (Sun et al. 2016) |
| `bvd_tbv_fv` | 7 | BVD per-cell TBV variant (FV) |
| `bvd_e6_mp5_fv` | 6 | BVD explicit 6th-order + MP5 (FV) |
| `bvd_cu_fv` | 6 | BVD central-upwind HOCUS (Chamarthi & Frankel 2021) |
| `mood_fv` | 5 | MOOD a posteriori (Clain et al. 2011) |

### ES-WENO (Energy-Stable)
| Method | Stencil | Description |
|--------|---------|-------------|
| `esweno3_fv` / `esweno3_pw` | 3 | ES-WENO3 (Yamaleev & Carpenter 2009) |

### Entropy-stable scalar
| Method | Stencil | Description |
|--------|---------|-------------|
| `es_scalar_quad_fv` | 5 | Entropy-stable eta=u^2/2 |
| `es_scalar_log_fv` | 5 | Entropy-stable eta=u*log(u) |
| `es_scalar_cubic_fv` | 5 | Entropy-stable eta=u^4/4 |
| `es_scalar_quad_pw` | 5 | Entropy-stable eta=u^2/2 (PW) |
| `es_scalar_log_pw` | 5 | Entropy-stable eta=u*log(u) (PW) |
| `es_scalar_cubic_pw` | 5 | Entropy-stable eta=u^4/4 (PW) |

### Low-dissipation
| Method | Stencil | Description |
|--------|---------|-------------|
| `hybrid_linear_weno_fv` | 5 | Hybrid linear-WENO-Z, C_tau=1.0 (FV) |
| `hybrid_linear_weno_mild_fv` | 5 | Hybrid linear-WENO-Z, C_tau=0.5 mild (FV) |
| `hybrid_linear_weno_strong_fv` | 5 | Hybrid linear-WENO-Z, C_tau=2.0 strong (FV) |

### ROUND
| Method | Stencil | Description |
|--------|---------|-------------|
| `round_a_fv` | 3 | ROUND-A aggressive |
| `round_b_fv` | 3 | ROUND-B balanced |
| `round_c_fv` | 3 | ROUND-C smooth |

### Very-High-Order
| Method | Stencil | Description |
|--------|---------|-------------|
| `vhoweno9_fv` | 9 | VHO-WENO9 9th-order (Gerolymos et al. 2009) |
| `vhoweno11_fv` | 11 | VHO-WENO11 11th-order (Gerolymos et al. 2009) |

### LS-WENO (Log-Space / Physics-Informed Hybrid)
| Method | Stencil | Description |
|--------|---------|-------------|
| `lsweno5h_fv` | 5 | LS-WENO5-H log-space hybrid |
| `lsweno5hp_fv` | 5 | LS-WENO5-HP physics-informed hybrid |

### WENO5-Z Sign-Preserving
| Method | Stencil | Description |
|--------|---------|-------------|
| `weno5z_sp_fv` / `weno5z_sp_pw` | 5 | WENO5-Z sign-preserving |

## Hydro driver

A 1+1D special-relativistic hydrodynamics driver is included for end-to-end
testing of reconstruction methods on shock-tube problems.

```python
from hydro_driver.shock_tube import ShockTubeConfig, get_problem, evolve

cfg = get_problem("sod", ncells=100, cfl=0.3,
                  recon_method="weno5z_fv", riemann_solver="llf")
sol = evolve(cfg)
```

Subpackage modules:
- `hydro_driver.eos` -- Gamma-law ideal gas EOS, P2C/C2P in 1+1D
- `hydro_driver.sr_hd` -- SR fluxes, wave-speeds, LLF/HLLE Riemann solvers
- `hydro_driver.shock_tube` -- Config, grid/IC setup, RK3 timestepping, evolution driver

![Sod shock tube comparison](figs/profile_sod_comparison.png)

## Visualization

Scripts in `viz/` produce diagnostic plots from hydro evolution or reconstruction
on realistic density profiles loaded from GR-Athena++ tab output.

![NS surface profile FV reconstruction](figs/ns_surface_semilog_01_fv.png)

```bash
# Shock-tube evolution profiles (written to figs/)
python viz/run_sod.py     # Sod shock tube
python viz/run_blast.py   # Blast wave
python viz/run_collide.py # Colliding shocks

# Reconstruction error analysis on NS surface profile
python viz/ns_surface_profile.py    # FV methods
python viz/ns_surface_profile_pw.py # PW methods
```

Sample data (`viz/sample_data/`) contains GR-Athena++ TOV-MHD tab dumps at
N=128, 256, 512 resolution for surface profile visualization.

## Tests

```bash
python3 -m pytest tests/ -v
```

726 tests total:
- Per-method reconstruction unit tests with jump/smooth/exact interpolation verification
- Convergence tests for all FV and PW methods against analytic reference
- Properties tests (stencil width, face convention consistency)
- Hydro driver tests: SR EOS (enthalpy, sound speed, P2C/C2P roundtrip),
  flux and Riemann solver correctness, shock-tube evolution integration (Sod, Lax, etc.)

## References

- Reconstruction method-to-paper mapping: `REFS.md`
