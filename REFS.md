# REFS.md -- Method reference mapping

Each reconstruction method in pyrecon is mapped to its source:
a published paper, or marked as "Synthetic / novel" if it is
invented / experimental for this library.

## Literature -- WENO

| Method | Paper |
|--------|-------|
| weno3_fv / weno3_pw | Jiang & Shu, JCP 126, 202-228 (1996), 3rd-order reduction |
| weno3z_fv / weno3z_pw | Borges et al., JCP 227, 3191-3211 (2008), 3rd-order reduction |
| weno5_fv / weno5_pw | Jiang & Shu, JCP 126, 202-228 (1996) |
| weno5z_fv / weno5z_pw | Borges et al., JCP 227, 3191-3211 (2008) |
| weno5d_si_fv / weno5d_si_pw | Don et al., JCP 448, 110724 (2022) -- scale-invariant |
| weno5_ha_js_fv | Ha et al., JCP 232, 68-86 (2013) -- new smoothness indicator with JS weights |
| weno5cz_fv | Barreto et al., arXiv:2311.09332 (2023) -- centered Z weighting |
| weno5_bc_fv | Barreto et al., arXiv:2311.09332 (2023) -- biased-centered weighting |
| weno5zcplus_fv | Centered Z-type WENO variant |
| weno5z_ns_fv | Ha et al., JCP 232, 68-86 (2013) smoothness |
| weno5m_fv / weno5m_pw | Henrick, Aslam & Powers, JCP 207, 542-567 (2005) -- mapped WENO |
| weno7_fv / weno7_pw | Balsara & Shu, JCP 160, 405-452 (2000) |
| weno7z_fv / weno7z_pw | Borges-type Z for 7th order |
| weno_ao53_fv / weno_ao53_pw | Balsara, Garain & Shu, JCP 326, 780-804 (2016) -- adaptive order |
| aweno5_fv | Wang, Don & Wang, Computers & Fluids 251, 105743 (2023) -- adaptive WENO |
| adweno5_fv | Curvature-sharpened WENO5-Z FV (pyrecon custom) |

## Literature -- WENO-C (Two-Layer Combined WENO)

| Method | Paper |
|--------|-------|
| wenoc5_fv / wenoc5z_fv | Hossein Mahmoodi Darian, arXiv:2410.09502 (2024) -- 5th-order combined |
| wenoc7_fv / wenoc7z_fv | Hossein Mahmoodi Darian, arXiv:2410.09502 (2024) -- 7th-order combined |

## Literature -- TENO

| Method | Paper |
|--------|-------|
| teno5_fv / teno5_pw | Takagi, Fu, Wakimura & Xiao, JCP 452, 110899 (2022) |
| teno5_mc2_fv / teno5_mc2_pw | TENO5 with MC2 fallback |
| teno5_koren_fv / teno5_koren_pw | TENO5 with Koren limiter fallback |
| teno_a_fv / teno_a_pw | Adaptive-dissipation TENO5 with beta-ratio sensor (pyrecon custom) |
| teno_hybrid_fv / teno_hybrid_pw | Fu, CiCP 26, 973-1007 (2019) -- hybrid TENO with discontinuity indicator |
| teno_thinc_fv / teno_thinc_pw | Takagi, Fu, Wakimura & Xiao, JCP 452, 110899 (2022) -- TENO-THINC |
| vho_teno8_aa_pw | Fu, CMAME 387, 114193 (2021) -- VHO-TENO8-AA |
| vho_teno10_aa_pw | Fu, CMAME 387, 114193 (2021) -- VHO-TENO10-AA |
| teno_m_va_fv | TENO-M with Van Albada limiter (pyrecon custom) |
| teno_m_tvd5_fv | TENO-M with 5th-order TVD limiter (pyrecon custom) |
| teno_m_mp_fv | TENO-M with MP limiter (pyrecon custom) |

## Literature -- CTENO / CTENOZ (Central TENO)

| Method | Paper |
|--------|-------|
| cteno5_fv | Ma, Chong, Feng, Zhang, Wang, Zhou, arXiv:2312.17042 (2023) -- hard ENO-like cutoff |
| cteno5z_fv | Ma et al., arXiv:2312.17042 (2023) -- WENOZ-inspired tau variant |

## Literature -- ENO-MR (Multi-Resolution ENO)

| Method | Paper |
|--------|-------|
| eno_mr3_fv | Hua Shen, arXiv:2311.15504 (2023) -- 3rd-order multi-resolution |
| eno_mr5_fv | Hua Shen, arXiv:2311.15504 (2023) -- 5th-order multi-resolution |
| eno_mr7_fv | Hua Shen, arXiv:2311.15504 (2023) -- 7th-order multi-resolution |

## Literature -- GENO (Path Function Blending)

| Method | Paper |
|--------|-------|
| geno5_fv | Zhao & Xu, arXiv:2507.20461 (2025) -- generalized ENO with tanh blending |

## Literature -- CWENO / Central WENO

| Method | Paper |
|--------|-------|
| central_weno_fv | Levy, Puppo, Russo, M2AN 33(3), 547-571 (1999) -- original CWENO3 |
| cweno3_fv | Cravero, Puppo, Semplice & Visconti, Math. Comp. 87, 1689-1719 (2017) -- CWENO3 |
| cweno5_fv | Cravero et al. (2017) -- CWENO5 extension (5-point sub-stencils) |
| cweno5_capdeville_fv | Capdeville, JCP 227, 2977-3014 (2008) -- CWENO5 with quadratic sub-stencils, non-uniform meshes |
| cweno_z3_fv | CWENO3-Z WENO-Z-like variant (pyrecon extension) |
| cweno_z5_fv | CWENO5-Z WENO-Z-like variant (pyrecon extension) |

## Literature -- CENO

| Method | Paper |
|--------|-------|
| ceno3_fv | Central ENO (various) |
| ceno5_fv | Central ENO (various) |

## Literature -- PPM

| Method | Paper |
|--------|-------|
| ppm_fv | Colella & Woodward (1984); Peterson & Hammett, SIAM J. Sci. Comput. 35, B576 (2013) |
| ppmx_fv | Colella & Sekora, JCP 227, 7069-7076 (2008) |

## Literature -- MP (Monotonicity-Preserving)

| Method | Paper |
|--------|-------|
| mp3_fv | Suresh & Huynh, JCP 136, 83-99 (1997) |
| mp5_fv | Suresh & Huynh, JCP 136, 83-99 (1997) |
| mp7_fv | Suresh & Huynh, JCP 136, 83-99 (1997) |
| mp5_r_fv | Rider & Margolin, JCP 174, 473-488 (2001); He et al., Computers & Fluids (2016) -- refined limiter |
| mp5_mp_fv | Ha & Lee, Computers & Fluids 197, 104345 (2020) -- modified MP for multi-phase |

## Literature -- Linear

| Method | Paper |
|--------|-------|
| lin_vl_fv | van Leer (1979) |
| lin_mc2_fv | van Leer (1977), monotonized central |
| donate_fv | First-order donor cell (standard) |

## Literature -- BVD / MOOD / THINC

| Method | Paper |
|--------|-------|
| thinc_bvd_fv / thinc_bvd_pw | Deng et al., JCP 386, 323-352 (2019); Cheng et al., JCP 421, 109738 (2020) |
| bvd_fv | Sun, Inaba & Xiao, JCP 322, 309-325 (2016) |
| bvd_tbv_fv | BVD per-cell TBV variant (Sun/Inaba/Xiao 2016, Remark 6) |
| bvd_e6_mp5_fv | BVD explicit 6th-order + MP5 alpha=4 (pyrecon extension) |
| bvd_cu_fv | Chamarthi & Frankel, JCP 427, 110067 (2021) -- central-upwind BVD |
| mood_fv | Clain, Diot & Loubere, JCP 230, 4028-4050 (2011) -- a posteriori MOOD |

## Literature -- Energy-Stable / Entropy-Stable

| Method | Paper |
|--------|-------|
| esweno3_fv / esweno3_pw | Yamaleev & Carpenter, JCP 228, 3025-3047 (2009) |

## Literature -- Entropy-Stable Scalar

| Method | Paper |
|--------|-------|
| es_scalar_quad_fv / es_scalar_quad_pw | Duan & Tang, Adv. Appl. Math. Mech. 12, 1-29 (2020) -- quadratic entropy eta=u^2/2 |
| es_scalar_log_fv / es_scalar_log_pw | Duan & Tang (2020) -- logarithmic entropy eta=u*log(u) |
| es_scalar_cubic_fv / es_scalar_cubic_pw | Duan & Tang (2020) -- cubic entropy eta=u^4/4 |

## Literature -- ROUND (Unified NVD Framework)

| Method | Paper |
|--------|-------|
| round_a_fv | Deng, J. Comput. Phys. 481, 112052 (2023) -- aggressive nonlinear |
| round_b_fv | Deng (2023) -- balanced with TVD property |
| round_c_fv | Deng (2023) -- smooth with enhanced resolving power |

## Literature -- Low-Dissipation Shock-Capturing

| Method | Paper |
|--------|-------|
| hybrid_linear_weno_fv | Hybrid linear-WENO-Z convex blend, C_tau=1.0 (pyrecon custom) |
| hybrid_linear_weno_mild_fv | Hybrid linear-WENO-Z, C_tau=0.5 mild variant (pyrecon custom) |
| hybrid_linear_weno_strong_fv | Hybrid linear-WENO-Z, C_tau=2.0 strong variant (pyrecon custom) |

## Literature -- Hybrid Flux MP

| Method | Paper |
|--------|-------|
| hybrid_mp_mc2_fv | Ahn & Lee, Math. Probl. Eng. 2019, 4590956 (2019) -- hybrid flux MP5-MUSCL (reconstruction-level adaptation) |

## Literature -- Lagrange

| Method | Paper |
|--------|-------|
| lag6_fv / lag6_pw | 6th-order Lagrange interpolation (standard) |

## Literature -- Very-High-Order

| Method | Paper |
|--------|-------|
| vhoweno9_fv | Gerolymos, Senechal & Vallet, JCP 228, 8481-8524 (2009) -- VHO-WENO 9th order |
| vhoweno11_fv | Gerolymos et al. (2009) -- VHO-WENO 11th order |

## Synthetic / Novel

These methods were invented for pyrecon. They are not traced to a specific
published paper. They combine known building blocks (WENO weights, stencils,
smoothness indicators) in novel configurations or serve as experimental
variants.

| Method | Description |
|--------|-------------|
| lsweno5h_fv | Synthetic -- WENO5-Z with log-space fallbacks at sharp features (drop threshold >20x) |
| lsweno5hp_fv | Synthetic -- WENO5-Z with physics-informed log-space fallbacks (plateau + exponential fit) |
| weno5z_p2_fv | WENO5-Z with Z-exponent p=2 (pyrecon variant) |
| weno5z_sp_fv / weno5z_sp_pw | Synthetic -- WENO5-Z with branchless sign preservation (positivity at same-sign data) |
