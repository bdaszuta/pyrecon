"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Visualization: ns surface profile (PW), paginated.
"""
import math
import sys
sys.path.insert(0, '.')
from pyrecon.interface import reconstruct_array, list_methods
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Test functions (retained for reference)
# ---------------------------------------------------------------------------

def ns_profile(x, x0=0.45, scale_height=0.01, amplitude=1e-3, floor=1e-14):
    """Polytropic TOV-like interior + sharp exponential crust."""
    if x <= x0:
        rho_central = amplitude * 3.0
        rho_surface = amplitude
        xi = x / x0
        return rho_surface + (rho_central - rho_surface) * (1.0 - xi**2)**1.5
    else:
        return max(amplitude * math.exp(-(x - x0) / scale_height), floor)


def ns_profile_fit(x, rho_c=1.28e-3, a=1.20, p=2.2, b=2.50, q=18.0,
                   x_crust=0.99, crust_scale=0.003, floor=5e-15):
    """Fitted profile: exp2 interior + exponential crust. Symmetric."""
    ax = abs(x)
    if ax <= x_crust:
        return max(rho_c * math.exp(-(a * ax**p + b * ax**q)), floor)
    else:
        rho_t = rho_c * math.exp(-(a * x_crust**p + b * x_crust**q))
        return max(rho_t * math.exp(-(ax - x_crust) / crust_scale), floor)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXCLUDE = 6          # boundary-adjacent cells excluded from error
METHODS_PER_PAGE = 12
NROWS, NCOLS = 3, 4  # 3x4 grid
PV_FAMILY = 'pw'
PV_PREFACE = ['lag6_pw', 'weno5z_pw', 'teno5_pw']
TAB_PATH = 'viz/sample_data/0128_z4c_tov_mhd.block0.out7.00000.tab'

# ---------------------------------------------------------------------------
# Method discovery
# ---------------------------------------------------------------------------


def _discover_methods(suffix, preface):
    """Return ordered list of (name, label) for all methods matching suffix.

    preface entries come first; remaining methods sorted alphabetically.
    Labels are cleaned registry descriptions.
    """
    all_methods = list_methods()
    matching = []
    for name, _sw, desc in all_methods:
        if name.endswith(suffix):
            label = desc
            for tag in (' (FV)', ' (PW)'):
                if label.endswith(tag):
                    label = label[:-len(tag)]
            matching.append((name, label))

    matching.sort(key=lambda x: x[0])

    desc_lookup = dict(matching)

    result = []
    seen = set()
    for p in preface:
        if p in desc_lookup:
            result.append((p, desc_lookup[p]))
            seen.add(p)

    for name, label in matching:
        if name not in seen:
            result.append((name, label))

    return result

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_hydro_data():
    """Load hydro reference data and return preprocessed dict."""
    from scipy.interpolate import PchipInterpolator

    hydro_raw = np.loadtxt(TAB_PATH, skiprows=2)
    Xh = hydro_raw[:, 1]
    rhoh = hydro_raw[:, 2]
    ipeak_h = np.argmax(rhoh)
    rho_c_hydro = rhoh[ipeak_h]
    floor = rhoh.min()
    peak_x = Xh[ipeak_h]
    right = Xh >= peak_x
    x_shifted = Xh[right] - peak_x
    norm_h = (rhoh[right] - floor) / (rho_c_hydro - floor)
    hw_hydro = x_shifted[np.where(norm_h < 0.01)[0][0]]

    xh_right = Xh[right]
    rhoh_right = rhoh[right]
    x_cells = (xh_right - peak_x) / hw_hydro

    cell_avgs = rhoh_right  # pointwise cell-center values
    xc = x_cells
    x_faces = np.zeros(len(x_cells) + 1)
    x_faces[1:-1] = 0.5 * (x_cells[1:] + x_cells[:-1])
    x_faces[0] = x_cells[0] - 0.5 * (x_cells[1] - x_cells[0])
    x_faces[-1] = x_cells[-1] + 0.5 * (x_cells[-1] - x_cells[-2])

    interp_exact = PchipInterpolator(x_cells, cell_avgs, extrapolate=False)

    def fn_exact(x):
        ax = abs(x)
        if ax > x_cells[-1]:
            return floor
        v = interp_exact(ax)
        return float(v) if not np.isnan(v) else floor

    xf = np.linspace(x_faces[0], x_faces[-1], 500)
    exact = np.array([fn_exact(x) for x in xf])
    xL, xR = x_faces[0], x_faces[-1]
    N = len(cell_avgs)
    dx = (xR - xL) / N
    surface_cells = 3

    idx_10pct = np.where(norm_h < 0.1)[0]
    x0 = (x_shifted[idx_10pct[0]] / hw_hydro) if len(idx_10pct) else 0.5

    def fn(x):
        ax = abs(x)
        if ax > x_cells[-1]:
            return floor
        idx = np.searchsorted(x_cells, ax)
        if idx >= len(x_cells):
            return float(cell_avgs[-1])
        if idx == 0:
            return float(cell_avgs[0])
        if ax - x_cells[idx-1] < x_cells[idx] - ax:
            return float(cell_avgs[idx-1])
        return float(cell_avgs[idx])

    i_interior = slice(EXCLUDE, N - EXCLUDE)

    i_surface = ((x_faces >= 0.85) & (x_faces <= 1.15)).nonzero()[0]
    if len(i_surface) == 0:
        i_surface = ((x_faces >= 0.80) & (x_faces <= 1.20)).nonzero()[0]

    return {
        'cell_avgs': cell_avgs, 'x_cells': xc,
        'x_faces': x_faces, 'fn_exact': fn_exact,
        'xf': xf, 'exact': exact,
        'xL': xL, 'xR': xR, 'N': N, 'dx': dx,
        'surface_cells': surface_cells, 'x0': x0,
        'rho_c_hydro': rho_c_hydro, 'floor': floor,
        'hw_hydro': hw_hydro, 'norm_h': norm_h,
        'x_shifted': x_shifted, 'rhoh_right': rhoh_right,
        'peak_x': peak_x, 'Xh_right': xh_right,
        'fn': fn, 'i_interior': i_interior,
        'i_surface': i_surface,
    }

# ---------------------------------------------------------------------------
# Per-page rendering: panels 1, 2, 3
# ---------------------------------------------------------------------------


def _render_page(data, methods, page_num, family):
    """Render semilog grid, error-vs-position, and bar chart for one page."""
    cell_avgs = data['cell_avgs']
    xc = data['x_cells']
    x_faces = data['x_faces']
    fn_exact = data['fn_exact']
    xf = data['xf']
    exact = data['exact']
    xL = data['xL']
    xR = data['xR']
    N = data['N']
    dx = data['dx']
    surface_cells = data['surface_cells']
    x0 = data['x0']
    rho_c_hydro = data['rho_c_hydro']
    floor = data['floor']
    i_interior = data['i_interior']

    n_methods = len(methods)

    # ---- Panel 1: Semilog overview ----
    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(16, 12),
                             sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, (method, label) in enumerate(methods):
        ax = axes[idx]
        zl, zr = reconstruct_array(method, list(cell_avgs), periodic=False)
        zl = np.array(zl)
        zr = np.array(zr)

        ax.semilogy(xf, np.abs(exact) + 1e-30, 'k-', lw=0.5)
        ax.semilogy(xc, np.abs(cell_avgs) + 1e-30, 's-', ms=1.5, lw=0.8,
                    color='blue', alpha=0.50)

        mask = (x_faces > 0.70) & (x_faces < 1.30)
        ax.semilogy(x_faces[mask], np.abs(zl[mask]) + 1e-30,
                    'r.-', ms=1.5, lw=0.8)
        ax.semilogy(x_faces[mask], np.abs(zr[mask]) + 1e-30,
                    '.-', ms=1.5, lw=0.6, color='purple', alpha=0.7)

        ax.set_title(label, fontsize=9)
        ax.set_xlim(0.75, 1.25)
        ax.set_ylim(1e-17, 2e-3)
        if idx >= n_methods - NCOLS:
            ax.set_xlabel('x', fontsize=8)
        if idx % NCOLS == 0:
            ax.set_ylabel('|u|', fontsize=8)

        # Inset: zoom on atmosphere tail
        axins = ax.inset_axes([0.12, 0.12, 0.35, 0.28])
        mask_tail = (x_faces >= 1.05) & (x_faces <= 1.15)
        xf_tail = x_faces[mask_tail]
        zl_tail = zl[mask_tail]
        zr_tail = zr[mask_tail]
        exact_tail = np.array([fn_exact(x) for x in xf_tail])
        mask_cells_tail = (xc >= 1.04) & (xc <= 1.16)

        axins.semilogy(xf_tail, exact_tail + 1e-30, 'k-', lw=0.6)
        if mask_cells_tail.any():
            axins.semilogy(
                xc[mask_cells_tail],
                cell_avgs[mask_cells_tail] + 1e-30,
                's', ms=1.0, color='blue', alpha=0.5, zorder=0)
        axins.semilogy(xf_tail, zl_tail + 1e-30, 'r.', ms=1.5)
        axins.semilogy(
            xf_tail, zr_tail + 1e-30, '.', ms=1.5,
            color='purple', alpha=0.7)

        axins.set_xlim(1.05, 1.15)
        axins.set_ylim(1e-16, 1e-14)
        axins.tick_params(labelsize=5)
        ax.indicate_inset_zoom(axins, edgecolor='gray', alpha=0.4)

    # Hide unused axes on last page
    for idx in range(n_methods, NROWS * NCOLS):
        axes[idx].set_visible(False)

    nl = '\n'
    fig.suptitle(
        f'Hydro profile (POINTWISE): N={N}, rho_c={rho_c_hydro:.2e}, '
        f'floor={floor:.1e}{nl}'
        f'(black line = model, blue squares = cell averages, '
        f'red dots = |uL|, purple dots = |uR|)',
        fontsize=11, y=0.997)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.08,
                        wspace=0.15, hspace=0.35)
    fig.savefig(f'figs/ns_surface_semilog_{page_num:02d}_{family}.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ---- Panel 2: Error vs position ----
    fig2, ax2 = plt.subplots(figsize=(14, 6))

    exact_faces = np.array([fn_exact(x) for x in x_faces])
    subset_methods = methods[:min(6, n_methods)]
    colors = plt.cm.tab10(np.linspace(0, 1, len(subset_methods)))

    for (method, label), c in zip(subset_methods, colors):
        zl, _ = reconstruct_array(method, list(cell_avgs), periodic=False)
        zl = np.array(zl)
        abs_err = np.abs(zl - exact_faces)
        ax2.semilogy(x_faces[i_interior], abs_err[i_interior] + 1e-30,
                     '-', color=c, lw=1, alpha=0.7, label=label)

    ax2.axvline(x=x0, color='k', ls='--', lw=0.5, alpha=0.3)
    ax2.axvspan(x0, x0 + surface_cells * dx, color='red', alpha=0.06)
    ax2.set_xlabel('x', fontsize=11)
    ax2.set_ylabel('|reconstructed - exact|', fontsize=11)
    ax2.set_title(f'Reconstruction error vs position (pointwise)\\n'
                  f'(coloured lines = methods, dashed = surface at x={x0}, '
                  f'red shade = surface zone)', fontsize=10)
    ax2.legend(fontsize=8, loc='upper right', ncol=2)
    ax2.set_xlim(xL, xR)
    ax2.set_ylim(1e-17, 1e-1)

    # Inset: zoom on surface feature
    axins = ax2.inset_axes([0.55, 0.15, 0.40, 0.35])
    for (method, _label), c in zip(subset_methods, colors):
        zl, _ = reconstruct_array(method, list(cell_avgs), periodic=False)
        zl = np.array(zl)
        abs_err = np.abs(zl - exact_faces)
        mask = (x_faces >= 0.90) & (x_faces <= 1.10)
        axins.semilogy(x_faces[mask], abs_err[mask] + 1e-30,
                       '-', color=c, lw=1.2, alpha=0.85)
    axins.axvspan(x0, x0 + surface_cells * dx, color='red', alpha=0.08)
    axins.axvline(x=x0, color='k', ls='--', lw=0.5, alpha=0.3)
    axins.set_xlim(0.90, 1.10)
    axins.set_ylim(1e-17, 1e-1)
    axins.tick_params(labelsize=6)
    ax2.indicate_inset_zoom(axins, edgecolor='gray', alpha=0.5)

    fig2.tight_layout()
    fig2.savefig(
        f'figs/ns_surface_error_vs_x_{page_num:02d}_{family}.png',
        dpi=150, bbox_inches='tight')
    plt.close(fig2)

    # ---- Panel 3: Error + undershoot bar chart ----
    errors_surf = []
    errors_full = []
    undershoot_zl = []
    undershoot_zr = []

    i_surface = data['i_surface']

    for method, _ in methods:
        zl, zr = reconstruct_array(method, list(cell_avgs), periodic=False)
        zl = np.array(zl)
        zr = np.array(zr)
        abs_err = np.abs(zl - exact_faces)
        errors_surf.append(np.mean(abs_err[i_surface]))
        errors_full.append(np.mean(abs_err[i_interior]))
        undershoot_zl.append(
            abs(min(0.0, float(np.min(zl[EXCLUDE:N+1-EXCLUDE])))))
        undershoot_zr.append(
            abs(min(0.0, float(np.min(zr[EXCLUDE:N-EXCLUDE])))))

    fig3, (ax3a, ax3b, ax3c) = plt.subplots(1, 3, figsize=(20, 6))

    y_pos = range(len(errors_surf))
    names = [label for _, label in methods]

    colours_surf = []
    for uz in undershoot_zl:
        if uz > 1e-6:
            colours_surf.append('#8B0000')
        elif uz > 1e-10:
            colours_surf.append('#d62728')
        elif uz > 0:
            colours_surf.append('#ff7f0e')
        else:
            colours_surf.append('#2ca02c')

    ax3a.barh(y_pos, errors_surf, color=colours_surf,
              edgecolor='black', alpha=0.85)
    ax3a.set_yticks(y_pos)
    ax3a.set_yticklabels(names, fontsize=8)
    ax3a.set_xlabel('L1 error at surface', fontsize=10)
    ax3a.set_xscale('log')
    ax3a.set_title(
        f'Surface [{x_faces[i_surface[0]]:.3f}, '
        f'{x_faces[i_surface[-1]]:.3f}]',
        fontsize=10)
    ax3a.invert_yaxis()

    ax3b.barh(y_pos, errors_full, color=colours_surf,
              edgecolor='black', alpha=0.85)
    ax3b.set_xlabel('L1 error (full domain)', fontsize=10)
    ax3b.set_xscale('log')
    ax3b.set_title(f'Interior [{EXCLUDE}:{N-EXCLUDE}]', fontsize=10)

    undershoot_max = [
        max(uz, ur) for uz, ur in zip(undershoot_zl, undershoot_zr)]
    undershoot_plot = [max(u, 1e-18) for u in undershoot_max]
    ax3c.barh(y_pos, undershoot_plot, color=colours_surf,
              edgecolor='black', alpha=0.85)
    ax3c.set_xlabel('max |min(reconstructed, 0)|', fontsize=10)
    ax3c.set_xscale('log')
    ax3c.set_title('Undershoot severity', fontsize=10)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ca02c', label='Safe'),
        Patch(facecolor='#ff7f0e', label='Tiny (<1e-10)'),
        Patch(facecolor='#d62728', label='Moderate (<1e-6)'),
        Patch(facecolor='#8B0000', label='Severe (>1e-6)'),
    ]
    ax3c.legend(handles=legend_elements, fontsize=7, loc='lower right')

    fig3.suptitle(
        f'Reconstruction error & undershoot (pointwise) '
        f'(N={N}, {surface_cells}-cell surface)',
        fontsize=13)
    fig3.tight_layout()
    fig3.savefig(
        f'figs/ns_surface_errors_{page_num:02d}_{family}.png',
        dpi=150, bbox_inches='tight')
    plt.close(fig3)

    header = (
        f"{'Method':16s}  {'Surface L1':>10s}  "
        f"{'Full L1':>10s}  {'Undershoot':>12s}")
    print(f"\nPage {page_num}:")
    print(header)
    print(f"{'-'*16}  {'-'*10}  {'-'*10}  {'-'*12}")
    for (_m, label), es, ef, um in sorted(
        zip(methods, errors_surf, errors_full, undershoot_max),
        key=lambda x: x[1]):
        uflag = "SAFE" if um < 1e-16 else f"{um:.1e}"
        print(f"  {label:16s}  {es:10.1e}  {ef:10.1e}  {uflag:>12s}")

# ---------------------------------------------------------------------------
# Panel 4: Full-profile overview (once only)
# ---------------------------------------------------------------------------


def _render_full_profile(data, family):
    """Render full-profile overview (method-independent)."""
    fn_exact = data['fn_exact']
    dx = data['dx']
    floor = data['floor']
    x_shifted = data['x_shifted']
    rhoh_right = data['rhoh_right']
    hw_hydro = data['hw_hydro']

    x_hydro = x_shifted / hw_hydro
    y_hydro = rhoh_right

    xL_ext, xR_ext = 0.0, 2.0
    N_ext = int((xR_ext - xL_ext) / dx)
    xR_ext = xL_ext + N_ext * dx
    xf_full = np.linspace(xL_ext, xR_ext, 4000)
    exact_full = np.array([fn_exact(x) for x in xf_full])

    fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(14, 10), sharex=True,
                                       gridspec_kw={'hspace': 0.0})
    rho_c_fn = fn_exact(0.0)

    ax4a.semilogy(
        xf_full, np.abs(exact_full) + 1e-40, 'k-', lw=1.5,
        label='Model profile')
    ax4a.semilogy(
        x_hydro, y_hydro + 1e-40, '.-', ms=3, lw=1.0,
        color='green', label='Hydro data')
    ax4a.set_ylabel('|density|', fontsize=12)
    ax4a.set_title(
        f'Profile comparison (pointwise) '
        f'(N_ext={N_ext}, dx={dx:.4f}, rho_c={rho_c_fn:.2e})',
        fontsize=13)
    ax4a.legend(fontsize=10, loc='lower left')
    ax4a.set_xlim(xL_ext, xR_ext)
    ax4a.set_ylim(bottom=floor / 10)

    ax4b.plot(
        xf_full, exact_full / rho_c_fn, 'k-', lw=1.5,
        label='Model profile')
    ax4b.plot(x_hydro, y_hydro / rho_c_fn, '.-', ms=3, lw=1.0,
              color='green', label='Hydro data')
    ax4b.set_xlabel('x = r / R', fontsize=12)
    ax4b.set_ylabel('density / rho_c', fontsize=12)

    fig4.tight_layout()
    fig4.savefig(
        f'figs/ns_surface_full_profile_{family}.png', dpi=150,
        bbox_inches='tight')
    plt.close(fig4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(pages=None):
    """Generate ns_surface_profile_pw plots for all or specified pages.

    Parameters
    ----------
    pages : list of int or None
        Page numbers to generate (1-indexed). None = all pages.
    """
    data = _load_hydro_data()
    methods = _discover_methods('_pw', PV_PREFACE)

    N = data['N']
    rho_c_hydro = data['rho_c_hydro']
    floor_val = data['floor']

    print(
        f"N={N}, rho_c={rho_c_hydro:.2e}, "
        f"floor={floor_val:.1e}")
    print(f"Total methods: {len(methods)}, {METHODS_PER_PAGE} per page")

    total_pages = (len(methods) + METHODS_PER_PAGE - 1) // METHODS_PER_PAGE

    if pages is None:
        pages = range(1, total_pages + 1)
    else:
        pages = sorted(set(pages))
        pages = [p for p in pages if 1 <= p <= total_pages]

    for page_num in pages:
        start = (page_num - 1) * METHODS_PER_PAGE
        page_methods = methods[start:start + METHODS_PER_PAGE]
        if not page_methods:
            break
        _render_page(data, page_methods, page_num, PV_FAMILY)

    _render_full_profile(data, PV_FAMILY)

    print(f"\nSaved: figs/ns_surface_full_profile_{PV_FAMILY}.png")
    for page_num in pages:
        print(f"Saved: figs/ns_surface_semilog_{page_num:02d}_{PV_FAMILY}.png")
    for page_num in pages:
        print(f"Saved: figs/ns_surface_error_vs_x_{page_num:02d}_{PV_FAMILY}.png")
    for page_num in pages:
        print(f"Saved: figs/ns_surface_errors_{page_num:02d}_{PV_FAMILY}.png")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        pages = [int(a) for a in sys.argv[1:]]
    else:
        pages = None
    main(pages)
#
# :D
#
