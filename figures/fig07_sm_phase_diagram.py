"""Fig. 7: the SM phase diagram (fig. 4 of arXiv:1307.3536) redrawn in terms of the weak-scale couplings: x axis
u = lambda_0 - lambda_c(y0_best), y axis y_0 = y_t(M_t), with lambda_0 = lambda(M_t).

lambda_c(y_0) is the linearised stability bound, eq. (2.42) of the paper; on the x axis it is always evaluated at the
best-fit y_0, so x = 0 is the critical lambda_0 at the measured top mass. The figure shows lambda_c(y0_best) (solid
blue), lines at |u_0| = 10^-1 and 10^-2 (dashed blue), and the 68% and 95% CL ellipses of (lambda_0, y_0).

Reads data/sm_weak_grid.npz (made by compute/sm_weak_grid.py). Writes output/sm_weak_phase_diagram_decades.pdf.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the nearcrit package

import numpy as np
import matplotlib.pyplot as plt

from nearcrit import style
from nearcrit.sm_rge import DMH, DMT, LAMC_BEST, STABILITY_FROM_LAMBDA_EFF, U_BEST, Y0_BEST
from nearcrit.style import LINEWIDTH_IN

style.use()

DATA = style.data_file("sm_weak_grid.npz", made_by="compute/sm_weak_grid.py")

# Colours of the phases, close to those of fig. 4 of arXiv:1307.3536.
C_STAB, C_META, C_INST, C_NOEW = "#8ff08f", "#ffff8c", "#ff9a9a", "#ff5c5c"
# lambda_c and the |u_0| lines: a bright blue, so that they are not taken for the (black) phase boundaries.
C_LAMC = "#0039e6"


def load():
    """The grids of both panels, {"wide": {...}, "zoom": {...}}."""
    data = np.load(DATA, allow_pickle=False)
    return {p: {k[len(p) + 1:]: data[k] for k in data.files if k.startswith(p + "_")} for p in ("wide", "zoom")}
def sm_ellipse(delta_chi2=2.30, n=200):
    """CL region of (u, y_0) from eqs. (55) and (57), propagating the errors of M_t, alpha_3, M_h linearly."""
    # Jacobian of (55), (57) with respect to (M_t, alpha_3 in units of its error, M_h).
    J = np.array([[-0.00004, 0.0, 0.00206],
                  [0.00556, -0.00042, 0.0]])
    C = J @ np.diag([DMT**2, 1.0, DMH**2]) @ J.T
    L = np.linalg.cholesky(C)
    phi = np.linspace(0, 2 * np.pi, n)
    dx = np.sqrt(delta_chi2) * L @ np.array([np.cos(phi), np.sin(phi)])
    return U_BEST + dx[0], Y0_BEST + dx[1]


def draw_panel(ax, d):
    """Fill the phases of one panel (see nearcrit/sm_rge.py for the criteria).

    The Planck-scale dominated region of fig. 4 is not drawn: in these coordinates it is a sliver along the
    stability edge, narrower than a grid cell, because lambda(M_Pl) varies ~16 times faster than lambda_0 there.
    """
    U, Y = np.meshgrid(d["u"], d["y0"])
    lo, hi = -1e9, 1e9
    min_lam = d["min_lam_eff"] if STABILITY_FROM_LAMBDA_EFF else d["min_lam"]
    min_lam = np.where(np.isnan(min_lam), hi, min_lam)  # lambda ran into a Landau pole: positive
    ax.set_facecolor(C_STAB)
    ax.contourf(U, Y, min_lam, levels=[lo, 0], colors=[C_META])
    ax.contourf(U, Y, d["log10_p0"], levels=[0, hi], colors=[C_INST])

    # No EW vacuum: lambda_0 < 0.
    ax.axvspan(-hi, -LAMC_BEST, color=C_NOEW, lw=0, zorder=2.5)

    # Phase boundaries, drawn over the lambda_c bands so they stay visible through them.
    kw = dict(colors="k", linewidths=0.8, zorder=2.8)
    ew = U > -LAMC_BEST
    ax.contour(U, Y, np.where(ew, min_lam, np.nan), levels=[0], **kw)
    ax.contour(U, Y, np.where(ew, d["log10_p0"], np.nan), levels=[0], **kw)
    ax.axvline(-LAMC_BEST, color="k", lw=0.8, zorder=2.8)


def draw_decades(ax, exponents):
    """lambda_c(y0_best) (x = 0) and lines at |u_0| = 10^k for k in `exponents`, labelled on a top axis.

    Only the u > 0 lines are labelled (with 10^k, under a single |u_0| label): the lines crowd near u = 0.
    10^-3 is not drawn in the zoom, where +-10^-3 would merge with the u = 0 line. The best-fit lambda_0 line is left out: at u = -0.0092 it would sit on top of the u = -10^-2 line, and
    the ellipse already marks it.
    """
    ax.axvline(0, color=C_LAMC, lw=1.2, zorder=2.7)
    for k in exponents:
        for sgn in (-1, 1):
            ax.axvline(sgn * 10.0**k, color=C_LAMC, ls=(0, (4, 2)), lw=0.7, zorder=2.7)
    top = ax.secondary_xaxis("top")
    ticks = [sgn * 10.0**k for k in exponents for sgn in (-1, 1)]
    top.set_xticks(ticks, ["" if t < 0 else rf"$10^{{{k}}}$" for k in exponents for t in (-1, 1)],
                   fontsize=9, color=C_LAMC)
    top.tick_params(colors=C_LAMC, length=2, pad=1)


def plot_weak(data):
    """Draw both panels. Returns (fig, (ax_wide, ax_zoom)) so that data can be overlaid."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(LINEWIDTH_IN, 2.9))

    for ax, name in ((ax1, "wide"), (ax2, "zoom")):
        d = data[name]
        draw_panel(ax, d)
        draw_decades(ax, (-1, -2) if name == "wide" else (-2,))
        ax.set_xlim(d["u"][0], d["u"][-1])
        ax.set_ylim(d["y0"][0] if name == "zoom" else 0, d["y0"][-1])
        ax.set_xlabel(r"$u = \lambda_0 - \lambda_c$")
    for dchi2, color in ((5.99, "0.6"), (2.30, "0.25")):  # 95% and 68% CL
        ax2.fill(*sm_ellipse(dchi2), color=color, lw=0, zorder=3)
    ax2.annotate("SM best fit", xy=(U_BEST - 0.0015, Y0_BEST + 0.004), xytext=(-0.052, 0.956), ha="center", va="center",
                 fontsize=9, arrowprops=dict(arrowstyle="-|>", color="k", lw=0.5, mutation_scale=5), zorder=3)

    labels1 = [(0.12, 0.45, "Stability", 0), (-0.05, 1.12, "Instability", 0),
               (-0.168, 0.62, "No EW vacuum", 90), (-0.075, 0.79, "Metastability", 30)]
    for x, y, s, rot in labels1:
        ax1.text(x, y, s, ha="center", va="center", fontsize=9, rotation=rot)
    labels2 = [(0.045, 0.89, "Stability", 0), (-0.045, 0.995, "Instability", 0),
               (-0.056, 0.895, "Metastability", 42)]
    for x, y, s, rot in labels2:
        ax2.text(x, y, s, ha="center", va="center", fontsize=9, rotation=rot)
    for ax in (ax1, ax2):
        ax.text(0.0, 1.02, r"$|u_0|$", fontsize=9, ha="left", va="bottom", color=C_LAMC, transform=ax.transAxes)
    ax1.set_ylabel(r"$y_0$")
    ax1.set_yticks(np.arange(0, 1.26, 0.25), [rf"${v:.2f}$" for v in np.arange(0, 1.26, 0.25)])
    ax2.set_yticks([0.90, 0.95, 1.00], [r"$0.90$", r"$0.95$", r"$1.00$"])
    fig.tight_layout()
    return fig, (ax1, ax2)


if __name__ == "__main__":
    fig, (ax_wide, ax_zoom) = plot_weak(load())
    # Overlay your own points here, e.g.
    #   ax_zoom.plot(my_u, my_y0, "o", color="C4", ms=3)
    style.savefig(fig, "sm_weak_phase_diagram_decades.pdf")
