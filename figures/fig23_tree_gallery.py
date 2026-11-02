"""Fig. 23: the tree-level Higgs-singlet potential in each potential phase: our version of Fig. 2 of Enguita et al.
(arXiv:2503.03825), in our phase names (0, Ia, ..., IIIc).

Each panel is the potential over the plane of normalized fields x = |H|/v_0, y = |S|/w_0, with the curves where
dV/d|H| = 0 (solid) and dV/d|S| = 0 (dashed), including the axes, and the extrema P_k marked by type. The parameters
are those of the original figure; Ia and IIa both correspond to its phase I, whose parameters make P_1 and P_2
degenerate, so here lambda_S (lambda_H) is raised to make P_1 (P_2) the global minimum. check_panels() checks that
every panel is in the phase it is labelled with.

The potential is written in the barred fields hbar^2 = lambda_H |H|^2 / mu_H^2 and sbar^2 = lambda_S |S|^2 / mu_S^2,
whose squares carry the signs of the mu^2: hbar^2 = sign(mu_H^2) x^2.

Writes output/2D_gallery_potentials.pdf.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the nearcrit package

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from nearcrit import style
from nearcrit.singlet_tree import PHASE_LABELS, tree_potential_phase_map_Omega
from nearcrit.style import LINEWIDTH_IN

style.use()
def V(hbar, sbar, mu2H, mu2S, kappa, OmegaH, OmegaS):
    return (mu2H * mu2S / kappa) * (- hbar**2 / OmegaH
                                    - sbar**2 / OmegaS
                                    + hbar**2 * sbar**2 / (OmegaH * OmegaS)
                                    + hbar**4 / (2 * OmegaH)
                                    + sbar**4 / (2 * OmegaS))

def dVh_eq_0(hbar, sbar, OmegaS):
    return - 1.0 + hbar**2 + sbar**2 / OmegaS

def dVs_eq_0(hbar, sbar, OmegaH):
    return - 1.0 + sbar**2 + hbar**2 / OmegaH


# The extrema bar P_k = (|H|_k / v_0, |S|_k / w_0) of the draft, in the plotted coordinates; None where they do
# not exist.
def P0():
    return 0.0, 0.0

def P1(mu2H):
    return (1.0, 0.0) if mu2H > 0 else None

def P2(mu2S):
    return (0.0, 1.0) if mu2S > 0 else None

def P3(mu2H, mu2S, OmegaH, OmegaS):
    x2 = np.sign(mu2H) * OmegaH * (OmegaS - 1.0) / (OmegaH * OmegaS - 1.0)
    y2 = np.sign(mu2S) * OmegaS * (OmegaH - 1.0) / (OmegaH * OmegaS - 1.0)
    return (np.sqrt(x2), np.sqrt(y2)) if x2 > 0 and y2 > 0 else None


def bar_fields(x, y, mu2H, mu2S):
    """(hbar, sbar) at x = |H|/v_0, y = |S|/w_0. They are imaginary for a negative mu^2, so that their squares
    carry its sign; everything above depends on them only through their squares."""
    return np.sqrt(np.sign(mu2H) + 0j) * x, np.sqrt(np.sign(mu2S) + 0j) * y


def V_xy(x, y, p):
    return V(*bar_fields(x, y, p["mu2H"], p["mu2S"]), **p).real


def classify(x, y, p, eps=1e-4):
    """'min', 'max' or 'saddle' from the Hessian in (x, y). V is even in x and y, so this also holds on the axes."""
    f = lambda a, b: V_xy(a, b, p)
    fxx = (f(x + eps, y) - 2 * f(x, y) + f(x - eps, y)) / eps**2
    fyy = (f(x, y + eps) - 2 * f(x, y) + f(x, y - eps)) / eps**2
    fxy = (f(x + eps, y + eps) - f(x + eps, y - eps) - f(x - eps, y + eps) + f(x - eps, y - eps)) / (4 * eps**2)
    ev = np.linalg.eigvalsh([[fxx, fxy], [fxy, fyy]])
    return "min" if ev.min() > 0 else "max" if ev.max() < 0 else "saddle"


def extrema(p):
    """{k: (x, y, kind)} for the extrema that exist, with kind 'global', 'local', 'max' or 'saddle'."""
    points = {0: P0(), 1: P1(p["mu2H"]), 2: P2(p["mu2S"]), 3: P3(p["mu2H"], p["mu2S"], p["OmegaH"], p["OmegaS"])}
    out = {k: (*P, classify(*P, p)) for k, P in points.items() if P is not None}
    minima = [k for k, (_, _, kind) in out.items() if kind == "min"]
    k_global = min(minima, key=lambda k: V_xy(*out[k][:2], p))
    for k in minima:
        out[k] = (*out[k][:2], "global" if k == k_global else "local")
    return out


# The parameters of the original figure, with mu^2 = +-1 and kappa = +-1 (only the Omegas and the signs of the
# mu^2 matter here, up to the overall scale). In brackets the phase of the original figure.
PANELS = {
    "0":    dict(OmegaH=1/2, OmegaS=1/2, mu2H=-1, mu2S=-1, kappa=1),   # (V)
    "Ia":   dict(OmegaH=1/2, OmegaS=3/4, mu2H=1,  mu2S=1,  kappa=1),   # (I), with lambda_S = 3/8
    "Ib":   dict(OmegaH=1/2, OmegaS=3/2, mu2H=1,  mu2S=1,  kappa=1),   # (IV)
    "Ic":   dict(OmegaH=-1/2, OmegaS=-1/2, mu2H=1, mu2S=-1, kappa=1),  # (IV')
    "IIa":  dict(OmegaH=3/4, OmegaS=1/2, mu2H=1,  mu2S=1,  kappa=1),   # (I), with lambda_H = 3/8
    "IIb":  dict(OmegaH=3/2, OmegaS=1/2, mu2H=1,  mu2S=1,  kappa=1),   # (III)
    "IIc":  dict(OmegaH=-1/2, OmegaS=-1/2, mu2H=-1, mu2S=1, kappa=1),  # (III')
    "IIIa": dict(OmegaH=3/2, OmegaS=3/2, mu2H=1,  mu2S=1,  kappa=1),   # (II)
    "IIIb": dict(OmegaH=2,   OmegaS=3/4, mu2H=-1, mu2S=1,  kappa=-1),  # (II')
    "IIIc": dict(OmegaH=3/4, OmegaS=2,   mu2H=1,  mu2S=-1, kappa=-1),  # (II'')
}

# The extrema of each potential phase, as in the table in the draft: {kind: points}.
EXPECTED = {
    "0":    dict(global_=[0]),
    "Ia":   dict(global_=[1], local=[2], max=[0], saddle=[3]),
    "Ib":   dict(global_=[1], max=[0], saddle=[2]),
    "Ic":   dict(global_=[1], saddle=[0]),
    "IIa":  dict(global_=[2], local=[1], max=[0], saddle=[3]),
    "IIb":  dict(global_=[2], max=[0], saddle=[1]),
    "IIc":  dict(global_=[2], saddle=[0]),
    "IIIa": dict(global_=[3], max=[0], saddle=[1, 2]),
    "IIIb": dict(global_=[3], saddle=[0, 2]),
    "IIIc": dict(global_=[3], saddle=[0, 1]),
}


def check_panels():
    """Each panel is in the phase it is labelled with, by the phase map and by its extrema, and is bounded below
    (lambda_H, lambda_S > 0, and lambda_H lambda_S > kappa^2 / 4 if kappa < 0)."""
    for label, p in PANELS.items():
        code = int(tree_potential_phase_map_Omega(p["OmegaH"], p["OmegaS"], p["mu2H"], p["mu2S"]))
        assert PHASE_LABELS.get(code) == label, (label, PHASE_LABELS.get(code))
        found = {}
        for k, (_, _, kind) in extrema(p).items():
            found.setdefault("global_" if kind == "global" else kind, []).append(k)
        assert found == EXPECTED[label], (label, found)
        lambdaH = p["mu2H"] * p["kappa"] * p["OmegaH"] / (2 * p["mu2S"])
        lambdaS = p["mu2S"] * p["kappa"] * p["OmegaS"] / (2 * p["mu2H"])
        assert lambdaH > 0 and lambdaS > 0 and (p["kappa"] > 0 or lambdaH * lambdaS > p["kappa"]**2 / 4), label


# Potential: one light-to-dark hue (low V light), so that the red minima stand out; thin darker contours.
CMAP = LinearSegmentedColormap.from_list("V", ["#f4f7fb", "#9db7d5", "#3d6a9e"])
CONTOUR_COLOR = "#1f3a5c"
CURVE_STYLE = {"h": dict(color="k", lw=1.5, ls="-"), "s": dict(color="k", lw=1.5, ls="--")}
# Extrema: minima red as in the 1D figures (open for a local minimum), maxima and saddles black. The maximum is
# always P_0, in the corner, so it is a square.
MARKERS = {
    "global": dict(marker="o", ms=6, color="C3", mec="white", mew=0.6),
    "local":  dict(marker="o", ms=6, color="C3", mfc="white", mew=1.2),
    "max":    dict(marker="s", ms=5, color="k", mec="white", mew=0.6),
    "saddle": dict(marker="D", ms=4.5, color="k", mfc="white", mew=1.0),
}
MARKER_LABELS = {"global": "Global minimum", "local": "Local minimum", "max": "Maximum", "saddle": "Saddle point"}
# Where each point's label sits, in points from the point.
LABEL_OFFSETS = {0: (4, 4, "left", "bottom"), 1: (-3, 5, "right", "bottom"), 2: (5, -3, "left", "top"),
                 3: (5, 3, "left", "bottom")}
# Per panel, where the default would leave the plot or sit on a curve.
LABEL_OFFSETS_PANEL = {("Ia", 2): (5, 5, "left", "bottom"), ("IIa", 1): (5, 5, "left", "bottom"),
                       ("IIIb", 2): (5, 5, "left", "bottom"), ("IIIb", 3): (-4, 5, "right", "bottom"),
                       ("IIIc", 1): (5, 5, "left", "bottom"), ("IIIc", 3): (-5, 3, "right", "bottom")}
XY_MAX = 1.45
FIG_HEIGHT_IN = 5.4
TOP_FACTOR = 2.5  # the colour scale ends at this many times the range of V over the extrema


def plot_panel(ax, label, p, n=400):
    g = np.linspace(0, XY_MAX, n)
    X, Y = np.meshgrid(g, g)
    Z = V_xy(X, Y, p)
    # Evenly spaced levels from the lowest V up to a cap set by the extrema, beyond which the colour saturates,
    # so that the quartic growth towards the far corner does not wash out the region of the extrema.
    points = extrema(p)
    V_ext = [V_xy(x, y, p) for x, y, _ in points.values()]
    # Most of the panel (85%) is below the cap, so that it does not saturate where V is steep.
    top = Z.max() if len(V_ext) == 1 else min(Z.max(), max(Z.min() + TOP_FACTOR * (max(V_ext) - Z.min()),
                                                           np.quantile(Z, 0.85)))
    levels = np.linspace(Z.min(), top, 21)
    ax.contourf(X, Y, Z, levels=levels, cmap=CMAP, extend="max", zorder=0)
    ax.contour(X, Y, Z, levels=levels, colors=CONTOUR_COLOR, linewidths=0.25, alpha=0.6, zorder=1)

    # The nontrivial zeros of the derivatives, and the axes, where the trivial ones are.
    hbar, sbar = bar_fields(X, Y, p["mu2H"], p["mu2S"])
    for F, which in [(dVh_eq_0(hbar, sbar, p["OmegaS"]).real, "h"), (dVs_eq_0(hbar, sbar, p["OmegaH"]).real, "s")]:
        if F.min() < 0 < F.max():
            ax.contour(X, Y, F, levels=[0], colors=CURVE_STYLE[which]["color"],
                       linewidths=CURVE_STYLE[which]["lw"], linestyles=CURVE_STYLE[which]["ls"], zorder=2)
    ax.plot([0, 0], [0, XY_MAX], clip_on=False, zorder=2, **CURVE_STYLE["h"])
    ax.plot([0, XY_MAX], [0, 0], clip_on=False, zorder=2, **CURVE_STYLE["s"])

    for k, (x, y, kind) in points.items():
        ax.plot(x, y, ls="none", clip_on=False, zorder=5, **MARKERS[kind])
        dx, dy, ha, va = LABEL_OFFSETS_PANEL.get((label, k), LABEL_OFFSETS[k])
        ax.annotate(rf"$\bar P_{k}$", (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                    fontsize=9, zorder=6)

    ax.set_xlim(0, XY_MAX)
    ax.set_ylim(0, XY_MAX)
    ax.set_aspect("equal")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    # Long enough to stick out from under the markers at 1.
    ax.tick_params(length=5.25, width=0.8)
    ax.set_title(f"Phase {label}")


def add_key(ax):
    """The legend for the markers and curves, and the colour scale below it, in the empty panel slots ax."""
    ax.axis("off")
    handles = [Line2D([], [], ls="none", label=MARKER_LABELS[kind], **style) for kind, style in MARKERS.items()]
    handles += [Line2D([], [], label=r"$\partial V/\partial |H| = 0$", **CURVE_STYLE["h"]),
                Line2D([], [], label=r"$\partial V/\partial |S| = 0$", **CURVE_STYLE["s"])]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.92), ncols=1, fontsize=9,
              frameon=False, handlelength=2.0, labelspacing=0.7)
    # The colour scale, from the lowest V in each panel up to the cap of plot_panel.
    cax = ax.inset_axes([0.15, 0.2, 0.7, 0.04])
    cax.imshow(np.linspace(0, 1, 256)[None, :], cmap=CMAP, aspect="auto", extent=(0, 1, 0, 1))
    cax.set_xticks([0, 1], ["low", "high"], fontsize=9)
    cax.set_yticks([])
    cax.set_title(r"$V$", fontsize=9, pad=3)


if __name__ == "__main__":
    check_panels()
    # One row per family, a, b, c across; phase 0 at the top of a fourth column, with the key below it. (Short
    # enough to fit on a page with its caption and parameter table.)
    fig, axes = plt.subplots(3, 4, figsize=(LINEWIDTH_IN, FIG_HEIGHT_IN), sharex=True, sharey=True)
    layout = [["Ia", "Ib", "Ic", "0"], ["IIa", "IIb", "IIc", None], ["IIIa", "IIIb", "IIIc", None]]
    for row_axes, row in zip(axes, layout):
        for ax, label in zip(row_axes, row):
            if label is not None:
                plot_panel(ax, label, PANELS[label])
    for ax in axes[-1, :3]:
        ax.set_xlabel(r"$|H|/v_0$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$|S|/w_0$")
    # Phase 0 is a sketch of the single minimum: ticks but no labels (the other panels have them).
    axes[0, 3].tick_params(labelleft=False, labelbottom=False)
    fig.tight_layout(w_pad=0.4, h_pad=1.2)
    key_pos = [axes[2, 3].get_position(), axes[1, 3].get_position()]
    axes[1, 3].remove()
    axes[2, 3].remove()
    add_key(fig.add_axes([key_pos[0].x0, key_pos[0].y0, key_pos[0].width, key_pos[1].y1 - key_pos[0].y0]))
    style.savefig(fig, "2D_gallery_potentials.pdf")
    r = fig.canvas.get_renderer()
    print("panel sizes (in):", sorted({(round(ax.get_window_extent(r).width / fig.dpi, 2),
                                         round(ax.get_window_extent(r).height / fig.dpi, 2)) for ax in axes[:, :3].flat}))
