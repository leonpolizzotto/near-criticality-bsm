"""Gallery of the potential phases of the RG-improved one-loop Higgs-singlet potential that do not exist at tree
level: the loop analogue of Fig. 23. Two figures, one panel per phase, each at a parameter
point well inside it in the phase maps (PANELS below; the point is re-classified when the script runs, and the
script stops if it is not in the phase it is labelled with):

  - Fig. 31, loop_gallery_gs_sm_rg.pdf: the phases with a ground state, from Figs. 25-30.
      Id, IIId   origin max, H-axis min, S-axis saddle, and a mixed min-saddle pair; ground state on the H axis
                 (Id) or at the mixed minimum (IIId). Tree phase Ib plus a radiatively generated pair.
      Ie, I'e    origin max, H axis min-saddle-min, S-axis saddle: Ib plus a second, radiatively generated H-axis
                 minimum at large field, separated by a barrier. The ground state is the inner, electroweak-like
                 minimum (Ie) or the outer one (I'e, a separate quantum phase: the ground state jumps between the
                 two in a first-order transition).
      If, IIIf   the same as d for mu_S^2 < 0 (kappa < 0): Ic (origin saddle, H-axis min, nothing on the S axis)
                 plus a mixed min-saddle pair; ground state on the H axis (If) or at the mixed minimum (IIIf).
      Ig, I'g    the same as e for mu_S^2 < 0 (kappa > 0): Ic plus a second, large-field H-axis minimum.
      IIIh, III'h  mu_S^2 < 0, kappa < 0: IIIc (origin saddle, H-axis saddle, mixed min) plus a second mixed
                 minimum and a saddle; the ground state is the inner (IIIh) or the outer (III'h) mixed minimum.
      IIIi       mu_S^2 > 0, kappa < 0: the same for IIIa (origin max, saddles on both axes). Phase i is a sliver
                 (about 1e-4 wide in Omega_S) along the NGS edge at kappa = -0.5, r = 1, too thin to see in
                 Fig. 27; the ground state there is the inner or the outer mixed minimum.
  - Fig. 32, loop_gallery_ngs_sm_rg.pdf: the NGS potentials.
      NGS_...    V on the disk |H|^2 + |S|^2 <= Lambda^2 (Lambda = 10 TeV) has a local minimum on its edge: a
                 valley of V crosses the cutoff with V still falling outward, so there is no ground state below
                 Lambda (in the panels: the light, low-V region along the H axis at the right edge); the subscript
                 lists the local minima that remain (other than the origin), deepest first. Most are a tree phase
                 plus the barrier the top loop puts on the H axis: NGS_HS and NGS_SH = Ia/IIa (the H-axis or the
                 S-axis minimum deeper), NGS_H = Ib, NGS_M = IIIa, NGS_HM and NGS_MH = Id (the H-axis or the mixed
                 minimum deeper), each plus that barrier. In NGS_S and NGS_empty the H-axis minimum is gone
                 altogether (lambda_H too small). Only the minima define an NGS phase, so NGS_S also covers a sliver
                 near Omega_H = Omega_S = 1 where the S-axis minimum comes with two H-axis saddles instead of an
                 off-axis one (tree phase IIb plus the barrier); it is not shown separately.

Each panel shows the potential over (|H|/v_0, |S|/w_0), with v_0^2 = mu_H^2/lambda_H and w_0^2 = |mu_S^2|/lambda_S
(the tree-level VEVs where they exist), as in Fig. 23. Each panel is focused on its own extrema, so the ranges differ from panel
to panel (see the ticks): an axis is linear from 0 to a bit past the outermost extremum when they all lie within
twice the tree VEV, and symlog (linear up to 1, logarithmic beyond) out to 1.8 times the outermost one when they
span more, as for the large-field barriers and minima. No axis goes past the cutoff Lambda except in Ie, I'e and
I'g; in the NGS panels the H axis with a barrier runs all the way to it, to show V falling beyond the barrier. In Ie,
I'e, Ig and I'g the H axis is linear and runs to Lambda, to show the outer minimum and V rising again beyond it; in Ie,
I'e and I'g it runs a little past Lambda (hatched beyond it), to 1.15 Lambda or to just short of the Landau pole where
that is closer (the couplings are frozen beyond it). In Ig the rise is already clear inside Lambda. The colour is arcsinh(V / V_0), with
V_0 = mu_H^4 / (4 lambda_H) the depth of the tree-level H minimum, so that both the electroweak-scale structure and
the much deeper large-field region are visible; levels are evenly spaced in it. The curves are the zeros of
dV/d|H| (solid) and dV/d|S| (dashed) off the axes; the axes themselves are zeros of the other derivative.

Physics and inputs as in nearcrit/loop_maps.py (RG-improved, SM couplings at M_t, Lambda = 10 TeV).


Reads data/loop_gallery_points.npz (made by compute/loop_gallery_points.py) for the points of phases d, e, g and h.
At each point the phase is re-checked (loop_phase_map at that single point) before it is drawn.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the nearcrit package

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from nearcrit import data_io, style
from nearcrit import singlet_loop as L
from nearcrit.loop_maps import (
    CUTOFF, GALLERY_NGS_KEYS as NGS_KEYS, GALLERY_PANELS, KIND_TO_QUANTUM, LOOP_RG, MU, ground_state_is_outer,
    potential_phase,
)
from nearcrit.style import LINEWIDTH_IN

style.use()

PANELS = dict(GALLERY_PANELS)   # the points left as None are filled in from data/loop_gallery_points.npz (fill_points)
TITLES = {
    "Id": "Phase Id", "IIId": "Phase IIId", "Ie": "Phase Ie", "I'e": r"Phase I$'$e",
    "If": "Phase If", "IIIf": "Phase IIIf", "Ig": "Phase Ig", "I'g": r"Phase I$'$g",
    "IIIh": "Phase IIIh", "III'h": r"Phase III$'$h", "IIIi": "Phase IIIi",
    "NGS_HS": r"NGS$_{HS}$", "NGS_SH": r"NGS$_{SH}$", "NGS_H": r"NGS$_H$",
    "NGS_S": r"NGS$_S$", "NGS_M": r"NGS$_M$",
    "NGS_HM": r"NGS$_{HM}$", "NGS_MH": r"NGS$_{MH}$", "NGS_0": r"NGS$_\emptyset$",
}
# (layout, key slot, height in inches, output)
FIGURES = [
    ([["Id", "IIId", None],
      ["Ie", "I'e", "If"],
      ["IIIf", "Ig", "I'g"],
      ["IIIh", "III'h", "IIIi"]], (0, 2), 8.2, "loop_gallery_gs_sm_rg.pdf"),
    ([["NGS_HS", "NGS_SH", "NGS_H"],
      ["NGS_HM", "NGS_MH", "NGS_M"],
      ["NGS_S", "NGS_0", None]], (2, 2), 6.5, "loop_gallery_ngs_sm_rg.pdf"),
]

# Styles as in Fig. 23.
CMAP = LinearSegmentedColormap.from_list("V", ["#f4f7fb", "#9db7d5", "#3d6a9e"])
CONTOUR_COLOR = "#1f3a5c"
CURVE_STYLE = {"h": dict(color="k", lw=1.2, ls="-"), "s": dict(color="k", lw=1.2, ls="--")}
MARKERS = {
    "global": dict(marker="o", ms=5, color="C3", mec="white", mew=0.6),
    "local":  dict(marker="o", ms=5, color="C3", mfc="white", mew=1.1),
    "max":    dict(marker="s", ms=4.5, color="k", mec="white", mew=0.6),
    "saddle": dict(marker="D", ms=4, color="k", mfc="white", mew=0.9),
}
MARKER_LABELS = {"global": "Global minimum", "local": "Local minimum", "max": "Maximum", "saddle": "Saddle point"}
LINTHRESH = 1.0
# Phases with a large-field H-axis minimum: a linear H axis that runs to Lambda (LARGE_FIELD_H), and in PAST_CUTOFF_H a
# little past it (hatched beyond it), to show V rising again beyond the outer minimum -- to PAST_CUTOFF times Lambda,
# or to just short of the Landau pole if that comes first (the couplings are frozen beyond it, so V there means
# nothing). In Ig the rise is already clear inside Lambda.
LARGE_FIELD_H = ("Ie", "I'e", "Ig", "I'g")
PAST_CUTOFF_H = ("Ie", "I'e", "I'g")
PAST_CUTOFF = 1.15
BEYOND_CUTOFF_STYLE = dict(facecolor="none", edgecolor="k", hatch="////", lw=0)  # hatch: keeps the colours


def fill_points():
    """Fill in the points left as None in PANELS from data/loop_gallery_points.npz."""
    data = data_io.load(style.data_file("loop_gallery_points.npz", made_by="compute/loop_gallery_points.py"))
    for label, v in PANELS.items():
        if v[2] is None:
            kappa, r, _, _, st, q = v
            assert data["want"][label] == (kappa, r, st, q), (label, "data/loop_gallery_points.npz is out of date")
            oh, os_, margin = data["points"][label]
            PANELS[label] = (kappa, r, oh, os_, st, q)
            print(f"{label}: kappa = {kappa}, r = {r}, Omega_H = {oh:.4f}, Omega_S = {os_:.4f} (margin {margin:.0f} cells)")


def point(label):
    """The loop_phase_map result at one panel's point, checked against the phase it is labelled with."""
    kappa, r, oh, os_, st, q = PANELS[label]
    lamH, lamS = L.lambdas_from_tree_Omegas(r, kappa, oh, os_)
    res = L.loop_phase_map(r, kappa, np.array([lamH]), np.array([lamS]), MU, **LOOP_RG)
    found = ("NGS" if res["gs_kind"][0] == "" else
             KIND_TO_QUANTUM[res["gs_kind"][0]] + ("'" if ground_state_is_outer(res, 0) else ""))
    assert res["structure"][0] == st and found == q, (label, res["structure"][0], found)
    assert bool(res["index_ok"][0]), label
    if q == "NGS":
        key = potential_phase(res["structure"][0], "", True, {}, order=res["order"][0])[1]
        assert key == NGS_KEYS[label], (label, key)
    return res


def symlog_samples(top, n_lin=90, n_log=220):
    """Field values from 0 to top, evenly spaced on a symlog axis with linthresh 1."""
    return np.unique(np.concatenate([np.linspace(0, LINTHRESH, n_lin),
                                     np.logspace(0, np.log10(top), n_log)]))


def axis_spec(values, cap, to_cutoff=False):
    """Range, scale, samples and ticks for one field axis, focused on the extrema at these (normalized) field
    values: linear from 0 to a bit past the outermost one if they all lie within twice the tree VEV, otherwise
    symlog (linear up to 1, logarithmic beyond) out to 1.8 times the outermost one, never past the cutoff."""
    outer = max([v for v in values if v > 0], default=0.0)
    if outer <= 2.0:
        top = max(1.35 * outer, 1.4)
        step = 0.5 if top <= 2.2 else 1.0
        ticks = list(np.arange(0, top + 1e-9, step))
        return dict(scale="linear", top=top, samples=np.linspace(0, top, 320), ticks=ticks)
    top = cap if to_cutoff else min(1.8 * outer, cap)
    ticks = [t for t in (0, 1, 10, 100, 1000) if t <= top]
    return dict(scale="symlog", top=top, samples=symlog_samples(top), ticks=ticks)


def classify_markers(pts, res):
    """Marker kind of each extremum; the global minimum is the ground state found by loop_phase_map."""
    gs = (res["gs_Hbar"][0], res["gs_Sbar"][0])
    kinds = []
    for a, b, kind in pts:
        if kind == "min":
            kind = "global" if np.isfinite(gs[0]) and np.hypot(a - gs[0], b - gs[1]) < 1e-6 * (1 + a + b) else "local"
        kinds.append(kind)
    return kinds


def draw_map(ax, U, W, X, Y, P):
    """arcsinh(V / V_0) and the off-axis zeros of the two derivatives, on a grid of plotted coordinates (U, W) with
    fields X = |H|^2, Y = |S|^2."""
    V = L.potential(X, Y, P) - L.potential(0.0, 0.0, P)
    V0 = float(P.muH2[0]**2 / (4.0 * P.lamH[0]))
    Z = np.arcsinh(V / V0)
    levels = np.linspace(np.quantile(Z, 0.01), np.quantile(Z, 0.97), 22)
    ax.contourf(U, W, Z, levels=levels, cmap=CMAP, extend="both", zorder=0)
    ax.contour(U, W, Z, levels=levels, colors=CONTOUR_COLOR, linewidths=0.2, alpha=0.6, zorder=1)
    # dV/d|H| = 2|H| dV/dx, so off the axes its zeros are those of dV/dx (likewise for S).
    gx, gy = L.gradient(X, Y, P)
    for F, which in [(gx, "h"), (gy, "s")]:
        if np.nanmin(F) < 0 < np.nanmax(F):
            ax.contour(U, W, F, levels=[0], colors=CURVE_STYLE[which]["color"], linewidths=CURVE_STYLE[which]["lw"],
                       linestyles=CURVE_STYLE[which]["ls"], zorder=2)


def past_cutoff_top(P, v0):
    """How far past the cutoff the H axis of a PAST_CUTOFF_H panel runs, in |H|/v0: PAST_CUTOFF Lambda, or 99% of the
    way to the Landau pole on the H axis if that is closer."""
    top = PAST_CUTOFF * CUTOFF
    t_np = float(P.rg.t_np[0])
    if np.isfinite(t_np):
        top = min(top, 0.99 * np.sqrt(P.rg.mu02 * np.expm1(2.0 * t_np)))
    return top / v0


def plot_panel(ax, label):
    """One panel: the potential at the point of phase label, its derivative zero curves and its extrema."""
    res = point(label)
    P = res["model"]
    xbar, ybar, _ = (float(v[0]) for v in L._scales(P))
    v0, w0 = np.sqrt(xbar), np.sqrt(ybar)

    # Extrema, in the plotted (tree-normalized) fields; they set each axis' range.
    ex = res["extrema"][0]
    pts = [(0.0, 0.0, ex["origin"])]
    pts += [(np.sqrt(x / xbar), 0.0, k) for x, k in ex["H"]]
    pts += [(0.0, np.sqrt(y / ybar), k) for y, k in ex["S"]]
    pts += [(np.sqrt(x / xbar), np.sqrt(y / ybar), k) for x, y, k in ex["M"]]
    ax.minorticks_off()
    ax.tick_params(length=3.5, width=0.7)
    ax.set_box_aspect(1)
    ax.set_title(TITLES[label])

    # Without a ground state, or with a large-field H minimum, the H axis runs on to the cutoff: to show V
    # falling beyond the barrier, or rising again beyond the minimum.
    large, past = label in LARGE_FIELD_H, label in PAST_CUTOFF_H
    sx = axis_spec([a for a, _, _ in pts], past_cutoff_top(P, v0) if past else CUTOFF / v0,
                   to_cutoff=PANELS[label][5] == "NGS" or large)
    if large:
        # linear, so that the outer minimum and the rise beyond it get room (denser samples near the origin)
        top = sx["top"]
        sx = dict(scale="linear", top=top, ticks=list(np.arange(0, 0.8 * CUTOFF / v0, 5.0)),
                  samples=np.unique(np.concatenate([np.linspace(0, 3, 150), np.linspace(0, top, 500)])))
    sy = axis_spec([b for _, b, _ in pts], CUTOFF / w0)
    U, W = np.meshgrid(sx["samples"], sy["samples"])
    draw_map(ax, U, W, (U * v0)**2, (W * w0)**2, P)
    for (a, b, _), kind in zip(pts, classify_markers(pts, res)):
        ax.plot(a, b, ls="none", clip_on=False, zorder=5, **MARKERS[kind])

    for spec, set_scale, set_lim, set_ticks in [(sx, ax.set_xscale, ax.set_xlim, ax.set_xticks),
                                                (sy, ax.set_yscale, ax.set_ylim, ax.set_yticks)]:
        if spec["scale"] == "symlog":
            set_scale("symlog", linthresh=LINTHRESH, linscale=0.6)
        set_lim(0, spec["top"])
        set_ticks(spec["ticks"], [f"{t:g}" for t in spec["ticks"]])
    ax.minorticks_off()
    if large:
        # Tick the cutoff, and hatch beyond it.
        cut = CUTOFF / v0
        if past:
            with plt.rc_context({"hatch.linewidth": 0.7}):
                ax.axvspan(cut, sx["top"], zorder=3, **BEYOND_CUTOFF_STYLE)
            ax.axvline(cut, color="0.35", lw=0.6, ls=":", zorder=3)
        ticks = [t for t in sx["ticks"] if t < 0.75 * cut] + [cut]
        ax.set_xticks(ticks, [f"{t:g}" for t in ticks[:-1]] + [r"$\Lambda$"])


def add_key(ax, ground_state=True):
    """The key; without the global-minimum marker when no panel has a ground state."""
    ax.axis("off")
    handles = [Line2D([], [], ls="none", label=MARKER_LABELS[k], **s) for k, s in MARKERS.items()
               if ground_state or k != "global"]
    handles += [Line2D([], [], label=r"$\partial V/\partial |H| = 0$", **CURVE_STYLE["h"]),
                Line2D([], [], label=r"$\partial V/\partial |S| = 0$", **CURVE_STYLE["s"])]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncols=1, fontsize=9, frameon=False,
              handlelength=2.0, labelspacing=0.55)
    cax = ax.inset_axes([0.12, 0.02, 0.76, 0.035])
    cax.imshow(np.linspace(0, 1, 256)[None, :], cmap=CMAP, aspect="auto", extent=(0, 1, 0, 1))
    cax.set_xticks([0, 1], ["low", "high"], fontsize=9)
    cax.set_yticks([])
    cax.set_title(r"$\mathrm{arcsinh}(V/V_0)$", fontsize=9, pad=3)


def make_gallery(layout, key_slot, height, output):
    """One gallery figure: the panels in layout, the key in key_slot; saved as output/<output>."""
    fig, axes = plt.subplots(len(layout), len(layout[0]), figsize=(LINEWIDTH_IN, height))
    for row_axes, row in zip(axes, layout):
        for ax, label in zip(row_axes, row):
            if label is not None:
                plot_panel(ax, label)
            elif (list(axes.flat).index(ax) // len(layout[0]), list(row_axes).index(ax)) != tuple(key_slot):
                ax.remove()
    # x labels under the lowest panel of each column
    for c in range(len(layout[0])):
        rows = [i for i in range(len(layout)) if layout[i][c] is not None]
        if rows:
            axes[rows[-1], c].set_xlabel(r"$|H|/v_0$")
    for row_axes, row in zip(axes, layout):
        if row[0] is not None:
            row_axes[0].set_ylabel(r"$|S|/w_0$")
    fig.tight_layout(w_pad=0.6, h_pad=1.0)
    ground_state = any(PANELS[l][5] != "NGS" for row in layout for l in row if l is not None)
    key_ax = axes[key_slot]
    pos = key_ax.get_position()
    key_ax.remove()
    add_key(fig.add_axes([pos.x0 - 0.02, pos.y0 - 0.02, pos.width + 0.04, pos.height + 0.04]), ground_state)
    style.savefig(fig, output)
    return fig




if __name__ == "__main__":
    fill_points()
    for layout, key_slot, height, output in FIGURES:
        make_gallery(layout, key_slot, height, output)
    print("Points:")
    for label, (kappa, r, oh, os_, _, _) in PANELS.items():
        lamH, lamS = L.lambdas_from_tree_Omegas(r, kappa, oh, os_)
        print(f"  {label:7s} kappa = {kappa:<5g} r = {r:<4g} Omega_H = {oh:.4f} Omega_S = {os_:.4f} "
              f"lambda_H = {lamH:.4f} lambda_S = {lamS:.4f}")
