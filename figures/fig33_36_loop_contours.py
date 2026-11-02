"""Figs. 33-36: the ground state of the RG-improved one-loop Higgs-singlet potential along an elliptical contour in the
tree-level (Omega_H, Omega_S) plane, at fixed r = mu_S^2 / mu_H^2 and kappa: the one-loop version of Fig. 24.

  - Top left: the phase map of a panel of Figs. 25-30 (a closer view around the contour if one is given) with the
    contour, a dot at its start (theta = 0) and an arrow showing its direction.
  - The other panels, against theta along the contour: the ground-state energy E_0 = V_min - V(0); the position of
    the global minimum in tree-normalized fields, Hbar = H / sqrt(mu_H^2 / lambda_H) and Sbar = S / sqrt(mu_S^2 /
    lambda_S); and the first three derivatives of E_0 projected on the unit tangent t of the contour,
    (t.grad)^n E_0 for n = 1, 2, 3 (derivatives in Omega). A first-order transition is a jump in the first
    derivative, a second-order one a jump in the second only. The background is the quantum phase (colour) and the
    potential phase (hatch) at each point, with lines at the transitions as on the phase map, and gray dash-dot lines
    where a tree-level mass in the loop crosses zero at the minimum (see compute/loop_contours.py).

Fig. 33: r = 1, kappa = 0.5, round the triple point (map: Fig. 25).
Fig. 34: r = 2, kappa = 0.5, through phase d (I d and III d) (map: zoom (0.5, 2.0)).
Fig. 35: r = -1, kappa = -1 (map: Fig. 28). E_0 and its derivatives in phase III are 10^3-10^6 times those in phase I,
         so the panels against theta are on a symmetric log scale, linear within the size of the values in phase I.
Fig. 36: r = -0.5, kappa = -0.5, across the first-order line III -> III' of phase h (map: Fig. 30 (c)); the outer
         minimum is ~100 times deeper, so the panels are on a symmetric log scale.

Reads data/loop_contour_<name>.npz (made by compute/loop_contours.py) and the phase maps (compute/loop_phase_maps.py).
Writes output/loop_contour_<name>_sm_rg.pdf.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the nearcrit package

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from nearcrit import data_io, style
from nearcrit.loop_maps import (
    HATCH_COLOR, HATCH_RC, POTENTIAL_TRANSITION_STYLE, ZOOMS, ZOOMS_SIGNS, _LEGEND_KW, add_legends, draw_panel,
    hatch_of, label_panels, phase_color, present_entries,
)
from nearcrit.singlet_tree import TRANSITION_LINESTYLE
from nearcrit.style import LINEWIDTH_IN, set_xlabel_right, set_ylabel_top

style.use()

MAPS = "compute/loop_phase_maps.py"
CONTOUR_COLOR = "C3"
THETA_TRANSITION_LW = 1.0
# Where a tree-level mass in the loop crosses zero at the minimum (see tree_mass_zeros): thin gray dash-dot lines in
# the panels against theta, and a cross on the contour on the phase map
CROSSING_STYLE = dict(color="0.25", lw=0.8, ls=(0, (4, 1.5, 1, 1.5)))
CROSSING_MARKER = dict(marker="x", color="0.15", ms=4.5, mew=1.0, ls="none")
CROSSING_LABEL = r"Tree-level $m^2 = 0$"
THETA_TICK_LABELS = {-1: r"$-\pi/2$", 0: r"$0$", 1: r"$\pi/2$", 2: r"$\pi$", 3: r"$3\pi/2$"}
LIMIT_DIST = 2e-3         # the y limits come from the points at least this far (in Omega) from a quantum transition
                          # (limit_dist in CONTOURS for a contour smaller than that)

# Drawing options of each contour (its geometry is in its data file): the phase map it is drawn on (map: a zoom, or the
# grid panel of the given data file), where the direction arrow sits (theta), a closer view of the map with its ticks,
# a symmetric log scale for the panels against theta, where the H/S legend goes, and limit_dist for a contour smaller
# than LIMIT_DIST.
DRAWING = {
    "r1_k05": dict(map="grid", grid="loop_grid_rg.npz", arrow_at=np.pi / 4),
    "r2_k05_zoom": dict(map="zoom", arrow_at=4.6, view=((0.79, 0.95), (0.975, 1.06)), ticks=([0.8, 0.9], [1.0, 1.04])),
    "rm1_km1": dict(map="grid", grid="loop_grid_rg_kneg_mSneg.npz", arrow_at=np.pi / 4, symlog=True,
                    legend_loc="upper right"),
    "rm05_km05_zoom": dict(map="zoom_signs", arrow_at=np.pi / 4, view=((0.5822, 0.5862), (1.7675, 1.7725)),
                           ticks=([0.583, 0.585], [1.768, 1.771]), symlog=True, limit_dist=1.5e-4),
}


def output_name(name):
    """File name of the figure of contour name."""
    return f"loop_contour_{name}_sm_rg.pdf"


def load_contour(name):
    """The settings (geometry and drawing options) and the data of contour name."""
    saved = data_io.load(style.data_file(f"loop_contour_{name}.npz", made_by="compute/loop_contours.py"))
    c = {**saved["contour"], **DRAWING[name]}
    d = saved["data"]
    d["limit_dist"] = c.get("limit_dist", LIMIT_DIST)
    return c, d


# --- Drawing ----------------------------------------------------------------------------------------------------------
def draw_contour(ax, d, color=CONTOUR_COLOR, arrow_at=np.pi / 4):
    """The contour on the phase map, with a dot at its start (theta = 0) and an arrow showing its direction."""
    ax.plot(d["OH"], d["OS"], color=color, lw=1.5, zorder=7)
    i0 = np.argmin(np.abs(d["theta"]))
    ax.plot(d["OH"][i0], d["OS"][i0], "o", color=color, ms=3.5, zorder=8)
    j = np.argmin(np.abs(d["theta"] - arrow_at))
    ax.annotate("", xy=(d["OH"][j], d["OS"][j]), xytext=(d["OH"][j - 1], d["OS"][j - 1]), zorder=8,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5, mutation_scale=10))
    for th, _ in d["crossings"]:
        OH, OS = np.interp(th, d["theta"], d["OH"]), np.interp(th, d["theta"], d["OS"])
        ax.plot(OH, OS, zorder=9, **CROSSING_MARKER)


def decorate_along_contour(ax, d):
    """Background of a panel against theta: the quantum phase colour and potential phase hatch between the
    transitions, a line at each transition (purple potential, black solid first order, black dashed second order),
    a gray dash-dot line where a tree-level mass in the loop crosses zero at the minimum, a zero line, and the theta
    axis."""
    theta = d["theta"]
    cuts = [theta[0]] + [t["theta"] for t in d["transitions"]] + [theta[-1]]
    for a, b in zip(cuts[:-1], cuts[1:]):
        i = np.argmin(np.abs(theta - 0.5 * (a + b)))
        q, letter = d["Q"][i], d["P"][i]
        ax.axvspan(a, b, facecolor=phase_color(q), lw=0, zorder=0)
        hatch = hatch_of(q, letter)
        if hatch:
            ax.axvspan(a, b, facecolor="none", edgecolor=HATCH_COLOR, hatch=hatch, lw=0, zorder=0.5)
    for t in d["transitions"]:
        if t["kind"] == "potential":
            ax.axvline(t["theta"], color=POTENTIAL_TRANSITION_STYLE["colors"],
                       lw=POTENTIAL_TRANSITION_STYLE["linewidths"], zorder=1)
        else:
            ax.axvline(t["theta"], color="k", lw=THETA_TRANSITION_LW, ls=TRANSITION_LINESTYLE[t["kind"]], zorder=2)
    for th, _ in d["crossings"]:
        ax.axvline(th, zorder=2, **CROSSING_STYLE)
    ax.axhline(0, color="gray", lw=0.5, zorder=1)
    ax.set_xlim(theta[0], theta[-1])
    quarters = np.arange(np.ceil(theta[0] / (np.pi / 2)), np.ceil(theta[-1] / (np.pi / 2) - 0.1)).astype(int)
    ax.set_xticks(quarters * np.pi / 2, [THETA_TICK_LABELS[k] for k in quarters])
    ax.set_yticks([0], [r"$0$"])
    set_xlabel_right(ax, r"$\theta$")


def symlog_axis(ax, values, far):
    """A symmetric log y axis, linear within the median |value| (in practice the size of the values in the largest
    phase), with ticks at 0 and at up to two powers of ten on each side; limits padded in the scaled coordinates."""
    v = np.abs(values[np.isfinite(values)])
    lin = 10.0 ** np.ceil(np.log10(np.median(v[v > 0])))
    ax.set_yscale("symlog", linthresh=lin, linscale=1.0)
    lo, hi = min(np.nanmin(values[far]), 0), max(np.nanmax(values[far]), 0)
    T = ax.yaxis.get_transform()
    a, b = T.transform([lo, hi])
    pad = 0.06 * (b - a)
    ax.set_ylim(*T.inverted().transform([a - pad, b + pad]))
    ticks, labels = [0.0], [r"$0$"]
    for sgn, end in ((1, hi), (-1, lo)):
        ks = np.arange(np.log10(lin), np.floor(np.log10(max(abs(end), lin))) + 1).astype(int)
        ks = ks[::-max(2, int(np.ceil(len(ks) / 2)))]    # at most two decades a side, 2+ apart, keeping the largest
        for k in ks:
            if 10.0**k <= abs(end) * 1.0001:
                ticks.append(sgn * 10.0**k)
                labels.append(("$-" if sgn < 0 else "$") + (f"10^{{{k}}}$" if k != 0 else "1$"))
    ax.set_yticks(ticks, labels)
    ax.minorticks_off()


def plot_along_contour(ax, d, values, color=CONTOUR_COLOR, set_limits=True, symlog=False, ends=None, **kw):
    """values against theta, not joined across quantum transitions (where the global minimum changes). ends (from
    transition_ends): the value of each branch at the transitions, which each piece then runs up to."""
    theta = d["theta"]
    cuts = [t["theta"] for t in d["transitions"] if t["kind"] != "potential"]
    at = {(th, side): v for th, side, v in (ends or [])}
    xs, ys = [], []
    for a, b in zip([-np.inf] + cuts, cuts + [np.inf]):
        sel = (theta > a) & (theta < b)
        px, py = list(theta[sel]), list(values[sel])
        if (a, "after") in at:
            px, py = [a] + px, [at[(a, "after")]] + py
        if (b, "before") in at:
            px, py = px + [b], py + [at[(b, "before")]]
        xs += px + [np.nan]
        ys += py + [np.nan]
    line, = ax.plot(xs, ys, color=color, lw=1.5, zorder=3, **kw)
    if set_limits:
        # from the points away from the quantum transitions: where a derivative diverges, it runs off the panel
        far = d["dist"] > d.get("limit_dist", LIMIT_DIST)
        if symlog:
            symlog_axis(ax, values, far)
            return line
        lo, hi = min(np.nanmin(values[far]), 0), max(np.nanmax(values[far]), 0)
        span = hi - lo
        ax.set_ylim(lo - (0.15 if lo == 0 else 0.08) * span, hi + (0.15 if hi == 0 else 0.08) * span)
    return line


def make_figure(name, c, d, height_in=7.9):
    """The figure of contour name, with its settings c (geometry and DRAWING options) and data d."""
    r, kappa = c["r"], c["kappa"]
    if c["map"] in ("zoom", "zoom_signs"):
        zooms, data = (ZOOMS_SIGNS, "loop_zoom_signs.npz") if c["map"] == "zoom_signs" else (ZOOMS, "loop_zoom.npz")
        panel = data_io.load(style.data_file(data, made_by=MAPS))["panels"][(kappa, r)]
        ticks = zooms[(kappa, r)]["ticks"]
    else:
        panels = data_io.load(style.data_file(c["grid"], made_by=MAPS))["panels"]
        panel = panels[(kappa, r)]
        ticks = None
    # the panel and its refined insets (keys (kappa, r, m)), drawn instead of it in the inset
    group = {(kappa, r): panel}
    if c["map"] == "grid":
        group.update({kk: p for kk, p in panels.items() if len(kk) == 3 and kk[:2] == (kappa, r)})
    labelled, _ = label_panels(group)
    Q, P = labelled[(kappa, r)]
    fines = [(group[kk], *labelled[kk]) for kk in sorted(group) if len(kk) == 3]

    with plt.rc_context(HATCH_RC):
        fig, axes = plt.subplots(3, 2, figsize=(LINEWIDTH_IN, height_in))
        axPD, axX, axE, ax1, ax2, ax3 = axes.flat

        draw_panel(axPD, panel, Q, P, r, kappa, ticks=ticks, fines=fines)
        if "view" in c:    # a closer view of the map around the contour
            axPD.set_xlim(c["view"][0])
            axPD.set_ylim(c["view"][1])
            axPD.set_xticks(c["ticks"][0], [f"${v:g}$" for v in c["ticks"][0]])
            axPD.set_yticks(c["ticks"][1], [f"${v:g}$" for v in c["ticks"][1]])
            # the map's lines and fills are clipped to the whole zoom window by their own clip path; clip them to the
            # axes as well
            for art in axPD.collections + axPD.patches + axPD.lines:
                art.set_clip_box(axPD.bbox)
        draw_contour(axPD, d, arrow_at=c["arrow_at"])
        set_xlabel_right(axPD, r"$\Omega_H$")
        set_ylabel_top(axPD, r"$\Omega_S$")

        for ax in (axE, axX, ax1, ax2, ax3):
            decorate_along_contour(ax, d)
        symlog = c.get("symlog", False)
        plot_along_contour(axE, d, d["E"], symlog=symlog, ends=d["ends"]["E"])
        if symlog:     # the 0 tick is at the top end of the axis: the label goes beside it
            axE.set_ylabel(r"$E_0$", labelpad=2)
        else:
            set_ylabel_top(axE, r"$E_0$")

        hl = plot_along_contour(axX, d, d["Hbar"], set_limits=False, ends=d["ends"]["Hbar"])
        sl = plot_along_contour(axX, d, d["Sbar"], color="k", set_limits=False, ends=d["ends"]["Sbar"])
        top = max(np.nanmax(d["Hbar"]), np.nanmax(d["Sbar"]))
        axX.set_ylim(-0.15 * top, 1.3 * top)
        yt = [0, 10] if top > 11 else [0, 1]
        axX.set_yticks(yt, [f"${v}$" for v in yt])
        axX.legend([hl, sl], [r"$\bar H$", r"$\bar S$"], loc=c.get("legend_loc", "center right"), fontsize=9, ncols=2, frameon=True,
                   framealpha=0.85, edgecolor="none", handlelength=1.4, borderpad=0.3, columnspacing=1.0)

        for ax, key, label in ((ax1, "d1", r"$\hat t\cdot\nabla E_0$"),
                               (ax2, "d2", r"$(\hat t\cdot\nabla)^2 E_0$"),
                               (ax3, "d3", r"$(\hat t\cdot\nabla)^3 E_0$")):
            plot_along_contour(ax, d, d[key], symlog=symlog)
            ax.set_ylabel(label, labelpad=2)

        quantum, entries = present_entries({(kappa, r): (Q, P), **{kk: labelled[kk] for kk in labelled if len(kk) == 3}})
        h = fig.get_figheight()
        # Below the keys, a key of its own for the tree-level masses crossing zero (not a phase transition)
        y0 = 0.06 / h
        if d["crossings"]:
            leg = fig.legend(handles=[Line2D([], [], **{**CROSSING_MARKER, **CROSSING_STYLE}, label=CROSSING_LABEL
                                             + " at the minimum (one-loop IR log)")],
                             **{**_LEGEND_KW, "handlelength": 2.4})
            rend = fig.canvas.get_renderer()
            w = leg.get_window_extent(rend).width / fig.dpi / fig.get_figwidth()
            leg.set_bbox_to_anchor((0.5 - w / 2, y0), transform=fig.transFigure)
            y0 += leg.get_window_extent(rend).height / fig.dpi / h + 0.1 / h
        keys_top = add_legends(fig, y0=y0, quantum_phases=quantum, potential_entries=entries)
        fig.tight_layout(rect=(0, keys_top + 0.05 / h, 1, 1), h_pad=1.2, w_pad=1.5)
        style.savefig(fig, output_name(name))
    return fig


def report(name, d):
    """Print the transitions and the tree-level mass crossings along contour name."""
    print(f"{name}: transitions along the contour")
    for t in d["transitions"]:
        print(f"  theta = {t['theta']:.6f} (+- {t['width']:.0e}) at (Omega_H, Omega_S) = ({t['OH']:.4f}, {t['OS']:.4f}):"
              f" {t['before']} -> {t['after']}, {t['kind']}, ground state moves by {t['jump']:.3g}")
    for th, mass in d["crossings"]:
        print(f"  theta = {th:.4f}: the tree-level mass of the {mass} crosses zero at the minimum")
    new = sorted(set(d["P"]) & set(d["new_letters"].values()))
    print("  radiatively generated phases on the contour:", new or "none")



if __name__ == "__main__":
    for name in DRAWING:
        c, data = load_contour(name)
        report(name, data)
        make_figure(name, c, data)
