"""Fig. 13: the (a, c) phase diagram of the single-log potential (4.8), and the ground state along c = 1.

Writes output/1d_example_running_1_E0.pdf."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the nearcrit package

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from nearcrit import style
from nearcrit.style import LINEWIDTH_IN, set_xlabel_right, set_ylabel_top
from nearcrit.toy_running import (
    A_PM_STYLE, CONTOUR_COLOR, EXTREMA_LINE_STYLE, GS_HEIGHT_IN, HATCH_COLOR, hatch_legend_handles,
    N_EXTREMA_HATCHES, PHASE_ALPHA, PHASE_COLOR_NGS, r_FO_single_log, r_SP_single_log, single_log_minima,
    single_log_n_extrema, single_log_phase_rgb, TRANSITION_STYLE,
)

style.use(amsmath=True)
plt.rcParams["hatch.linewidth"] = 0.3
def plot_single_log_ground_state(r_lim, c_contour, filename, a_lim=(-0.4, 0.3), c_lim=(-0.6, 1.3), color=CONTOUR_COLOR,
                                 n_r=4001):
    """The (a, c) phase diagram with the line c = c_contour, a / c_contour in r_lim, and the ground state along it."""
    height = GS_HEIGHT_IN - 2.1  # the circle figure without its bottom row
    axes = plt.figure(figsize=(LINEWIDTH_IN, height)).subplot_mosaic([["PD", "x"], ["E", "dE"]])
    fig = axes["PD"].figure
    axPD, axX, axE, axdE = (axes[k] for k in ("PD", "x", "E", "dE"))

    # Phase diagram: wedges between the rays r = a/c = const (exact polygons, not pixels)
    (a_lo, a_hi), (c_lo, c_hi) = a_lim, c_lim
    phase_wedges = [
        ([a_lo, a_hi, a_hi, a_lo], [c_lo, c_lo, 0, 0], 1 - PHASE_ALPHA * (1 - PHASE_COLOR_NGS)),
        ([0, r_FO_single_log * c_hi, a_lo, a_lo], [0, c_hi, c_hi, 0], single_log_phase_rgb(False)),
        ([0, a_hi, a_hi, r_FO_single_log * c_hi], [0, 0, c_hi, c_hi], single_log_phase_rgb(True)),
    ]
    for a_w, c_w, rgb in phase_wedges:
        axPD.fill(a_w, c_w, color=rgb, lw=0, zorder=0)
    # Extrema at x > 0 for c < 0 are the same as for c > 0 (V changes sign), at r = a/c
    extrema_wedges = [
        ([0, r_SP_single_log * c_hi, a_lo, a_lo], [0, c_hi, c_hi, 0], 0),
        ([0, 0, r_SP_single_log * c_hi], [0, c_hi, c_hi], 2),
        ([0, a_hi, a_hi, 0], [0, 0, c_hi, c_hi], 1),
        ([0, a_lo, a_lo, 0], [0, 0, c_lo, c_lo], 1),
        ([0, 0, r_SP_single_log * c_lo], [0, c_lo, c_lo], 2),
        ([0, r_SP_single_log * c_lo, a_hi, a_hi], [0, c_lo, c_lo, 0], 0),
    ]
    for a_w, c_w, n in extrema_wedges:
        axPD.fill(a_w, c_w, facecolor="none", edgecolor=HATCH_COLOR, hatch=N_EXTREMA_HATCHES[n], lw=0, zorder=0.5)
    axPD.axhline(0, color="gray", lw=0.5)
    axPD.plot([0, 0], c_lim, **EXTREMA_LINE_STYLE)  # x = 0 turns from minimum to maximum
    axPD.plot(r_SP_single_log * np.array(c_lim), c_lim, **A_PM_STYLE)  # spinodal: metastable minimum and maximum
    axPD.plot([0, r_FO_single_log * c_hi], [0, c_hi], color="k", lw=1.5)  # first order, 0 <-> II
    axPD.plot(a_lim, [0, 0], color="k", lw=1.5)  # c = 0: the edge of the region with no ground state
    # the contour, with a dot at its start and an arrow showing its direction (as draw_contour)
    a_line = c_contour * np.array(r_lim)
    axPD.plot(a_line, [c_contour, c_contour], color=color, lw=1.5)
    axPD.plot(a_line[0], c_contour, "o", color=color, ms=3.5, zorder=4)
    a_arrow = a_line[0] + 0.8 * np.diff(a_line)[0]  # in phase II, clear of the rays
    axPD.annotate("", xy=(a_arrow, c_contour), xytext=(a_arrow - 1e-3, c_contour),
                  arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5, mutation_scale=10))
    axPD.set_xticks([0], [r"$0$"])
    axPD.set_yticks([0, c_contour], [r"$0$", rf"${c_contour:g}$"])
    axPD.set_xlim(a_lim)
    axPD.set_ylim(c_lim)
    set_xlabel_right(axPD, r"$a$")
    set_ylabel_top(axPD, r"$c$")

    # Along the contour, against a/c
    r = np.linspace(*r_lim, n_r)
    ground, meta, in_II = single_log_minima(r)
    scales = (1, c_contour, 1)  # x, E0, dE0/da from x, e0, e0'
    i_FO = np.searchsorted(r, r_FO_single_log)  # the ground state jumps here
    panels = ((axX, r"$x_\text{min}$"), (axE, r"$E_0$"), (axdE, r"$\partial E_0/\partial a$"))
    for k, (ax, label) in enumerate(panels):
        ax.plot(np.insert(r, i_FO, np.nan), np.insert(scales[k] * ground[k], i_FO, np.nan), color=color, lw=1.5,
                zorder=3)
        ax.plot(np.insert(r, i_FO, np.nan), np.insert(scales[k] * meta[k], i_FO, np.nan), color=color, lw=1.2,
                ls=(0, (1, 1.5)), zorder=2.5)
        lo, hi = np.nanmin(scales[k] * ground[k]), np.nanmax(scales[k] * ground[k])
        if ax is axE:  # include the metastable minimum, above 0
            hi = np.nanmax(scales[k] * meta[k])
        lo, hi = min(lo, 0), max(hi, 0)
        pad = 0.08 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
        if ax is axE:  # room above 0 for its tick label, clear of the y label at the top
            ax.set_ylim(lo - pad, max(hi + pad, -0.15 * lo))
        if ax is axdE:
            ax.set_ylabel(label, labelpad=2)
        else:
            set_ylabel_top(ax, label)
        # Background: phase colour and hatching by the number of extrema, and lines where these change
        r_edges = np.array([r_lim[0], r_SP_single_log, r_FO_single_log, 0, r_lim[1]])
        for r_from, r_to in zip(r_edges[:-1], r_edges[1:]):
            r_mid = 0.5 * (r_from + r_to)
            ax.axvspan(r_from, r_to, color=single_log_phase_rgb(r_mid > r_FO_single_log), lw=0, zorder=0)
            ax.axvspan(r_from, r_to, facecolor="none", edgecolor=HATCH_COLOR,
                       hatch=N_EXTREMA_HATCHES[single_log_n_extrema(r_mid)], lw=0, zorder=0.5)
        ax.axvline(r_SP_single_log, zorder=1, **A_PM_STYLE)
        ax.axvline(0, zorder=1, **EXTREMA_LINE_STYLE)
        ax.axvline(r_FO_single_log, zorder=2, **TRANSITION_STYLE["first"])
        ax.axhline(0, color="gray", lw=0.5, zorder=1)
        ax.set_xlim(r_lim)
        ax.set_yticks([0], [r"$0$"])
        # ticks at the spinodal and at the transition, (a/c)_crit = F(e^{-1/2}) = -1/(2e), their labels pushed apart
        # (they are close)
        ax.set_xticks([r_SP_single_log, r_FO_single_log, 0], [r"$-e^{-3/2}$", r"$(a/c)_\text{crit}$", r"$0$"])
        ax.get_xticklabels()[0].set_ha("right")
        ax.get_xticklabels()[1].set_ha("left")
        set_xlabel_right(ax, r"$a/c$")
    x_top = 1.08 * np.nanmax(ground[0])
    axX.set_ylim(-0.05 * x_top, x_top)
    axX.set_yticks([0, np.exp(-0.5)], [r"$0$", r"$e^{-1/2}$"])  # where x jumps to at the transition

    fig.tight_layout(rect=(0, 0, 1, 1 - 0.79 / height))  # the legends take the top 0.79 in
    line_handles = [
        Line2D([], [], color=color, lw=1.5, label=rf"Contour ($c = {c_contour:g}$)"),
        Line2D([], [], color=color, lw=1.2, ls=(0, (1, 1.5)), label="Metastable minimum"),
        Line2D([], [], **{**TRANSITION_STYLE["first"], "lw": 1.5}, label="First-order transition"),
    ]
    fig.legend(handles=line_handles, loc="upper center", fontsize=9, frameon=False, ncol=3, handlelength=2.5,
               columnspacing=1.5, bbox_to_anchor=(0.5, 1.0))
    phase_handles = [Patch(color=single_log_phase_rgb(False), label="0"), Patch(color=single_log_phase_rgb(True), label="II"),
                     Patch(color=1 - PHASE_ALPHA * (1 - PHASE_COLOR_NGS), label="NGS")]
    fig.legend(handles=phase_handles, loc="upper right", fontsize=9, frameon=False, ncol=3, columnspacing=1.0,
               title="Quantum phase", title_fontsize=9, bbox_to_anchor=(0.42, 1 - 0.28 / height))
    fig.legend(handles=hatch_legend_handles()[:3], loc="upper left", fontsize=9, frameon=False, ncol=3,
               columnspacing=1.0, handleheight=1.4, title="Number of nontrivial extrema", title_fontsize=9,
               bbox_to_anchor=(0.5, 1 - 0.28 / height))
    style.savefig(fig, filename)


plot_single_log_ground_state((-0.3, 0.12), 1.0, "1d_example_running_1_E0.pdf", a_lim=(-0.35, 0.2))
