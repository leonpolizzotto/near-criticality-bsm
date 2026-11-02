"""Fig. 12: F(x) (eq. (4.10)) and the potential (4.8) for several a/c; the extrema of V are where F = a/c.

Writes output/1d_example_running_1.pdf."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the nearcrit package

import numpy as np
import matplotlib.pyplot as plt

from nearcrit import style
from nearcrit.style import LINEWIDTH_IN, set_xlabel_right, set_ylabel_top
from nearcrit.toy_running import (
    a_over_c_CRIT_1, a_over_c_CRIT_2, extrema_positions, F, plot_extrema, V4p8,
)

style.use(amsmath=True)

# One a/c in each regime and at each critical value. The value between the two critical values is both
# < a_over_c_CRIT_1 and > a_over_c_CRIT_2.
a_over_c_lines = [
    (0.15, r"$a/c > 0$", "C0", "-"),
    (a_over_c_CRIT_1, r"$a/c = 0$", "C1", "--"),
    (a_over_c_CRIT_2 / 2, r"$-e^{-3/2} < a/c < 0$", "C2", "-"),
    (a_over_c_CRIT_2, r"$a/c = -e^{-3/2}$", "C3", "--"),
    (1.5 * a_over_c_CRIT_2, r"$a/c < -e^{-3/2}$", "C4", "-"),
]

x = np.linspace(1e-6, 1.3, 1000)

# F above V on a shared x axis, as in the H/G figure below. V4p8' = 2x (F - a/c), so the extrema of V sit directly
# below the points where the horizontal line of the same colour crosses F; dotted lines join them (drawn in each panel
# separately). The minimum of F, x = e^{-3/4}, and its zero, x = e^{-1/4}, are the extrema of a/c = -e^{-3/2} and
# a/c = 0, so the dotted lines of those two mark them.
F_lim, V_lim = (-0.4, 0.5), (-0.24, 0.5)
fig, (axF, axV) = plt.subplots(2, 1, figsize=(LINEWIDTH_IN, 4.8), sharex=True, height_ratios=[1, 1.55])
axF.plot(x, F(x), color="k", lw=1.5)
for a_over_c, label, color, ls in a_over_c_lines:
    axF.axhline(a_over_c, color=color, ls=ls, lw=1.0)
    axV.plot(x, V4p8(x, a_over_c), color=color, ls=ls, lw=1.5, label=label)
    xe = extrema_positions(a_over_c)
    plot_extrema(axF, xe, F(xe), color)
    plot_extrema(axV, xe, V4p8(xe, a_over_c), color)
    for x_e in xe:
        axF.plot([x_e, x_e], [F_lim[0], a_over_c], color=color, lw=0.6, ls=":", zorder=1)
        axV.plot([x_e, x_e], [V4p8(x_e, a_over_c), V_lim[1]], color=color, lw=0.6, ls=":", zorder=1)
for ax, lim in ((axF, F_lim), (axV, V_lim)):
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_ylim(lim)
axF.set_yticks([0, a_over_c_CRIT_2], [r"$0$", r"$-e^{-3/2}$"])
axV.set_yticks([0], [r"$0$"])
set_ylabel_top(axF, r"$F(x)$")
set_ylabel_top(axV, r"$V(x)$")
axV.set_xlim(0, x[-1])
axV.set_xticks([0, np.exp(-0.75), np.exp(-0.25)], [r"$0$", r"$e^{-3/4}$", r"$e^{-1/4}$"])
set_xlabel_right(axV, r"$x$")

# Inset zooming in on the maximum of V near x = 0 for -e^{-3/2} < a/c < 0, which is far too shallow (V ~ 0.002) to see
# at the scale of the panel. The inset sits directly above the zoomed region with the same x scale, so the dotted line
# from that maximum runs straight through it (it is drawn again inside the inset, which hides the panel behind it).
x_zoom = np.linspace(1e-6, 0.38, 400)
inset = axV.inset_axes([0, 0.52, x_zoom[-1] / x[-1], 0.44])
inset.set_ylim(-0.004, 0.006)
for a_over_c, label, color, ls in a_over_c_lines:
    inset.plot(x_zoom, V4p8(x_zoom, a_over_c), color=color, ls=ls, lw=1.2)
    xe = extrema_positions(a_over_c)
    xe = xe[xe < x_zoom[-1]]
    plot_extrema(inset, xe, V4p8(xe, a_over_c), color)
    for x_e in xe:
        inset.plot([x_e, x_e], [V4p8(x_e, a_over_c), inset.get_ylim()[1]], color=color, lw=0.6, ls=":", zorder=1)
inset.axhline(0, color="gray", lw=0.5)
inset.set_xlim(0, x_zoom[-1])
inset.set_xticks([])
inset.set_yticks([])
for spine in inset.spines.values():
    spine.set_linewidth(0.6)
# Box around the zoomed region, joined to the inset. The region is far too flat to see at the scale of the
# panel, so the box is given a minimum height (a fraction of the panel).
box_height = max(np.diff(inset.get_ylim())[0], 0.04 * np.diff(V_lim)[0])
box = (0, np.mean(inset.get_ylim()) - box_height / 2, x_zoom[-1], box_height)
indicator = axV.indicate_inset(box, inset_ax=inset, edgecolor="0.3", lw=0.6, alpha=1)
for connector in indicator.connectors:
    connector.set(lw=0.6, color="0.3")

fig.tight_layout(h_pad=0.3)
fig.legend(*axV.get_legend_handles_labels(), loc="upper center", ncol=5, fontsize=9, frameon=False,
           handlelength=1.8, columnspacing=1.0, bbox_to_anchor=(0.5, 1.0))
fig.subplots_adjust(top=0.93)
style.savefig(fig, "1d_example_running_1.pdf")
