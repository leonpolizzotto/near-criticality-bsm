"""Fig. 15: H(y) (eq. (4.21)) above G_b(y) (eq. (4.20)) for several b; the extrema of G_b are where H = b.

Writes output/1d_example_running_HG_stacked.pdf."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the nearcrit package

import numpy as np
import matplotlib.pyplot as plt

from nearcrit import style
from nearcrit.style import LINEWIDTH_IN, set_xlabel_right, set_ylabel_top
from nearcrit.toy_running import (
    b0, b_CRIT, G, H, plot_extrema,
)

style.use(amsmath=True)
plt.rcParams["hatch.linewidth"] = 0.3
# Second figure (2x2): H and G on top, V4p17 (placeholder so far) and the (a, b) phase diagram below.
# G' = 4y (b - H), so the extrema of G are where H = b: ln y = (-3 +- sqrt(5 - 16 b)) / 4. H has its maximum
# b_CRIT = 5/16 at y = e^{-3/4}; for b < b_CRIT the root below e^{-3/4} is a maximum of G and the one above a
# minimum. At b = b0 that minimum is G = 0, at y = e^{-1/4}. The values of G there are a_plus_minus(b, +-1).
def G_extrema_positions(b):
    """Positions y > 0 of the extrema of G(., b), i.e. the solutions of H(y) = b."""
    if np.isclose(b, b_CRIT, rtol=0, atol=1e-12):
        return np.array([np.exp(-0.75)])
    if b > b_CRIT:
        return np.array([])
    root = np.sqrt(5 - 16*b)
    return np.exp((-3 + np.array([-root, root])) / 4)


# One b in each regime and at each critical value, coloured as the a/c values of the first figure.
b_lines = [
    (b_CRIT + 0.125, r"$b > b_\text{crit}$", "C0", "-"),
    (b_CRIT, r"$b = b_\text{crit}$", "C1", "--"),
    ((b_CRIT + b0) / 2, r"$b_0 < b < b_\text{crit}$", "C2", "-"),
    (b0, r"$b = b_0$", "C3", "--"),
    (b0 - (b_CRIT - b0) / 2, r"$b < b_0$", "C4", "-"),
]


# H above G on a shared y axis. G' = 4y (b - H), so the extrema of G sit directly below the points where the
# horizontal line of the same colour crosses H; dotted lines join them (drawn in each panel separately). (In ln y,
# H = -1/4 - 3L/2 - L^2 is a parabola, symmetric about its maximum b_crit at L = -3/4, but a linear y axis shows G = 0
# at y = 0 and matches the potential figures.) The G axis is ticked at a_crit = a_+(b_crit) = a_-(b_crit) = e^{-3/2},
# where the maximum and minimum of G merge into the saddle at y = e^{-3/4}; a light grey line through it shows this.
y_HG = np.linspace(1e-6, np.exp(0.3), 1000)
figHG, (axH2, axG2) = plt.subplots(2, 1, figsize=(LINEWIDTH_IN, 4.8), sharex=True, height_ratios=[1, 1.55])
axH2.plot(y_HG, H(y_HG), color="k", lw=1.5)
H_lim, G_lim = (-0.2, 0.5), (-0.22, 0.38)
for b, label, color, ls in b_lines:
    axH2.axhline(b, color=color, ls=ls, lw=1.0)
    axG2.plot(y_HG, G(y_HG, b), color=color, ls=ls, lw=1.5, label=label)
    ye = G_extrema_positions(b)
    plot_extrema(axH2, ye, H(ye), color)  # minima of G filled, maxima and the inflection point open
    plot_extrema(axG2, ye, G(ye, b), color)
    for y_e in ye:
        axH2.plot([y_e, y_e], [H_lim[0], b], color=color, lw=0.6, ls=":", zorder=1)
        axG2.plot([y_e, y_e], [G(y_e, b), G_lim[1]], color=color, lw=0.6, ls=":", zorder=1)
for ax, lim in ((axH2, H_lim), (axG2, G_lim)):
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_ylim(lim)
axH2.set_yticks([b0, b_CRIT], [r"$b_0$", r"$b_\text{crit}$"])
axG2.axhline(np.exp(-1.5), color="0.75", lw=0.6, zorder=0.5)
axG2.set_yticks([0, np.exp(-1.5)], [r"$0$", r"$a_\text{crit}$"])
set_ylabel_top(axH2, r"$H$")
set_ylabel_top(axG2, r"$G_b$")
axG2.set_xlim(0, y_HG[-1])
axG2.set_xticks([0, np.exp(-0.75)], [r"$0$", r"$e^{-3/4}$"])
set_xlabel_right(axG2, r"$y$")
figHG.tight_layout(h_pad=0.3)
figHG.legend(*axG2.get_legend_handles_labels(), loc="upper center", ncol=5, fontsize=9, frameon=False,
             handlelength=1.8, columnspacing=1.0, bbox_to_anchor=(0.5, 1.0))
figHG.subplots_adjust(top=0.93)
style.savefig(figHG, "1d_example_running_HG_stacked.pdf")
