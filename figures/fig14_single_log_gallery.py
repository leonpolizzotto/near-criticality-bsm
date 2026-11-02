"""Fig. 14: the potential (4.8) in each of its potential phases.

Writes output/1d_example_running_1_gallery.pdf."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the nearcrit package

import numpy as np
import matplotlib.pyplot as plt

from nearcrit import style
from nearcrit.style import LINEWIDTH_IN, set_xlabel_right, set_ylabel_top
from nearcrit.toy_running import (
    extrema_positions, EXTREMUM_MARKERS, r_FO_single_log, r_SP_single_log, V4p8, V_limits,
)

style.use(amsmath=True)
plt.rcParams["hatch.linewidth"] = 0.3
# Gallery of the potential phases of the single log potential V = -a x^2 + c x^4 ln x (as c V4p8(x, a/c)), named as
# above: for c > 0 by the ground state phase (0 or II) and the number of extrema at x > 0, for c < 0 NGS and that number.
# The extrema at x > 0 are at the same a/c for either sign of c (where F = a/c), with minima and maxima swapped; x = 0 is
# a minimum where a < 0. With no ground state (c < 0) every minimum is metastable. The top row (c = 1) is in increasing
# a/c: 00 (a/c < -e^{-3/2}), 02 (up to the transition at -1/(2e)), II2 (up to 0), II1. The bottom row (c = -1) sits
# between them in the same order: NGS0 (a/c < -e^{-3/2}), NGS2 (up to 0), NGS1. II2 and NGS2 are at a/c = -0.15, where
# the maximum near x = 0 (V ~ 0.004) is not too shallow to see.
def all_extrema_single_log(r, sign_c):
    """Every extremum of c V4p8(., r) at x >= 0, |c| = 1, as (x, V, kind) like all_extrema."""
    xs = np.concatenate(([0.0], extrema_positions(r)))
    is_min = np.concatenate(([sign_c * r < 0], sign_c * (4 * np.log(xs[1:]) + 3) > 0))  # F' = x (4 ln x + 3)
    Vs = np.concatenate(([0.0], V4p8(xs[1:], r, sign_c)))
    ground = np.argmin(np.where(is_min, Vs, np.inf)) if sign_c > 0 else -1  # no ground state for c < 0
    kinds = ["ground" if i == ground else "metastable" if m else "max" for i, m in enumerate(is_min)]
    return xs, Vs, kinds


gallery_single_log = {  # mosaic label: (a/c, sign of c, potential phase, number of extrema at x > 0)
    "A": (-0.3, 1, "00", 0),
    "B": ((r_SP_single_log + r_FO_single_log) / 2, 1, "02", 2),
    "C": (-0.15, 1, "II2", 2),
    "D": (0.15, 1, "II1", 1),
    "E": (-0.3, -1, "NGS0", 0),
    "F": (-0.15, -1, "NGS2", 2),
    "G": (0.15, -1, "NGS1", 1),
}
# Each panel spans two columns of eight; the bottom row is offset by one, so its panels sit between those above
axesGal1 = plt.figure(figsize=(LINEWIDTH_IN, 3.8)).subplot_mosaic("AABBCCDD;.EEFFGG.")
figGal1 = axesGal1["A"].figure
for key, (r, sign_c, phase, n) in gallery_single_log.items():
    ax = axesGal1[key]
    xs, Vs, kinds = all_extrema_single_log(r, sign_c)
    assert len(xs) - 1 == n, (r, sign_c, xs)
    # one extremum at x > 0: a wider x range, to show the potential rising (c > 0) or falling (c < 0) past it
    x_hi = (1.45 if n == 1 else 1.3) * xs.max() if n > 0 else 1.1
    x_plot = np.linspace(1e-9, x_hi, 800)
    V_plot = V4p8(x_plot, r, sign_c)
    ax.plot(x_plot, V_plot, color="C0", lw=1.2)
    for x_e, V_e, kind in zip(xs, Vs, kinds):
        ax.plot(x_e, V_e, "o", zorder=3, clip_on=False, **EXTREMUM_MARKERS[kind])
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlim(0, x_hi)
    if n == 0:  # only x = 0: show the curve rising (c > 0) or falling (c < 0) across the panel
        V_lo, V_hi = (-0.4 * V_plot.max(), V_plot.max()) if sign_c > 0 else (V_plot.min(), -0.4 * V_plot.min())
    else:
        V_lo, V_hi = V_limits(x_plot, V_plot, Vs)
    ax.set_ylim(V_lo, V_hi)
    ax.set_xticks([0], [r"$0$"])
    ax.set_yticks([0], [r"$0$"])
    ax.set_title(phase)
for key in ("A", "E"):
    set_ylabel_top(axesGal1[key], r"$V$")
for key in gallery_single_log:  # all of them: the label takes no extra space, beside the 0 tick
    set_xlabel_right(axesGal1[key], r"$x$")
figGal1.tight_layout()
style.savefig(figGal1, "1d_example_running_1_gallery.pdf")
