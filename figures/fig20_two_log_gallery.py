"""Fig. 20: the potential (4.19) in each of its potential phases.

Writes output/1d_example_running_gallery.pdf."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the nearcrit package

import numpy as np
import matplotlib.pyplot as plt

from nearcrit import style
from nearcrit.style import LINEWIDTH_IN, set_xlabel_right, set_ylabel_top
from nearcrit.toy_running import (
    a_minus, a_plus, all_extrema, b_CRIT, EXTREMUM_MARKERS, first_order_a_P1_P2, V_curve, V_limits,
)

style.use(amsmath=True)
plt.rcParams["hatch.linewidth"] = 0.3

# Gallery: the potential in each of the six potential phases, named by quantum phase and number of extrema at y > 0
# (00, 02, I3, II2, II3), except that the one-extremum region, which joins I and II round the critical point, is a
# single supercritical phase SC1. Along b = 1/8: 0 < a_- = 0.0711 < a_FO = 0.1101 < a_+ = 0.1751 (SC1 | I3 | II3 |
# SC1); along b = -1/16: a_- = -0.1707 < a_FO(P0/P2) = -0.0665 < 0 (00 | 02 | II2). The points are at the middle of
# each band, and SC1 is above the critical point.
b_mid, b_low = 1 / 8, -1 / 16
L_FO_low = (-1 + np.sqrt(1 - 4 * b_low)) / 2  # P0/P2 degenerate: b = -L (L + 1), a = -L e^{2L}
a_FO_low = -L_FO_low * np.exp(2 * L_FO_low)
a_FO_mid = first_order_a_P1_P2(b_mid)
gallery = [  # (a, b, potential phase, number of extrema at y > 0)
    (a_minus(b_low) - 0.06, b_low, "00", 0),
    ((a_minus(b_low) + a_FO_low) / 2, b_low, "02", 2),
    ((a_minus(b_mid) + a_FO_mid) / 2, b_mid, "I3", 3),
    (a_FO_low / 2, b_low, "II2", 2),
    ((a_FO_mid + a_plus(b_mid)) / 2, b_mid, "II3", 3),
    (np.exp(-1.5), b_CRIT + 0.09, "SC1", 1),
]
figGal, axesGal = plt.subplots(2, 3, figsize=(LINEWIDTH_IN, 3.8))
for ax, (a, b, phase, n) in zip(axesGal.flat, gallery):
    ys, Vs, kinds = all_extrema(a, b)
    assert len(ys) - 1 == n, (a, b, ys)
    y_hi = 1.3 * ys.max() if ys.max() > 0 else 0.8
    if phase in ("00", "SC1"):  # one minimum: a wider y range, to show the potential rising clearly
        y_hi = 2.1 * ys.max() if ys.max() > 0 else 1.1
    y_plot = np.linspace(0, y_hi, 800)
    V_plot = V_curve(y_plot, a, b)
    ax.plot(y_plot, V_plot, color="C0", lw=1.2)
    for y_e, V_e, kind in zip(ys, Vs, kinds):
        ax.plot(y_e, V_e, "o", zorder=3, clip_on=False, **EXTREMUM_MARKERS[kind])
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlim(0, y_hi)
    V_lo, V_hi = V_limits(y_plot, V_plot, Vs)
    if phase in ("00", "SC1"):
        V_hi = max(V_hi, V_plot.max())
    ax.set_ylim(V_lo, V_hi)
    ax.set_xticks([0], [r"$0$"])
    ax.set_yticks([0], [r"$0$"])
    ax.set_title(phase)
for ax in axesGal[:, 0]:
    set_ylabel_top(ax, r"$V$")
for ax in axesGal[-1, :]:
    set_xlabel_right(ax, r"$y$")
figGal.tight_layout()
style.savefig(figGal, "1d_example_running_gallery.pdf")
