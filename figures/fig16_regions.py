"""Fig. 16: the (a, b) plane of the potential (4.19) split into the regions referred to in the text, each shaded
and labelled:
  A: b < b_crit and a_-(b) < a < a_+(b)   (three extrema at y > 0)
  B: a > 0, outside A
  C: a < 0, outside A
The diagram is drawn at half the text width, with the key to its right; the script prints the width to include it at.

Writes output/1d_example_running_regions.pdf.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the nearcrit package

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

from nearcrit import style
from nearcrit.style import LINEWIDTH_IN, set_xlabel_right, set_ylabel_top
from nearcrit.toy_running import a_minus, a_plus, b0, b_CRIT

style.use(amsmath=True)

HALF_LINEWIDTH_IN = 0.5 * LINEWIDTH_IN
a_CRIT = np.exp(-1.5)  # a_+ = a_- at b_CRIT

# Region names, written upright so they are not mistaken for the parameters a and b.
REGION_NAMES = [r"\textrm{A}", r"\textrm{B}", r"\textrm{C}"]
REGION_DEFINITIONS = [
    r"$b<b_\text{crit},\ a_-<a<a_+$",
    r"$a>0$, not A",
    r"$a<0$, not A",
]
REGION_COLORS = ["#CC6677", "#88CCEE", "#DDCC77"]
REGION_ALPHA = 0.45
# Opaque blends with white, so overlapping band edges leave no seams.
REGION_RGB = [tuple(1 - REGION_ALPHA * (1 - np.array(to_rgb(c)))) for c in REGION_COLORS]


def plot_regions(fig, ax, x0, a_lim, b_lim, label_positions, tick_b0):
    a_lo, a_hi = a_lim
    b_lo, b_hi = b_lim
    b_below = np.linspace(b_lo, b_CRIT, 800)
    b_upper = np.array([b_CRIT, b_hi])
    b_lower = b_below[b_below <= b0]
    b_middle = b_below[b_below >= b0]
    # (b, a from, a to, region) bands for fill_betweenx. a_- and a_+ are clipped to the plot, since a_- runs far to
    # the left at large negative b.
    clip = lambda a: np.clip(a, a_lo, a_hi)
    bands = [
        (b_below, clip(a_minus(b_below)), clip(a_plus(b_below)), 0),
        (b_below, clip(a_plus(b_below)), a_hi, 1),
        (b_middle, 0, a_minus(b_middle), 1),
        (b_upper, 0, a_hi, 1),
        (b_lower, a_lo, clip(a_minus(b_lower)), 2),
        (b_middle, a_lo, 0, 2),
        (b_upper, a_lo, 0, 2),
    ]
    for b, a_from, a_to, n in bands:
        ax.fill_betweenx(b, a_from, a_to, color=REGION_RGB[n], lw=0.5)  # edge hides seams

    # No boundary lines between the regions (they would look like transitions); only the zero axes and the critical point.
    ax.axhline(0, color="gray", lw=0.5, zorder=1)
    ax.axvline(0, color="gray", lw=0.5, zorder=1)
    ax.plot(a_CRIT, b_CRIT, "o", color="k", ms=4, zorder=3)

    for name, (a, b) in zip(REGION_NAMES, label_positions):
        ax.text(a, b, rf"{name}", ha="center", va="center", fontsize=11)

    ax.set_xticks([0, a_CRIT], [r"$0$", r"$a_\text{crit}$"])
    if tick_b0:
        ax.set_yticks([0, b0, b_CRIT], [r"$0$", r"$b_0$", r"$b_\text{crit}$"])
    else:
        ax.set_yticks([0, b_CRIT], [r"$0$", r"$b_\text{crit}$"])
    ax.set_xlim(a_lim)
    ax.set_ylim(b_lim)
    set_xlabel_right(ax, r"$a$")
    set_ylabel_top(ax, r"$b$")

    handles = [Patch(facecolor=c, edgecolor="none", label=f"{name}: {definition}")
               for c, name, definition in zip(REGION_RGB, REGION_NAMES, REGION_DEFINITIONS)]
    return fig.legend(handles=handles, title="Region", fontsize=9, title_fontsize=9, frameon=False, handlelength=1.4,
                      handleheight=1.4, handletextpad=0.5, borderaxespad=0.0, alignment="left", loc="center left",
                      bbox_to_anchor=(x0 + 0.04 / FIG_WIDTH_IN, 0.55))


# The diagram (with its axis labels) takes half the text width; the key sits to its right, so the figure is wider:
# include it at width=FIG_WIDTH_IN/LINEWIDTH_IN.
LEGEND_WIDTH_IN = 1.9
FIG_WIDTH_IN = HALF_LINEWIDTH_IN + LEGEND_WIDTH_IN
A_LIM, B_LIM = (-0.65, 0.65), (-0.65, 0.65)
LABEL_POSITIONS = [(-0.12, -0.42), (0.42, 0.1), (-0.38, 0.3)]  # of A, B, C

fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 2.9))
legend = plot_regions(fig, ax, HALF_LINEWIDTH_IN / FIG_WIDTH_IN, A_LIM, B_LIM, LABEL_POSITIONS, tick_b0=False)
fig.tight_layout(rect=(0, 0, HALF_LINEWIDTH_IN / FIG_WIDTH_IN, 1))
style.savefig(fig, "1d_example_running_regions.pdf")
r = fig.canvas.get_renderer()
print("legend width (in):", round(legend.get_window_extent(r).width / fig.dpi, 2))
print(f"include with width={FIG_WIDTH_IN / LINEWIDTH_IN:.3f}\\linewidth")
