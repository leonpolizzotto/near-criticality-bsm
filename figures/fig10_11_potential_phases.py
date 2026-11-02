"""Fig. 10: the potential phases of the quartic toy model and the transitions between them.
Fig. 11: its quantum (colour) and potential (hatch) phase diagram.

Writes output/1d_example_potential_phases.pdf and output/1d_example_potential_phase_diagram.pdf."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the nearcrit package

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

from nearcrit import style
from nearcrit.style import LINEWIDTH_IN, set_xlabel_right
from nearcrit.toy_quartic import V1D_example, plot_minima

style.use()

# Potential phases of the untilted potential. Unlike the quantum phases, these only depend on
# the signs of mu2 and lambda, so a phase does not need to have a minimum (phase NGS0 has none).
# Each entry is (mu2, lambda, title) with mu2, lambda in {-1, 0, 1}.
potential_phases = [
    [
        (1.0, 1.0, "Potential Phase 1\n" r"$\mu^2>0,\ \lambda>0$"),
        (0.0, 1.0, r"Transition 1$\to$2" "\n" r"$\mu^2=0,\ \lambda>0$"),
        (-1.0, 1.0, "Potential Phase 2\n" r"$\mu^2<0,\ \lambda>0$"),
    ],
    [
        (1.0, 0.0, r"Transition 1$\to$NGS1" "\n" r"$\mu^2>0,\ \lambda=0$"),
        (0.0, 0.0, "Double Critical Point\n" r"$\mu^2=0,\ \lambda=0$"),
        (-1.0, 0.0, r"Transition 2$\to$NGS0" "\n" r"$\mu^2<0,\ \lambda=0$"),
    ],
    [
        (1.0, -1.0, "Potential Phase NGS1\n" r"$\mu^2>0,\ \lambda<0$"),
        (0.0, -1.0, r"Transition NGS1$\to$NGS0" "\n" r"$\mu^2=0,\ \lambda<0$"),
        (-1.0, -1.0, "Potential Phase NGS0\n" r"$\mu^2<0,\ \lambda<0$"),
    ],
]

phi = np.linspace(-2.0, 2.0, 2001)

# The y limits are shared within a row; the middle row is centered on V=0 and the last row has
# V=0 near the top, since those potentials open downward.
row_ylims = [(-0.6, 1.6), (-2.2, 2.2), (-1.6, 0.6)]
fig, axes = plt.subplots(3, 3, figsize=(LINEWIDTH_IN, 6.8), sharex=True, sharey="row")

for i, ax, (mu2, lambda_, title) in zip(np.repeat(range(3), 3), axes.flat, sum(potential_phases, [])):
    V = V1D_example(phi, mu2, lambda_)
    ax.plot(phi, V, color="C0", lw=1.5)
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    plot_minima(ax, phi, V, lambda_)
    ax.set_title(title)
    ax.set_xlim(phi[0], phi[-1])
    ax.set_ylim(*row_ylims[i])
    ax.set_xticks([])
    ax.set_yticks([])

for ax in axes[-1]:
    ax.set_xlabel(r"$\phi$")
for ax in axes[:, 0]:
    ax.set_ylabel(r"$V(\phi)$")

fig.tight_layout()
style.savefig(fig, "1d_example_potential_phases.pdf")


# Potential phase diagram. It is a simple diagram, so it is included at half the line width
# (width=0.5\linewidth): draw it at that width so the 11 pt text stays the size of the body text.
# The color is the quantum phase (as in plot_phase_diagram) and the black hatch is the potential phase. There
# are no labels on the diagram: two legends below it give the colors and the hatches separately.
HALF_LINEWIDTH_IN = 0.5 * LINEWIDTH_IN
HATCH_COLOR = "0.5"  # gray hatching with boxed legend handles, as in Figs. 13-20


def plot_potential_phase_diagram(fig, ax, x0):
    # (polygon x, polygon y, color, hatch) for each potential phase. Phases NGS1 and NGS0 are both
    # No Ground State, so they share a color.
    # Phases 1 and 2 have the same hatch density (the same number of characters), and the legend uses
    # the very same strings.
    """Fig. 11 on ax: colour = quantum phase, hatch = potential phase, with the two legends to its right
    (x0 = their left edge in figure coordinates). Returns the two legends."""
    hatches = ["//", r"\\", "--", "xx"]  # phases 1, 2, NGS0, NGS1
    regions = [
        ([0, 1, 1, 0], [0, 0, 1, 1], "C0", hatches[0]),
        ([-1, 0, 0, -1], [0, 0, 1, 1], "C1", hatches[1]),
        ([-1, 0, 0, -1], [-1, -1, 0, 0], "gray", hatches[2]),
        ([0, 1, 1, 0], [-1, -1, 0, 0], "gray", hatches[3]),
    ]
    for x, y, color, hatch in regions:
        # Fill at alpha 0.3 as in plot_phase_diagram. The hatch is a separate, unfilled patch: in a PDF the
        # alpha of a filled patch also applies to its hatch in some viewers (it came out grey).
        ax.fill(x, y, facecolor=(*to_rgb(color), 0.3), lw=0)
        ax.fill(x, y, facecolor="none", edgecolor=HATCH_COLOR, hatch=hatch, lw=0)
    # Every axis is a boundary between potential phases (unlike the quantum phases, where mu2 = 0 is not
    # a transition for lambda < 0).
    ax.plot([0, 0], [-1, 1], color="k", lw=1.5)
    ax.plot([-1, 1], [0, 0], color="k", lw=1.5)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    set_xlabel_right(ax, r"$\mu^2$")
    ax.set_ylabel(r"$\lambda$", rotation=0, ha="right", va="top")
    ax.yaxis.set_label_coords(-0.03, 1.0)
    ax.set_xticks([0])
    ax.set_yticks([0])
    ax.set_xticklabels(["0"])
    ax.set_yticklabels(["0"])
    # Legends: the colors (quantum phases) and the hatches (potential phases), separately, to the right of
    # the diagram. They are placed in figure coordinates, x0 being the left edge of the legend column.
    color_handles = [Patch(facecolor=(*to_rgb(c), 0.3), edgecolor="none", label=label)
                     for c, label in [("C0", "I"), ("C1", "II"), ("gray", "NGS")]]
    hatch_handles = [Patch(facecolor="white", edgecolor=HATCH_COLOR, hatch=h, lw=0.5, label=label)
                     for h, label in zip(hatches, ["1", "2", "NGS0", "NGS1"])]
    kw = dict(frameon=False, handlelength=1.4, handleheight=1.4, handletextpad=0.5, borderaxespad=0.0,
              alignment="center", loc="center left")
    leg_q = fig.legend(handles=color_handles, title="Quantum phase", bbox_to_anchor=(x0, 0.76), **kw)
    leg_p = fig.legend(handles=hatch_handles, title="Potential phase", bbox_to_anchor=(x0, 0.30), **kw)
    return leg_q, leg_p


# The diagram (with its axis labels) takes the same width as the half-width figure was drawn at, and the
# legends sit to its right, so the figure is wider than 0.5 \linewidth: include it at width=FIG_WIDTH_IN/LINEWIDTH_IN.
LEGEND_WIDTH_IN = 1.3
FIG_WIDTH_IN = HALF_LINEWIDTH_IN + LEGEND_WIDTH_IN
with plt.rc_context({"hatch.linewidth": 0.3}):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 2.9))
    legends = plot_potential_phase_diagram(fig, ax, x0=HALF_LINEWIDTH_IN / FIG_WIDTH_IN)
    fig.tight_layout(rect=(0, 0, HALF_LINEWIDTH_IN / FIG_WIDTH_IN, 1))
    style.savefig(fig, "1d_example_potential_phase_diagram.pdf")
    # Report how wide the legends are, to choose LEGEND_WIDTH_IN.
    r = fig.canvas.get_renderer()
    print("legend widths (in):", [round(l.get_window_extent(r).width / fig.dpi, 2) for l in legends])
    print(f"include with width={FIG_WIDTH_IN / LINEWIDTH_IN:.3f}\\linewidth")
