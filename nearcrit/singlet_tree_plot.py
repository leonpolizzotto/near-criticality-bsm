"""Drawing of Higgs-singlet phase maps, shared by the tree-level (Figs. 21, 22, 24) and loop-level (Figs. 25-36)
figures: the colour is the quantum phase and the gray hatch the potential phase (the letter a, b, c; the quantum phase
0 contains a single potential phase and is not hatched). Quantum phase transitions are black, solid for first order
and dashed for second order; potential phase transitions that are not quantum ones are purple. The keys sit below the
panels.
"""
from itertools import combinations

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from nearcrit.singlet_tree import (
    PHASE_ALPHA, PHASE_COLORS, PHASE_LABELS, QUANTUM_PHASE_LABELS, QUANTUM_PHASE_OF_POTENTIAL, TRANSITION_LINESTYLE,
    _fill_phase_seams, _transition_order,
)
from nearcrit.style import region_color
# Colours of the quantum phases, as in the earlier phase maps of tree_quantum_phases_fns.py: there each potential
# phase has its own shade, and a quantum phase is the middle shade of its family (Ib, IIb, IIIb; 0), drawn at
# the same PHASE_ALPHA. They are drawn as opaque colours over white, so that neighbouring regions leave no seams.
_SHADE = {label: PHASE_COLORS[code] for code, label in PHASE_LABELS.items()}
QUANTUM_COLORS = {"0": _SHADE["0"], "I": _SHADE["Ib"], "II": _SHADE["IIb"], "III": _SHADE["IIIb"]}
NGS_COLOR = "gray"
HATCH_COLOR = "0.5"  # gray hatching with boxed legend handles, as in 1d_example_running.py
# Potential phases by their letter within the quantum phase (lines only, since dot hatches come out black).
HATCHES = {"a": "///", "b": "\\\\\\", "c": "---"}
# Potential phase transitions that are not quantum phase transitions, as A_PM_STYLE in 1d_example_running.py.
POTENTIAL_TRANSITION_STYLE = dict(colors="#8e24aa", linewidths=1.1)


def quantum_color(label):
    """The fill colour of the quantum phase with this label."""
    return region_color(QUANTUM_COLORS[label], PHASE_ALPHA)



def draw_phase_map(ax, X, Y, PH):
    """Quantum phase colours, potential phase hatches and the transitions of the potential phase map PH on the
    grid (X, Y). Code 0 (no phase) is not drawn, so whatever is underneath shows through."""
    # Points exactly on a boundary belong to no phase; give them the phase next to them.
    PH = _fill_phase_seams(PH)
    QP = np.vectorize(QUANTUM_PHASE_OF_POTENTIAL.get)(PH)
    for q in np.unique(QP[QP > 0]):
        ax.contourf(X, Y, (QP == q).astype(float), levels=[0.5, 1.5],
                    colors=[quantum_color(QUANTUM_PHASE_LABELS[q])], zorder=1)
    # The hatch of each potential phase, as an unfilled patch.
    for p in np.unique(PH[PH > 0]):
        hatch = HATCHES.get(PHASE_LABELS[p][-1])
        if hatch is None:
            continue
        cs = ax.contourf(X, Y, (PH == p).astype(float), levels=[0.5, 1.5], colors="none",
                         hatches=[hatch], zorder=2)
        cs.set_edgecolor(HATCH_COLOR)
        cs.set_linewidth(0)

    ax.axhline(0, color="gray", lw=0.5, zorder=3)
    ax.axvline(0, color="gray", lw=0.5, zorder=3)

    # Phase transitions, one contour per pair of neighbouring potential phases: black for a quantum phase
    # transition (solid first order, dashed second order), purple for a potential phase transition within one
    # quantum phase. Short stubs where several regions meet at a point are grid artefacts, not boundaries.
    codes = [int(c) for c in np.unique(PH) if c != 0]
    for p1, p2 in combinations(codes, 2):
        Z = np.where(PH == p1, 1.0, np.where(PH == p2, 0.0, np.nan))
        if QUANTUM_PHASE_OF_POTENTIAL[p1] == QUANTUM_PHASE_OF_POTENTIAL[p2]:
            cs = ax.contour(X, Y, Z, levels=[0.5], zorder=3.5, **POTENTIAL_TRANSITION_STYLE)
        else:
            cs = ax.contour(X, Y, Z, levels=[0.5], colors="k", linewidths=1.5,
                            linestyles=TRANSITION_LINESTYLE[_transition_order(p1, p2)], zorder=4)
        cs.set_paths([path for path in cs.get_paths() if len(path.vertices) >= 5])


def add_legends(fig, y0, ngs=True, quantum_phases=("0", "I", "II", "III"), gap_in=0.4, row_gap_in=0.12):
    """The keys, centred below the panels, in two rows: the transition lines (black solid and dashed for first and
    second order quantum phase transitions, purple for potential phase transitions) on top, and below them the
    colours (quantum phases) and the hatches (potential phases), side by side. y0 is the bottom of the keys in
    figure coordinates; returns their top. ngs adds the NGS colour, for figures that have an NGS region, and
    quantum_phases lists the quantum phases to show (those in the figure)."""
    color_handles = [Patch(facecolor=quantum_color(label), edgecolor="none", label=label)
                     for label in quantum_phases]
    if ngs:
        color_handles.append(Patch(facecolor=region_color(NGS_COLOR), edgecolor="none", label="NGS"))
    hatch_handles = [Patch(facecolor="white", edgecolor=HATCH_COLOR, hatch=h, lw=0.5, label=letter)
                     for letter, h in HATCHES.items()]
    line_handles = [
        Line2D([], [], color="k", lw=1.5, ls=TRANSITION_LINESTYLE["first"], label="Quantum, first order"),
        Line2D([], [], color="k", lw=1.5, ls=TRANSITION_LINESTYLE["second"], label="Quantum, second order"),
        Line2D([], [], color=POTENTIAL_TRANSITION_STYLE["colors"], lw=POTENTIAL_TRANSITION_STYLE["linewidths"],
               label="Potential"),
    ]
    kw = dict(frameon=False, handlelength=1.4, handleheight=1.4, handletextpad=0.5, columnspacing=1.2,
              borderaxespad=0.0, borderpad=0.0, alignment="center", loc="lower left")
    leg_q = fig.legend(handles=color_handles, title="Quantum phase", ncols=len(color_handles), **kw)
    leg_p = fig.legend(handles=hatch_handles, title="Potential phase", ncols=len(hatch_handles), **kw)
    leg_t = fig.legend(handles=line_handles, title="Phase transition", ncols=len(line_handles),
                       **{**kw, "handlelength": 2.4, "columnspacing": 1.8})
    # Measure the legends, then place them: the pair centred on the bottom row, the lines centred above it.
    r = fig.canvas.get_renderer()
    W, H = fig.get_figwidth(), fig.get_figheight()
    size = lambda leg: (leg.get_window_extent(r).width / fig.dpi / W, leg.get_window_extent(r).height / fig.dpi / H)
    (w_q, h_q), (w_p, h_p), (w_t, h_t) = size(leg_q), size(leg_p), size(leg_t)
    gap = gap_in / W
    x0 = 0.5 - (w_q + gap + w_p) / 2
    leg_q.set_bbox_to_anchor((x0, y0), transform=fig.transFigure)
    leg_p.set_bbox_to_anchor((x0 + w_q + gap, y0), transform=fig.transFigure)
    y_t = y0 + max(h_q, h_p) + row_gap_in / H
    leg_t.set_bbox_to_anchor((0.5 - w_t / 2, y_t), transform=fig.transFigure)
    return y_t + h_t


# Hatch settings for these figures (used as plt.rc_context(HATCH_RC)): thin gray lines, as in Figs. 13-20.
HATCH_RC = {"hatch.linewidth": 0.3, "hatch.color": HATCH_COLOR}
