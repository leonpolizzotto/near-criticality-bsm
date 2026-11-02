"""Figs. 17 and 19: two contours joining the same two points of the (a, b) plane of the potential (4.19), passing
above and below the critical point.

Fig. 17: the phase diagram, y_min, E0 and its first three derivatives along the contours.
Fig. 19: the potential at points along each contour.

Writes output/1d_example_running_crossover.pdf and output/1d_example_running_crossover_potentials.pdf."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the nearcrit package

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.lines import Line2D
from scipy.optimize import brentq

from nearcrit import style as figstyle  # (the code below uses 'style' for line styles)
from nearcrit.style import LINEWIDTH_IN, set_xlabel_right, set_ylabel_top
from nearcrit.toy_running import (
    a_minus, a_plus, A_PM_STYLE, all_extrema, b_CRIT, CONTOUR_COLOR, decorate_along_contour,
    draw_ground_state_phase_diagram, EXTREMUM_MARKERS, first_order_a_P1_P2, ground_state, ground_state_derivatives,
    ground_state_third_derivatives, GS_HEIGHT_IN, hatch_legend_handles, n_minima_region, phase_color,
    phase_legend_handles, shade_along_contour, TRANSITION_STYLE, transitions_along, V4p17, V_curve,
)

figstyle.use(amsmath=True)
plt.rcParams["hatch.linewidth"] = 0.3
# Crossover: the circle of radius CO_RADIUS around the critical point, counterclockwise, laid out as the ground state
# figures above. It is split into two contours joining the same two points, the
# bottom and the left end (theta = pi, Phase I): contour 1 (-pi/2 <= theta <= pi) goes round the right and over the top
# of the critical point, a crossover from II to I along which E0 is analytic and the potential keeps a single minimum;
# contour 2 (pi <= theta <= 3 pi/2) goes back from I to II below it, across a_-, the first-order line (the global
# minimum jumps from P1 to P2, two minima trading places) and a_+. The circle stays at a > 0, so there is no Phase 0, and
# the contours are told apart by colour and linestyle (blue is free). The bottom row shows the potential at points along
# each, coloured by the ground state phase as in the phase diagram (full-strength colours), with the same points marked
# on the phase diagram and on y_min. The theta panels run from 0 to 2 pi (contour 1 is the two stretches outside
# pi < theta < 3 pi/2), which puts the first-order transition near the middle.
CO_RADIUS = 0.18
CO_HALF_STYLES = (dict(color="b", ls="--"), dict(color=CONTOUR_COLOR, ls="-"))  # contour 1, contour 2
n_theta_CO = 4001
theta_CO = np.linspace(0, 2 * np.pi, n_theta_CO)
a_CO, b_CO = np.exp(-1.5) + CO_RADIUS * np.cos(theta_CO), b_CRIT + CO_RADIUS * np.sin(theta_CO)
E0_CO, (Ea_CO, Eb_CO), (Eaa_CO, Eab_CO, Ebb_CO), P_CO, E_CO = ground_state_derivatives(a_CO, b_CO)
y_CO = ground_state(a_CO, b_CO)[1]
# along the unit tangent (t_a, t_b) = (-sin, cos)
t_a_CO, t_b_CO = -np.sin(theta_CO), np.cos(theta_CO)
dE_CO = t_a_CO * Ea_CO + t_b_CO * Eb_CO
ddE_CO = t_a_CO**2 * Eaa_CO + 2 * t_a_CO * t_b_CO * Eab_CO + t_b_CO**2 * Ebb_CO
Eaaa_CO, Eaab_CO, Eabb_CO, Ebbb_CO = ground_state_third_derivatives(a_CO, b_CO)
dddE_CO = (t_a_CO**3 * Eaaa_CO + 3 * t_a_CO**2 * t_b_CO * Eaab_CO + 3 * t_a_CO * t_b_CO**2 * Eabb_CO
           + t_b_CO**3 * Ebbb_CO)
transitions_CO = transitions_along(a_CO, b_CO, P_CO, E_CO)
breaks_CO = [i for i, kind in transitions_CO if kind == "first"]
halves_CO = ((theta_CO <= np.pi) | (theta_CO >= 3 * np.pi / 2), (theta_CO >= np.pi) & (theta_CO <= 3 * np.pi / 2))
print(f"Crossover circle, radius {CO_RADIUS}:")
for i, kind in transitions_CO:
    print(f"  {kind} transition at theta = {theta_CO[i]:.3f}, (a, b) = ({a_CO[i]:.3f}, {b_CO[i]:.3f}): "
          f"P{P_CO[i - 1]} -> P{P_CO[i]}")
# The points where the potentials are shown (besides the three transitions on contour 2). Contour 1: every pi/4.
# Contour 2: its ends and one point between each pair of transitions, where both minima exist.
# where contour 2 crosses a_-, the first-order line and a_+ (each once, in this order), refined from the grid step
crossing_theta_CO = lambda boundary_a, i: brentq(
    lambda th: np.exp(-1.5) + CO_RADIUS * np.cos(th) - boundary_a(b_CRIT + CO_RADIUS * np.sin(th)),
    theta_CO[i - 1], theta_CO[i], xtol=1e-14)
i_minus, i_plus = np.flatnonzero(np.diff(n_minima_region(a_CO, b_CO)[1])) + 1
theta_minus, theta_FO, theta_plus = (crossing_theta_CO(f, i) for f, i in ((a_minus, i_minus),
                                                                          (first_order_a_P1_P2, breaks_CO[0]),
                                                                          (a_plus, i_plus)))
theta_samples_CO = (
    np.linspace(-np.pi / 2, np.pi, 7),
    np.array([np.pi, (theta_minus + theta_FO) / 2, (theta_FO + theta_plus) / 2, 3 * np.pi / 2]),
)

on_contour = lambda values, half: np.insert(np.where(half, values, np.nan), breaks_CO, np.nan)  # nan off it
# cyan, used nowhere else in these figures, outlined in black so it shows on both phase colours
CONTOUR_END_MARKER = dict(marker="x", color="#00e5ff", ms=7, mew=1.6, ls="none", zorder=5,
                          path_effects=[patheffects.Stroke(linewidth=3.2, foreground="k"), patheffects.Normal()])
# the potentials at the ends, in the same colour
CONTOUR_END_LINE = dict(zorder=2.8, path_effects=[patheffects.Stroke(linewidth=2.4, foreground="k"),
                                                  patheffects.Normal()])


def draw_crossover_phase_diagram(ax):
    """The phase diagram zoomed on the critical point, from the zero axes (so they are ticked) to the same distance
    beyond the circle in a and b, with the two contours, an arrow on each, and their common end points (crosses)."""
    draw_ground_state_phase_diagram(ax, a_lim=(0, np.exp(-1.5) + CO_RADIUS + 0.04),
                                    b_lim=(0, b_CRIT + CO_RADIUS + 0.04))
    for half, style, theta_samples in zip(halves_CO, CO_HALF_STYLES, theta_samples_CO):
        ax.plot(on_contour(a_CO, half), on_contour(b_CO, half), lw=1.5, zorder=3, **style)
        # arrow in the middle of the longest stretch between two of the points where the potentials are drawn
        k = np.argmax(np.diff(theta_samples))
        j = max(np.searchsorted(theta_CO, ((theta_samples[k] + theta_samples[k + 1]) / 2) % (2 * np.pi)), 1)
        ax.annotate("", xy=(a_CO[j], b_CO[j]), xytext=(a_CO[j - 1], b_CO[j - 1]), zorder=3,
                    arrowprops=dict(arrowstyle="-|>", color=style["color"], lw=1.5, mutation_scale=10))
    for theta in (np.pi, 3 * np.pi / 2):
        ax.plot(np.exp(-1.5) + CO_RADIUS * np.cos(theta), b_CRIT + CO_RADIUS * np.sin(theta), **CONTOUR_END_MARKER)


# Six panels as the ground state figures above: the phase diagram, y_min, E0 and its first three derivatives along the
# contour. The potentials along it are in a figure of their own, below.
axesCO = plt.figure(figsize=(LINEWIDTH_IN, GS_HEIGHT_IN)).subplot_mosaic(
    [["PD", "y"], ["E", "dE"], ["ddE", "dddE"]])
figCO = axesCO["PD"].figure
axPD, axY, axE, axdE, axddE, axdddE = (axesCO[k] for k in ("PD", "y", "E", "dE", "ddE", "dddE"))
draw_crossover_phase_diagram(axPD)
panels_CO = ((axY, y_CO, r"$y_\text{min}$"), (axE, E0_CO, r"$E_0$"), (axdE, dE_CO, r"$\hat t\cdot\nabla E_0$"),
             (axddE, ddE_CO, r"$\hat t^{\mathsf{T}}\,\nabla^2 E_0\,\hat t$"),
             (axdddE, dddE_CO, r"$(\hat t\cdot\nabla)^3 E_0$"))
for ax, values, label in panels_CO:
    decorate_along_contour(ax, theta_CO, a_CO, b_CO, transitions_CO)
    # vertical, beside the axis: the long derivative labels to save width, and y_min and E0 so that they clear their
    # tick labels (the 1 tick of y_min and the 0 tick of E0 sit near the top of the panel, where a label would go)
    ax.set_ylabel(label, labelpad=2)
for half, style in zip(halves_CO, CO_HALF_STYLES):
    for ax, values, _ in panels_CO:
        ax.plot(np.insert(theta_CO, breaks_CO, np.nan), on_contour(values, half), lw=1.5, zorder=3, **style)
# E0 is concave and <= 0, and so is its projected Hessian: room above 0 so the tick label stays clear of the top
for ax in (axE, axddE):
    y_lo = ax.get_ylim()[0]
    ax.set_ylim(y_lo, -0.15 * y_lo)
# The third derivative jumps to ~120 just past the first-order line (the new minimum is still close to its spinodal),
# against a crossover peak of ~50: the top is set by the peak, and the jump runs off it
d3_peak = np.nanmax(dddE_CO[np.abs(theta_CO - theta_FO) > 0.2])
axdddE.set_ylim(axdddE.get_ylim()[0], 1.5 * d3_peak)
y_top = 1.1 * y_CO.max()
axY.set_ylim(-0.05 * y_top, y_top)
axY.set_yticks(np.arange(0, y_top, 0.5), [rf"${v:g}$" for v in np.arange(0, y_top, 0.5)])
for ax, _, _ in panels_CO:  # phase colours, now the limits are set
    shade_along_contour(ax, theta_CO, phase_color(y_CO, P_CO, b_CO))

# The potentials exactly at the potential phase transitions on contour 2, in the colour of the line crossed there
# (purple a_+-, black first-order): at a_- a minimum and a maximum appear at an inflection point, H(y) = b with
# ln y = (-3 + s) / 4, and at a_+ the minimum near the origin and the maximum merge there, at ln y = (-3 - s) / 4 (open
# diamonds); on the first-order line the two minima are degenerate (both filled).
INFLECTION_MARKER = dict(marker="D", color="k", mfc="white", ms=3.5, ls="none")
CO_TRANSITION_POINTS = [(theta_minus, A_PM_STYLE["color"], +1), (theta_FO, TRANSITION_STYLE["first"]["color"], 0),
                        (theta_plus, A_PM_STYLE["color"], -1)]  # (theta, colour, sign of s at the inflection point)
# The panels of the potentials figure, a row for each contour in its direction: (theta, title, style), with style
# "end" (the colour of the end crosses), "phase" (coloured by the ground state phase) or (colour, sign of s at the
# inflection point, 0 for none) at a transition
CO_PANELS = (
    [(theta, rf"$\theta={label}$", "end" if k in (0, len(theta_samples_CO[0]) - 1) else "phase")
     for k, (theta, label) in enumerate(zip(theta_samples_CO[0], (r"-\pi/2", r"-\pi/4", "0", r"\pi/4", r"\pi/2",
                                                                  r"3\pi/4", r"\pi")))],
    [(np.pi, r"$\theta=\pi$", "end"), (theta_minus, r"$a=a_-$", CO_TRANSITION_POINTS[0][1:]),
     (theta_samples_CO[1][1], "", "phase"), (theta_FO, "First order", CO_TRANSITION_POINTS[1][1:]),
     (theta_samples_CO[1][2], "", "phase"), (theta_plus, r"$a=a_+$", CO_TRANSITION_POINTS[2][1:]),
     (3 * np.pi / 2, r"$\theta=3\pi/2$", "end")],
)


def crossover_point_color(theta, style):
    """Colour of the potential at theta: that of the end crosses, of the ground state phase (full strength), or of the
    transition line crossed there."""
    if style == "end":
        return CONTOUR_END_MARKER["color"]
    if style == "phase":
        a, b = np.exp(-1.5) + CO_RADIUS * np.cos(theta), b_CRIT + CO_RADIUS * np.sin(theta)
        _, y_g, P_g, _ = ground_state(a, b)
        return phase_color(np.array(y_g), np.array(P_g), np.array(b), alpha=1.0)
    return style[0]


def draw_crossover_potential(ax, theta, style, y_plot):
    """The potential at theta on the circle, with its minima (filled: global, open: metastable, both filled on the
    first-order line) and, at a_+-, the inflection point where a minimum and a maximum appear or merge (diamond). The
    potentials at the ends are outlined in black, like their crosses on the phase diagram."""
    a, b = np.exp(-1.5) + CO_RADIUS * np.cos(theta), b_CRIT + CO_RADIUS * np.sin(theta)
    color = crossover_point_color(theta, style)
    line_style, marker_style = (CONTOUR_END_LINE, dict(mec="k", mew=0.6)) if style == "end" else ({}, {})
    ax.plot(y_plot, V_curve(y_plot, a, b), color=color, lw=1.3, **line_style)
    inflection_sign = style[1] if isinstance(style, tuple) else 0
    y_inflection = np.exp((-3 + inflection_sign * np.sqrt(5 - 16 * b)) / 4) if inflection_sign else None
    V_g = ground_state(a, b)[0]
    for y_e, V_e, kind in zip(*all_extrema(a, b)):
        if y_inflection is not None and abs(y_e / y_inflection - 1) < 1e-3:
            continue  # the pair being born or annihilated, found (or not) by the root finder just off it
        if kind != "max":
            kind = "ground" if abs(V_e - V_g) < 1e-12 else kind
            ax.plot(y_e, V_e, "o", zorder=3, **{**EXTREMUM_MARKERS[kind], "color": color, "ms": 4, **marker_style})
    if y_inflection is not None:
        ax.plot(y_inflection, V4p17(y_inflection, a, b), zorder=3, **{**INFLECTION_MARKER, "ms": 3})
    ax.axhline(0, color="gray", lw=0.5)


POINT_MARKER = dict(marker="o", mec="k", mew=0.5, ms=4, ls="none", zorder=4)

# Key, as in the ground state figures: the two contours and the transition line, then the phases and hatches present
CO_KEY_IN = 0.86  # height of the key at the top
figCO.tight_layout(rect=(0, 0, 1, 1 - CO_KEY_IN / GS_HEIGHT_IN))
line_handles = [
    Line2D([], [], lw=1.5, label=r"Contour 1, $-\pi/2 \le \theta \le \pi$", **CO_HALF_STYLES[0]),
    Line2D([], [], lw=1.5, label=r"Contour 2, $\pi \le \theta \le 3\pi/2$", **CO_HALF_STYLES[1]),
    Line2D([], [], **{**TRANSITION_STYLE["first"], "lw": 1.5}, label="First-order transition"),
    Line2D([], [], **CONTOUR_END_MARKER, label="Ends"),
]
figCO.legend(handles=line_handles, loc="upper center", fontsize=9, frameon=False, ncol=4, handlelength=1.8,
             handletextpad=0.5, columnspacing=0.8, borderaxespad=0.2, bbox_to_anchor=(0.5, 1.0))
figCO.legend(handles=phase_legend_handles()[1:], loc="upper right", fontsize=9, frameon=False, ncol=2,
             columnspacing=1.0, title="Quantum phase", title_fontsize=9, bbox_to_anchor=(0.42, 1 - 0.3 / GS_HEIGHT_IN))
figCO.legend(handles=[hatch_legend_handles()[n] for n in (1, 3)], loc="upper left", fontsize=9, frameon=False,
             ncol=2, columnspacing=1.0, handleheight=1.4, title="Number of nontrivial extrema", title_fontsize=9,
             bbox_to_anchor=(0.5, 1 - 0.3 / GS_HEIGHT_IN))
figstyle.savefig(figCO, "1d_example_running_crossover.pdf")

# The potentials along the two contours of the crossover figure as small multiples: a row for each contour, one panel
# per point in order along it (arrows between them), the points marked on the phase diagram top left (as in the
# crossover figure, to recall the contours), with the key beside it. All panels share their axes. The V range is a
# compromise: the transition structure on contour 2 is shallow (the barrier at the first-order line is ~0.01, the
# inflection point at a_- at V = 0.015, which the top clears), while the wells on contour 1 go down to 0.17. It takes
# in every well but the deepest (theta = -pi/4, 0.165), and the end at 3 pi/2 (0.092, shown again as the first panel
# of contour 1).
CO_Y_MAX = 1.25
CO_V_RANGE = (-0.12, 0.03)
CO_V_HEIGHT_IN = 7.4
n_col_COV = max(len(row) for row in CO_PANELS)
figCOV = plt.figure(figsize=(LINEWIDTH_IN, CO_V_HEIGHT_IN))
gs_COV = figCOV.add_gridspec(2, 1, height_ratios=[1.75, 5.15], hspace=0.25, left=0.075, right=0.985, top=0.99,
                             bottom=0.035)
gs_top_COV = gs_COV[0].subgridspec(1, 2, width_ratios=[1, 1.75], wspace=0.2)
axPD_COV, axKey = figCOV.add_subplot(gs_top_COV[0]), figCOV.add_subplot(gs_top_COV[1])
draw_crossover_phase_diagram(axPD_COV)
for row in CO_PANELS:  # the points of the panels (the ends under their crosses)
    for theta, _, style in row:
        axPD_COV.plot(np.exp(-1.5) + CO_RADIUS * np.cos(theta), b_CRIT + CO_RADIUS * np.sin(theta),
                      color=crossover_point_color(theta, style), **POINT_MARKER)
gs_V_COV = gs_COV[1].subgridspec(2, n_col_COV, hspace=0.42, wspace=0.14)
y_plot_COV = np.linspace(0, CO_Y_MAX, 1200)
for r, row in enumerate(CO_PANELS):
    for c, (theta, title, style) in enumerate(row):
        ax = figCOV.add_subplot(gs_V_COV[r, c])
        draw_crossover_potential(ax, theta, style, y_plot_COV)
        ax.set_xlim(0, CO_Y_MAX)
        ax.set_ylim(CO_V_RANGE)
        ax.set_xticks([])
        ax.set_yticks([0], [r"$0$"] if c == 0 else [])
        ax.tick_params(axis="y", length=2 if c == 0 else 0, pad=2)
        ax.set_title(title, fontsize=9, pad=3)
        if c == 0:  # at the top, as set_ylabel_top, but further out: the panels are narrow
            set_ylabel_top(ax, r"$V$")
            ax.yaxis.set_label_coords(-0.12, 1.0)
            ax.text(-0.3, 1.17, f"Contour {r + 1}", transform=ax.transAxes, ha="left", va="bottom")
        if (r, c) == (len(CO_PANELS) - 1, len(row) - 1):  # one y label (all panels share their axes)
            set_xlabel_right(ax, r"$y$")
        if c < len(row) - 1:  # arrow to the next panel
            ax.annotate("", xy=(1.12, 0.5), xytext=(1.01, 0.5), xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="-|>", color="0.3", lw=0.8, mutation_scale=7))
# Key, beside the phase diagram: the contours, the first-order transition (and the potential at it on contour 2) and
# the end points (the purple a_+- lines and the potentials at them are left to the caption); below, the quantum phases
# (as the background of the phase diagram, and at full strength as the colours of the potentials) and the number of
# extrema
axKey.axis("off")
line_handles = [
    Line2D([], [], lw=1.5, label=r"Contour 1, $-\pi/2 \le \theta \le \pi$", **CO_HALF_STYLES[0]),
    Line2D([], [], lw=1.5, label=r"Contour 2, $\pi \le \theta \le 3\pi/2$", **CO_HALF_STYLES[1]),
    Line2D([], [], lw=1.5, color=TRANSITION_STYLE["first"]["color"], label="First-order transition"),
    Line2D([], [], **{**CONTOUR_END_MARKER, "ls": "-", "lw": 1.2}, label="Ends of the contours"),
]
key_lines = axKey.legend(handles=line_handles, loc="upper left", fontsize=9, frameon=False, handlelength=2.2,
                         bbox_to_anchor=(0.0, 1.03))
axKey.add_artist(key_lines)
key_phases = axKey.legend(handles=phase_legend_handles()[1:], loc="lower left", fontsize=9, frameon=False, ncol=2,
                          columnspacing=0.8, title="Quantum phase", title_fontsize=9, bbox_to_anchor=(0.0, 0.0),
                          borderaxespad=0.2)
axKey.add_artist(key_phases)
axKey.legend(handles=[hatch_legend_handles()[n] for n in (1, 3)], loc="lower right", fontsize=9, frameon=False,
             ncol=2, columnspacing=0.8, handleheight=1.4, title="Number of nontrivial extrema", title_fontsize=9,
             bbox_to_anchor=(1.0, 0.0), borderaxespad=0.2)
figstyle.savefig(figCOV, "1d_example_running_crossover_potentials.pdf")
