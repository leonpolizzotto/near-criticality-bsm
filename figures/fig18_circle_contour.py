"""Fig. 18: the ground state of the potential (4.19) along the circle a^2 + b^2 = 0.6^2 around the origin.

Writes output/1d_example_running_E0_circle.pdf."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the nearcrit package

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from matplotlib.lines import Line2D

from nearcrit import style
from nearcrit.style import LINEWIDTH_IN, set_ylabel_top
from nearcrit.toy_running import (
    b_CRIT, B_WIDOM_END, CONTOUR_COLOR, decorate_along_contour, draw_contour, draw_ground_state_phase_diagram,
    FAR_INSET_BOUNDS, ground_state, ground_state_derivatives, ground_state_third_derivatives, GS_HEIGHT_IN,
    HATCH_COLOR, hatch_legend_handles, ln_y_widom, N_EXTREMA_HATCHES, n_minima_region, PEAK_INSET_LS, phase_color,
    phase_legend_handles, shade_along_contour, TRANSITION_STYLE, transitions_along,
)

style.use(amsmath=True)
plt.rcParams["hatch.linewidth"] = 0.3


def plot_ground_state_along(a_c, b_c, A, B, title, filename, color, ls, other_contours=(), zoom_inset=True,
                            peak_inset=None, n_theta=4001):
    """The ground state figure along the ellipse (a_c, b_c, A, B), drawn in colour and linestyle (color, ls). The
    other_contours, (a_c, b_c, A, B, color, ls) each, are also drawn on the phase diagram, for comparison. zoom_inset
    adds an inset zoomed in on the divergence of the third derivative at a continuous transition, and peak_inset
    (theta from, theta to) one zoomed in on a narrow but finite peak of the third derivative (at the crossover)."""
    theta = np.linspace(0, 2 * np.pi, n_theta)
    a = a_c + A * np.cos(theta)
    b = b_c + B * np.sin(theta)
    E0, (Ea, Eb), (Eaa, Eab, Ebb), P, E = ground_state_derivatives(a, b)
    y_min = ground_state(a, b)[1]  # position of the global minimum (0 on P0)
    # unit tangent, in the direction of increasing theta
    t_a, t_b = -A * np.sin(theta), B * np.cos(theta)
    t_norm = np.hypot(t_a, t_b)
    t_a, t_b = t_a / t_norm, t_b / t_norm
    dE = t_a * Ea + t_b * Eb
    ddE = t_a**2 * Eaa + 2 * t_a * t_b * Eab + t_b**2 * Ebb
    Eaaa, Eaab, Eabb, Ebbb = ground_state_third_derivatives(a, b)
    dddE = t_a**3 * Eaaa + 3 * t_a**2 * t_b * Eaab + 3 * t_a * t_b**2 * Eabb + t_b**3 * Ebbb

    transitions = transitions_along(a, b, P, E)
    breaks = [i for i, kind in transitions if kind == "first"]  # the curves are not joined across a jump
    # nor is the third derivative across a continuous transition, where it diverges on the y > 0 side
    breaks_d3 = sorted(i for i, _ in transitions)
    # Widom line crossings: above b_CRIT (and below its end), where ln y_min passes the line's ln y
    above = (b > b_CRIT) & (b < B_WIDOM_END)
    with np.errstate(divide="ignore", invalid="ignore"):
        side = np.sign(np.log(y_min) - ln_y_widom(np.clip(b, b_CRIT, B_WIDOM_END)))
    widom_crossings = np.flatnonzero(above[1:] & above[:-1] & (np.diff(side) != 0)) + 1
    theta_between = lambda i: 0.5 * (theta[i - 1] + theta[i])
    print(f"{title}:")
    for i, kind in transitions:
        print(f"  {kind} transition at theta = {theta_between(i):.3f}, (a, b) = ({a[i]:.3f}, {b[i]:.3f}): "
              f"P{P[i - 1]} -> P{P[i]}")
    for i in widom_crossings:
        print(f"  Widom line at theta = {theta_between(i):.3f}, (a, b) = ({a[i]:.3f}, {b[i]:.3f})")

    axes = plt.figure(figsize=(LINEWIDTH_IN, GS_HEIGHT_IN)).subplot_mosaic(
        [["PD", "y"], ["E", "dE"], ["ddE", "dddE"]])
    fig = axes["PD"].figure
    axPD, axY, axE, axdE, axddE, axdddE = (axes[k] for k in ("PD", "y", "E", "dE", "ddE", "dddE"))

    draw_ground_state_phase_diagram(axPD)
    for contour in (*other_contours, (a_c, b_c, A, B, color, ls)):
        draw_contour(axPD, *contour)

    # Along the contour: background coloured and hatched as the phase diagram at each point, grey lines where the
    # number of extrema changes, and black lines at the ground state transitions (solid: first order, dashed:
    # continuous). The Widom line is not drawn (only the colour fade follows it); its crossings are printed.
    contour_colors = phase_color(y_min, P, b)
    panels = ((axY, y_min, r"$y_\text{min}$"), (axE, E0, r"$E_0$"), (axdE, dE, r"$\hat t\cdot\nabla E_0$"),
              (axddE, ddE, r"$\hat t^{\mathsf{T}}\,\nabla^2 E_0\,\hat t$"),
              (axdddE, dddE, r"$(\hat t\cdot\nabla)^3 E_0$"))
    for ax, values, label in panels:
        decorate_along_contour(ax, theta, a, b, transitions)
        ax_breaks = breaks_d3 if ax is axdddE else breaks
        ax.plot(np.insert(theta, ax_breaks, np.nan), np.insert(values, ax_breaks, np.nan), color=color, ls=ls, lw=1.5,
                zorder=3)
        if ax in (axdE, axddE, axdddE):  # long labels: vertical, beside the axis, to save width
            ax.set_ylabel(label, labelpad=2)
        else:
            set_ylabel_top(ax, label)
    # E0 is a minimum over y of functions linear in (a, b), so it is concave: E0 <= 0 and the projected Hessian is
    # <= 0. Leave room above 0 so its tick label stays clear of the y label at the top.
    for ax in (axE, axddE):
        y_lo = ax.get_ylim()[0]
        ax.set_ylim(y_lo, -0.15 * y_lo)
    # y_min: 0 on P0 (a little room below so the curve along 0 is clear)
    y_top = max(1.1, 1.08 * y_min.max())
    axY.set_ylim(-0.05 * y_top, y_top)
    axY.set_yticks(np.arange(0, y_top, 0.5), [rf"${v:g}$" for v in np.arange(0, y_top, 0.5)])
    # The third derivative diverges at the continuous transition. At fixed b, with L = ln y, both a and E_aaa are exact
    # functions of L: a = e^{2L} (2b + L + 2L^2) and E_aaa = (3/2 + 2L) / (8 e^{2L} D^3), D = b + 1/4 + 3L/2 + L^2.
    # a -> 0+ is L -> -inf, where E_aaa ~ -e^{-2L} / (4 |L|^5) -> -inf (asymptotically 1 / (2 a L^3) ~ 4 / (a ln^3 a),
    # approached only slowly, with 1/L corrections), while on P0 it is 0. The tangent is (-1, 0) there, so the contour
    # derivative goes to +inf. It only exceeds the crossover peak within ~1e-4 of the transition, far finer than the
    # theta grid, so it is shown in an inset zoomed in on the transition. The panel limits are set by the rest of the
    # contour.
    def shade_inset(inset, x_lim, theta_0):
        """Background of an inset with x = theta - theta_0, coloured and hatched as the main panel."""
        x_bg = np.linspace(*x_lim, 801)
        th_bg = theta_0 + 0.5 * (x_bg[1:] + x_bg[:-1])
        a_bg, b_bg = a_c + A * np.cos(th_bg), b_c + B * np.sin(th_bg)
        _, y_bg, P_bg, _ = ground_state(a_bg, b_bg)
        inset.imshow(phase_color(y_bg, P_bg, b_bg)[None], extent=(*x_lim, 0, 1), transform=inset.get_xaxis_transform(),
                     aspect="auto", interpolation="nearest", zorder=0)
        n_bg = n_minima_region(a_bg, b_bg)[1]
        changes = np.flatnonzero(np.diff(n_bg)) + 1
        for start, end in zip(np.concatenate(([0], changes)), np.concatenate((changes, [len(n_bg)]))):
            inset.axvspan(x_bg[start], x_bg[end], facecolor="none", edgecolor=HATCH_COLOR,
                          hatch=N_EXTREMA_HATCHES[n_bg[start]], lw=0, zorder=0.5)

    i_continuous = [i for i, kind in transitions if kind == "continuous"]
    away = np.ones(n_theta, bool)
    for i in i_continuous:
        away &= np.abs(theta - theta_between(i)) > 0.15
    d3_lo, d3_hi = np.nanmin(dddE[away]), np.nanmax(dddE[away])
    pad = 0.05 * (d3_hi - d3_lo)
    axdddE.set_ylim(min(d3_lo, 0) - pad, max(d3_hi, 0) + pad)
    for i in i_continuous:
        theta_c = brentq(lambda th: a_c + A * np.cos(th), theta[i - 1], theta[i], xtol=1e-15)  # at a = 0
        # On the main panel, the y > 0 side continued on a log-spaced grid up to the transition, so it visibly runs off
        # the top there (while the P0 side starts from 0)
        side = -1 if P[i - 1] > 0 else 1  # the y > 0 side
        th_approach = theta_c + side * np.logspace(np.log10(abs(theta[i] - theta[i - 1])), -12, 200)
        a_ap, b_ap = a_c + A * np.cos(th_approach), b_c + B * np.sin(th_approach)
        ta, tb = -A * np.sin(th_approach), B * np.cos(th_approach)
        ta, tb = ta / np.hypot(ta, tb), tb / np.hypot(ta, tb)
        T = ground_state_third_derivatives(a_ap, b_ap)
        i_last = i - 1 if side < 0 else i  # last grid point on the y > 0 side
        axdddE.plot(np.append(theta[i_last], th_approach),
                    np.append(dddE[i_last], ta**3 * T[0] + 3 * ta**2 * tb * T[1] + 3 * ta * tb**2 * T[2] + tb**3 * T[3]),
                    color=color, ls=ls, lw=1.5, zorder=3)
        if not zoom_inset:
            continue
        # Inset: the last 2e-3 of theta before the transition (and a little after), on linear axes, where the
        # divergence becomes visible. It sits in the P0 stretch, on the side of the zero line with more room.
        zoom = 2e-3
        offsets = np.concatenate((np.linspace(zoom, 1e-4, 200), np.logspace(-4, -12, 200)))
        th_in = theta_c + side * offsets
        a_in, b_in = a_c + A * np.cos(th_in), b_c + B * np.sin(th_in)
        ta, tb = -A * np.sin(th_in), B * np.cos(th_in)
        ta, tb = ta / np.hypot(ta, tb), tb / np.hypot(ta, tb)
        T = ground_state_third_derivatives(a_in, b_in)
        d3_in = ta**3 * T[0] + 3 * ta**2 * tb * T[1] + 3 * ta * tb**2 * T[2] + tb**3 * T[3]
        lo, hi = axdddE.get_ylim()
        zero_at = -lo / (hi - lo)  # height of the zero line in axes fraction
        # With a peak inset too, that one takes this place (next to its box) and this one goes to the top right.
        if peak_inset is not None:
            inset = axdddE.inset_axes(FAR_INSET_BOUNDS)
        else:
            inset = axdddE.inset_axes([0.31, 0.42 if zero_at < 0.5 else 0.22, 0.27, 0.45])
        inset.plot(side * offsets, d3_in, color=color, ls=ls, lw=1.2)
        inset.plot([0, -side * zoom / 4], [0, 0], color=color, ls=ls, lw=1.2)  # P0 side: 0
        inset.axvline(0, **{**TRANSITION_STYLE["continuous"], "lw": 0.8})
        inset.axhline(0, color="gray", lw=0.5, zorder=1)
        y_top = 1.3 * d3_in[np.argmin(np.abs(offsets - 1e-4))]  # the curve leaves the top just inside 1e-4
        x_lim = sorted((side * zoom, -side * zoom / 4))  # theta - theta_c
        inset.set_xlim(x_lim)
        inset.set_ylim(-0.08 * y_top, y_top)
        inset.set_xticks([])
        inset.set_yticks([])
        for spine in inset.spines.values():
            spine.set_linewidth(1.2)
        shade_inset(inset, x_lim, theta_c)  # background coloured and hatched as the main panel
        inset.set_xlim(x_lim)
        inset.set_ylim(-0.08 * y_top, y_top)
        # Box on the main panel marking the zoomed region: far too narrow to see at its true width (2.5e-3), so it is
        # given a minimum width, and it is cut off at the top of the panel.
        box_width = max(np.diff(x_lim)[0], 0.035 * 2 * np.pi)
        box_lo, box_hi = max(-0.08 * y_top, lo), min(y_top, hi - 0.02 * (hi - lo))
        rect = (theta_c + np.mean(x_lim) - box_width / 2, box_lo, box_width, box_hi - box_lo)
        indicator = axdddE.indicate_inset(rect, inset_ax=inset, edgecolor="k", lw=0.8, alpha=1)
        for connector in indicator.connectors:
            connector.set(lw=0.6, color="0.2")
            # With a peak inset too, no connectors (too cluttered): the boxes, the backgrounds and the border styles
            # say which inset is which.
            if peak_inset is not None:
                connector.set_visible(False)
    if peak_inset is not None:
        # Inset: the crossover peak of the third derivative, which is narrow on the scale of the panel but finite and
        # smooth (here on a grid 10 times finer). It sits just right of its box, low enough for the connector of the
        # other inset to pass above it.
        th_pk = np.linspace(*peak_inset, 1001)
        a_pk, b_pk = a_c + A * np.cos(th_pk), b_c + B * np.sin(th_pk)
        ta, tb = -A * np.sin(th_pk), B * np.cos(th_pk)
        ta, tb = ta / np.hypot(ta, tb), tb / np.hypot(ta, tb)
        T = ground_state_third_derivatives(a_pk, b_pk)
        d3_pk = ta**3 * T[0] + 3 * ta**2 * tb * T[1] + 3 * ta * tb**2 * T[2] + tb**3 * T[3]
        inset = axdddE.inset_axes([0.31, 0.25, 0.27, 0.45])
        inset.plot(th_pk, d3_pk, color=color, ls=ls, lw=1.2)
        inset.axhline(0, color="gray", lw=0.5, zorder=1)
        pk_lo, pk_hi = min(d3_pk.min(), 0), d3_pk.max()
        pk_pad = 0.08 * (pk_hi - pk_lo)
        shade_inset(inset, peak_inset, 0)
        inset.set_xlim(peak_inset)
        inset.set_ylim(pk_lo - pk_pad, pk_hi + pk_pad)
        inset.set_xticks([])
        inset.set_yticks([])
        for spine in inset.spines.values():  # dotted, as its box, to tell it from the other inset
            spine.set(linewidth=1.2, linestyle=PEAK_INSET_LS, capstyle="round")  # round caps: dots, not dashes
        lo, hi = axdddE.get_ylim()
        box_lo, box_hi = max(pk_lo - pk_pad, lo), min(pk_hi + pk_pad, hi - 0.02 * (hi - lo))  # inside the panel
        rect = (peak_inset[0], box_lo, np.diff(peak_inset)[0], box_hi - box_lo)
        indicator = axdddE.indicate_inset(rect, inset_ax=inset, edgecolor="k", lw=0.8, ls=PEAK_INSET_LS, alpha=1)
        indicator.rectangle.set_capstyle("round")
        for connector in indicator.connectors:
            connector.set_visible(False)
    for ax, _, _ in panels:  # phase colours, now the limits are set
        shade_along_contour(ax, theta, contour_colors)

    # Key to the lines and shading, above the panels: the lines on one row, the phase colours and hatches below
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.79 / GS_HEIGHT_IN))  # the legends take the top 0.79 in
    line_handles = [
        Line2D([], [], color=color, ls=ls, lw=1.5, label="Contour"),
        Line2D([], [], **{**TRANSITION_STYLE["first"], "lw": 1.5}, label="First-order transition"),
        Line2D([], [], **{**TRANSITION_STYLE["continuous"], "lw": 1.5}, label="Continuous transition"),
    ]
    fig.legend(handles=line_handles, loc="upper center", fontsize=9, frameon=False, ncol=3, handlelength=2.5,
               columnspacing=1.5, bbox_to_anchor=(0.5, 1.0))
    fig.legend(handles=phase_legend_handles(), loc="upper right", fontsize=9, frameon=False, ncol=3,
               columnspacing=1.0, title="Quantum phase", title_fontsize=9, bbox_to_anchor=(0.42, 1 - 0.28 / GS_HEIGHT_IN))
    fig.legend(handles=hatch_legend_handles(), loc="upper left", fontsize=9, frameon=False, ncol=4,
               columnspacing=1.0, handleheight=1.4, title="Number of nontrivial extrema", title_fontsize=9,
               bbox_to_anchor=(0.5, 1 - 0.28 / GS_HEIGHT_IN))
    style.savefig(fig, filename)


# radius 0.6: the contour crosses the Widom line at b = 0.538 (just below where it ends, b = 9/16), far enough from the
# critical point that the crossover is visibly smooth
plot_ground_state_along(0.0, 0.0, 0.6, 0.6, r"$a^2 + b^2 = 0.6^2$", "1d_example_running_E0_circle.pdf",
                        CONTOUR_COLOR, "-", peak_inset=(0.85, 1.3))
