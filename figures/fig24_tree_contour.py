"""Fig. 24: the minimum of the tree-level Higgs-singlet potential along an elliptical contour in the
(Omega_H, Omega_S) plane, at fixed mu_H^2 = mu_S^2 = 2, kappa = 0.1.

  - Top left: the phase map of the (Omega_H, Omega_S) plane (as Fig. 21) with the contour, a dot at its start
    (theta = 0) and an arrow showing its direction.
  - The other panels, against theta along the contour: V_min, and its first and second derivatives projected on the
    unit tangent of the contour, t.grad V_min and t^T Hess(V_min) t (exact, from nearcrit/singlet_tree.py). As V_min is
    continuous, a first-order transition is a jump in the first and a second-order one a jump in only the second.
    The background is the quantum phase (colour) and the potential phase (hatch) at each point, with lines at the
    transitions as on the phase map.

Writes output/tree_contour_Omega.pdf.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the nearcrit package

import numpy as np
import matplotlib.pyplot as plt

from nearcrit import style
from nearcrit.singlet_tree import (
    PHASE_LABELS, QUANTUM_PHASE_LABELS, QUANTUM_PHASE_OF_POTENTIAL, TRANSITION_LINESTYLE, Vmin_Omega,
    _directional_deriv_fns, _transition_order, ddVmin_OmegaH, ddVmin_OmegaH_OmegaS, ddVmin_OmegaS, dVmin_OmegaH,
    dVmin_OmegaS, is_bounded_below_Omega, tree_potential_phase_map_Omega,
)
from nearcrit.singlet_tree_plot import (
    HATCH_COLOR, HATCH_RC, HATCHES, POTENTIAL_TRANSITION_STYLE, add_legends, draw_phase_map, quantum_color,
)
from nearcrit.style import LINEWIDTH_IN, set_ticks, set_xlabel_right, set_ylabel_top

style.use()

CONTOUR_COLOR = "C3"  # as in 1d_example_running.py
# Transition lines in the panels against theta, as in 1d_example_running.py (thinner than on the phase map).
THETA_TRANSITION_LW = 1.0
THETA_TICK_LABELS = {-1: r"$-\pi/2$", 0: r"$0$", 1: r"$\pi/2$", 2: r"$\pi$", 3: r"$3\pi/2$"}


def contour_data(muH2, muS2, kappa, OmegaH_lim, OmegaS_lim, theta_lim, n_theta=4000):
    """Everything along the ellipse inscribed in OmegaH_lim x OmegaS_lim, Omega = centre + semi-axis * (cos, sin)
    theta, for theta in theta_lim: the coordinates, the potential phase, the minimum P (which extremum is the global
    minimum), V_min and its two projected derivatives."""
    x0, a = np.mean(OmegaH_lim), np.diff(OmegaH_lim)[0] / 2
    y0, b = np.mean(OmegaS_lim), np.diff(OmegaS_lim)[0] / 2
    theta = np.linspace(*theta_lim, n_theta)
    OH, OS = x0 + a * np.cos(theta), y0 + b * np.sin(theta)
    assert is_bounded_below_Omega(OH, OS, muH2, muS2, kappa).all()
    V, P = Vmin_Omega(OH, OS, muH2, muS2, kappa, return_P=True)
    (d1, _), (d2, _) = _directional_deriv_fns((dVmin_OmegaH, dVmin_OmegaS),
                                              (ddVmin_OmegaH, ddVmin_OmegaH_OmegaS, ddVmin_OmegaS), (x0, y0, a, b))
    PH = tree_potential_phase_map_Omega(OH, OS, muH2, muS2)
    # A point exactly on a boundary belongs to no phase (code 0); give it the phase before it.
    for i in np.flatnonzero(PH == 0):
        PH[i] = PH[i - 1] if i > 0 else PH[np.flatnonzero(PH)[0]]
    return dict(theta=theta, OH=OH, OS=OS, PH=PH, P=P, V=V,
                dV=d1(P, OH, OS, muH2, muS2, kappa), ddV=d2(P, OH, OS, muH2, muS2, kappa))


def draw_contour(ax, c, color=CONTOUR_COLOR):
    """The contour on the phase map, with a dot at its start (theta = 0) and an arrow showing its direction, as
    draw_contour in 1d_example_running.py (here at theta = pi/4, clear of the boundaries through the start)."""
    ax.plot(c["OH"], c["OS"], color=color, lw=1.5, zorder=5)
    i0 = np.argmin(np.abs(c["theta"]))
    ax.plot(c["OH"][i0], c["OS"][i0], "o", color=color, ms=3.5, zorder=6)
    j = np.argmin(np.abs(c["theta"] - np.pi / 4))
    ax.annotate("", xy=(c["OH"][j], c["OS"][j]), xytext=(c["OH"][j - 1], c["OS"][j - 1]), zorder=6,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5, mutation_scale=10))


def decorate_along_contour(ax, c):
    """Background of a panel against theta: the quantum phase colour and the potential phase hatch at each point, a
    purple line where only the potential phase changes and a black one at a quantum phase transition (solid first
    order, dashed second order), a zero line, and the theta axis."""
    theta, PH = c["theta"], c["PH"]
    edges = np.concatenate(([theta[0]], 0.5 * (theta[1:] + theta[:-1]), [theta[-1]]))
    changes = np.flatnonzero(np.diff(PH)) + 1
    for start, end in zip(np.concatenate(([0], changes)), np.concatenate((changes, [len(theta)]))):
        p = PH[start]
        ax.axvspan(edges[start], edges[end], facecolor=quantum_color(QUANTUM_PHASE_LABELS[
            QUANTUM_PHASE_OF_POTENTIAL[p]]), lw=0, zorder=0)
        hatch = HATCHES.get(PHASE_LABELS[p][-1])
        if hatch:
            ax.axvspan(edges[start], edges[end], facecolor="none", edgecolor=HATCH_COLOR, hatch=hatch, lw=0,
                       zorder=0.5)
    for i in changes:
        p1, p2 = PH[i - 1], PH[i]
        if QUANTUM_PHASE_OF_POTENTIAL[p1] == QUANTUM_PHASE_OF_POTENTIAL[p2]:
            ax.axvline(edges[i], color=POTENTIAL_TRANSITION_STYLE["colors"],
                       lw=POTENTIAL_TRANSITION_STYLE["linewidths"], zorder=1)
        else:
            ax.axvline(edges[i], color="k", lw=THETA_TRANSITION_LW,
                       ls=TRANSITION_LINESTYLE[_transition_order(p1, p2)], zorder=2)
    ax.axhline(0, color="gray", lw=0.5, zorder=1)
    ax.set_xlim(theta[0], theta[-1])
    # ticks every pi/2 inside the panel, but none at its right end, where a label would collide with the axis label
    quarters = np.arange(np.ceil(theta[0] / (np.pi / 2)), np.ceil(theta[-1] / (np.pi / 2) - 0.1)).astype(int)
    ax.set_xticks(quarters * np.pi / 2, [THETA_TICK_LABELS[k] for k in quarters])
    ax.set_yticks([0], [r"$0$"])
    set_xlabel_right(ax, r"$\theta$")


def plot_along_contour(ax, c, values, color=CONTOUR_COLOR):
    """values against theta, not joined where the global minimum changes (the derivatives jump there)."""
    breaks = np.flatnonzero(np.diff(c["P"])) + 1
    ax.plot(np.insert(c["theta"], breaks, np.nan), np.insert(values, breaks, np.nan), color=color, lw=1.5, zorder=3)
    # limits: the curve, and 0 (the zero line), padded; more room beyond 0 when the curve is all on one side of it,
    # so that the 0 tick label stays clear of the y label at the top
    lo, hi = min(np.nanmin(values), 0), max(np.nanmax(values), 0)
    span = hi - lo
    ax.set_ylim(lo - (0.15 if lo == 0 else 0.08) * span, hi + (0.15 if hi == 0 else 0.08) * span)


def plot_contour_figure(muH2, muS2, kappa, OmegaH_lim, OmegaS_lim, plane_lim, theta_lim, filename,
                        height_in=6.1):
    c = contour_data(muH2, muS2, kappa, OmegaH_lim, OmegaS_lim, theta_lim)
    for i in np.flatnonzero(np.diff(c["PH"])) + 1:
        print(f"theta = {0.5 * (c['theta'][i - 1] + c['theta'][i]):.4f} at (OmegaH, OmegaS) = "
              f"({c['OH'][i]:.3f}, {c['OS'][i]:.3f}): {PHASE_LABELS[c['PH'][i - 1]]} -> {PHASE_LABELS[c['PH'][i]]}, "
              f"P{c['P'][i - 1]} -> P{c['P'][i]}")

    fig, axes = plt.subplots(2, 2, figsize=(LINEWIDTH_IN, height_in))
    axPD, axV, axdV, axddV = axes.flat

    # The phase map of the quadrant the contour is in (lambda_H, lambda_S > 0 fixes the signs of the Omegas).
    g = np.linspace(*plane_lim, 800)
    X, Y = np.meshgrid(g, g)
    PH = tree_potential_phase_map_Omega(X, Y, muH2, muS2)
    draw_phase_map(axPD, X, Y, PH)
    draw_contour(axPD, c)
    axPD.set_xlim(plane_lim)
    axPD.set_ylim(plane_lim)
    axPD.set_aspect("equal")
    set_ticks(axPD, [0, 1])
    set_xlabel_right(axPD, r"$\Omega_H$")
    set_ylabel_top(axPD, r"$\Omega_S$")
    axPD.set_title(rf"$\mu_H^2 = {muH2:g},\ \mu_S^2 = {muS2:g},\ \kappa = {kappa:g}$")

    panels = ((axV, c["V"], r"$V_{\min}$"), (axdV, c["dV"], r"$\hat t\cdot\nabla V_{\min}$"),
              (axddV, c["ddV"], r"$\hat t^{\mathsf{T}}\,\nabla^2 V_{\min}\,\hat t$"))
    for ax, values, label in panels:
        decorate_along_contour(ax, c)
        plot_along_contour(ax, c, values)
        if ax is axV:
            set_ylabel_top(ax, label)
        else:  # long labels: vertical, beside the axis, to save width
            ax.set_ylabel(label, labelpad=2)

    h = fig.get_figheight()
    # Only the quantum phases on the phase map (the contour stays inside it).
    present = {QUANTUM_PHASE_LABELS[QUANTUM_PHASE_OF_POTENTIAL[p]] for p in np.unique(PH[PH > 0])}
    keys_top = add_legends(fig, y0=0.08 / h, ngs=False,
                           quantum_phases=[q for q in ("0", "I", "II", "III") if q in present])
    fig.tight_layout(rect=(0, keys_top + 0.05 / h, 1, 1))
    style.savefig(fig, filename)


if __name__ == "__main__":
    with plt.rc_context(HATCH_RC):
        # The values of the original figure (tree_quantum_phases.ipynb). theta = 0 is on the boundary Omega_S = 1, so
        # the panels start a little before it, to show the transition there; they still cover the contour once.
        plot_contour_figure(muH2=2.0, muS2=2.0, kappa=0.1, OmegaH_lim=(0.25, 1.75), OmegaS_lim=(0.5, 1.5),
                            plane_lim=(0.0, 2.0), theta_lim=(-np.pi / 8, 2 * np.pi - np.pi / 8),
                            filename="tree_contour_Omega.pdf")
