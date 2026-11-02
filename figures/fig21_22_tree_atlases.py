"""Figs. 21 and 22: tree-level phase atlases of the Higgs-singlet model.

Fig. 21: the (Omega_H, Omega_S) plane, one panel per sign of (mu_H^2, mu_S^2). In this section lambda_H, lambda_S > 0,
so Omega_H and Omega_S have the same sign: the two quadrants where they do not are left white. Within the other two,
the region where the potential is unbounded below (Omega_H Omega_S < 1 with kappa < 0) is No Ground State (NGS), gray.
Fig. 22: the (mu_H^2, mu_S^2) plane, one panel per sign of kappa, at lambda_H = 0.3, lambda_S = 0.7, |kappa| = 0.5.
The boundaries are rays from the origin whose angles depend on these values.

Writes output/tree_phase_atlas_Omega.pdf and output/tree_phase_atlas_mu.pdf.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the nearcrit package

import numpy as np
import matplotlib.pyplot as plt

from nearcrit import style
from nearcrit.singlet_tree import is_bounded_below_lambda, tree_potential_phase_map_lambda, tree_potential_phase_map_Omega
from nearcrit.singlet_tree_plot import HATCH_RC, NGS_COLOR, add_legends, draw_phase_map
from nearcrit.style import LINEWIDTH_IN, region_color, set_ticks, set_xlabel_right, set_ylabel_top

style.use()


def plot_Omega_panel(ax, sH, sS, omega_max=2.0, ngrid=800):
    """Phases in the (Omega_H, Omega_S) plane for sign(mu_H^2) = sH and sign(mu_S^2) = sS.

    The phase map depends on the mus only through their signs, so this is the exact map. Keep ngrid even:
    with an odd one, Omega = 0 and Omega = 1 are grid lines, and the transitions along them are lost."""
    g = np.linspace(-omega_max, omega_max, ngrid)
    X, Y = np.meshgrid(g, g)
    # NGS underneath the two quadrants where Omega_H and Omega_S have the same sign, the quantum phases on top
    # of it. The other two quadrants need lambda_H lambda_S < 0, which is excluded here, and stay white.
    for corner in [(0, 0), (-omega_max, -omega_max)]:
        ax.add_patch(plt.Rectangle(corner, omega_max, omega_max, facecolor=region_color(NGS_COLOR), lw=0,
                                   zorder=0))
    draw_phase_map(ax, X, Y, tree_potential_phase_map_Omega(X, Y, sH, sS))
    ax.set_xlim(-omega_max, omega_max)
    ax.set_ylim(-omega_max, omega_max)
    ax.set_aspect("equal")
    ax.set_title(rf"$\mu_H^2 {'>' if sH > 0 else '<'} 0,\ \mu_S^2 {'>' if sS > 0 else '<'} 0$")
    # The axes, and Omega = 1, where the potential phases change.
    set_ticks(ax, [0, 1])


def plot_mu_panel(ax, lambdaH, lambdaS, kappa, mu_max=3.0, ngrid=800):
    """Phases in the (mu_H^2, mu_S^2) plane at fixed lambda_H, lambda_S, kappa (bounded below, so there is no
    NGS region). Keep ngrid even, so that mu^2 = 0 is not a grid line."""
    assert is_bounded_below_lambda(lambdaH, lambdaS, kappa)
    g = np.linspace(-mu_max, mu_max, ngrid)
    X, Y = np.meshgrid(g, g)
    draw_phase_map(ax, X, Y, tree_potential_phase_map_lambda(lambdaH, lambdaS, X, Y, kappa))
    ax.set_xlim(-mu_max, mu_max)
    ax.set_ylim(-mu_max, mu_max)
    ax.set_aspect("equal")
    ax.set_title(rf"$\kappa = {kappa:g},\ \lambda_H = {lambdaH:g},\ \lambda_S = {lambdaS:g}$")
    # The boundaries are rays from the origin, so 0 is the only critical value on the axes.
    set_ticks(ax, [0])


def finish_atlas(fig, axes, xlabel, ylabel, name, ngs=True):
    """Axis labels on the outer panels, the keys below the panels, layout, save. The figure height is chosen so
    that the panels, which are square, fill the text width with no spare height."""
    axes = np.atleast_2d(axes)
    for ax in axes[-1]:
        set_xlabel_right(ax, xlabel)
    for ax in axes[:, 0]:
        set_ylabel_top(ax, ylabel)
    h = fig.get_figheight()
    keys_top = add_legends(fig, y0=0.08 / h, ngs=ngs)
    fig.tight_layout(rect=(0, keys_top + 0.05 / h, 1, 1))
    style.savefig(fig, f"tree_phase_atlas_{name}.pdf")
    # Report the layout, to choose the figure height.
    r = fig.canvas.get_renderer()
    print(name, "keys height (in):", round((keys_top + 0.05 / h) * h, 2))
    print(name, "panel sizes (in):", [(round(ax.get_window_extent(r).width / fig.dpi, 2),
                                       round(ax.get_window_extent(r).height / fig.dpi, 2)) for ax in axes.flat])


if __name__ == "__main__":
    with plt.rc_context(HATCH_RC):
        fig, axes = plt.subplots(2, 2, figsize=(LINEWIDTH_IN, 7.3), sharex=True, sharey=True)
        # Laid out like the original figure: (+,+) (-,+) / (+,-) (-,-).
        for (sH, sS), ax in zip([(1, 1), (-1, 1), (1, -1), (-1, -1)], axes.flat):
            plot_Omega_panel(ax, sH, sS)
        finish_atlas(fig, axes, r"$\Omega_H$", r"$\Omega_S$", "Omega")

        # The values of the original figure (tree_quantum_phases.ipynb).
        fig, axes = plt.subplots(1, 2, figsize=(LINEWIDTH_IN, 4.2), sharex=True, sharey=True)
        for kappa, ax in zip([0.5, -0.5], axes):
            plot_mu_panel(ax, lambdaH=0.3, lambdaS=0.7, kappa=kappa)
        finish_atlas(fig, axes, r"$\mu_H^2$", r"$\mu_S^2$", "mu", ngs=False)
