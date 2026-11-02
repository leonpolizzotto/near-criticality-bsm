"""Fig. 4: the phase diagram of the quartic toy model, and its ground state energy and first two derivatives in mu2.

Writes output/1d_example_phase_diagram_energy.pdf."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the nearcrit package

import numpy as np
import matplotlib.pyplot as plt

from nearcrit import style
from nearcrit.style import LINEWIDTH_IN, set_xlabel_right
from nearcrit.toy_quartic import E0, dE0_dmu2, d2E0_dmu22, mark_critical_point, plot_phase_diagram

style.use()

# Ground state energy and its derivatives as a function of mu2 (lambda > 0).
# The functions are not defined at mu2 = 0, so each side is evaluated separately.
lambda_ = 1.0
eps = 1e-9
mu2_neg = np.linspace(-2.0, -eps, 400)
mu2_pos = np.linspace(eps, 2.0, 400)

# (function, y label, y limits) for E0 and its first two derivatives with respect to mu2.
energy_curves = [
    (E0, r"$E_0$", (-1.1, 0.15)),
    (dE0_dmu2, r"$\displaystyle\frac{\partial E_0}{\partial \mu^2}$", (-0.1, 1.1)),
    (d2E0_dmu22, r"$\displaystyle\frac{\partial^2 E_0}{\partial(\mu^2)^2}$", (-0.7, 0.2)),
]


def plot_energy_panel(ax, k):
    """Plot the k-th entry of energy_curves against mu2 on ax; the title says the energy is at lambda > 0."""
    f, label, ylim = energy_curves[k]
    for mu2 in (mu2_neg, mu2_pos):
        ax.plot(mu2, [f(m, lambda_) for m in mu2], color="C0", lw=1.5)
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_ylabel(label, rotation=0, ha="right", va="center")
    set_xlabel_right(ax, r"$\mu^2$")
    ax.set_title(r"$\lambda>0$")  # the parameter the energy is at goes in the title, not the axis label
    ax.set_xlim(mu2_neg[0], mu2_pos[-1])
    ax.set_ylim(*ylim)
    mark_critical_point(ax)
    ax.set_yticks([])
    if f is d2E0_dmu22:
        # Mark the jump in the second derivative at the transition.
        for mu2 in (-eps, eps):
            ax.plot(0, f(mu2, lambda_), "o", mfc="white", color="C3", ms=6, zorder=3)


# Phase diagram next to the ground state energy and its derivatives.
fig, axes = plt.subplots(2, 2, figsize=(LINEWIDTH_IN, 4.9))
plot_phase_diagram(axes[0, 0], label_bbox=True)
for k, ax in enumerate(axes.flat[1:]):
    plot_energy_panel(ax, k)
fig.tight_layout()
style.savefig(fig, "1d_example_phase_diagram_energy.pdf")
