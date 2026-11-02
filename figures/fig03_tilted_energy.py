"""Fig. 3: the ground state energy of the tilted quartic potential and its first two derivatives in delta.

Writes output/1d_example_tilted_energy.pdf."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the nearcrit package

import numpy as np
import matplotlib.pyplot as plt

from nearcrit import style
from nearcrit.style import LINEWIDTH_IN, set_xlabel_right
from nearcrit.toy_quartic import E0_tilted, dE0_ddelta, d2E0_ddelta2, mark_critical_point

style.use()

# Ground state energy and its derivatives as a function of delta (mu2 < 0, lambda > 0).
mu2 = -1.0
lambda_ = 1.0
eps = 1e-9
# The first derivative jumps at delta = 0, so each side is evaluated separately.
delta_max = 1.5
delta_neg = np.linspace(-delta_max, -eps, 400)
delta_pos = np.linspace(eps, delta_max, 400)

tilted_energy_curves = [
    (E0_tilted, r"$E_0$"),
    (dE0_ddelta, r"$\displaystyle\frac{\partial E_0}{\partial\delta}$"),
    (d2E0_ddelta2, r"$\displaystyle\frac{\partial^2 E_0}{\partial\delta^2}$"),
]


def plot_tilted_energy_panel(ax, k):
    """Plot the k-th entry of tilted_energy_curves against delta on ax (at the fixed mu2, lambda)."""
    f, label = tilted_energy_curves[k]
    for delta in (delta_neg, delta_pos):
        ax.plot(delta, [f(mu2, lambda_, d) for d in delta], color="C0", lw=1.5)
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_ylabel(label, rotation=0, ha="right", va="center")
    set_xlabel_right(ax, r"$\delta$")
    ax.set_title(r"$\mu^2<0,\ \lambda>0$")  # the parameters the energy is at go in the title
    ax.set_xlim(delta_neg[0], delta_pos[-1])
    mark_critical_point(ax)
    ax.set_yticks([])
    if f is dE0_ddelta:
        # Mark the jump in the first derivative at the transition.
        for delta in (-eps, eps):
            ax.plot(0, f(mu2, lambda_, delta), "o", mfc="white", color="C3", ms=6, zorder=3)


fig, axes = plt.subplots(1, 3, figsize=(LINEWIDTH_IN, 2.3), sharex=True)
for k, ax in enumerate(axes):
    plot_tilted_energy_panel(ax, k)
fig.tight_layout()
style.savefig(fig, "1d_example_tilted_energy.pdf")
