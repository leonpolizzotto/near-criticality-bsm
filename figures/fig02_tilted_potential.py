"""Fig. 2: the tilted quartic potential (with a phi^3 term) across the first-order transition at delta = 0.

Writes output/1d_example_tilted.pdf."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the nearcrit package

import numpy as np
import matplotlib.pyplot as plt

from nearcrit import style
from nearcrit.style import LINEWIDTH_IN
from nearcrit.toy_quartic import V1D_example_tilted, local_minima

style.use()

# Tilted potential (phi^3 term) at mu2 < 0, lambda > 0, where the broken phase always exists.
# Phase I (delta > 0) has its global minimum at phi < 0, phase II (delta < 0) at phi > 0, and
# at delta = 0 the two minima are degenerate.
mu2 = -1.0
lambda_ = 1.0
delta0 = 0.5

tilted = [
    (delta0, "Phase I\n" r"$\mu^2<0,\ \lambda>0,\ \delta>0$"),
    (0.0, "First Order Phase Transition\n" r"$\mu^2<0,\ \lambda>0,\ \delta=0$"),
    (-delta0, "Phase II\n" r"$\mu^2<0,\ \lambda>0,\ \delta<0$"),
]

phi = np.linspace(-2.0, 2.0, 2001)


def plot_tilted_potential_panel(ax, k):
    """Plot the tilted potential V(phi) for the k-th entry of tilted on ax."""
    delta, title = tilted[k]
    V = V1D_example_tilted(phi, mu2, lambda_, delta)
    ax.plot(phi, V, color="C0", lw=1.5)
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    pm, Vm = local_minima(phi, V)
    # Filled markers for ground states (degenerate at delta = 0), open for metastable minima.
    is_ground = np.isclose(Vm, Vm.min(), atol=1e-6)
    ax.plot(pm[is_ground], Vm[is_ground], "o", color="C3", ms=6, zorder=3)
    ax.plot(pm[~is_ground], Vm[~is_ground], "o", mfc="white", color="C3", ms=6, zorder=3)
    ax.set_title(title)
    ax.set_xlabel(r"$\phi$")
    ax.set_xlim(phi[0], phi[-1])
    ax.set_ylim(-0.8, 1.6)
    ax.set_xticks([])
    ax.set_yticks([])


fig, axes = plt.subplots(1, 3, figsize=(LINEWIDTH_IN, 2.3), sharex=True, sharey=True)
for k, ax in enumerate(axes):
    plot_tilted_potential_panel(ax, k)
axes[0].set_ylabel(r"$V(\phi)$")
fig.tight_layout()
style.savefig(fig, "1d_example_tilted.pdf")
