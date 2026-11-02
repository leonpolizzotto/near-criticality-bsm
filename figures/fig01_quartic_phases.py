"""Fig. 1: the quantum phases of the quartic toy model, and the transitions between them.

Writes output/1d_example.pdf."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the nearcrit package

import numpy as np
import matplotlib.pyplot as plt

from nearcrit import style
from nearcrit.style import LINEWIDTH_IN
from nearcrit.toy_quartic import V1D_example, plot_minima

style.use()
# Quantum phases (the ones with a ground state) and the transitions between them. Each panel is
# (title, [(mu2, lambda, label), ...]); several curves in one panel are distinguished by label.
quantum_phases = [
    [
        ("Quantum Phase I\n" r"$\mu^2>0,\ \lambda>0$", [(1.0, 1.0, None)]),
        (r"Transition I$\to$II" "\n" r"$\mu^2=0,\ \lambda>0$", [(0.0, 1.0, None)]),
        ("Quantum Phase II\n" r"$\mu^2<0,\ \lambda>0$", [(-1.0, 1.0, None)]),
    ],
    [
        (r"Transition I$\to$NGS" "\n" r"$\mu^2>0,\ \lambda=0$", [(1.0, 0.0, None)]),
        (
            "No Ground State (NGS)\n" r"$\lambda<0$",
            [
                (-1.0, -1.0, r"$\mu^2<0$"),
                (0.0, -1.0, r"$\mu^2=0$"),
                (1.0, -1.0, r"$\mu^2>0$"),
            ],
        ),
        (r"Transition II$\to$NGS" "\n" r"$\mu^2<0,\ \lambda=0$", [(-1.0, 0.0, None)]),
    ],
]

phi = np.linspace(-2.0, 2.0, 2001)

# The y limits are shared within a row; the second row is centered on V=0 since it contains both
# upward and downward opening potentials.
row_ylims = [(-0.6, 1.6), (-2.6, 2.6)]
fig, axes = plt.subplots(2, 3, figsize=(LINEWIDTH_IN, 4.4), sharex=True, sharey="row")

for i, row in enumerate(quantum_phases):
    for ax, (title, curves) in zip(axes[i], row):
        for j, (mu2, lambda_, label) in enumerate(curves):
            V = V1D_example(phi, mu2, lambda_)
            ax.plot(phi, V, color=f"C{j}", lw=1.5, label=label)
            # With several curves in a panel, colour the circle like its curve.
            plot_minima(ax, phi, V, lambda_, color="C3" if len(curves) == 1 else f"C{j}")
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        if len(curves) > 1:
            ax.legend(loc="upper left", frameon=False, fontsize=9, handlelength=1.2, labelspacing=0.2)
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
style.savefig(fig, "1d_example.pdf")
