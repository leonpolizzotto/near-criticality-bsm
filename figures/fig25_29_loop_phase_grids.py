"""Figs. 25-29: phase diagrams of the RG-improved one-loop Higgs-singlet potential in the tree-level (Omega_H, Omega_S)
plane, rows kappa = 0.25, 0.5, 1 and columns r = mu_S^2 / mu_H^2 = 0.5, 1, 2 (with the signs given below), each panel
over 0 < |Omega_H|, |Omega_S| < 2 in the quadrant sign(Omega) = sign(kappa) sign(mu_S^2) where lambda_H, lambda_S > 0.

Fig. 25: mu_S^2 > 0, kappa > 0.
Fig. 26: the same, zoomed in on the tree-level corner Omega_H = Omega_S = 1 (0.7 < Omega_H < 1.1,
         0.96 < Omega_S < 1.14), where the radiatively generated potential phase d appears. The window is not square in
         Omega, so the panels are stretched; the tick labels give the scale.
Fig. 27: mu_S^2 > 0, kappa < 0.
Fig. 28: mu_S^2 < 0, kappa < 0.
Fig. 29: mu_S^2 < 0, kappa > 0.

Reads data/loop_grid_rg*.npz and data/loop_corner.npz (made by compute/loop_phase_maps.py). Writes
output/loop_phase_grid_sm_rg{,_kneg,_kneg_mSneg,_mSneg}.pdf and output/loop_phase_corner_sm_rg.pdf.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the nearcrit package

import matplotlib.pyplot as plt

from nearcrit import data_io, style
from nearcrit.loop_maps import KAPPA_VALUES, R_VALUES, make_grid_figure

style.use()

MADE_BY = "compute/loop_phase_maps.py"
# (data file, sign of kappa, sign of mu_S^2, output, ticks); the quadrant with Omega < 0 is ticked at -1 and 0
GRIDS = [
    ("loop_grid_rg.npz", 1, 1, "loop_phase_grid_sm_rg.pdf", None),                            # Fig. 25
    ("loop_grid_rg_kneg.npz", -1, 1, "loop_phase_grid_sm_rg_kneg.pdf", ([-1, 0], [-1, 0])),     # Fig. 27
    ("loop_grid_rg_kneg_mSneg.npz", -1, -1, "loop_phase_grid_sm_rg_kneg_mSneg.pdf", None),     # Fig. 28
    ("loop_grid_rg_mSneg.npz", 1, -1, "loop_phase_grid_sm_rg_mSneg.pdf", ([-1, 0], [-1, 0])),   # Fig. 29
]
CORNER_TICKS = ([0.8, 1.0], [1.0, 1.1])                                                        # Fig. 26

if __name__ == "__main__":
    for name, kappa_sign, r_sign, output, ticks in GRIDS:
        panels = data_io.load(style.data_file(name, MADE_BY))["panels"]
        fig = make_grid_figure(panels, [kappa_sign * k for k in KAPPA_VALUES], [r_sign * r for r in R_VALUES], output,
                               ticks=ticks)
        plt.close(fig)
    panels = data_io.load(style.data_file("loop_corner.npz", MADE_BY))["panels"]
    make_grid_figure(panels, KAPPA_VALUES, R_VALUES, "loop_phase_corner_sm_rg.pdf", ticks=CORNER_TICKS)
