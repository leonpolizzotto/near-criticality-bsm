"""Fig. 30: zooms of the RG-improved one-loop phase maps onto the first-order jumps to a second, outer minimum at the
edge of NGS, too thin to see in the full grids. One row:
  (a) r = 2, kappa = 1 (Fig. 25): phase e, a second, radiatively generated H-axis minimum, to which the ground state
      jumps (I -> I', a first-order transition within quantum phase I) before V runs away below Lambda.
  (b) r = -2, kappa = 1 at the NGS edge (Fig. 29): phase g, a second H-axis minimum and the jump to it (I -> I'), the
      mirror of (a).
  (c) r = -0.5, kappa = -0.5 next to the NGS edge (Fig. 28): phase h, a second mixed minimum, to which the ground state
      jumps (III -> III', first order) before V runs away below Lambda.
The windows are not square in Omega, so the panels are stretched; the tick labels give the scale.

Reads data/loop_zoom.npz and data/loop_zoom_signs.npz (made by compute/loop_phase_maps.py). Writes
output/loop_phase_zoom_outer_sm_rg.pdf.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the nearcrit package

import matplotlib.pyplot as plt

from nearcrit import data_io, style
from nearcrit.loop_maps import (
    HATCH_RC, ZOOMS, ZOOMS_SIGNS, add_legends, draw_panel, label_panels, present_entries,
)
from nearcrit.style import LINEWIDTH_IN, set_xlabel_left, set_xlabel_right, set_ylabel_bottom, set_ylabel_top

style.use()

MADE_BY = "compute/loop_phase_maps.py"
def make_figure(panels, zooms, output, height_in, ylabel_all=False, w_pad=2.0, box_aspect=1.0):
    """One row of zoom windows, in the order of zooms, with the keys below. ylabel_all labels Omega_S on every panel
    (for panels with different scales), not only on the first. box_aspect is the panels' height / width."""
    labelled, new_letters = label_panels(panels)
    with plt.rc_context(HATCH_RC):
        fig, axes = plt.subplots(1, len(zooms), figsize=(LINEWIDTH_IN, height_in))
        for ax, (key, z), tag in zip(axes, zooms.items(), "abc"):
            kappa, r = key
            Q, P = labelled[key]
            draw_panel(ax, panels[key], Q, P, r, kappa, ticks=z["ticks"])
            ax.set_box_aspect(box_aspect)
            ax.set_title(rf"({tag})\ $r = {r:g},\ \kappa = {kappa:g}$")
            # labels at the far end of each axis from where Omega = 0 lies (as in loop_phase_grid.py)
            (h0, h1), (s0, s1) = z["window"]
            if h1 <= 0:
                set_xlabel_left(ax, r"$\Omega_H$")
            else:
                set_xlabel_right(ax, r"$\Omega_H$")
            if s1 <= 0:
                set_ylabel_bottom(ax, r"$\Omega_S$")
            elif ylabel_all or ax is axes[0]:
                set_ylabel_top(ax, r"$\Omega_S$")

        quantum, entries = present_entries(labelled)

        h = fig.get_figheight()
        keys_top = add_legends(fig, y0=0.06 / h, quantum_phases=quantum, potential_entries=entries)
        fig.tight_layout(rect=(0, keys_top + 0.05 / h, 1, 1), w_pad=w_pad)
        style.savefig(fig, output)
        rend = fig.canvas.get_renderer()
        print("keys height (in):", round((keys_top + 0.05 / h) * h, 2))
        print("panel sizes (in):", [(round(ax.get_window_extent(rend).width / fig.dpi, 2),
                                     round(ax.get_window_extent(rend).height / fig.dpi, 2)) for ax in axes])
    print("Radiatively generated potential phases:")
    for structure, letter in new_letters.items():
        print(f"  {letter}: origin {structure[0]}; H axis {list(structure[1])}; S axis {list(structure[2])}; "
              f"off axes {list(structure[3])}")
    return fig


if __name__ == "__main__":
    panels = data_io.load(style.data_file("loop_zoom.npz", MADE_BY))["panels"]
    signs = data_io.load(style.data_file("loop_zoom_signs.npz", MADE_BY))["panels"]
    outer = {(1.0, 2.0): ZOOMS[(1.0, 2.0)], (1.0, -2.0): ZOOMS_SIGNS[(1.0, -2.0)], (-0.5, -0.5): ZOOMS_SIGNS[(-0.5, -0.5)]}
    make_figure({**{k: v for k, v in panels.items() if k == (1.0, 2.0)}, **signs}, zooms=outer,
                output="loop_phase_zoom_outer_sm_rg.pdf", height_in=4.5, ylabel_all=True, w_pad=1.2,
                box_aspect=1.5)
