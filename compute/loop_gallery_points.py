"""Data for Figs. 31-32: the parameter points of the gallery panels of the radiatively generated phases d, e, g and h,
which are too thin to pick by hand. Each is the cell of a zoom map (data/loop_zoom.npz, data/loop_zoom_signs.npz)
farthest from the boundary of its phase.

Writes data/loop_gallery_points.npz (a few seconds).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the nearcrit package

import numpy as np
from scipy import ndimage

from nearcrit import data_io, style
from nearcrit.loop_maps import GALLERY_PANELS, GALLERY_ZOOMS, label_panels, omega_grid

OUTPUT = style.DATA_DIR / "loop_gallery_points.npz"


def find_points():
    """The points left as None in GALLERY_PANELS (phases d, e, g and h), from the zoom maps: the cell farthest from
    the phase's boundary. Returns {label: (Omega_H, Omega_S, distance to the boundary in cells)} and the
    specification {label: (kappa, r, structure, quantum phase)} they were found for."""
    todo = {label: v for label, v in GALLERY_PANELS.items() if v[2] is None}
    files = sorted({GALLERY_ZOOMS[label[-1]] for label in todo})
    found = {}
    for f in files:
        zoom = data_io.load(style.data_file(f, made_by="compute/loop_phase_maps.py"))["panels"]
        labelled, _ = label_panels(zoom)
        for label, (kappa, r, _, _, st, q) in todo.items():
            if GALLERY_ZOOMS[label[-1]] != f:
                continue
            p = zoom[(kappa, r)]
            Q = labelled[(kappa, r)][0]
            mask = np.array([[p["structure"][i, j] == st and Q[i, j] == q for j in range(p["n"])]
                             for i in range(p["n"])])
            dist = ndimage.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
            i, j = np.unravel_index(np.argmax(dist), dist.shape)
            OH, OS, _, _ = omega_grid(p["window"], p["n"])
            found[label] = (float(OH[i, j]), float(OS[i, j]), float(dist[i, j]))
    want = {label: (v[0], v[1], v[4], v[5]) for label, v in todo.items()}
    return found, want


if __name__ == "__main__":
    found, want = find_points()
    for label, (oh, os_, margin) in found.items():
        kappa, r = want[label][:2]
        print(f"{label}: kappa = {kappa}, r = {r}, Omega_H = {oh:.4f}, Omega_S = {os_:.4f} (margin {margin:.0f} cells)")
    data_io.save(OUTPUT, dict(points=found, want=want))
    print("Saved", OUTPUT)
