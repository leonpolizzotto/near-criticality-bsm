"""Data for Figs. 25-36: the RG-improved one-loop phase maps of the Higgs-singlet model (physics, conventions and
method in nearcrit/loop_maps.py).

  data/loop_grid_rg.npz             Fig. 25: rows kappa = 0.25, 0.5, 1, columns r = mu_S^2 / mu_H^2 = 0.5, 1, 2,
                                    each over 0 < Omega_H, Omega_S < 2 on a 160 x 160 grid, with finer insets
                                    (REFINE_RG) where a phase is narrower than a grid cell.
  data/loop_grid_rg_kneg.npz        Fig. 27: the same |kappa| and r with kappa < 0, in the quadrant
  data/loop_grid_rg_mSneg.npz       Fig. 29: ... with mu_S^2 < 0,     sign(Omega) = sign(kappa) sign(mu_S^2)
  data/loop_grid_rg_kneg_mSneg.npz  Fig. 28: ... with both,           where lambda_H, lambda_S > 0; these three
                                    get finer insets around every short piece of line (AUTO_REFINE).
  data/loop_corner.npz              Fig. 26: the 3 x 3 panels over 0.7 < Omega_H < 1.1, 0.96 < Omega_S < 1.14.
  data/loop_zoom.npz                the zoom windows ZOOMS (Figs. 30, 34 and the gallery points).
  data/loop_zoom_signs.npz          the zoom windows ZOOMS_SIGNS (Figs. 30, 36 and the gallery points).

Each file holds {"key": the settings, "panels": {(kappa, r): panel, (kappa, r, m): refined inset m}}. A panel is saved
as soon as it is computed, and a run picks up the panels already in a file with the same key, so an interrupted run
resumes where it stopped (delete a file to recompute it).

This is slow: a 160 x 160 panel takes ~5 min on one core, and all files together a few hours.

Run `python compute/loop_phase_maps.py [--only grid|corner|zoom]`. The process caps its own memory (--max-mem-gb,
default 16; Linux only) so that a runaway computation stops with a MemoryError instead of exhausting the machine.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the nearcrit package

import argparse

from nearcrit import data_io, style
from nearcrit import singlet_loop as L
from nearcrit.loop_maps import (
    AUTO_N, FULL_WINDOW, KAPPA_VALUES, LOOP_RG, MU, NGRID, NZOOM, OMEGA_MAX, R_VALUES, REFINE_RG, ZOOMS, ZOOMS_SIGNS,
    add_inner_cuts, add_outer_ground_states, auto_refine_insets, compute_panel, label_panels, refine_window,
)

CORNER_WINDOW = ((0.7, 1.1), (0.96, 1.14))
NCORNER = 160


def load(path, key):
    """The panels already computed in path with this key ({} if none)."""
    if path.exists():
        saved = data_io.load(path)
        if saved["key"] == key:
            return saved["panels"]
    return {}


def save(path, key, panels):
    """Save the panels computed so far, with their key."""
    style.DATA_DIR.mkdir(exist_ok=True)
    data_io.save(path, dict(key=key, panels=panels))


def finish(path, key, panels):
    """Add the inner cuts and the outer ground states that are missing, and save."""
    add_inner_cuts(panels, LOOP_RG)        # lines just inside the NGS edge
    add_outer_ground_states(panels, LOOP_RG)
    save(path, key, panels)


def grid(kappa_sign=1, r_sign=1):
    """One 3 x 3 grid (Figs. 25, 27-29): the same |kappa| and |r| with these signs, in the quadrant where lambda_H,
    lambda_S > 0, with the refined insets."""
    signs = ("_kneg" if kappa_sign < 0 else "") + ("_mSneg" if r_sign < 0 else "")
    path = style.DATA_DIR / f"loop_grid_rg{signs}.npz"
    kappa_values = [kappa_sign * k for k in KAPPA_VALUES]
    r_values = [r_sign * r for r in R_VALUES]
    window = ((-OMEGA_MAX, 0.0), (-OMEGA_MAX, 0.0)) if kappa_sign * r_sign < 0 else FULL_WINDOW
    key = dict(R=r_values, K=kappa_values, N=NGRID, O=OMEGA_MAX, MU=MU, region=L.SEARCH_REGION, boundaries="solved-v4",
               vacuum="Omega", loop={k: v for k, v in LOOP_RG.items() if k != "species"},
               species=[tuple(s) for s in LOOP_RG["species"]])
    panels = load(path, key)
    for k in kappa_values:
        for r in r_values:
            if (k, r) in panels:
                continue
            panels[(k, r)] = compute_panel(r, k, window=window, loop=LOOP_RG)
            save(path, key, panels)
    # Refined insets (REFINE_RG, then for the figures with other signs automatic tiles) are stored under (kappa, r, m)
    # with their own settings; others (from older settings) are dropped.
    specs = {}
    for (k, r), fines in REFINE_RG.items():
        if k in kappa_values and r in r_values:
            specs[(k, r)] = [dict(f) for f in fines]
    if signs:        # Fig. 25 keeps only its manual insets
        base = {kk: p for kk, p in panels.items() if len(kk) == 2}
        add_outer_ground_states(base, LOOP_RG)
        labelled, _ = label_panels(base)
        for kk, p in base.items():
            manual = [f["inset"] for f in specs.get(kk, [])]
            for inset in auto_refine_insets(p, *labelled[kk], exclude=manual):
                specs.setdefault(kk, []).append(dict(inset=inset, n=(AUTO_N, AUTO_N)))
    want = {(k, r, m): fine for (k, r), fines in specs.items() for m, fine in enumerate(fines)}
    for kk in [kk for kk in panels if len(kk) == 3]:
        if kk not in want or panels[kk].get("inset") != want[kk]["inset"] or panels[kk].get("n") != want[kk]["n"]:
            del panels[kk]
    todo = [kk for kk in want if kk not in panels]
    for count, (k, r, m) in enumerate(todo):
        fine = want[(k, r, m)]
        print(f"refined inset {count + 1} of {len(todo)}: kappa = {k}, r = {r}, {fine['inset']}", flush=True)
        panels[(k, r, m)] = compute_panel(r, k, window=refine_window(fine["inset"]), n=fine["n"], loop=LOOP_RG)
        panels[(k, r, m)]["inset"] = fine["inset"]
        save(path, key, panels)
    finish(path, key, panels)


def corner():
    """The 3 x 3 panels over CORNER_WINDOW, where the radiatively generated phase d appears (Fig. 26)."""
    path = style.DATA_DIR / "loop_corner.npz"
    key = dict(W=CORNER_WINDOW, N=NCORNER, R=R_VALUES, K=KAPPA_VALUES, MU=MU, region=L.SEARCH_REGION,
               boundaries="solved-v4", vacuum="Omega", loop={k: v for k, v in LOOP_RG.items()})
    panels = load(path, key)
    for kappa in KAPPA_VALUES:
        for r in R_VALUES:
            if (kappa, r) in panels:
                continue
            panels[(kappa, r)] = compute_panel(r, kappa, window=CORNER_WINDOW, n=NCORNER, loop=LOOP_RG)
            save(path, key, panels)
    finish(path, key, panels)


def zoom(zooms, name):
    """The zoom windows {(kappa, r): dict(window, ticks[, n])}, at NZOOM x NZOOM unless n is given."""
    path = style.DATA_DIR / name
    key = dict(Z={k: (v["window"], v["n"]) if "n" in v else v["window"] for k, v in zooms.items()}, N=NZOOM, MU=MU,
               region=L.SEARCH_REGION, boundaries="solved-v4", vacuum="Omega", loop={k: v for k, v in LOOP_RG.items()})
    panels = load(path, key)
    for (kappa, r), z in zooms.items():
        if (kappa, r) in panels:
            continue
        panels[(kappa, r)] = compute_panel(r, kappa, window=z["window"], n=z.get("n", NZOOM), loop=LOOP_RG)
        save(path, key, panels)
    finish(path, key, panels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--only", choices=("grid", "corner", "zoom"), help="only these maps")
    parser.add_argument("--max-mem-gb", type=float, default=16.0, help="memory cap for this process (Linux)")
    args = parser.parse_args()
    try:
        import resource
        cap = int(args.max_mem_gb * 2**30)
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    except (ImportError, ValueError, OSError):
        print("note: could not cap the memory of this process")
    if args.only in (None, "grid"):
        for kappa_sign, r_sign in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
            grid(kappa_sign, r_sign)
    if args.only in (None, "corner"):
        corner()
    if args.only in (None, "zoom"):
        zoom(ZOOMS, "loop_zoom.npz")
        zoom(ZOOMS_SIGNS, "loop_zoom_signs.npz")
