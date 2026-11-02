"""Make the figures of "A framework for near-criticality in BSM physics".

    python make_figures.py                 all figures, from the data in data/ (a few minutes)
    python make_figures.py 7 25-29         only these figures (by number in the paper)
    python make_figures.py --compute       first recompute all of data/ (about 9 hours; see README.md), then all figures
    python make_figures.py --compute 30    first recompute only the data of Fig. 30, then Fig. 30
    python make_figures.py --tests         run the tests in tests/
    python make_figures.py --no-tex        draw the text without LaTeX (also automatic if LaTeX is not installed)

Each script runs in its own process, exactly as when it is run by hand, and writes its PDFs to output/.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Each data file (in data/) and the command that computes it, in the order they depend on each other (the gallery
# points come from the zoom maps).
COMPUTE = {
    "sm_weak_grid": ["compute/sm_weak_grid.py"],
    **{name: ["compute/loop_phase_maps.py", name]
       for name in ("loop_grid_rg", "loop_grid_rg_kneg", "loop_grid_rg_mSneg", "loop_grid_rg_kneg_mSneg",
                    "loop_corner", "loop_zoom", "loop_zoom_signs")},
    "loop_gallery_points": ["compute/loop_gallery_points.py"],
    **{f"loop_contour_{name}": ["compute/loop_contours.py", name]
       for name in ("r1_k05", "r2_k05_zoom", "rm1_km1", "rm05_km05_zoom")},
}

# The data each figure needs, including what that data is computed from (figures not listed need none).
DATA = {
    7: ["sm_weak_grid"],
    25: ["loop_grid_rg"],
    26: ["loop_corner"],
    27: ["loop_grid_rg_kneg"],
    28: ["loop_grid_rg_kneg_mSneg"],
    29: ["loop_grid_rg_mSneg"],
    30: ["loop_zoom", "loop_zoom_signs"],
    31: ["loop_zoom", "loop_zoom_signs", "loop_gallery_points"],
    32: ["loop_zoom", "loop_zoom_signs", "loop_gallery_points"],
    33: ["loop_grid_rg", "loop_contour_r1_k05"],
    34: ["loop_zoom", "loop_contour_r2_k05_zoom"],
    35: ["loop_grid_rg_kneg_mSneg", "loop_contour_rm1_km1"],
    36: ["loop_zoom_signs", "loop_contour_rm05_km05_zoom"],
}


# Each plotting script and the figures (numbered as in the paper) it makes.
FIGURES = {
    "fig01_quartic_phases.py": [1],
    "fig02_tilted_potential.py": [2],
    "fig03_tilted_energy.py": [3],
    "fig04_phase_diagram_energy.py": [4],
    "fig05_rg_flow.py": [5],
    "fig06_critical_surface.py": [6],
    "fig07_sm_phase_diagram.py": [7],
    "fig08_09_sextic_levels.py": [8, 9],
    "fig10_11_potential_phases.py": [10, 11],
    "fig12_single_log_extrema.py": [12],
    "fig13_single_log_ground_state.py": [13],
    "fig14_single_log_gallery.py": [14],
    "fig15_H_and_G.py": [15],
    "fig16_regions.py": [16],
    "fig17_19_crossover.py": [17, 19],
    "fig18_circle_contour.py": [18],
    "fig20_two_log_gallery.py": [20],
    "fig21_22_tree_atlases.py": [21, 22],
    "fig23_tree_gallery.py": [23],
    "fig24_tree_contour.py": [24],
    "fig25_29_loop_phase_grids.py": [25, 26, 27, 28, 29],
    "fig30_loop_zooms.py": [30],
    "fig31_32_loop_galleries.py": [31, 32],
    "fig33_36_loop_contours.py": [33, 34, 35, 36],
}


def parse_numbers(args):
    """Figure numbers from arguments like 7 or 25-29."""
    numbers = set()
    for a in args:
        lo, _, hi = a.partition("-")
        numbers |= set(range(int(lo), int(hi or lo) + 1))
    return numbers


def run(path, *args):
    """Run one script in its own process (stops if it fails)."""
    t0 = time.time()
    print(f"--- {' '.join([str(path.relative_to(ROOT)), *args])}", flush=True)
    subprocess.run([sys.executable, str(path), *args], check=True)
    print(f"    ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("figures", nargs="*", help="figure numbers or ranges (default: all)")
    parser.add_argument("--compute", action="store_true", help="recompute the data of these figures first (slow)")
    parser.add_argument("--tests", action="store_true", help="run the tests instead")
    parser.add_argument("--no-tex", action="store_true", help="draw the text without LaTeX")
    args = parser.parse_args()
    if args.tests:
        for path in sorted((ROOT / "tests").glob("test_*.py")):
            run(path)
        sys.exit()
    wanted = parse_numbers(args.figures)
    if args.compute:
        needed = {d for n in (wanted or DATA) for d in DATA.get(n, [])}
        for name, (script, *script_args) in COMPUTE.items():
            if name in needed:
                run(ROOT / script, *script_args)
    for script, numbers in FIGURES.items():
        if not wanted or wanted & set(numbers):
            run(ROOT / "figures" / script, *(["--no-tex"] if args.no_tex else []))
    print(f"Figures are in {ROOT / 'output'}")
