# A framework for near-criticality in BSM physics: figures

Code to reproduce the figures of the paper *A framework for near-criticality in BSM physics*
([arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX)).

The scripts in `figures/` draw every figure of the paper, one script per figure or per small group of closely related
figures. The slow computations (the SM running on a grid, and the one-loop phase maps of the Higgs-singlet model)
are done by the scripts in `compute/`, whose results are stored in `data/` and committed, so the figures can be
remade in a few minutes without them.

## Installation

The code needs Python 3 (3.12 was used) and a LaTeX installation, because the figures are typeset with LaTeX
(matplotlib's `text.usetex`): `latex` and `dvipng`, with the `type1cm`, `type1ec` (cm-super) and `amsmath` packages,
all included in a full TeX Live.

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the exact versions the figures were made and verified with.

## Reproducing the figures

```
python make_figures.py              # all figures, from data/ (about 4 minutes)
python make_figures.py 7 25-29      # only some figures, by their number in the paper
python make_figures.py --compute    # recompute data/ first (slow, see below), then all figures
python make_figures.py --tests      # run the tests (about a minute)
```

The figures are written to `output/`, under the file names used in the paper's source. Every script can also be run
on its own, from any directory, e.g. `python figures/fig07_sm_phase_diagram.py`.

| Figure | Script | Output (in `output/`) | Data (in `data/`) |
|---|---|---|---|
| 1 | `figures/fig01_quartic_phases.py` | `1d_example.pdf` | – |
| 2 | `figures/fig02_tilted_potential.py` | `1d_example_tilted.pdf` | – |
| 3 | `figures/fig03_tilted_energy.py` | `1d_example_tilted_energy.pdf` | – |
| 4 | `figures/fig04_phase_diagram_energy.py` | `1d_example_phase_diagram_energy.pdf` | – |
| 5 | `figures/fig05_rg_flow.py` | `1d_example_phase_diagram_flow_t.pdf` | – |
| 6 | `figures/fig06_critical_surface.py` | `gamma_surface_critical_coordinates.pdf` | – |
| 7 | `figures/fig07_sm_phase_diagram.py` | `sm_weak_phase_diagram_decades.pdf` | `sm_weak_grid.npz` |
| 8, 9 | `figures/fig08_09_sextic_levels.py` | `1d_example_6_energy_levels_delta.pdf`, `1d_example_6_energy_levels_mu2.pdf` | – |
| 10, 11 | `figures/fig10_11_potential_phases.py` | `1d_example_potential_phases.pdf`, `1d_example_potential_phase_diagram.pdf` | – |
| 12 | `figures/fig12_single_log_extrema.py` | `1d_example_running_1.pdf` | – |
| 13 | `figures/fig13_single_log_ground_state.py` | `1d_example_running_1_E0.pdf` | – |
| 14 | `figures/fig14_single_log_gallery.py` | `1d_example_running_1_gallery.pdf` | – |
| 15 | `figures/fig15_H_and_G.py` | `1d_example_running_HG_stacked.pdf` | – |
| 16 | `figures/fig16_regions.py` | `1d_example_running_regions.pdf` | – |
| 17, 19 | `figures/fig17_19_crossover.py` | `1d_example_running_crossover.pdf`, `1d_example_running_crossover_potentials.pdf` | – |
| 18 | `figures/fig18_circle_contour.py` | `1d_example_running_E0_circle.pdf` | – |
| 20 | `figures/fig20_two_log_gallery.py` | `1d_example_running_gallery.pdf` | – |
| 21, 22 | `figures/fig21_22_tree_atlases.py` | `tree_phase_atlas_Omega.pdf`, `tree_phase_atlas_mu.pdf` | – |
| 23 | `figures/fig23_tree_gallery.py` | `2D_gallery_potentials.pdf` | – |
| 24 | `figures/fig24_tree_contour.py` | `tree_contour_Omega.pdf` | – |
| 25 | `figures/fig25_29_loop_phase_grids.py` | `loop_phase_grid_sm_rg.pdf` | `loop_grid_rg.npz` |
| 26 | `figures/fig25_29_loop_phase_grids.py` | `loop_phase_corner_sm_rg.pdf` | `loop_corner.npz` |
| 27 | `figures/fig25_29_loop_phase_grids.py` | `loop_phase_grid_sm_rg_kneg.pdf` | `loop_grid_rg_kneg.npz` |
| 28 | `figures/fig25_29_loop_phase_grids.py` | `loop_phase_grid_sm_rg_kneg_mSneg.pdf` | `loop_grid_rg_kneg_mSneg.npz` |
| 29 | `figures/fig25_29_loop_phase_grids.py` | `loop_phase_grid_sm_rg_mSneg.pdf` | `loop_grid_rg_mSneg.npz` |
| 30 | `figures/fig30_loop_zooms.py` | `loop_phase_zoom_outer_sm_rg.pdf` | `loop_zoom.npz`, `loop_zoom_signs.npz` |
| 31, 32 | `figures/fig31_32_loop_galleries.py` | `loop_gallery_gs_sm_rg.pdf`, `loop_gallery_ngs_sm_rg.pdf` | `loop_gallery_points.npz` |
| 33 | `figures/fig33_36_loop_contours.py` | `loop_contour_r1_k05_sm_rg.pdf` | `loop_contour_r1_k05.npz`, `loop_grid_rg.npz` |
| 34 | `figures/fig33_36_loop_contours.py` | `loop_contour_r2_k05_zoom_sm_rg.pdf` | `loop_contour_r2_k05_zoom.npz`, `loop_zoom.npz` |
| 35 | `figures/fig33_36_loop_contours.py` | `loop_contour_rm1_km1_sm_rg.pdf` | `loop_contour_rm1_km1.npz`, `loop_grid_rg_kneg_mSneg.npz` |
| 36 | `figures/fig33_36_loop_contours.py` | `loop_contour_rm05_km05_zoom_sm_rg.pdf` | `loop_contour_rm05_km05_zoom.npz`, `loop_zoom_signs.npz` |

## Recomputing the data

Only Fig. 7 and Figs. 25-36 need precomputed data. Each compute script writes its files to `data/`:

| Script | Writes | For figures | Time (one core) |
|---|---|---|---|
| `compute/sm_weak_grid.py` | `sm_weak_grid.npz`: the SM phase on two grids of (u, y_0), from the 3-loop running up to M_Pl | 7 | 2.5 min |
| `compute/loop_phase_maps.py` | `loop_grid_rg*.npz`, `loop_corner.npz`, `loop_zoom*.npz`: the one-loop phase maps | 25-36 | ~8.5 h |
| `compute/loop_gallery_points.py` | `loop_gallery_points.npz`: the parameter points of the gallery panels of phases d, e, g, h | 31, 32 | < 1 s |
| `compute/loop_contours.py` | `loop_contour_*.npz`: the ground state and its derivatives along the four contours | 33-36 | ~1.5 min per contour |

`compute/loop_gallery_points.py` reads the zoom maps, so it runs after `compute/loop_phase_maps.py`
(`make_figures.py --compute` runs them in this order). The phase maps are the slow step: each 160 x 160 panel takes
about 5 minutes on one core. `compute/loop_phase_maps.py` saves every panel as soon as it is computed and resumes from
the panels already in `data/` with the same settings, so it can be interrupted; delete a file to recompute it. Its
`--only grid|corner|zoom` option computes one kind of map, so several can run in parallel.

Measured times for `compute/loop_phase_maps.py`: the four grids 0.7, 0.75, 1.1 and 3.1 h, the corner 1.3 h, the two
sets of zoom windows 0.3 and 0.8 h (each on one core, with the jobs sharing one machine). See Verification below for
how the recomputed data compare with the committed data.

## Layout

```
make_figures.py     entry point
figures/            one plotting script per figure (or group of figures); they read data/ and write output/
compute/            the slow computations; they write data/
data/               their results, as compressed .npz files (nearcrit/data_io.py reads them; no pickle)
nearcrit/           code shared by several scripts:
  style.py              figure size and fonts (the paper's text width, 11 pt LaTeX), axis-label helpers, paths
  toy_quartic.py        the quartic toy model (Figs. 1-5, 10, 11)
  toy_running.py        the running-coupling toy potentials (Figs. 12-20)
  sm_rge.py             SM running and vacuum-stability classification, after arXiv:1307.3536 (Fig. 7)
  singlet_tree.py       the tree-level Higgs-singlet potential, its extrema and phases (Figs. 21-24)
  singlet_tree_plot.py  drawing of Higgs-singlet phase maps (Figs. 21-36)
  singlet_loop.py       the one-loop (RG-improved Coleman-Weinberg) Higgs-singlet potential and its extrema
  loop_boundaries.py    phase boundaries solved from the conditions that define them
  loop_maps.py          the one-loop phase maps: computation, labelling, drawing (Figs. 25-36)
  data_io.py            saving and loading the data files
tests/              checks of the physics code (python make_figures.py --tests)
```

## Tests

`tests/` checks the physics code:
- `test_sm_rge.py`: the SM running against arXiv:1307.3536 (couplings at M_Pl, eqs. (61a)-(61e); the scales of
  sec. 4.2; the stability bound, eq. (64)).
- `test_singlet_loop.py`: the one-loop potential. Its derivatives agree with finite differences; with the loop
  switched off it reproduces the tree-level phase map; the RG improvement is consistent with the one-loop beta
  functions.
- `test_loop_boundaries.py`: the solved phase boundaries agree with the phase map just either side of them.
- `test_tree_gallery.py`: every panel of Fig. 23 is in the potential phase it is labelled with.

Run them with `python make_figures.py --tests`, one at a time with `python tests/<file>.py`, or with pytest.

## Verification

Every figure made by this repository from the committed `data/` was compared with the version in the paper, pixel
by pixel at 300 dpi, and they are identical.

All of `data/` was also recomputed from scratch with the scripts in `compute/`:
- `sm_weak_grid.npz`, the gallery points and the four contours are bit-identical to the committed files.
- The phase maps agree to a relative 4e-12 (rounding in the solved boundaries), with one exception: the committed
  `loop_zoom.npz` has no inner cuts (the lines just inside the NGS edge) for the panel kappa = 1, r = 2, while a fresh
  computation finds them. The committed file was made before the last change to the inner cuts, and
  `compute/loop_phase_maps.py` only adds them where they are missing.
- From the recomputed data, every figure is identical to the paper's except Fig. 30, whose panel (a) gains a
  potential-transition line along the edge of the NGS region.

## Citation

If you use this code, please cite the paper:

```
@article{nearcriticality,
    title   = "{A framework for near-criticality in BSM physics}",
    author  = "[AUTHORS]",
    journal = "[JOURNAL]",
    eprint  = "XXXX.XXXXX",
    archivePrefix = "arXiv",
    year    = "[YEAR]"
}
```

## License

[LICENSE TO BE CHOSEN], see `LICENSE`.
