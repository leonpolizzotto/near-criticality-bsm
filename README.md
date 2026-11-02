# A framework for near-criticality in BSM physics: figures

Code to reproduce the figures of the paper *A framework for near-criticality in BSM physics*
([arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX)).

The scripts in `figures/` draw every figure of the paper, one script per figure or per small group of closely related
figures. The slow computations (the SM running on a grid, and the one-loop phase maps of the Higgs-singlet model)
are done by the scripts in `compute/`, whose results are stored in `data/` and committed, so the figures can be
remade in a few minutes without them.

## Installation

The code needs Python 3 (3.12 was used). The paper's figures are typeset with LaTeX (matplotlib's `text.usetex`),
which needs `latex` and `dvipng`, with the `type1cm`, `type1ec` (cm-super) and `amsmath` packages, all included in a
full TeX Live. Without LaTeX the figures are still drawn: matplotlib then typesets the text itself, in the same
Computer Modern fonts, so they look slightly different from the paper's. This happens automatically if LaTeX or one
of its packages is missing, and can be asked for with `--no-tex`.

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the exact versions the figures were made and verified with.

## Reproducing the figures

```
python make_figures.py                 # all figures, from data/ (about 4 minutes)
python make_figures.py 7 25-29         # only some figures, by their number in the paper
python make_figures.py --compute 30    # recompute the data of Fig. 30 (about 1 hour, see below), then draw it
python make_figures.py --compute       # recompute all of data/ (about 9 hours), then draw all figures
python make_figures.py --tests         # run the tests (about a minute)
python make_figures.py --no-tex        # draw the text without LaTeX (any figure script also takes --no-tex)
```

Most figures take a few seconds each. Figs. 31-32 take about 15 s, because the gallery script evaluates the one-loop
potential at each panel's point. The first run takes longest, while LaTeX builds its cache of the labels.

The figures are written to `output/`, under the file names used in the paper's source. Every script can also be run
on its own, from any directory, e.g. `python figures/fig07_sm_phase_diagram.py`.

## Recomputing the data

Only Fig. 7 and Figs. 25-36 need precomputed data. `python make_figures.py --compute N` recomputes everything that
Fig. N needs (including the files that data is computed from) and then draws it. The compute scripts can also be run
directly, for one file at a time:

| Data file (in `data/`) | Computed by | Used by figures | Time |
|---|---|---|---|
| `sm_weak_grid.npz` | `compute/sm_weak_grid.py` | 7 | 2.5 min |
| `loop_grid_rg.npz` | `compute/loop_phase_maps.py loop_grid_rg` | 25, 33 | 1.1 h |
| `loop_corner.npz` | `compute/loop_phase_maps.py loop_corner` | 26 | 1.3 h |
| `loop_grid_rg_kneg.npz` | `compute/loop_phase_maps.py loop_grid_rg_kneg` | 27 | 1.25 h |
| `loop_grid_rg_kneg_mSneg.npz` | `compute/loop_phase_maps.py loop_grid_rg_kneg_mSneg` | 28, 35 | 3.1 h |
| `loop_grid_rg_mSneg.npz` | `compute/loop_phase_maps.py loop_grid_rg_mSneg` | 29 | 0.75 h |
| `loop_zoom.npz` | `compute/loop_phase_maps.py loop_zoom` | 30-32, 34 | 18 min |
| `loop_zoom_signs.npz` | `compute/loop_phase_maps.py loop_zoom_signs` | 30-32, 36 | 45 min |
| `loop_gallery_points.npz` | `compute/loop_gallery_points.py` (from the two zoom files) | 31, 32 | seconds |
| `loop_contour_<name>.npz` | `compute/loop_contours.py <name>` | 33-36 | 1.5 min each |

What the files contain:
- `sm_weak_grid.npz`: the SM phase on two grids of (u, y_0), from the 3-loop running up to M_Pl.
- `loop_*` maps: the one-loop phase maps, 160 x 160 points per panel.
- `loop_gallery_points.npz`: the parameter points of the gallery panels of phases d, e, g and h.
- `loop_contour_*.npz`: the ground state and its derivatives along the four contours.

`compute/loop_phase_maps.py` saves each panel as soon as it is computed, to `data/<name>.partial.npz`. The finished
file replaces `data/<name>.npz` only at the end. If a run is interrupted, running the same command again resumes from
the partial file.

**About the times.** They were measured on a 2025 laptop (Intel Core Ultra 7 255H), with each script using one
processor core. The speed of a single core is what matters, not the number of cores, so:
- another recent laptop or desktop should take about as long (within a factor of 2);
- an older computer takes longer.

The scripts are independent apart from the gallery points, which need the two zoom files. You can run several at
once, in separate terminals, to use more cores: the whole of `data/` then takes about 3 hours instead of 9.

## Layout

```
make_figures.py     entry point
figures/            one plotting script per figure (or group of figures); they read data/ and write output/
compute/            the slow computations; they write data/
data/               their results, as compressed .npz files (nearcrit/data_io.py reads them)
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

## How this code was written

Most of the code in this repository was written by Claude, an AI model made by Anthropic, using Claude Code, working
under the direction of the authors. The authors wrote the potentials and most of the explicit expressions for the
phase boundaries of the single-field models and of the tree-level two-field model. Claude turned these into the
figures, and chose and implemented the numerical methods for the one-loop two-field model (finding its extrema and
solving for its phase boundaries). Claude also reorganized the code into this repository and checked that every figure
reproduces the paper's. The authors designed the computations and reviewed the code and its results, and they take
full responsibility for both.

## Citation

If you use this code, please cite the paper:

```
@article{nearcriticality,
    title   = "{A framework for near-criticality in BSM physics}",
    author  = "Lodman, Jackie and Polizzotto, Leon and Steingasser, Thomas",
    journal = "[JOURNAL]",
    eprint  = "XXXX.XXXXX",
    archivePrefix = "arXiv",
    year    = "[YEAR]"
}
```

## License

MIT, see `LICENSE`.
