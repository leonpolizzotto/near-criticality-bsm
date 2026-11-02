"""Figure setup shared by all figures, and where figures and data live.

The paper is an 11 pt jheppub article, so the text width is 430.2 pt. The figures are drawn at that width with
11 pt LaTeX fonts and included with width=\\linewidth, so no rescaling happens.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"

LINEWIDTH_IN = 430.20639 / 72.27

RC = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "savefig.dpi": 300,
}


def use(amsmath=False):
    """Set the paper's rcParams. amsmath adds \\usepackage{amsmath} to the LaTeX preamble (for \\text)."""
    plt.rcParams.update(RC)
    if amsmath:
        plt.rcParams.update({"text.latex.preamble": r"\usepackage{amsmath}"})


def savefig(fig, name, **kwargs):
    """Save fig as output/<name>, creating output/ if needed."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUTPUT_DIR / name, **kwargs)


def set_xlabel_right(ax, label):
    """x axis label at the right end of the axis, just below it (out of the way of the tick labels)."""
    ax.set_xlabel(label, ha="right", va="top")
    ax.xaxis.set_label_coords(1.0, -0.03)


def set_xlabel_left(ax, label):
    """x axis label at the left end of the axis, just below it (for axes where 0 is at the right end)."""
    ax.set_xlabel(label, ha="left", va="top")
    ax.xaxis.set_label_coords(0.0, -0.03)


def set_ylabel_top(ax, label):
    """Horizontal y axis label at the top end of the axis, just left of it."""
    ax.set_ylabel(label, rotation=0, ha="right", va="top")
    ax.yaxis.set_label_coords(-0.03, 1.0)


def set_ylabel_bottom(ax, label):
    """Horizontal y axis label at the bottom end of the axis (for axes where 0 is at the top end)."""
    ax.set_ylabel(label, rotation=0, ha="right", va="bottom")
    ax.yaxis.set_label_coords(-0.03, 0.0)


def set_ticks(ax, values):
    """Tick both axes at these values only (the axes and other critical values), labelled."""
    labels = [f"{v:g}" for v in values]
    ax.set_xticks(values, labels)
    ax.set_yticks(values, labels)


def region_color(color, alpha=0.3):
    """The colour of a fill at alpha over white."""
    return alpha * np.array(to_rgb(color)) + 1 - alpha


def data_file(name, made_by):
    """Path of data/<name>. Stops with a message naming the compute script made_by if the file is missing."""
    path = DATA_DIR / name
    if not path.exists():
        raise SystemExit(f"{path} is missing: run `python {made_by}` first (see README.md).")
    return path
