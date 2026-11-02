"""Figure setup shared by all figures, and where figures and data live.

The paper is an 11 pt jheppub article, so the text width is 430.2 pt. The figures are drawn at that width with
11 pt LaTeX fonts and included with width=\\linewidth, so no rescaling happens.

Without LaTeX (option --no-tex of any figure script or of make_figures.py, or automatically if LaTeX or one of its
packages is missing) the text is drawn by matplotlib itself, in its Computer Modern fonts. The figures then look
slightly different from the paper's.
"""
import os
import re
import shutil
import sys
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


# Without LaTeX: matplotlib's own Computer Modern, for the text and for the maths.
RC_NO_TEX = {
    "text.usetex": False,
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,     # cmr10 has no minus sign; mathtext draws it
}


def use(amsmath=False):
    """Set the paper's rcParams. amsmath adds \\usepackage{amsmath} to the LaTeX preamble (for \\text).
    Falls back to drawing the text without LaTeX if asked to (--no-tex) or if LaTeX does not work."""
    plt.rcParams.update(RC)
    if amsmath:
        plt.rcParams.update({"text.latex.preamble": r"\usepackage{amsmath}"})
    if "--no-tex" in sys.argv[1:] or os.environ.get("NEARCRIT_NO_TEX") or not _latex_works():
        plt.rcParams.update(RC_NO_TEX)
        _translate_labels()


# The LaTeX in the labels that matplotlib's mathtext spells differently.
_MATHTEXT = [(re.compile(r"\\displaystyle\s*\\frac"), r"\\dfrac"),
             (re.compile(r"\\le(?![a-zA-Z])"), r"\\leq"),
             (re.compile(r"\\ge(?![a-zA-Z])"), r"\\geq")]


def _translate_labels():
    """Make every text set from now on use the mathtext spelling of _MATHTEXT (without LaTeX only)."""
    from matplotlib.text import Text
    if getattr(Text.set_text, "translated", False):
        return
    set_text = Text.set_text

    def translated(self, s):
        if isinstance(s, str):
            for pattern, repl in _MATHTEXT:
                s = pattern.sub(repl, s)
        set_text(self, s)
    translated.translated = True
    Text.set_text = translated


def _latex_works():
    """Whether matplotlib can typeset with LaTeX (with the preamble set now); prints why not if it cannot."""
    if shutil.which("latex") is None:
        print("note: LaTeX not found; drawing the text without it (the figures differ slightly from the paper's)")
        return False
    from matplotlib.texmanager import TexManager
    try:
        TexManager().make_dvi(r"$x$", 11)
    except Exception as e:  # noqa: BLE001 (any failure of latex: a missing package, ...)
        lines = str(e).strip().splitlines()
        reason = next((line for line in lines if line.startswith("!")), lines[0] if lines else "")  # LaTeX's error
        print(f"note: LaTeX failed ({reason}); drawing the text without it (the figures differ slightly from the "
              f"paper's)")
        return False
    return True


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
