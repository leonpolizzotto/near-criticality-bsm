"""Check that every panel of Fig. 23 is in the potential phase it is labelled with, both by the tree-level phase map
and by its extrema, and is bounded below (check_panels in figures/fig23_tree_gallery.py). Run
`python tests/test_tree_gallery.py`, or with pytest.
"""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_tree_gallery_panels():
    spec = importlib.util.spec_from_file_location("fig23", ROOT / "figures" / "fig23_tree_gallery.py")
    fig23 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fig23)
    fig23.check_panels()


if __name__ == "__main__":
    test_tree_gallery_panels()
    print("test_tree_gallery_panels: passed")
