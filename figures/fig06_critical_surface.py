"""Fig. 6: sketch of a critical surface C in gamma-parameter space with adapted coordinates (sigma^A, t).

The 3D scene is projected orthographically onto a plain 2D axes by hand, so the drawing order
(surface, grid lines, arrows) and the right-angle markers are fully under control.

Writes output/gamma_surface_critical_coordinates.pdf.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the nearcrit package

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyArrowPatch

from nearcrit import style
from nearcrit.style import LINEWIDTH_IN

style.use()

# ---------------------------------------------------------------- projection
# Orthographic camera: VIEW points from the scene towards the viewer. gamma^1 runs to the right,
# gamma^3 up and gamma^2 towards the viewer (down and to the left on the page).
VIEW = np.array([0.22, 1.0, 0.55])
VIEW /= np.linalg.norm(VIEW)
RIGHT = np.array([VIEW[1], -VIEW[0], 0.0])
RIGHT /= np.linalg.norm(RIGHT)
UP = np.array([0.0, 0.0, 1.0]) - VIEW[2] * VIEW
UP /= np.linalg.norm(UP)


def proj(p):
    """Project 3D points (..., 3) to page coordinates (..., 2)."""
    p = np.asarray(p, float)
    return np.stack([p @ RIGHT, p @ UP], axis=-1)


def depth(p):
    """Distance towards the viewer (larger = in front)."""
    return np.asarray(p, float) @ VIEW


# ---------------------------------------------------------------- the surface
# C is a wavy patch over the rectangle G1_RANGE x G2_RANGE of the (gamma^1, gamma^2) plane, so its edges
# follow the gamma axes. The coordinates sigma^A on it are rotated by THETA against gamma^1, gamma^2:
# the sigma grid is drawn clipped to the patch, and is visibly not aligned with the gamma axes.
G1_RANGE = (0.10, 2.10)
G2_RANGE = (0.05, 0.95)
THETA = np.deg2rad(20)              # rotation of the sigma grid against the gamma axes
CENTER = np.array([1.10, 0.50])     # sigma = 0 in the (gamma^1, gamma^2) plane
E1 = np.array([np.cos(THETA), np.sin(THETA)])
E2 = np.array([-np.sin(THETA), np.cos(THETA)])


def height(g1, g2):
    """gamma^3 on C as a function of (gamma^1, gamma^2)."""
    return (0.50 + 0.12 * np.sin(1.6 * np.pi * g1 + 0.4) + 0.10 * np.cos(1.8 * np.pi * g2)
            + 0.12 * g1 * (1 - g2) + 0.08 * np.sin(np.pi * g1) * np.sin(np.pi * g2))


def patch(x, y):
    """Point on C at fractional position (x, y) in [0, 1]^2 of the gamma^1, gamma^2 rectangle."""
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    g1 = G1_RANGE[0] + (G1_RANGE[1] - G1_RANGE[0]) * x
    g2 = G2_RANGE[0] + (G2_RANGE[1] - G2_RANGE[0]) * y
    return np.stack([g1, g2, height(g1, g2)], axis=-1)


def surface(s1, s2):
    """Point on C at surface coordinates (sigma^1, sigma^2), in gamma units."""
    s1, s2 = np.broadcast_arrays(np.asarray(s1, float), np.asarray(s2, float))
    g1 = CENTER[0] + s1 * E1[0] + s2 * E2[0]
    g2 = CENTER[1] + s1 * E1[1] + s2 * E2[1]
    return np.stack([g1, g2, height(g1, g2)], axis=-1)


def sigma_of(x, y):
    """Surface coordinates of the patch point (x, y)."""
    d = patch(x, y)[:2] - CENTER
    return d @ E1, d @ E2


def inside(P, tol=1e-9):
    return ((P[..., 0] >= G1_RANGE[0] - tol) & (P[..., 0] <= G1_RANGE[1] + tol)
            & (P[..., 1] >= G2_RANGE[0] - tol) & (P[..., 1] <= G2_RANGE[1] + tol))


def tangents(s1, s2, h=1e-5):
    """Coordinate tangent vectors d/dsigma^1, d/dsigma^2 at a point."""
    d1 = (surface(s1 + h, s2) - surface(s1 - h, s2)) / (2 * h)
    d2 = (surface(s1, s2 + h) - surface(s1, s2 - h)) / (2 * h)
    return d1, d2


def normal(s1, s2):
    d1, d2 = tangents(s1, s2)
    n = np.cross(d1, d2)
    n /= np.linalg.norm(n)
    return n if n[2] > 0 else -n


# ---------------------------------------------------------------- styles
SURF_COLOR = to_rgb("C0")
GRID_KW = dict(color="C0", lw=0.7, solid_capstyle="round")
EDGE_KW = dict(color="C0", lw=1.5, solid_joinstyle="round")
FRAME_COLOR = "#0b3d6e"                    # darker blue: sigma^A arrows stand out from grid and edge
FRAME_KW = dict(color=FRAME_COLOR, lw=1.5)   # sigma^A arrows
T_KW = dict(color="black", lw=1.2)           # t arrows
AXIS_KW = dict(color="black", lw=0.9)
LABEL_BOX = dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75)  # labels on the surface


def arrow(ax, p0, p1, z=5, **kw):
    """Straight 2D arrow between two page points."""
    style = dict(arrowstyle="-|>", mutation_scale=9, shrinkA=0, shrinkB=0)
    ax.add_patch(FancyArrowPatch(p0, p1, zorder=z, **style, **kw))


def curve_arrow(ax, pts, z=5, **kw):
    """Polyline in page coordinates with an arrowhead at its end."""
    ax.plot(pts[:-1, 0], pts[:-1, 1], zorder=z, solid_capstyle="round", **kw)
    arrow(ax, pts[-6], pts[-1], z=z, color=kw["color"], lw=kw["lw"])


def draw_surface(ax, n1=90, n2=70):
    """Opaque, lightly shaded surface, drawn back to front (painter's algorithm)."""
    x, y = np.meshgrid(np.linspace(0, 1, n1), np.linspace(0, 1, n2), indexing="ij")
    P = patch(x, y)
    quads = np.stack([P[:-1, :-1], P[1:, :-1], P[1:, 1:], P[:-1, 1:]], axis=2)  # (n1-1, n2-1, 4, 3)
    quads = quads.reshape(-1, 4, 3)
    # Lambert shading with light from the upper left, mapped onto a light-to-mid blue.
    nrm = np.cross(quads[:, 1] - quads[:, 0], quads[:, 3] - quads[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm *= np.sign(nrm[:, 2:3])
    light = np.array([-0.5, 0.6, 1.0])
    light /= np.linalg.norm(light)
    shade = np.clip(nrm @ light, 0, 1)
    shade = (shade - shade.min()) / (np.ptp(shade) + 1e-12)
    white = np.ones(3)
    colors = [(1 - (0.13 + 0.17 * (1 - s))) * white + (0.13 + 0.17 * (1 - s)) * np.array(SURF_COLOR)
              for s in shade]
    order = np.argsort(depth(quads.mean(axis=1)))
    pc = PolyCollection(proj(quads[order]), facecolors=np.array(colors)[order],
                        edgecolors="none", antialiased=False, zorder=2, rasterized=True)
    ax.add_collection(pc)


GRID_SPACING = 0.17


def draw_grid(ax, through, spacing=GRID_SPACING):
    """sigma grid with lines through the point `through` (in sigma), clipped to the patch."""
    corners = np.array([[G1_RANGE[i], G2_RANGE[j]] for i in (0, 1) for j in (0, 1)]) - CENTER
    smin, smax = (corners @ np.stack([E1, E2], 1)).min(0), (corners @ np.stack([E1, E2], 1)).max(0)
    for a in (0, 1):                                 # a = 0: lines of constant sigma^1
        ks = np.arange(np.ceil((smin[a] - through[a]) / spacing), (smax[a] - through[a]) / spacing + 1)
        run = np.linspace(smin[1 - a], smax[1 - a], 800)
        for c in through[a] + spacing * ks:
            P = surface(c, run) if a == 0 else surface(run, c)
            P[~inside(P)] = np.nan
            ax.plot(*proj(P).T, zorder=3, **GRID_KW)
    s = np.linspace(0, 1, 200)
    edge = np.concatenate([patch(s, 0), patch(1, s), patch(s[::-1], 1), patch(0, s[::-1])])
    ax.plot(*proj(edge).T, zorder=3, **EDGE_KW)


def right_angle(ax, p, n, tvec, size=0.05, z=6):
    """Small square in the plane of the normal n and the tangent tvec at p (drawn in 3D, then projected)."""
    tvec = tvec / np.linalg.norm(tvec)
    corners = np.array([p + size * tvec, p + size * (tvec + n), p + size * n])
    ax.plot(*proj(corners).T, color="black", lw=0.6, zorder=z)


def draw_t_arrow(ax, sig, length=0.30, label=True, side="right"):
    """Normal arrow at sigma = sig, with a right-angle mark on the given side (page left/right).

    The mark's tangent leg is n x VIEW: it is tangent to C and lies in the page plane, so the mark is an
    exact right angle on the page as well as in 3D."""
    s1, s2 = sig
    p = surface(s1, s2)
    n = normal(s1, s2)
    tvec = np.cross(n, VIEW)
    if (proj(tvec)[0] > 0) != (side == "right"):
        tvec = -tvec
    right_angle(ax, p, n, tvec)
    arrow(ax, proj(p), proj(p + length * n), z=6, **T_KW)
    if label:
        tip = proj(p + length * n)
        ax.text(tip[0] + 0.012, tip[1] + 0.005, r"$t$", ha="left", va="bottom", zorder=7)
    return p


def draw_frame(ax, sig, L1, L2, label1_below=False, label2_below=False):
    """sigma^1, sigma^2 arrows along the grid lines through sig, with labels at their tips."""
    s1, s2 = sig
    curve_arrow(ax, proj(surface(s1 + np.linspace(0, L1, 60), s2)), z=5, **FRAME_KW)
    curve_arrow(ax, proj(surface(s1, s2 + np.linspace(0, L2, 60))), z=5, **FRAME_KW)
    q1 = proj(surface(s1 + L1, s2))
    q2 = proj(surface(s1, s2 + L2))
    if label1_below:   # keeps the label clear of things above the tip
        ax.text(q1[0] - 0.02, q1[1] - 0.025, r"$\sigma^1$", color=FRAME_COLOR, ha="right", va="top", zorder=7,
                bbox=LABEL_BOX)
    else:
        ax.text(q1[0] + 0.01, q1[1] + 0.012, r"$\sigma^1$", color=FRAME_COLOR, ha="left", va="bottom", zorder=7,
                bbox=LABEL_BOX)
    if label2_below:   # for a sigma^2 arrow running sideways (along a crest): label under its tip
        ax.text(q2[0], q2[1] - 0.03, r"$\sigma^2$", color=FRAME_COLOR, ha="center", va="top", zorder=7,
                bbox=LABEL_BOX)
    else:
        ax.text(q2[0] - 0.03, q2[1], r"$\sigma^2$", color=FRAME_COLOR, ha="right", va="center", zorder=7,
                bbox=LABEL_BOX)


def draw_axes(ax):
    o = np.zeros(3)
    ends = {0: 2.35, 1: 0.90, 2: 1.15}
    labels = {0: r"$\gamma^1$", 1: r"$\gamma^2$", 2: r"$\gamma^3$"}
    offsets = {0: (0.02, 0.0, "left", "center"), 1: (-0.02, -0.01, "right", "top"),
               2: (-0.02, 0.0, "right", "center")}
    for k in range(3):
        e = np.zeros(3)
        e[k] = ends[k]
        arrow(ax, proj(o), proj(e), z=1, **AXIS_KW)
        dx, dy, ha, va = offsets[k]
        q = proj(e)
        ax.text(q[0] + dx, q[1] + dy, labels[k], ha=ha, va=va)


def make_figure():
    fig, ax = plt.subplots(figsize=(LINEWIDTH_IN, 2.96))  # height-limited: this sets the scale
    ax.set_aspect("equal")
    ax.axis("off")

    # Reference point gamma_c and two further points, all on grid intersections, each with the local
    # frame: sigma^A along the surface, t transverse to it.
    base = sigma_of(0.42, 0.24)                      # the grid passes through this point

    def node(i, j):
        """sigma of the grid intersection i, j steps from base."""
        return (base[0] + i * GRID_SPACING, base[1] + j * GRID_SPACING)

    # Nodes chosen so each t arrow reads as perpendicular to C on the page, with visible sigma arrows.
    sc, s_left, s_right = node(0, 0), node(-4, 1), node(5, 0)

    draw_axes(ax)
    draw_surface(ax)
    draw_grid(ax, through=base)

    draw_frame(ax, sc, 2 * GRID_SPACING, 2 * GRID_SPACING)
    draw_frame(ax, s_left, 1.5 * GRID_SPACING, GRID_SPACING)
    draw_frame(ax, s_right, 1.5 * GRID_SPACING, 2.0 * GRID_SPACING)

    draw_t_arrow(ax, sc, length=0.45, side="left")
    draw_t_arrow(ax, s_left, length=0.34, side="left")
    draw_t_arrow(ax, s_right, length=0.36, side="left")

    qc = proj(surface(*sc))
    ax.plot(*qc, "o", color="black", ms=4.5, zorder=8)
    ax.text(qc[0] + 0.05, qc[1] + 0.04, r"$\gamma_c^a$", ha="left", va="bottom", zorder=8,
            bbox=LABEL_BOX)

    qC = proj(patch(1.0, 0.0))
    ax.text(qC[0] + 0.03, qC[1] + 0.01, r"$\mathcal{C}$", color="C0", ha="left", va="bottom")

    ax.autoscale_view()
    fig.tight_layout(pad=0.1)
    style.savefig(fig, "gamma_surface_critical_coordinates.pdf")
    return fig


if __name__ == "__main__":
    make_figure()
