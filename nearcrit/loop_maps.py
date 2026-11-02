"""One-loop phase maps of the Higgs-singlet model in the tree-level (Omega_H, Omega_S) plane: their computation,
labelling and drawing (Figs. 25-36).

Model: H is the SM Higgs doublet (3 Goldstones) and S a real doublet, i.e. an O(2) multiplet with 1 Goldstone,

    V_tree = -mu_H^2 H^2/2 - mu_S^2 S^2/2 + lambda_H H^4/4 + lambda_S S^4/4 + kappa H^2 S^2/4,

plus the one-loop MS-bar Coleman-Weinberg potential with the two scalars, their Goldstones, W, Z and the top,
RG-improved: evaluated at the field-dependent scale mu^2 = M_t^2 + H^2 + S^2, with all couplings run there at one loop
(see singlet_loop.py). The SM couplings are the MS-bar values at mu = M_t used for Fig. 7 (g = 0.64779,
g' = 0.35830, g_3 = 1.1666, y_t = 0.93690), and masses are in units of |mu_H| = M_h / sqrt(2) = 88.5 GeV. At each
point (Omega_H, Omega_S) the couplings are lambda_H = kappa Omega_H / (2 r), lambda_S = kappa r Omega_S / 2, with
r = mu_S^2 / mu_H^2. The SM value lambda_H = 0.126 sits at Omega_H = 2 r 0.126 / kappa.

Extrema are searched in the disk H^2 + S^2 <= Lambda^2, Lambda = 10 TeV. Where V on that disk has a local minimum on
its edge -- a valley of V crosses the cutoff with V still falling outward, in practice mostly along the H axis, where
the top loop turns the potential over at small lambda_H -- there is No Ground State below Lambda (NGS, gray). The edge
of that region therefore moves with Lambda; it is not a phase boundary and is drawn without a line. Points where a
coupling runs past 4 pi (towards its Landau pole) below Lambda are non-perturbative: they get no phase and are left
white ("NP").

Conventions as in the tree-level figures: colour = quantum phase (where the ground state is), hatch = potential phase,
black lines = quantum phase transitions (solid first, dashed second order), purple = potential phase transitions
within a quantum phase. Within NGS the hatch says which local (metastable) minima remain: on the H axis (the
electroweak-like vacuum), on the S axis, or off the axes. Thin dotted gray lines are the tree-level boundaries.

The phase boundaries are not grid contours: each is solved from the condition that defines it (loop_boundaries.py).
A potential phase transition is where a mass vanishes at an extremum -- the origin's mass parameter, an axis
extremum's transverse mass (Omega^(1) = 1), or two extrema merging (a fold) -- and a first-order quantum transition is
where two minima have the same energy. The cutoff enters through three more conditions: an extremum crossing the
circle H^2 + S^2 = Lambda^2, the edge of NGS (dV/d rho is zero along a valley of V crossing that circle), and the
edge of the non-perturbative region (a coupling reaches 4 pi one RG step above mu^2 = M_t^2 + Lambda^2). The grid only
says which condition holds between two neighbouring points; it is then solved for along the edge between them, and
the lines and the edges of the fills pass through the solutions. Where a phase is narrower than a grid cell, a finer
grid is computed and drawn there instead (REFINE_RG, or automatically for the figures with other signs), so that the
lines do not break up.

A panel is a dict of arrays over its grid (see compute_panel); compute/loop_phase_maps.py computes them and adds the
inner cuts and the "outer" ground states (add_inner_cuts, add_outer_ground_states).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path as MPath

from nearcrit import loop_boundaries as B
from nearcrit import singlet_loop as L
from nearcrit.singlet_tree import TRANSITION_LINESTYLE, TRANSITION_ORDER
from nearcrit.singlet_tree_plot import (
    HATCH_COLOR, HATCH_RC, HATCHES, NGS_COLOR, PHASE_ALPHA, POTENTIAL_TRANSITION_STYLE, quantum_color,
)
from nearcrit.style import (
    LINEWIDTH_IN, region_color, savefig, set_ticks, set_xlabel_left, set_xlabel_right, set_ylabel_bottom,
    set_ylabel_top,
)

# --- Physical inputs -------------------------------------------------------------------------------------------
MH, MT = 125.15, 173.34                 # GeV, as in Fig. 7
MU_H = MH / np.sqrt(2.0)                # |mu_H| in GeV: the unit of mass
MU = MT / MU_H                          # renormalization scale mu = M_t
CUTOFF_GEV = 1.0e4                      # Lambda: extrema are searched in H^2 + S^2 <= Lambda^2
CUTOFF = CUTOFF_GEV / MU_H
LAMBDA_SM = 0.12604
SM_COUPLINGS = dict(g=0.64779, gp=0.35830, g3=1.1666, yt=0.93690)   # MS-bar at M_t
# The one-loop potential: Goldstones of H and S, the cutoff and the extremum search, and the running SM couplings (W, Z
# and top enter through them rather than as fixed species).
LOOP_RG = dict(n_goldstone=3, n_goldstone_S=1, species=(), cutoff=CUTOFF, axis_decades=5.0, interior_decades=5.0,
               interior_n=48, rg=SM_COUPLINGS)
R_VALUES = [0.5, 1.0, 2.0]              # columns: r = mu_S^2 / mu_H^2
KAPPA_VALUES = [0.25, 0.5, 1.0]         # rows
OMEGA_MAX = 2.0
NGRID = 160                             # cell centres (k + 1/2) * OMEGA_MAX / NGRID, so Omega = 1 is a cell edge

# --- Potential phases ------------------------------------------------------------------------------------------
# A potential phase is an extremum structure (origin, H axis, S axis, off the axes). Structures of the tree-level
# potential keep their tree letter (a, b, c, hatched as in the tree-level figures); the others are radiatively
# generated and get the next letters. Without a ground state, the hatch is the set of local minima that remain
# (other than the origin), listed deepest first.
NEW_HATCHES = ["xxx", "+++", "///---", "\\\\\\---", "xxx---", "+++---"]   # d, e, f, g, h, i
# The keys list the minima deepest first, one letter per minimum; the same minima in the other depth order are
# hatched more densely, and a second minimum of one kind adds a diagonal.
NGS_HATCHES = {"H, S": "|||", "S, H": "||||||", "H": "///|||", "S": "\\\\\\|||", "M": "+++|||", "H, M": "---|||///",
               "M, H": "------||||||//////", "S, M": "xxx|||", "H, S, M": "---|||xxx", "H, H": "//////|||",
               "M, M": "+++|||\\\\\\", "none": None}
NGS_LABELS = {key: (r"NGS$_{" + key.replace(", ", "") + "}$" if key != "none" else r"NGS$_\emptyset$")
              for key in NGS_HATCHES}
KIND_TO_QUANTUM = {"0": "0", "H": "I", "S": "II", "M": "III"}
# The radiatively generated structures met so far, with fixed letters so that every figure uses the same ones
# (origin, H axis, S axis, off the axes). Any other structure gets the next free letter.
KNOWN_NEW = {
    ("max", ("min",), ("saddle",), ("min", "saddle")): "d",   # Ib plus a mixed minimum-saddle pair
    ("max", ("min", "saddle", "min"), ("saddle",), ()): "e",  # Ib plus a second H-axis minimum
    ("saddle", ("min",), (), ("min", "saddle")): "f",         # Ic plus a mixed minimum-saddle pair (mu_S^2 < 0)
    ("saddle", ("min", "saddle", "min"), (), ()): "g",        # Ic plus a second H-axis minimum (mu_S^2 < 0)
    ("saddle", ("saddle",), (), ("min", "min", "saddle")): "h",  # IIIc plus a second mixed minimum (mu_S^2 < 0)
    ("max", ("saddle",), ("saddle",), ("min", "min", "saddle")): "i",  # IIIa plus a second mixed minimum (kappa < 0)
}
TREE_LINE_STYLE = dict(colors="0.45", linewidths=0.6, linestyles=":")
# A ground state in a minimum that is not the innermost of its kind -- e.g. the second, radiatively generated
# H-axis minimum of phase e -- is a different quantum phase (primed), since the ground state jumps between the two
# in a first-order transition. It is drawn in the darkest shade of its family (the c shades of PHASE_COLORS).
PRIMED_COLORS = {"I": "#005E53", "II": "#A3323F", "III": "#24486E"}
QUANTUM_ORDER = ("0", "I", "I'", "II", "II'", "III", "III'", "NGS", "NP")


def phase_color(q):
    """Fill colour of quantum phase q (primed phases in the dark shade of their family)."""
    if q == "NGS":
        return region_color(NGS_COLOR)
    if q.endswith("'"):
        return region_color(PRIMED_COLORS[q[:-1]], PHASE_ALPHA)
    return quantum_color(q)


def tex_label(q):
    """Quantum phase label for LaTeX (primes typeset)."""
    return q.replace("'", "$'$")


def potential_phase(structure, gs_kind, runaway, new_letters, order=None):
    """(quantum phase, potential-phase label) of one point. new_letters collects the radiatively generated
    structures, so that each keeps one letter across the whole figure. Without a ground state the label lists the
    local minima left (not the origin) by kind, one letter per minimum, deepest first when their depth order (order,
    the minima as (kind, index) from deepest to shallowest) is given: NGS_HS and NGS_SH differ by which minimum is
    deeper, and NGS_HH has two minima on the H axis."""
    if structure is None:
        return "NP", ""
    if runaway or gs_kind == "":
        minima = [name for name, types in zip("HSM", structure[1:]) for t in types if t == "min"]
        if order is not None:
            minima = [kind for kind, _ in order if kind != "0"]
        key = ", ".join(minima) or "none"
        return "NGS", key
    label = L.TREE_SIGNATURES.get(structure + (gs_kind,))
    if label is not None:
        return KIND_TO_QUANTUM[gs_kind], label[-1] if label != "0" else ""
    if structure not in new_letters:
        free = [c for c in "defghijklmnopq" if c not in KNOWN_NEW.values() and c not in new_letters.values()]
        new_letters[structure] = KNOWN_NEW.get(structure) or free[0]
    return KIND_TO_QUANTUM[gs_kind], new_letters[structure]


# --- Phase maps --------------------------------------------------------------------------------------------------
FULL_WINDOW = ((0.0, OMEGA_MAX), (0.0, OMEGA_MAX))

# Local refinement. Where a phase is narrower than one grid cell, the grid cannot tell which condition holds where,
# and its lines break up. There a finer grid is computed (with the same conditions solved on its edges) and drawn
# instead of the coarse one: {(kappa, r): [dict(inset=window drawn from the fine grid, n=(nx, ny))]}. The fine grid
# extends one coarse cell beyond the inset on every side, so its lines and fills run across the inset's edge and
# are cut there exactly. Inset edges sit on coarse cell edges.
# Automatic refinement (used for the figures with other signs): tiles of AUTO_TILE x AUTO_TILE coarse cells around
# every short piece of line (see B.polylines), each computed on a grid AUTO_N / (AUTO_TILE + 2) times finer.
AUTO_TILE = 8
AUTO_N = 80
REFINE_RG = {
    # phase e (Fig. 30 (a)): the strip between the NGS edge and the fold, about one cell wide
    # (fine cells 0.00016 across, where the NGS edge, the first-order line and the fold come within 0.0006)
    (1.0, 2.0): [dict(inset=((0.0875, 0.1375), (1.15, 2.0)), n=(480, 140))],
    # its mirror for mu_S^2 < 0: phase g (Ic plus a second H-axis minimum) at Omega < 0
    (1.0, -2.0): [dict(inset=((-0.1375, -0.0875), (-2.0, -1.15)), n=(480, 140))],
}
_CELL = OMEGA_MAX / NGRID

# Zoom windows on the radiatively generated phases, at full panel resolution (Fig. 30, and the maps of Figs. 31-36).
NZOOM = 160
# (kappa, r): window ((Omega_H range), (Omega_S range)), ticks (Omega_H, Omega_S)
ZOOMS = {
    (0.5, 2.0): dict(window=((0.55, 0.95), (0.97, 1.12)), ticks=([0.6, 0.8], [1.0, 1.1])),
    (1.0, 2.0): dict(window=((0.095, 0.135), (1.10, 1.60)), ticks=([0.1, 0.12], [1.2, 1.4])),
}
ZOOMS_SIGNS = {
    # the strips here are ~1e-4 (IIIh) and ~1e-3 (III'h) wide in Omega_H: a short piece of the line, on a finer grid
    (-0.5, -0.5): dict(window=((0.578, 0.588), (1.755, 1.785)), ticks=([0.58, 0.585], [1.76, 1.77, 1.78]), n=320),
    (1.0, -2.0): dict(window=((-0.135, -0.095), (-1.70, -1.10)), ticks=([-0.12, -0.1], [-1.6, -1.4, -1.2])),
}


def auto_refine_insets(panel, Q, P, exclude=()):
    """Insets (tiles of AUTO_TILE x AUTO_TILE coarse cells) around the short pieces of line of one panel: where a
    strip narrower than a cell makes a line break up. Tiles overlapping an inset in exclude are left out."""
    window = panel.get("window", FULL_WINDOW)
    n = panel.get("n", NGRID)
    OH, OS, eh, es = omega_grid(window, n)
    classes = line_edges(panel, Q, P)
    all_c = np.logical_or.reduce([c for c, _ in classes.values()])
    all_r = np.logical_or.reduce([r for _, r in classes.values()])
    points = B.edge_points(panel["edges"]["pos_c"], panel["edges"]["pos_r"], OH[0], OS[:, 0])
    loose = []
    for act_c, act_r in classes.values():
        B.polylines(act_c, act_r, all_c, all_r, points, loose=loose)
    tiles = set()
    for t, i, j in loose:
        for ci, cj in ((i, j), (i, j + 1) if t == "c" else (i + 1, j)):
            tiles.add((ci // AUTO_TILE, cj // AUTO_TILE))
    out = []
    for ti, tj in sorted(tiles):
        h0, h1 = eh[AUTO_TILE * tj], eh[min(AUTO_TILE * (tj + 1), eh.size - 1)]
        s0, s1 = es[AUTO_TILE * ti], es[min(AUTO_TILE * (ti + 1), es.size - 1)]
        if any(h0 < b and a < h1 and s0 < d and c < s1 for (a, b), (c, d) in exclude):
            continue
        out.append(((float(h0), float(h1)), (float(s0), float(s1))))
    return out


def refine_window(inset):
    """The fine grid's window: the inset grown by one coarse cell on every side."""
    (h0, h1), (s0, s1) = inset
    return ((h0 - _CELL, h1 + _CELL), (s0 - _CELL, s1 + _CELL))


def omega_grid(window=None, n=None):
    """Cell centres (OH, OS) and edges (along Omega_H, along Omega_S) of an n x n grid (or nx x ny for n = (nx, ny))
    over the window ((OmegaH_min, OmegaH_max), (OmegaS_min, OmegaS_max)), by default the full panel."""
    (h0, h1), (s0, s1) = window or FULL_WINDOW
    nx, ny = (n, n) if np.isscalar(n or NGRID) else n
    nx, ny = nx or NGRID, ny or NGRID
    eh, es = np.linspace(h0, h1, nx + 1), np.linspace(s0, s1, ny + 1)
    OH, OS = np.meshgrid(0.5 * (eh[:-1] + eh[1:]), 0.5 * (es[:-1] + es[1:]))
    return OH, OS, eh, es


def panel_couplings(panel, r, kappa, X, Y):
    """(lambda_H, lambda_S, kappa) at the plot coordinates (X, Y) = (Omega_H, Omega_S) of a panel at fixed r and
    kappa."""
    lamH, lamS = L.lambdas_from_tree_Omegas(r, kappa, X, Y)
    return lamH, lamS, kappa


def compute_panel(r, kappa, window=None, n=None, loop=None):
    """The one-loop phase map of one panel, reduced to what the figure draws."""
    OH, OS, _, _ = omega_grid(window, n)
    lamH, lamS = L.lambdas_from_tree_Omegas(r, kappa, OH, OS)
    res = L.loop_phase_map(r, kappa, lamH, lamS, MU, verbose=True, **(loop or LOOP_RG))
    jumps = L.ground_state_jumps(res)
    edges = B.solve_edges(res, verbose=True, map_kwargs=loop or LOOP_RG)   # the boundaries, solved from their conditions
    return dict(structure=res["structure"], gs_kind=res["gs_kind"], runaway=res["runaway"], jumps=jumps, edges=edges,
                order=res["order"],
                OmegaH1=res["OmegaH1"], OmegaS1=res["OmegaS1"], nonperturbative=res["nonperturbative"],
                gs_Hbar=res["gs_Hbar"], gs_Sbar=res["gs_Sbar"], window=window or FULL_WINDOW, n=n or NGRID)


def inner_cuts(panel, r, kappa, loop=None, n_sub=12):
    """Edges between a point with a ground state and an NGS point across which a minimum also leaves the cutoff disk
    (the NGS side has fewer minima left than the other side has): the NGS strip in between, where that minimum
    remains, is narrower than the grid, so the potential transition within NGS (cut_H, cut_S or cut_M) would get no
    line. Solves that condition on these edges, as dict(pos_c, pos_r) (Omega on the edge, nan elsewhere)."""
    loop = loop or LOOP_RG
    OH, OS, _, _ = omega_grid(panel.get("window", FULL_WINDOW), panel.get("n", NGRID))
    ny, nx = OH.shape
    if loop.get("cutoff") is None:
        return dict(pos_c=np.full((ny, nx - 1), np.nan), pos_r=np.full((ny - 1, nx), np.nan))
    st, run = panel["structure"].ravel(), panel["runaway"].ravel()
    nonp = panel["nonperturbative"].ravel()
    a_all, b_all = B._edge_list(OH.shape)
    ngs_key = lambda s_: potential_phase(s_, "", True, {})[1]
    E = [e for e, (a, b) in enumerate(zip(a_all, b_all))
         if run[a] != run[b] and st[a] is not None and st[b] is not None and not nonp[a] and not nonp[b]
         and ngs_key(st[a]) != ngs_key(st[b])]
    pos = np.full(a_all.size, np.nan)
    if E:
        E = np.array(E)
        # bounded end first, NGS end second
        A = np.where(run[a_all[E]], b_all[E], a_all[E])
        Bn = np.where(run[a_all[E]], a_all[E], b_all[E])
        oh, os_ = OH.ravel(), OS.ravel()
        lamH, lamS, kap = panel_couplings(panel, r, kappa, np.concatenate([oh[A], oh[Bn]]),
                                          np.concatenate([os_[A], os_[Bn]]))
        res = L.loop_phase_map(r, kap, lamH, lamS, MU, **loop)
        k = E.size
        sts, ext = res["structure"], res["extrema"]
        mH1, mS1 = res["muH2_1"], res["muS2_1"]
        groups = {}
        for i in range(k):
            if sts[i] is None or sts[k + i] is None:
                continue
            ev, seed = B.structural_event(sts[i], sts[k + i], ext[i], ext[k + i],
                                          ((mH1[i], mH1[k + i]), (mS1[i], mS1[k + i])))
            if ev in ("cut_H", "cut_S", "cut_M"):
                groups.setdefault(ev, []).append((i, seed[0] if seed else np.nan))
        for ev, items in groups.items():
            I = np.array([i for i, _ in items])
            seeds = np.array([sd for _, sd in items], dtype=float) if ev == "cut_M" else None
            sol = B._solve_bracket(res["model"], I, k + I, ev, loop["cutoff"], seeds)
            # s runs from the bounded end; convert to Omega along the grid edge
            e = E[I]
            frm, to = A[I], Bn[I]
            cols = e < ny * (nx - 1)
            pos[e] = np.where(cols, oh[frm] + sol * (oh[to] - oh[frm]), os_[frm] + sol * (os_[to] - os_[frm]))
        # Where more than one condition is crossed (e.g. one minimum leaves the disk and another folds away), the
        # NGS part of the edge is sampled and solved like a grid (loop_boundaries.solve_edges); the line goes where
        # the remaining minima first change.
        left = np.flatnonzero(~np.isfinite(pos[E]))
        if left.size:
            e, frm, to = E[left], A[left], Bn[left]
            cols = e < ny * (nx - 1)
            coord = lambda i: np.where(cols, oh[i], os_[i])
            ngs_pos = np.concatenate([panel["edges"]["pos_c"].ravel(), panel["edges"]["pos_r"].ravel()])[e]
            with np.errstate(all="ignore"):
                s_ngs = np.where(np.isfinite(ngs_pos), (ngs_pos - coord(frm)) / (coord(to) - coord(frm)), 0.0)
            s0 = np.clip(s_ngs, 0.0, 1.0) + 1e-6
            t = s0[:, None] + np.linspace(0.0, 1.0, n_sub)[None, :] * (1.0 - s0)[:, None]
            sh = oh[frm][:, None] + t * (oh[to] - oh[frm])[:, None]
            ss = os_[frm][:, None] + t * (os_[to] - os_[frm])[:, None]
            lamH, lamS, kap = panel_couplings(panel, r, kappa, sh, ss)
            sub = L.loop_phase_map(r, kap, lamH, lamS, MU, **loop)
            sub["OmegaH"], sub["OmegaS"] = t.copy(), np.zeros_like(t)   # solved positions are fractions of the edge
            sol = B.solve_edges(sub, map_kwargs=loop, rows=False)["pos_c"]
            n_fb = 0
            for q in range(left.size):
                keys = [ngs_key(x) if x is not None else None for x in sub["structure"][q]]
                expected = ngs_key(st[frm[q]])
                if None in keys:
                    continue
                if keys[0] != expected:
                    u = s0[q]
                else:
                    m = next((m for m in range(n_sub - 1) if keys[m] != keys[m + 1]), None)
                    u = sol[q, m] if m is not None else np.nan
                if np.isfinite(u):
                    pos[e[q]] = coord(frm)[q] + u * (coord(to)[q] - coord(frm)[q])
                    n_fb += 1
            print(f"  inner cuts, sampled: {n_fb} of {left.size} edges")
        print(f"  inner cuts: {np.isfinite(pos).sum()} of {k} edges solved "
              f"({ {ev: len(v) for ev, v in groups.items()} })")
    nc = ny * (nx - 1)
    return dict(pos_c=pos[:nc].reshape(ny, nx - 1), pos_r=pos[nc:].reshape(ny - 1, nx))


def inner_jumps(panel, r, kappa, loop=None, n_sub=12):
    """Edges between a point with a ground state and an NGS point across which the ground state also changes before
    the NGS edge (a strip of another quantum phase there, narrower than the grid; the minimum it moves to may exist
    only inside the edge, e.g. one that enters through the cutoff circle). Each such edge is sampled at n_sub points
    up to its solved NGS crossing, and the sampled edge solved like a grid (loop_boundaries.solve_edges, subdividing
    where needed). Returns dict(first=dict(pos_c, pos_r), second=dict(pos_c, pos_r)): the first quantum transition
    on each edge, by its order (Omega on the edge, nan elsewhere)."""
    loop = loop or LOOP_RG
    OH, OS, _, _ = omega_grid(panel.get("window", FULL_WINDOW), panel.get("n", NGRID))
    ny, nx = OH.shape
    nc = ny * (nx - 1)
    out = {name: np.full(nc + (ny - 1) * nx, np.nan) for name in ("first", "second")}
    st, run, gk = panel["structure"].ravel(), panel["runaway"].ravel(), panel["gs_kind"].ravel()
    nonp = panel["nonperturbative"].ravel()
    a_all, b_all = B._edge_list(OH.shape)
    E = [e for e, (a, b) in enumerate(zip(a_all, b_all))
         if run[a] != run[b] and st[a] is not None and st[b] is not None and not nonp[a] and not nonp[b]
         and gk[b if run[a] else a] != ""]
    if E and loop.get("cutoff") is not None:
        E = np.array(E)
        A = np.where(run[a_all[E]], b_all[E], a_all[E])        # the end with a ground state
        Bn = np.where(run[a_all[E]], a_all[E], b_all[E])
        oh, os_ = OH.ravel(), OS.ravel()
        cols = E < nc
        coord = lambda i: np.where(cols, oh[i], os_[i])
        ngs_pos = np.concatenate([panel["edges"]["pos_c"].ravel(), panel["edges"]["pos_r"].ravel()])[E]
        with np.errstate(all="ignore"):
            s_ngs = np.where(np.isfinite(ngs_pos), (ngs_pos - coord(A)) / (coord(Bn) - coord(A)), 1.0)
        # samples from the bounded end to just before the NGS crossing
        t = np.linspace(0.0, 1.0, n_sub)[None, :] * (np.clip(s_ngs, 0.0, 1.0) * (1 - 1e-6))[:, None]
        sh = oh[A][:, None] + t * (oh[Bn] - oh[A])[:, None]
        ss = os_[A][:, None] + t * (os_[Bn] - os_[A])[:, None]
        lamH, lamS, kap = panel_couplings(panel, r, kappa, sh, ss)
        res = L.loop_phase_map(r, kap, lamH, lamS, MU, **loop)
        res["OmegaH"] = t.copy()          # so that the solved positions are fractions of the grid edge
        res["OmegaS"] = np.zeros_like(t)
        edges = B.solve_edges(res, map_kwargs=loop, rows=False)
        jc, _ = L.ground_state_jumps(res)
        g = res["gs_kind"]
        for k in range(E.size):
            for m in range(n_sub - 1):
                if (g[k, m] != g[k, m + 1] or jc[k, m]) and g[k, m] != "" and g[k, m + 1] != "":
                    branch = L.KIND_TO_BRANCH
                    first = jc[k, m] or (TRANSITION_ORDER.get(frozenset((branch[g[k, m]], branch[g[k, m + 1]])))
                                         == "first")
                    u = edges["pos_c"][k, m]    # fraction of the grid edge, from the bounded end
                    if np.isfinite(u):
                        out["first" if first else "second"][E[k]] = coord(A)[k] + u * (coord(Bn)[k] - coord(A)[k])
                    break
        print(f"  inner jumps: {np.isfinite(out['first']).sum()} first and {np.isfinite(out['second']).sum()} "
              f"second order, on {E.size} edges into NGS")
    return {name: dict(pos_c=v[:nc].reshape(ny, nx - 1), pos_r=v[nc:].reshape(ny - 1, nx)) for name, v in out.items()}


def add_inner_cuts(panels, loop=None):
    """Adds inner_cuts to the panels {(kappa, r[, m]): panel} that lack them; whether any were added."""
    missing = [kk for kk, p in panels.items() if "inner2" not in p or "inner_jumps2" not in p]
    for kk in missing:
        print(f"inner cuts: kappa = {kk[0]}, r = {kk[1]}" + (f", inset {kk[2]}" if len(kk) > 2 else ""), flush=True)
        if "inner2" not in panels[kk]:
            panels[kk].pop("inner", None)
            panels[kk]["inner2"] = inner_cuts(panels[kk], kk[1], kk[0], loop)
        panels[kk].pop("inner_jumps", None)
        panels[kk]["inner_jumps2"] = inner_jumps(panels[kk], kk[1], kk[0], loop)
    return bool(missing)


def outer_ground_states(panel, key, loop):
    """Where the ground state is not the innermost minimum of its kind (H axis, S axis or off the axes). Only points
    with more than one such minimum can qualify; they are few, so they are re-evaluated here (their positions are
    not stored) and checked against the stored structure."""
    kappa, r = key[:2]
    st, gk, run = panel["structure"], panel["gs_kind"], panel["runaway"]
    slot = {"H": 1, "S": 2, "M": 3}
    cand = [idx for idx in np.ndindex(gk.shape) if st[idx] is not None and not run[idx] and gk[idx] in slot
            and st[idx][slot[gk[idx]]].count("min") > 1]
    outer = np.zeros(gk.shape, dtype=bool)
    if not cand:
        return outer
    OH, OS, _, _ = omega_grid(panel.get("window"), panel.get("n"))
    rows, cols = np.array(cand).T
    lamH, lamS, kap = panel_couplings(panel, r, kappa, OH[rows, cols], OS[rows, cols])
    res = L.loop_phase_map(r, kap, lamH, lamS, MU, **loop)
    for i, idx in enumerate(cand):
        assert res["structure"][i] == st[idx], (key, idx)
        outer[idx] = ground_state_is_outer(res, i)
    return outer


def ground_state_is_outer(res, i):
    """Whether point i of a loop_phase_map result has its ground state in a minimum that is not the innermost
    (in tree-normalized fields) of its kind."""
    kind = res["gs_kind"][i]
    if kind not in ("H", "S", "M"):
        return False
    xbar, ybar, _ = L._scales(res["model"])
    ex = res["extrema"][i]
    if kind == "H":
        radii = sorted(np.sqrt(x / xbar[i]) for x, t in ex["H"] if t == "min")
    elif kind == "S":
        radii = sorted(np.sqrt(y / ybar[i]) for y, t in ex["S"] if t == "min")
    else:
        radii = sorted(np.hypot(np.sqrt(x / xbar[i]), np.sqrt(y / ybar[i])) for x, y, t in ex["M"] if t == "min")
    gs = np.hypot(res["gs_Hbar"][i], res["gs_Sbar"][i])
    return bool(abs(gs - radii[0]) > 1e-6 * (1 + gs))


def add_outer_ground_states(panels, loop=None):
    """Adds to each panel {(kappa, r[, m]): panel} that lacks it the mask "outer" of points whose ground state is not the innermost
    minimum of its kind (see outer_ground_states): those are primed quantum phases."""
    for key, p in panels.items():
        if "outer" not in p:
            p["outer"] = outer_ground_states(p, key, loop or LOOP_RG)


def label_panels(panels):
    """Quantum phase and potential-phase label of every point, with the new letters shared by all panels. The panels
    must have their "outer" masks (add_outer_ground_states)."""
    new_letters = {}
    labelled = {}
    for key in sorted(panels):
        p = panels[key]
        Q = np.empty(p["gs_kind"].shape, dtype=object)
        P = np.empty(p["gs_kind"].shape, dtype=object)
        for idx in np.ndindex(Q.shape):
            Q[idx], P[idx] = potential_phase(p["structure"][idx], p["gs_kind"][idx], p["runaway"][idx],
                                             new_letters, order=p["order"][idx] if "order" in p else None)
        outer = p["outer"]
        Q[outer] = np.array([q + "'" for q in Q[outer]], dtype=object)
        labelled[key] = (Q, P)
    return labelled, new_letters


# --- Gallery points (Figs. 31-32) ---------------------------------------------------------------------------
# label: (kappa, r, Omega_H, Omega_S, expected structure, expected quantum phase ('NGS' = no ground state below
# Lambda)). Chosen as the grid cell farthest from the phase's boundary; None = taken from the zoom maps
# (compute/loop_gallery_points.py).
GALLERY_PANELS = {
    "Id":     (0.5, 2.0, None, None, ("max", ("min",), ("saddle",), ("min", "saddle")), "I"),
    "IIId":   (0.5, 2.0, None, None, ("max", ("min",), ("saddle",), ("min", "saddle")), "III"),
    "Ie":     (1.0, 2.0, None, None, ("max", ("min", "saddle", "min"), ("saddle",), ()), "I"),
    "I'e":    (1.0, 2.0, None, None, ("max", ("min", "saddle", "min"), ("saddle",), ()), "I'"),
    "If":     (-0.5, -0.5, 1.98125, 0.45625, ("saddle", ("min",), (), ("min", "saddle")), "I"),
    "IIIf":   (-1.0, -0.5, 1.19375, 0.56875, ("saddle", ("min",), (), ("min", "saddle")), "III"),
    "Ig":     (1.0, -2.0, None, None, ("saddle", ("min", "saddle", "min"), (), ()), "I"),
    "I'g":    (1.0, -2.0, None, None, ("saddle", ("min", "saddle", "min"), (), ()), "I'"),
    "IIIh":   (-0.5, -0.5, None, None, ("saddle", ("saddle",), (), ("min", "min", "saddle")), "III"),
    "III'h":  (-0.5, -0.5, None, None, ("saddle", ("saddle",), (), ("min", "min", "saddle")), "III'"),
    "IIIi":   (-0.5, 1.0, -1.064, -0.99455, ("max", ("saddle",), ("saddle",), ("min", "min", "saddle")), "III"),
    "NGS_HS": (0.25, 2.0, 0.6063, 0.8688, ("max", ("min", "saddle"), ("min",), ("saddle",)), "NGS"),
    "NGS_SH": (0.25, 2.0, 0.7063, 0.2313, ("max", ("min", "saddle"), ("min",), ("saddle",)), "NGS"),
    "NGS_H":  (0.25, 2.0, 0.7063, 1.2688, ("max", ("min", "saddle"), ("saddle",), ()), "NGS"),
    "NGS_S":  (0.25, 2.0, 0.2313, 0.2313, ("max", (), ("min",), ("saddle",)), "NGS"),
    "NGS_M":  (0.25, 2.0, 0.9563, 1.5063, ("max", ("saddle", "saddle"), ("saddle",), ("min",)), "NGS"),
    "NGS_HM": (0.25, 2.0, 0.82875, 1.02694, ("max", ("min", "saddle"), ("saddle",), ("min", "saddle")), "NGS"),
    "NGS_MH": (0.25, 2.0, 0.9062, 1.0437, ("max", ("min", "saddle"), ("saddle",), ("min", "saddle")), "NGS"),
    "NGS_0":  (0.25, 2.0, 0.2313, 1.2312, ("max", (), ("saddle",), ()), "NGS"),
}
# The NGS key (minima deepest first, as in potential_phase) each NGS point must have.
GALLERY_NGS_KEYS = {"NGS_HS": "H, S", "NGS_SH": "S, H", "NGS_H": "H", "NGS_S": "S", "NGS_M": "M",
            "NGS_HM": "H, M", "NGS_MH": "M, H", "NGS_0": "none"}
# the zoom map each point left as None in GALLERY_PANELS is taken from, by its phase letter
GALLERY_ZOOMS = {"d": "loop_zoom.npz", "e": "loop_zoom.npz", "g": "loop_zoom_signs.npz", "h": "loop_zoom_signs.npz"}


# --- Drawing -----------------------------------------------------------------------------------------------------
def hatch_of(q, letter):
    """The hatch of potential phase letter in quantum phase q (None for no hatch)."""
    if q == "NP":
        return None
    if q == "NGS":
        return NGS_HATCHES[letter]
    if letter in HATCHES:
        return HATCHES[letter]
    return NEW_HATCHES["defghi".index(letter)] if letter else None


def line_edges(panel, Q, P):
    """Which edges between neighbouring grid points carry which line, as {name: (across columns, across rows)} with
    names "potential", "first" and "second" (boolean arrays of shape (ny, nx-1) and (ny-1, nx)).

    Potential transitions are changes of the potential phase within one quantum phase, and also places where two
    minima other than the ground state exchange depth order with the extrema otherwise unchanged (where there is no
    ground state, NGS, any two minima)."""
    kind = np.where(np.isin(Q, ["NGS", "NP"]), "", panel["gs_kind"]).astype(object)
    branch = L.KIND_TO_BRANCH
    table_first = np.vectorize(
        lambda a, b: a != "" and b != "" and (a == b or TRANSITION_ORDER.get(frozenset((branch[a], branch[b])))
                                              == "first"), otypes=[bool])

    st = panel["structure"]
    order = panel.get("order")
    if order is None:   # data from before the depth order was recorded
        order = np.full(st.shape, None, dtype=object)
    swapped = np.vectorize(
        lambda sa, sb, oa, ob, q: (sa is not None and sa == sb and oa is not None and ob is not None and oa != ob
                                   and (q == "NGS" or (len(oa) > 0 and len(ob) > 0 and oa[0] == ob[0]))),
        otypes=[bool])

    unprimed = np.vectorize(lambda q: q.rstrip("'"), otypes=[object])

    def classify(Qa, Qb, Pa, Pb, ka, kb, jump, sw):
        # A primed and an unprimed phase of one family without a jump (e.g. III' -> III where the inner minimum folds
        # away, so that the ground-state minimum becomes the innermost): the ground state does not move, so this is
        # a potential transition, not a quantum one.
        relabel = (Qa != Qb) & (unprimed(Qa) == unprimed(Qb)) & ~jump
        potential = (((Qa == Qb) & (Qa != "NP") & ((Pa != Pb) | sw)) | relabel)
        quantum = (Qa != Qb) & (ka != "") & (kb != "") & ~relabel
        first = quantum & (jump | table_first(ka, kb))
        return potential, first, quantum & ~first

    jc, jr = panel["jumps"]
    sw_c = swapped(st[:, :-1], st[:, 1:], order[:, :-1], order[:, 1:], Q[:, :-1])
    sw_r = swapped(st[:-1], st[1:], order[:-1], order[1:], Q[:-1])
    c = classify(Q[:, :-1], Q[:, 1:], P[:, :-1], P[:, 1:], kind[:, :-1], kind[:, 1:], jc, sw_c)
    r = classify(Q[:-1], Q[1:], P[:-1], P[1:], kind[:-1], kind[1:], jr, sw_r)
    return {"potential": (c[0], r[0]), "first": (c[1], r[1]), "second": (c[2], r[2])}


def _rect_path(windows, outer=None):
    """A path covering outer (if given) minus the windows ((h0, h1), (s0, s1)): the windows are wound the other way,
    so with the nonzero rule they are holes."""
    verts, codes = [], []

    def add(w, ccw):
        (h0, h1), (s0, s1) = w
        pts = [(h0, s0), (h1, s0), (h1, s1), (h0, s1)]
        pts = pts if ccw else pts[::-1]
        verts.extend(pts + [pts[0]])
        codes.extend([MPath.MOVETO] + [MPath.LINETO] * 3 + [MPath.CLOSEPOLY])

    if outer is not None:
        add(outer, True)
    for w in windows:
        add(w, outer is None)
    return MPath(verts, codes)


def _draw_map(ax, panel, Q, P, clip, clip_fills=None):
    """The fills, hatches and lines of one phase map, clipped to the path clip (data coordinates); the fills and
    hatches to clip_fills instead, if given."""
    OH, OS, _, _ = omega_grid(panel.get("window", FULL_WINDOW), panel.get("n", NGRID))
    clip_patch = PathPatch(clip, transform=ax.transData, facecolor="none", edgecolor="none")
    fill_patch = clip_patch if clip_fills is None else PathPatch(clip_fills, transform=ax.transData,
                                                                 facecolor="none", edgecolor="none")

    # Which edges between grid points carry a line: potential transitions (purple) between two potential phases of
    # one quantum phase, and quantum transitions (black) between two quantum phases with a ground state -- first
    # order (solid) where the ground state jumps, or where the tree-level table says so, second order (dashed)
    # otherwise. NGS has no ground state, so its edge, set by the cutoff, gets no line; like the edge of the
    # non-perturbative region, it is still solved from its condition, and the fills are snapped onto it.
    classes = line_edges(panel, Q, P)
    # Where a minimum leaves the cutoff disk just beyond the NGS edge, closer to it than the grid resolves, the edges
    # across both carry that potential transition (inner_cuts), drawn where its condition holds.
    # Likewise where the ground state jumps (first order) just before the NGS edge (inner_jumps).
    inner = {}
    if panel.get("inner2") is not None:
        inner["potential"] = panel["inner2"]
    if panel.get("inner_jumps2") is not None:
        inner.update(panel["inner_jumps2"])
    for name, ins in inner.items():
        pc, pr = classes[name]
        classes[name] = (pc | np.isfinite(ins["pos_c"]), pr | np.isfinite(ins["pos_r"]))
    all_c = np.logical_or.reduce([c for c, _ in classes.values()])
    all_r = np.logical_or.reduce([r for _, r in classes.values()])
    # Where each line crosses each edge: the solved condition (see loop_boundaries.py), else the edge midpoint.
    hc, sc = OH[0], OS[:, 0]
    edges = panel.get("edges")
    if edges is None:   # data from before the solved boundaries
        edges = dict(pos_c=np.full(all_c.shape, np.nan), pos_r=np.full(all_r.shape, np.nan))
    pos_c, pos_r = edges["pos_c"], edges["pos_r"]
    points = B.edge_points(pos_c, pos_r, hc, sc)
    class_points = {name: B.edge_points(np.where(np.isfinite(ins["pos_c"]), ins["pos_c"], pos_c),
                                        np.where(np.isfinite(ins["pos_r"]), ins["pos_r"], pos_r), hc, sc)
                    for name, ins in inner.items()}

    solved_c, solved_r = np.isfinite(pos_c), np.isfinite(pos_r)

    def snap(cs):
        """Move the fill's boundary onto the solved crossing points, so that fills and lines agree."""
        cs.set_paths(B.snap_paths(cs.get_paths(), hc, sc, pos_c, pos_r, solved_c, solved_r))
        cs.set_clip_path(fill_patch)
        cs.set_in_layout(False)          # always inside the axes; the clip path must not enter tight_layout

    # Colours: the quantum phases, NGS gray.
    for q in sorted(set(Q.flat) - {"NP"}):   # non-perturbative points stay white
        color = phase_color(q)
        snap(ax.contourf(OH, OS, (Q == q).astype(float), levels=[0.5, 1.5], colors=[color], zorder=1))
    # Hatches: the potential phases, as unfilled patches.
    codes = {pp: i for i, pp in enumerate(sorted(set(zip(Q.flat, P.flat))))}
    PH = np.vectorize(lambda q, p: codes[(q, p)])(Q, P)
    for (q, letter), c in codes.items():
        hatch = hatch_of(q, letter)
        if hatch is None:
            continue
        cs = ax.contourf(OH, OS, (PH == c).astype(float), levels=[0.5, 1.5], colors="none", hatches=[hatch],
                         zorder=2)
        cs.set_edgecolor(HATCH_COLOR)
        cs.set_linewidth(0)
        snap(cs)

    # The lines, through the solved crossing points; lines of every kind meet at common junction points.
    for name, (act_c, act_r) in classes.items():
        curves = B.polylines(act_c, act_r, all_c, all_r, class_points.get(name, points))
        if not curves:
            continue
        if name == "potential":
            style = dict(POTENTIAL_TRANSITION_STYLE, zorder=3.5)
        else:
            style = dict(colors="k", linewidths=1.5, linestyles=TRANSITION_LINESTYLE[name], zorder=4)
        lc = LineCollection(curves, **style)
        lc.set_clip_path(clip_patch)
        lc.set_in_layout(False)
        ax.add_collection(lc)


def draw_panel(ax, panel, Q, P, r, kappa, ticks=None, fines=()):
    """One phase map. ticks = (Omega_H ticks, Omega_S ticks); by default 0 and 1 on both, as in the atlases.
    fines = [(fine panel, its Q, its P)]: finer maps (see REFINE_RG), each drawn instead of this one in its inset."""
    window = panel.get("window", FULL_WINDOW)
    (h0, h1), (s0, s1) = window
    ax.add_patch(plt.Rectangle((h0, s0), h1 - h0, s1 - s0, facecolor="white", lw=0, zorder=0))
    insets = [f[0]["inset"] for f in fines]
    _draw_map(ax, panel, Q, P, _rect_path(insets, outer=window))
    for fine, Qf, Pf in fines:
        # The lines are cut exactly at the inset's edge. The fine fills reach half a coarse cell beyond it, over the
        # coarse ones (they are opaque and agree there), so that no anti-aliased seam shows along the edge.
        (a, b), (c, d) = fine["inset"]
        e = 0.5 * _CELL
        grown = ((max(a - e, h0), min(b + e, h1)), (max(c - e, s0), min(d + e, s1)))
        _draw_map(ax, fine, Qf, Pf, _rect_path([fine["inset"]]), clip_fills=_rect_path([grown]))

    # Tree-level boundaries (as in Fig. 21, mu_H^2 > 0), drawn exactly.
    tree = dict(color=TREE_LINE_STYLE["colors"], lw=TREE_LINE_STYLE["linewidths"], ls=TREE_LINE_STYLE["linestyles"],
                zorder=3)
    for xs, ys in tree_lines(np.sign(kappa), np.sign(r), window):
        ax.plot(xs, ys, **tree)

    ax.set_xlim(h0, h1)
    ax.set_ylim(s0, s1)
    ax.set_box_aspect(1)          # square panels; a zoom window that is not square is stretched
    ax.set_title(rf"$r = {r:g},\ \kappa = {kappa:g}$")
    if ticks is None:
        set_ticks(ax, [0, 1])
    else:
        ax.set_xticks(ticks[0], [f"${v:g}$" for v in ticks[0]])
        ax.set_yticks(ticks[1], [f"${v:g}$" for v in ticks[1]])


def tree_lines(sk, sr, window):
    """The tree-level boundaries in the window for sign(kappa) = sk and sign(mu_S^2) = sr (mu_H^2 > 0), as a list of
    (Omega_H values, Omega_S values):
      kappa > 0, mu_S^2 > 0 (Omega > 0): Omega_H = 1, Omega_S = 1 and the first-order line Omega_H = Omega_S;
      kappa > 0, mu_S^2 < 0 (Omega < 0): none (all Ic);
      kappa < 0, mu_S^2 > 0 (Omega < 0): Omega_H Omega_S = 1, inside which V is unbounded below;
      kappa < 0, mu_S^2 < 0 (Omega > 0): Omega_H Omega_S = 1, and Omega_H = 1 above it (Ic | IIIc)."""
    (h0, h1), (s0, s1) = window
    out = []
    if sk > 0 and sr > 0:
        out += [([1.0, 1.0], [s0, s1]), ([h0, h1], [1.0, 1.0])]
        lo, hi = max(h0, s0), min(h1, s1)
        if lo < hi:
            out.append(([lo, hi], [lo, hi]))
    if sk < 0:
        sgn = 1.0 if sr < 0 else -1.0          # the quadrant: sign(Omega) = sign(kappa) sign(mu_S^2)
        a = np.geomspace(1e-3, 1e3, 4001)
        x, y = sgn * a, sgn / a
        keep = (x >= h0) & (x <= h1) & (y >= s0) & (y <= s1)
        if keep.any():
            out.append((x[keep], y[keep]))
        if sr < 0:
            out.append(([1.0, 1.0], [max(1.0, s0), s1]))
    return out


_LEGEND_KW = dict(frameon=False, handlelength=1.4, handleheight=1.4, handletextpad=0.5, columnspacing=1.0,
                  borderaxespad=0.0, borderpad=0.0, alignment="center", loc="lower left", fontsize=9, title_fontsize=11)


def legend_handles(quantum_phases, potential_entries):
    """(line, colour, hatch) handles of the three keys: the lines (with the tree-level boundaries added), the
    quantum phases present (colours) and the potential phases (label, hatch)."""
    line_handles = [
        Line2D([], [], color="k", lw=1.5, ls=TRANSITION_LINESTYLE["first"], label="Quantum, first order"),
        Line2D([], [], color="k", lw=1.5, ls=TRANSITION_LINESTYLE["second"], label="Quantum, second order"),
        Line2D([], [], color=POTENTIAL_TRANSITION_STYLE["colors"], lw=POTENTIAL_TRANSITION_STYLE["linewidths"],
               label="Potential"),
        Line2D([], [], color=TREE_LINE_STYLE["colors"], lw=0.9, ls=":", label="Tree level"),
    ]
    color_handles = [Patch(facecolor=phase_color(q), edgecolor="none", label=tex_label(q)) for q in quantum_phases
                     if q != "NP"]
    if "NP" in quantum_phases:
        color_handles.append(Patch(facecolor="white", edgecolor="0.5", lw=0.5, label="Non-perturbative"))
    hatch_handles = [Patch(facecolor="white", edgecolor=HATCH_COLOR, hatch=h or "", lw=0.5, label=lab)
                     for lab, h in potential_entries]
    return line_handles, color_handles, hatch_handles


def present_entries(labelled):
    """(quantum phases, potential-phase key entries) present in the labelled panels {key: (Q, P)}."""
    present = set()
    for Q, P in labelled.values():
        present |= set(zip(Q.flat, P.flat))
    quantum = [q for q in QUANTUM_ORDER if any(p[0] == q for p in present)]
    letters = sorted({p for q, p in present if q not in ("NGS", "NP") and p})
    entries = [(lab, hatch_of("I", lab)) for lab in letters]
    entries += [(NGS_LABELS[k], NGS_HATCHES[k]) for k in NGS_HATCHES if ("NGS", k) in present]
    return quantum, entries


def add_legends(fig, y0, quantum_phases, potential_entries, gap_in=0.35, row_gap_in=0.1):
    """Keys centred below the panels, as add_legends in singlet_tree_plot.py, in three rows: the lines (with the
    tree-level boundaries added), the quantum phases (colours) and the potential phases (hatches). Returns the top
    of the keys in figure coordinates."""
    kw = _LEGEND_KW
    line_handles, color_handles, hatch_handles = legend_handles(quantum_phases, potential_entries)

    rows = int(np.ceil(len(hatch_handles) / 6))           # at most 6 entries per row, to fit the text width
    leg_p = fig.legend(handles=hatch_handles, title="Potential phase",
                       ncols=int(np.ceil(len(hatch_handles) / rows)), **kw)
    leg_q = fig.legend(handles=color_handles, title="Quantum phase", ncols=len(color_handles), **kw)
    leg_t = fig.legend(handles=line_handles, title="Phase transition", ncols=len(line_handles),
                       **{**kw, "handlelength": 2.4, "columnspacing": 1.4})
    r = fig.canvas.get_renderer()
    W, H = fig.get_figwidth(), fig.get_figheight()
    size = lambda leg: (leg.get_window_extent(r).width / fig.dpi / W, leg.get_window_extent(r).height / fig.dpi / H)
    y = y0
    for leg in (leg_p, leg_q, leg_t):
        w, h = size(leg)
        leg.set_bbox_to_anchor((0.5 - w / 2, y), transform=fig.transFigure)
        y += h + row_gap_in / H
    return y - row_gap_in / H


def make_grid_figure(panels, kappa_values, r_values, output, ticks=None):
    """The figure of the panels {(kappa, r): panel} (plus refined insets), rows kappa_values and columns r_values,
    saved as output/<output>. ticks = (Omega_H ticks, Omega_S ticks) as in draw_panel, for a figure over another
    window."""
    labelled, new_letters = label_panels(panels)
    with plt.rc_context(HATCH_RC):
        fig, axes = plt.subplots(len(kappa_values), len(r_values), figsize=(LINEWIDTH_IN, 8.2),
                                 sharex=True, sharey=True)
        for i, kappa in enumerate(kappa_values):
            for j, r in enumerate(r_values):
                Q, P = labelled[(kappa, r)]
                fines = [(panels[kk], *labelled[kk]) for kk in sorted(panels) if len(kk) == 3 and kk[:2] == (kappa, r)]
                draw_panel(axes[i, j], panels[(kappa, r)], Q, P, r, kappa, ticks=ticks, fines=fines)
        # Omega < 0: 0 is at the right and top ends, so label the other ends
        negative = max(panels[(kappa_values[0], r_values[0])]["window"][0]) <= 0
        for ax in axes[-1]:
            (set_xlabel_left if negative else set_xlabel_right)(ax, r"$\Omega_H$")
        for ax in axes[:, 0]:
            (set_ylabel_bottom if negative else set_ylabel_top)(ax, r"$\Omega_S$")

        quantum, entries = present_entries(labelled)

        h = fig.get_figheight()
        keys_top = add_legends(fig, y0=0.06 / h, quantum_phases=quantum, potential_entries=entries)
        fig.tight_layout(rect=(0, keys_top + 0.05 / h, 1, 1))
        savefig(fig, output)

        rend = fig.canvas.get_renderer()
        print("keys height (in):", round((keys_top + 0.05 / h) * h, 2))
        print("panel sizes (in):", sorted({(round(ax.get_window_extent(rend).width / fig.dpi, 2),
                                            round(ax.get_window_extent(rend).height / fig.dpi, 2)) for ax in axes.flat}))
    print("Radiatively generated potential phases:")
    for structure, letter in new_letters.items():
        print(f"  {letter}: origin {structure[0]}; H axis {list(structure[1])}; S axis {list(structure[2])}; "
              f"off axes {list(structure[3])}")
    print("lambda_H = lambda_SM at Omega_H =", {(k, r): round(2 * r * LAMBDA_SM / k, 3)
                                                 for k in kappa_values for r in r_values})
    return fig
