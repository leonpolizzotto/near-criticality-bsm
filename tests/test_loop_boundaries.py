"""Check of the solved phase boundaries (nearcrit/loop_boundaries.py): on a small grid through the radiatively
generated phase d, loop_phase_map is re-run just either side of each solved crossing point, and must match the grid
point on that side. Run `python tests/test_loop_boundaries.py` (about a minute), or with pytest.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the nearcrit package

import numpy as np

from nearcrit import singlet_loop as L
from nearcrit.loop_boundaries import (
    EVENTS, _BRACKET_EVENTS, _PAIR_EVENTS, _edge_list, solve_edges, structural_event,
)
from nearcrit.loop_maps import LOOP_RG, MU, omega_grid


def check_edges(res, edges, map_kwargs, delta=0.002, max_edges=None, rng=None):
    """Re-run loop_phase_map just either side of each solved crossing point (at s -/+ delta edge lengths) and check
    that the result matches the grid point on that side: the extremum structure for potential transitions, the
    ground-state kind and position for first-order ones. Returns (n_checked, n_ok, n_partial, failures): n_ok edges
    match on both sides, n_partial have a second transition within the edge (see below), failures is a list of
    (orientation, i, j, event)."""
    shape = res["gs_kind"].shape
    ny, nx = shape
    a_all, b_all = _edge_list(shape)
    ev = np.concatenate([edges["event_c"].ravel(), edges["event_r"].ravel()])
    pos = np.concatenate([edges["pos_c"].ravel(), edges["pos_r"].ravel()])
    E = np.flatnonzero(ev >= 0)
    if max_edges is not None and E.size > max_edges:
        E = (rng or np.random.default_rng(0)).choice(E, max_edges, replace=False)
    nc = ny * (nx - 1)
    P = res["model"]
    OH, OS = res["OmegaH"].ravel(), res["OmegaS"].ravel()
    prm = res["params"]
    r, kappa = prm["r"], prm["kappa"]
    A, B = a_all[E], b_all[E]
    across_cols = E < nc
    span = np.where(across_cols, OH[B] - OH[A], OS[B] - OS[A])
    start = np.where(across_cols, OH[A], OS[A])
    s = (pos[E] - start) / span
    sides = []
    for sgn in (-1.0, 1.0):
        sv = np.clip(s + sgn * delta, 0.0, 1.0)
        lamH = P.lamH[A] + sv * (P.lamH[B] - P.lamH[A])
        lamS = P.lamS[A] + sv * (P.lamS[B] - P.lamS[A])
        kap = P.kappa[A] + sv * (P.kappa[B] - P.kappa[A])
        sides.append(L.loop_phase_map(r, kap, lamH, lamS, prm["mu"], **map_kwargs))
    st, gk = res["structure"].ravel(), res["gs_kind"].ravel()
    Hb, Sb = res["gs_Hbar"].ravel(), res["gs_Sbar"].ravel()
    xbar, ybar, _ = L._scales(P)
    ok = np.ones(E.size, dtype=bool)
    od = res["order"].ravel()
    for k in range(E.size):
        for side, cell in zip(sides, (A[k], B[k])):
            if EVENTS[ev[E[k]]] == "ngs":
                if side["runaway"][k] != res["runaway"].ravel()[cell]:
                    ok[k] = False
            elif EVENTS[ev[E[k]]] == "np":
                if side["nonperturbative"][k] != res["nonperturbative"].ravel()[cell]:
                    ok[k] = False
            elif EVENTS[ev[E[k]]] == "swap":
                if side["structure"][k] != st[cell] or side["order"][k] != od[cell]:
                    ok[k] = False
            elif EVENTS[ev[E[k]]] == "fopt":
                if side["gs_kind"][k] != gk[cell]:
                    ok[k] = False
                    continue
                xs, ys = L._scales(side["model"].take(np.array([k])))[:2]
                g_side = np.array([side["gs_Hbar"][k] * np.sqrt(xs[0]), side["gs_Sbar"][k] * np.sqrt(ys[0])])
                g_cell = np.array([Hb[cell] * np.sqrt(xbar[cell]), Sb[cell] * np.sqrt(ybar[cell])])
                if np.hypot(*(g_side - g_cell)) > 0.3 * max(np.hypot(*g_cell), 1e-12):
                    ok[k] = False
            elif side["structure"][k] != st[cell]:
                ok[k] = False
    # Where a second transition lies within the same edge, one side of the solved point is a narrow intermediate
    # phase rather than the far grid point's. Such an edge still passes if the change across the solved point is
    # the claimed condition and the other side matches its grid point.
    partial = np.zeros(E.size, dtype=bool)
    for k in np.flatnonzero(~ok):
        name = EVENTS[ev[E[k]]]
        if name in _PAIR_EVENTS or name in _BRACKET_EVENTS:
            continue
        lo, hi = sides
        found, _ = structural_event(lo["structure"][k], hi["structure"][k], lo["extrema"][k], hi["extrema"][k],
                                    ((lo["muH2_1"][k], hi["muH2_1"][k]), (lo["muS2_1"][k], hi["muS2_1"][k])))
        one_side = lo["structure"][k] == st[A[k]] or hi["structure"][k] == st[B[k]]
        partial[k] = found == name and one_side
    fails = [("c" if E[k] < nc else "r",) + tuple(np.unravel_index(E[k] if E[k] < nc else E[k] - nc,
                                                                   (ny, nx - 1) if E[k] < nc else (ny - 1, nx)))
             + (EVENTS[ev[E[k]]],) for k in np.flatnonzero(~ok & ~partial)]
    return E.size, int(ok.sum()), int(partial.sum()), fails


def test_edges_phase_d():
    """A 12 x 12 grid over the zoom window of phase d (r = 2, kappa = 0.5)."""
    r, kappa = 2.0, 0.5
    OH, OS, _, _ = omega_grid(((0.55, 0.95), (0.97, 1.12)), 12)
    lamH, lamS = L.lambdas_from_tree_Omegas(r, kappa, OH, OS)
    res = L.loop_phase_map(r, kappa, lamH, lamS, MU, **LOOP_RG)
    edges = solve_edges(res, map_kwargs=LOOP_RG)
    n, n_ok, n_partial, fails = check_edges(res, edges, LOOP_RG)
    print(f"{n} solved edges: {n_ok} match on both sides, {n_partial} with a second transition in the edge, "
          f"{len(fails)} failures {fails}")
    assert n > 0 and not fails


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"{name}: passed")
