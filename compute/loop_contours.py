"""Data for Figs. 33-36: the ground state of the RG-improved one-loop Higgs-singlet potential along an elliptical contour
in the tree-level (Omega_H, Omega_S) plane, at fixed r = mu_S^2 / mu_H^2 and kappa: along the contour, the quantum and
potential phase, the transitions between them, the ground-state energy E_0 = V_min - V(0), the position of the global
minimum, and the first three derivatives of E_0 projected on the unit tangent t of the contour, (t.grad)^n E_0 for
n = 1, 2, 3 (derivatives in Omega). A first-order transition is a jump in the first derivative, a second-order one a
jump in the second only.

How it is computed:
  - The global minimum at every point of the contour from the full search of loop_phase_map (as the phase maps).
  - The transitions along the contour, where the quantum or potential phase changes, by bisection in theta; first
    order where the ground state jumps across it.
  - The derivatives by 5-point central differences along t (step at most H_STEP_REL times the smaller semi-axis,
    in Omega). At each stencil point the minimum is solved for by Newton, starting from the global minimum of the centre point and keeping its kind (on
    the H axis, on the S axis, or off the axes, in (H^2, S^2)). This follows that branch smoothly also where the
    stencil reaches across a first-order transition (as a metastable minimum), so each point gets the derivatives
    of its own phase. Near a quantum transition the step shrinks (to a tenth of the distance), so that the stencil
    stays within the phase, and more points are added there, so that the curves reach it. The step is checked
    against half of it.

At one loop E_0 is not analytic at a second-order transition where H -> 0 in the mixed minimum (II -> III): the top,
W and Z become massless there, and their m^4 ln m^2 terms make E_0 contain H^4 ln H^2. So on the III side the third
derivative diverges like 1 / distance (about -0.14 / distance in Omega at r = 1, kappa = 0.5; dominated by the top),
and the second like ln(distance). This is converged in the step, and it runs off the panel.

Within a phase, the tree-level mass of a state in the loop (the light h-s eigenvalue or a Goldstone) can cross zero
at the minimum, as these are only loop-sized there. Its m^4 ln m^2 then gives the second derivative a log and the
third a 1/m^2 pole: the spikes in the third derivative (e.g. at theta ~ 0.12 and 0.29 in phase III of the r = 2
contour; the script prints where). This is the Goldstone (IR) problem of the fixed-order one-loop potential, which
resummation cures; it is not numerical. Where S -> 0 (III -> I)
nothing becomes massless, as S has no gauge or Yukawa couplings, and the derivatives stay finite on both sides.

Same physics and inputs as the phase maps (nearcrit/loop_maps.py: RG-improved, Lambda = 10 TeV).

Writes data/loop_contour_<name>.npz for each contour (about 40 s each; `--only NAME` for one).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the nearcrit package

import argparse

import numpy as np

from nearcrit import data_io, style
from nearcrit import singlet_loop as L
from nearcrit.loop_maps import (
    CUTOFF, KIND_TO_QUANTUM, KNOWN_NEW, LOOP_RG, MU, SM_COUPLINGS, ground_state_is_outer, potential_phase,
)
from nearcrit.singlet_tree import TRANSITION_ORDER

N_THETA = 1200
H_STEP_REL = 8e-3        # largest finite-difference step along the tangent, in units of the smaller semi-axis
N_BISECT = 20            # bisection steps for each transition along the contour (to ~5e-9 in theta)
SIDE_DTHETA = 1e-6       # how far from a transition its two sides are read off
NEAR_DTHETA = (4e-4, 3e-2)   # extra points this close to each transition (in theta), geometrically spaced,
NEAR_MIN_DOMEGA = 2e-4       # but none closer than this in Omega (where the stencil, h >= H_MIN, would cross it)
N_NEAR = 16
H_MIN = 5e-5             # smallest finite-difference step (below it the third difference is noisy)
JUMP_TOL = 0.02          # relative jump of the ground state (in tree-normalized fields) that makes a transition first order

# name: the panel (r, kappa) and the ellipse Omega = centre + semi * (cos, sin) theta, over theta_lim (theta = 0 to 2 pi
# unless it starts on a boundary). newton_iter: Newton steps for the minimum at the stencil points (default 40).
# r1_k05 (Fig. 33): around the loop-level triple point (Omega_H, Omega_S) ~ (0.9, 1.0), wide enough to miss phase d
#   (within ~0.05 of it) and to stay clear of NGS (Omega_H < 0.22). theta = 0 is close to the II -> III line, so the
#   contour starts a little before it.
# r2_k05_zoom (Fig. 34): through phase d, the radiatively generated mixed minimum-saddle pair of the zoom (0.5, 2.0).
#   It crosses the d wedge (I d and III d, split by the first-order line) at Omega_S ~ 1.04, where it is still wide,
#   and the thin I d strip above Omega_S ~ 1.006 on the way back, and goes round both corners on Omega_S ~ 1.006
#   (Omega_H ~ 0.85 and 0.895).
# rm1_km1 (Fig. 35): r = -1, kappa = -1 (Fig. 28) round the corner where the second-order line III c -> I c
#   (Omega_H ~ 0.98) meets the first-order line I -> III f (to (2, 0.42)), through III c, the radiatively generated
#   phase f (a mixed minimum-saddle pair) and the thin I f strip next to the first-order line, at least ~0.03 from the
#   edge of NGS. Phase III is held up by the loops only here, along a nearly flat direction (Omega_H Omega_S ~ 1 with
#   kappa < 0), so its minimum is far out (H, S up to ~20 |mu_H|) and E_0 and its derivatives there are up to
#   10^3-10^6 times those in phase I.
# rm05_km05_zoom (Fig. 36; newton_iter: the outer minimum lies in a narrow, nearly flat valley, where Newton converges
#   slowly; 40 steps left relative gradients up to ~1e-1 at the stencil points): r = -0.5, kappa = -0.5, panel (c) of
#   Fig. 30: across the first-order line III -> III' (the ground state jumps to a second, outer mixed minimum, phase h)
#   twice, through the thin III h strip on its III side (the outer minimum metastable), at least ~4e-4 in Omega_H
#   (~13 cells) from the edge of NGS. The bands run nearly along Omega_S, so the ellipse is about as tall as it is
#   wide; the outer minimum is ~100 times deeper.
CONTOURS = {
    "r1_k05": dict(r=1.0, kappa=0.5, centre=(0.9, 1.0), semi=(0.5, 0.5), theta_lim=(-np.pi / 8, 2 * np.pi - np.pi / 8)),
    "r2_k05_zoom": dict(r=2.0, kappa=0.5, centre=(0.87, 1.015), semi=(0.05, 0.03), theta_lim=(0.0, 2 * np.pi)),
    "rm1_km1": dict(r=-1.0, kappa=-1.0, centre=(1.25, 0.95), semi=(0.3, 0.2), theta_lim=(0.0, 2 * np.pi)),
    "rm05_km05_zoom": dict(r=-0.5, kappa=-0.5, centre=(0.5842, 1.770), semi=(0.0014, 0.0015),
                           theta_lim=(0.0, 2 * np.pi), newton_iter=400),
}


def closed(c):
    """Whether theta_lim covers the whole ellipse (so distances in theta wrap round)."""
    return np.isclose(c["theta_lim"][1] - c["theta_lim"][0], 2 * np.pi)


# --- The contour ----------------------------------------------------------------------------------------------------
def ellipse(c, theta):
    """(Omega_H, Omega_S) and the unit tangent (t_H, t_S) at theta."""
    (x0, y0), (a, b) = c["centre"], c["semi"]
    OH, OS = x0 + a * np.cos(theta), y0 + b * np.sin(theta)
    tH, tS = -a * np.sin(theta), b * np.cos(theta)
    norm = np.hypot(tH, tS)
    return OH, OS, tH / norm, tS / norm


def global_minima(r, kappa, OH, OS, new_letters):
    """The full search at the points (OH, OS): quantum phase, potential-phase letter, kind of the ground state, its
    position (x, y) = (H^2, S^2) and in tree-normalized fields."""
    lamH, lamS = L.lambdas_from_tree_Omegas(r, kappa, OH, OS)
    res = L.loop_phase_map(r, kappa, lamH, lamS, MU, **LOOP_RG)
    n = len(OH)
    Q = np.empty(n, dtype=object)
    P = np.empty(n, dtype=object)
    for i in range(n):
        st, kind, run = res["structure"][i], res["gs_kind"][i], res["runaway"][i]
        if (st is not None and not run and kind != "" and L.TREE_SIGNATURES.get(st + (kind,)) is None
                and st not in KNOWN_NEW):
            # a structure met nowhere else: spurious extrema found right next to a boundary (in the bisection)
            Q[i], P[i] = KIND_TO_QUANTUM[kind], "?"
            continue
        Q[i], P[i] = potential_phase(st, kind, run, new_letters, order=res["order"][i])
        if ground_state_is_outer(res, i):
            Q[i] += "'"
    xbar, ybar, _ = L._scales(res["model"])
    Hb, Sb = res["gs_Hbar"], res["gs_Sbar"]
    return dict(Q=Q, P=P, kind=res["gs_kind"].astype(str), Hbar=Hb, Sbar=Sb, x=Hb**2 * xbar, y=Sb**2 * ybar,
                index_ok=res["index_ok"])


def find_transitions(c, theta, g, new_letters):
    """Where the (quantum, potential) phase changes between neighbouring contour points, located by bisection in
    theta, and classified as on the phase maps (line_edges in nearcrit/loop_maps.py): a change of the potential phase
    within one quantum phase is a potential transition (purple); a change of the quantum phase is first order where
    the ground state jumps (or the tree-level table says so), second order otherwise.

    Within ~1e-8 of a boundary the extremum search can return spurious extrema (along the nearly flat direction),
    so the phases on the two sides and the jump of the ground state are read off SIDE_DTHETA away from it."""
    lab = list(zip(g["Q"], g["P"]))
    idx = [i for i in range(len(theta) - 1) if lab[i] != lab[i + 1]]
    lo, hi = theta[idx].copy(), theta[np.array(idx) + 1].copy()
    lab_lo = [lab[i] for i in idx]
    for _ in range(N_BISECT):
        mid = 0.5 * (lo + hi)
        OH, OS, _, _ = ellipse(c, mid)
        gm = global_minima(c["r"], c["kappa"], OH, OS, new_letters)
        same = np.array([(gm["Q"][k], gm["P"][k]) == lab_lo[k] for k in range(len(mid))])
        lo, hi = np.where(same, mid, lo), np.where(same, hi, mid)
    th_c = 0.5 * (lo + hi)
    sides = {}
    for side, sgn in (("lo", -1), ("hi", 1)):
        OH, OS, _, _ = ellipse(c, th_c + sgn * SIDE_DTHETA)
        sides[side] = global_minima(c["r"], c["kappa"], OH, OS, new_letters)
    out = []
    for k, i in enumerate(idx):
        a = {key: v[k] for key, v in sides["lo"].items()}
        b = {key: v[k] for key, v in sides["hi"].items()}
        if (a["Q"], a["P"]) != lab[i] or (b["Q"], b["P"]) != lab[i + 1]:
            print(f"  warning: at theta = {th_c[k]:.6f} the sides are {(a['Q'], a['P'])}, {(b['Q'], b['P'])}, "
                  f"not {lab[i]}, {lab[i + 1]}")
        ga, gb = np.array([a["Hbar"], a["Sbar"]]), np.array([b["Hbar"], b["Sbar"]])
        jump = np.hypot(*(ga - gb)) > JUMP_TOL * max(np.hypot(*ga), np.hypot(*gb))
        (Qa, Pa), (Qb, Pb) = lab[i], lab[i + 1]
        if Qa == Qb or (Qa.rstrip("'") == Qb.rstrip("'") and not jump):
            kind = "potential"
        else:
            table = TRANSITION_ORDER.get(frozenset((L.KIND_TO_BRANCH[g["kind"][i]],
                                                    L.KIND_TO_BRANCH[g["kind"][i + 1]])))
            kind = "first" if jump or table == "first" else "second"
        OH, OS, _, _ = ellipse(c, th_c[k])
        out.append(dict(theta=th_c[k], width=hi[k] - lo[k], OH=float(OH), OS=float(OS), kind=kind,
                        before=lab[i], after=lab[i + 1], jump=float(np.hypot(*(ga - gb)))))
    return out


# --- E_0 and its derivatives along the tangent ----------------------------------------------------------------------
def branch_energies(r, kappa, OH, OS, kind, x0, y0, n_iter=40):
    """E_0 = V - V(0) at the minimum of the given kind ('H', 'S' or 'M') nearest to (x0, y0), at each point
    (OH, OS), by Newton on the gradient in (x, y) = (H^2, S^2): along the axis for 'H' and 'S', in the plane for
    'M'. Also the largest remaining gradient relative to the tree scale."""
    lamH, lamS = L.lambdas_from_tree_Omegas(r, kappa, OH, OS)
    Pm = L.make_model(r, kappa, lamH, lamS, MU, n_goldstone=LOOP_RG["n_goldstone"],
                      n_goldstone_S=LOOP_RG["n_goldstone_S"])
    Pm = L.attach_rg(Pm, SM_COUPLINGS, CUTOFF**2)
    xbar, ybar, Vs = L._scales(Pm)
    x, y = np.array(x0, dtype=float), np.array(y0, dtype=float)
    onH, onS, mixed = kind == "H", kind == "S", kind == "M"
    x[onS], y[onH] = 0.0, 0.0
    for _ in range(n_iter):
        gx, gy = L.gradient(x, y, Pm)
        ex, ey = 1e-6 * xbar, 1e-6 * ybar
        gxp, gyp = L.gradient(x + ex, y, Pm)
        gxm, gym = L.gradient(x - ex, y, Pm)
        gxq, gyq = L.gradient(x, y + ey, Pm)
        gxn, gyn = L.gradient(x, y - ey, Pm)
        Jxx, Jyx = (gxp - gxm) / (2 * ex), (gyp - gym) / (2 * ex)
        Jxy, Jyy = (gxq - gxn) / (2 * ey), (gyq - gyn) / (2 * ey)
        dx, dy = np.zeros_like(x), np.zeros_like(y)
        dx[onH] = -gx[onH] / Jxx[onH]
        dy[onS] = -gy[onS] / Jyy[onS]
        det = Jxx * Jyy - Jxy * Jyx
        dx[mixed] = (-(Jyy * gx - Jxy * gy) / det)[mixed]
        dy[mixed] = (-(-Jyx * gx + Jxx * gy) / det)[mixed]
        # no step longer than half the tree scale
        s = np.minimum(1.0, 0.5 / np.maximum(np.hypot(dx / xbar, dy / ybar), 1e-300))
        x, y = x + s * dx, y + s * dy
        if n_iter > 40 and np.nanmax(np.hypot(s * dx / xbar, s * dy / ybar)) < 1e-13:   # converged: stop early
            break
    gx, gy = L.gradient(x, y, Pm)
    resid = np.where(onH, np.abs(gx) * xbar, np.where(onS, np.abs(gy) * ybar, np.hypot(gx * xbar, gy * ybar))) / Vs
    E = L.potential(x, y, Pm) - L.potential(0.0, 0.0, Pm)
    return E, x, y, resid


def tangent_derivatives(c, OH, OS, tH, tS, g, h):
    """E_0 and (t.grad)^n E_0, n = 1, 2, 3, by 5-point central differences with step h (one per point), each point
    on the branch of its own global minimum."""
    n = len(OH)
    h = np.broadcast_to(np.asarray(h, dtype=float), (n,))
    steps = np.array([-2, -1, 0, 1, 2])[:, None] * h[None, :]
    sH = (OH[None, :] + steps * tH[None, :]).ravel()
    sS = (OS[None, :] + steps * tS[None, :]).ravel()
    tile = lambda v: np.tile(v, len(steps))
    E, x, y, resid = branch_energies(c["r"], c["kappa"], sH, sS, tile(g["kind"]), tile(g["x"]), tile(g["y"]),
                                     n_iter=c.get("newton_iter", 40))
    Em2, Em1, E0, Ep1, Ep2 = E.reshape(5, n)
    d1 = (Em2 - 8 * Em1 + 8 * Ep1 - Ep2) / (12 * h)
    d2 = (-Em2 + 16 * Em1 - 30 * E0 + 16 * Ep1 - Ep2) / (12 * h**2)
    d3 = (-Em2 + 2 * Em1 - 2 * Ep1 + Ep2) / (2 * h**3)
    return dict(E=E0, d1=d1, d2=d2, d3=d3, resid=resid.reshape(5, n).max(axis=0),
                x0=x.reshape(5, n)[2], y0=y.reshape(5, n)[2])


def tree_mass_zeros(c, theta, OH, OS, g):
    """Where a tree-level mass in the loop (the two h-s eigenvalues, or the Goldstone masses of H and S, at the running
    couplings) crosses zero at the ground state within a phase. There the m^4 ln m^2 of the one-loop potential makes
    the second derivative of E_0 log-divergent and the third a 1/m^2 pole: the Goldstone (IR) problem of the
    fixed-order potential, which resummation cures. As (theta, name) pairs, interpolated linearly between the points."""
    lamH, lamS = L.lambdas_from_tree_Omegas(c["r"], c["kappa"], OH, OS)
    Pm = L.attach_rg(L.make_model(c["r"], c["kappa"], lamH, lamS, MU, n_goldstone=LOOP_RG["n_goldstone"],
                                  n_goldstone_S=LOOP_RG["n_goldstone_S"]), SM_COUPLINGS, CUTOFF**2)
    x, y = g["x"], g["y"]
    Pf, xi2, _ = L._rg_fixed_model(Pm, L._rg_t(x, y, Pm))
    _, _, _, s, _, R = L._scalar_block(xi2 * x, y, Pf)
    masses = {"heavy scalar": s + R, "light scalar": s - R, "Goldstone of H": L._goldstone_mass(xi2 * x, y, Pf),
              "Goldstone of S": L._goldstone_mass_S(xi2 * x, y, Pf)}
    out = []
    for name, m2 in masses.items():
        for i in np.flatnonzero(np.diff(np.sign(m2)) != 0):
            if (g["Q"][i], g["P"][i], g["kind"][i]) == (g["Q"][i + 1], g["P"][i + 1], g["kind"][i + 1]):
                u = m2[i] / (m2[i] - m2[i + 1])          # linear interpolation of m^2 between the two points
                out.append((float(theta[i] + u * (theta[i + 1] - theta[i])), name))
    return sorted(out)


def contour_data(name, c):
    """Everything along contour c (see the module docstring): the points, the transitions between them, the
    ground state and the projected derivatives of E_0."""
    theta = np.linspace(*c["theta_lim"], N_THETA)
    OH, OS, tH, tS = ellipse(c, theta)
    new_letters = {}
    g = global_minima(c["r"], c["kappa"], OH, OS, new_letters)
    print(f"{name}: {N_THETA} points; failing the index check: {int((~g['index_ok']).sum())}; "
          f"kinds: {dict(zip(*np.unique(g['kind'], return_counts=True)))}", flush=True)
    trans = find_transitions(c, theta, g, new_letters)

    # More points towards each transition (geometrically closer), so that the curves reach it
    def offsets(th):
        speed = np.hypot(c["semi"][0] * np.sin(th), c["semi"][1] * np.cos(th))
        return np.geomspace(max(NEAR_DTHETA[0], NEAR_MIN_DOMEGA / speed), NEAR_DTHETA[1], N_NEAR)
    extra = np.concatenate([t["theta"] + sgn * offsets(t["theta"]) for t in trans for sgn in (-1, 1)])
    if closed(c):    # extra points past either end come round to the other end
        extra = theta[0] + np.mod(extra - theta[0], 2 * np.pi)
    extra = extra[(extra > theta[0]) & (extra < theta[-1])]
    theta = np.sort(np.concatenate((theta, extra)))
    OH, OS, tH, tS = ellipse(c, theta)
    g = global_minima(c["r"], c["kappa"], OH, OS, new_letters)
    # each point must be in the phase of its segment between transitions; drop any that is not (spurious extrema)
    cuts = np.array([t["theta"] for t in trans])
    seg = np.searchsorted(cuts, theta)
    seg_lab = [t["before"] for t in trans] + [trans[-1]["after"]]
    ok = np.array([(g["Q"][i], g["P"][i]) == seg_lab[seg[i]] for i in range(len(theta))])
    if not ok.all():
        print(f"  dropped {int((~ok).sum())} points not in the phase of their segment, at theta = "
              f"{np.round(theta[~ok], 6)}")
    theta, OH, OS, tH, tS = theta[ok], OH[ok], OS[ok], tH[ok], tS[ok]
    g = {k: v[ok] for k, v in g.items()}

    # The step: at most a tenth of the distance (in Omega) to the nearest quantum transition, so that the stencil
    # stays within the phase (E_0 need not be analytic at a second-order transition; see the docstring)
    quantum = np.array([t["theta"] for t in trans if t["kind"] != "potential"])
    speed = np.hypot(*(np.array(c["semi"])[:, None] * np.array([np.sin(theta), np.cos(theta)])))
    dtheta = np.abs(theta[:, None] - quantum[None, :])
    if closed(c):
        dtheta = np.minimum(dtheta, 2 * np.pi - dtheta)
    dist = np.min(dtheta, axis=1) * speed if quantum.size else np.full(theta.shape, np.inf)
    h = np.clip(dist / 10, H_MIN, H_STEP_REL * min(c["semi"]))
    d = tangent_derivatives(c, OH, OS, tH, tS, g, h)
    d_half = tangent_derivatives(c, OH, OS, tH, tS, g, h / 2)
    moved = np.hypot((np.sqrt(np.abs(d["x0"])) - np.sqrt(g["x"])), (np.sqrt(np.abs(d["y0"])) - np.sqrt(g["y"])))
    print(f"  {len(theta)} points; Newton: largest relative gradient left {np.nanmax(d['resid']):.1e}; "
          f"largest move of the centre minimum {np.nanmax(moved):.1e}")
    for k in ("d1", "d2", "d3"):
        rel = np.abs(d[k] - d_half[k]) / np.maximum(np.abs(d[k]), 1.0)
        print(f"  {k}: step h vs h/2, largest relative difference {np.nanmax(rel):.1e} "
              f"(at theta = {theta[np.nanargmax(rel)]:.4f})")
    return dict(theta=theta, OH=OH, OS=OS, tH=tH, tS=tS, new_letters=new_letters, transitions=trans, dist=dist,
                crossings=tree_mass_zeros(c, theta, OH, OS, g),
                **{k: g[k] for k in ("Q", "P", "kind", "Hbar", "Sbar", "x", "y")},
                E=d["E"], d1=d["d1"], d2=d["d2"], d3=d["d3"], d3_half=d_half["d3"])


def transition_ends(c, d):
    """E_0, Hbar and Sbar of each branch exactly at each quantum transition, from the nearest point on that side
    (Newton on its branch), so that the curves of E_0 and of the position end on the transition line: the nearest
    points are ~1e-3 away in theta, and E_0 can be very steep there (e.g. next to a jump to a far minimum), which
    otherwise looks like a step although E_0 is continuous. As {key: [(theta, side, value)]}, side "before"/"after"."""
    theta = d["theta"]
    out = {"E": [], "Hbar": [], "Sbar": []}
    for t in d["transitions"]:
        if t["kind"] == "potential":
            continue
        tc = t["theta"]
        OH, OS, _, _ = ellipse(c, np.array([tc]))
        lamH, lamS = L.lambdas_from_tree_Omegas(c["r"], c["kappa"], OH, OS)
        for side, sel in (("before", theta < tc), ("after", theta > tc)):
            idx = np.flatnonzero(sel)
            if idx.size == 0:
                continue
            i = idx[-1] if side == "before" else idx[0]
            E, x, y, _ = branch_energies(c["r"], c["kappa"], OH, OS, np.array([d["kind"][i]]), np.array([d["x"][i]]),
                                         np.array([d["y"][i]]), n_iter=c.get("newton_iter", 40))
            out["E"].append((tc, side, float(E[0])))
            out["Hbar"].append((tc, side, float(np.sqrt(max(x[0], 0.0) * lamH[0]))))           # xbar = mu_H^2 / lambda_H = 1 / lambda_H
            out["Sbar"].append((tc, side, float(np.sqrt(max(y[0], 0.0) * lamS[0] / abs(c["r"])))))   # ybar = |mu_S^2| / lambda_S
    return out



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--only", choices=list(CONTOURS), help="one contour only")
    args = parser.parse_args()
    for name in ([args.only] if args.only else CONTOURS):
        c = CONTOURS[name]
        d = contour_data(name, c)
        # recomputed from the final points along the contour (contour_data's are from before the extra points)
        d["crossings"] = tree_mass_zeros(c, d["theta"], d["OH"], d["OS"], d)
        d["ends"] = transition_ends(c, d)
        out = style.DATA_DIR / f"loop_contour_{name}.npz"
        data_io.save(out, dict(contour=c, data=d))
        print("Saved", out)
