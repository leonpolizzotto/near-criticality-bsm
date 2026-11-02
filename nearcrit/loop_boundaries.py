"""Phase boundaries of a loop_phase_map grid, solved from the conditions that define them.

The grid only says WHICH condition holds between two neighbouring points; WHERE it holds is then solved for along
the edge between them, with every other parameter fixed (lambda_H varies across a row, lambda_S down a column; or, in the (kappa,
lambda_S) maps, kappa across a row). The couplings are interpolated linearly along each edge. In
terms of the fields x = H^2, y = S^2, the conditions are:

  potential phase transitions (an extremum appears, disappears or changes type: a mass vanishes at an extremum)
    origin_H   origin, H direction:  dV/dx(0, 0) = 0                       (muH2^(1) = 0)
    origin_S   origin, S direction:  dV/dy(0, 0) = 0                       (muS2^(1) = 0)
    trans_H    H-axis extremum, S direction:  dV/dx(x, 0) = dV/dy(x, 0) = 0  (OmegaH^(1) = 1; an off-axis
               extremum leaves or joins the axis)
    trans_S    the mirror on the S axis                                    (OmegaS^(1) = 1)
    fold_H     two H-axis extrema merge:  dV/dx(x, 0) = d^2V/dx^2(x, 0) = 0
    fold_S     the mirror on the S axis
    fold_M     two off-axis extrema merge:  dV/dx = dV/dy = det Hess = 0
    swap       two minima that are not the ground state exchange depth order, with the extrema otherwise unchanged:
               grad V(a) = grad V(b) = 0, V(a) = V(b) (also between the local minima where there is no ground state)
    cut_H      an H-axis extremum crosses the cutoff circle:  dV/dx(cutoff^2, 0) = 0
    cut_S      the mirror on the S axis
    cut_M      an off-axis extremum crosses the cutoff circle:  dV/dx = dV/dy = 0 at x + y = cutoff^2
  first-order quantum phase transitions
    fopt       the ground state jumps between two minima a, b:  grad V(a) = grad V(b) = 0, V(a) = V(b)
  edges of the regions without a phase
    ngs        a valley of V crossing the cutoff circle starts to fall outward: dV/d(rho^2) = 0 at a local minimum of V
               along the circle (the smallest such slope, L.boundary_minimum_slope); beyond it V on the disk has a
               local minimum on its edge, and there is no ground state below the cutoff
    np         a coupling reaches 4 pi one RG step (dt) above the largest scale used, mu^2 = M_t^2 + cutoff^2
               (t_np = t_cut + dt): within that step the frozen couplings bend V just inside the cutoff

A second-order quantum transition is the potential transition in which the ground state itself loses a direction,
so it needs no condition of its own. The cut_* conditions and ngs are where the cutoff enters; they move with it.
With RG improvement, moving along an edge changes the couplings at M_t, so the
running is redone at every step (model_with_lambdas).

solve_edges(res) returns the solved crossing point of every edge where one condition is identified, polylines()
joins them into curves (lines meeting at junctions), and snap_paths() moves the vertices of filled contours onto the
same points, so that fills and lines agree. tests/test_loop_boundaries.py checks the solutions against
loop_phase_map just either side of each solved point.
"""
import numpy as np

from nearcrit import singlet_loop as L

EVENTS = ("origin_H", "origin_S", "trans_H", "trans_S", "fold_H", "fold_S", "fold_M", "fopt", "swap",
          "cut_H", "cut_S", "cut_M", "ngs", "np")
_PAIR_EVENTS = ("fopt", "swap")    # conditions on a pair of minima with equal V
# conditions on a single function of the couplings, solved by bisection along the edge: the edges of NGS and NP,
# and extrema crossing the cutoff circle (which can run out very fast, e.g. along a nearly flat valley for kappa < 0,
# too fast for Newton's method)
_BRACKET_EVENTS = ("ngs", "np", "cut_H", "cut_S", "cut_M")
NONE, FAILED, UNCHANGED = -1, -2, -3   # event codes: no single condition identified / not converged / no change
_NCOORD = {"0": 0, "H": 1, "S": 1, "M": 2}
_S_SLACK = 0.05                 # accept solutions this far (in edge lengths) outside the edge, then clip


# ---------------------------------------------------------------------------------------------------------------------
# Models with other couplings
# ---------------------------------------------------------------------------------------------------------------------

def model_with_lambdas(P, lamH, lamS, kappa=None):
    """P (1D coupling arrays) with lambda_H, lambda_S (and kappa, if given) replaced. With RG improvement the running
    is redone from the new couplings at M_t, with the same SM couplings, range and step as P's table."""
    Q = P._replace(lamH=np.asarray(lamH, dtype=float), lamS=np.asarray(lamS, dtype=float))
    if kappa is not None:
        Q = Q._replace(kappa=np.asarray(kappa, dtype=float))
    if P.rg is None:
        return Q
    Q = Q._replace(rg=None, pidx=None)
    table = L.RGTable(Q, P.rg.sm, P.rg.t_max_arg, dt=P.rg.dt, coupling_max=P.rg.coupling_max)
    return Q._replace(rg=table, pidx=np.arange(Q.muH2.size))


# ---------------------------------------------------------------------------------------------------------------------
# Which condition holds on each edge
# ---------------------------------------------------------------------------------------------------------------------

def _minima(ext, has_origin_min):
    """The minima of one grid point as (kind, coords) with absolute coords: () for the origin, (x,) on the H axis,
    (y,) on the S axis, (x, y) off the axes."""
    out = [("0", ())] if has_origin_min else []
    out += [("H", (x,)) for x, t in ext["H"] if t == "min"]
    out += [("S", (y,)) for y, t in ext["S"] if t == "min"]
    out += [("M", (x, y)) for x, y, t in ext["M"] if t == "min"]
    return out


def _closest(cands, kind, coords, exclude=None):
    """The candidate minimum of the given kind closest (in log fields) to coords, or None."""
    best, dbest = None, np.inf
    for c in cands:
        if c[0] != kind or (exclude is not None and c is exclude):
            continue
        d = np.sum((np.log(np.asarray(c[1])) - np.log(np.asarray(coords))) ** 2) if coords else 0.0
        if d < dbest:
            best, dbest = c, d
    return best


def _coords(ext, kind, index):
    """Absolute coords of the extremum (kind, index) of one grid point (see loop_phase_map's `order`)."""
    if kind == "0":
        return ()
    if kind == "H":
        return (ext["H"][index][0],)
    if kind == "S":
        return (ext["S"][index][0],)
    return (ext["M"][index][0], ext["M"][index][1])


def _gs_coords(kind, Hbar, Sbar, xb, yb):
    return {"0": (), "H": (Hbar**2 * xb,), "S": (Sbar**2 * yb,), "M": (Hbar**2 * xb, Sbar**2 * yb)}[kind]


def _classify(res, a, b, jump, scales):
    """(event, seed) for the edge between flat grid points a and b, or (None, None). The seed is the list of
    unknown field coordinates (absolute x or y) to start from; for fopt it is (kind_a, coords_a, kind_b, coords_b)."""
    st, ext = res["structure"].ravel(), res["extrema"].ravel()
    run, nonp = res["runaway"].ravel(), res["nonperturbative"].ravel()
    if nonp[a] != nonp[b]:
        return "np", []
    if st[a] is None or st[b] is None or nonp[a] or nonp[b]:
        return None, None
    if run[a] != run[b]:
        return "ngs", []
    xbar, ybar = scales
    kind = res["gs_kind"].ravel()
    if jump:
        Hb, Sb = res["gs_Hbar"].ravel(), res["gs_Sbar"].ravel()
        ka, kb = kind[a], kind[b]
        ga = _gs_coords(ka, Hb[a], Sb[a], xbar[a], ybar[a])
        gb = _gs_coords(kb, Hb[b], Sb[b], xbar[b], ybar[b])
        for base, (k1, g1), (k2, g2) in ((a, (ka, ga), (kb, gb)), (b, (kb, gb), (ka, ga))):
            cands = _minima(ext[base], ext[base]["origin"] == "min")
            m1 = _closest(cands, k1, g1)
            m2 = _closest(cands, k2, g2, exclude=m1)
            if m1 is not None and m2 is not None:
                return "fopt", (m1[0], m1[1], m2[0], m2[1])
        return None, None
    if st[a] == st[b]:
        # same extrema: two minima may have exchanged depth order
        oa, ob = res["order"].ravel()[a], res["order"].ravel()[b]
        if oa == ob or len(oa) != len(ob) or set(oa) != set(ob):
            return None, None
        diff = [k for k in range(len(oa)) if oa[k] != ob[k]]
        if len(diff) != 2:
            return None, None
        ev = "fopt" if diff[0] == 0 and not run[a] else "swap"
        (k1, i1), (k2, i2) = oa[diff[0]], oa[diff[1]]
        return ev, (k1, _coords(ext[a], k1, i1), k2, _coords(ext[a], k2, i2))
    mu1 = (res["muH2_1"].ravel()[[a, b]], res["muS2_1"].ravel()[[a, b]])
    return structural_event(st[a], st[b], ext[a], ext[b], mu1)


def structural_event(st_a, st_b, ext_a, ext_b, mu1):
    """(event, seed) for a change of extremum structure between two parameter points: which single condition
    (EVENTS) is crossed between them, or (None, None) if it is not one of them (e.g. two at once). mu1 is
    ((muH2_1 at a, at b), (muS2_1 at a, at b))."""
    ext = {0: ext_a, 1: ext_b}
    a, b = 0, 1
    (oa, ha, sa, ma), (ob, hb, sb, mb) = st_a, st_b
    if oa != ob:
        mH, mS = np.sign(mu1[0]), np.sign(mu1[1])
        if mH[0] != mH[1] and mS[0] == mS[1]:
            return "origin_H", []
        if mS[0] != mS[1] and mH[0] == mH[1]:
            return "origin_S", []
        return None, None
    if sa == sb and ma == mb and abs(len(ha) - len(hb)) == 2:
        more = a if len(ha) > len(hb) else b
        q = np.array([x for x, _ in ext[more]["H"]])
        k = np.argmin(np.diff(np.log(q)))
        return "fold_H", [np.sqrt(q[k] * q[k + 1])]
    if ha == hb and ma == mb and abs(len(sa) - len(sb)) == 2:
        more = a if len(sa) > len(sb) else b
        q = np.array([y for y, _ in ext[more]["S"]])
        k = np.argmin(np.diff(np.log(q)))
        return "fold_S", [np.sqrt(q[k] * q[k + 1])]
    if ha == hb and sa == sb and abs(len(ma) - len(mb)) == 2:
        more = a if len(ma) > len(mb) else b
        roots = ext[more]["M"]
        best, pair = np.inf, None
        for p in range(len(roots)):
            for q in range(p + 1, len(roots)):
                if {roots[p][2], roots[q][2]} in ({"min", "saddle"}, {"max", "saddle"}):
                    d = np.hypot(np.log(roots[p][0] / roots[q][0]), np.log(roots[p][1] / roots[q][1]))
                    if d < best:
                        best, pair = d, (roots[p], roots[q])
        if pair is None:
            return None, None
        return "fold_M", [np.sqrt(pair[0][0] * pair[1][0]), np.sqrt(pair[0][1] * pair[1][1])]
    if sa == sb and len(ha) == len(hb) and abs(len(ma) - len(mb)) == 1:
        diff = [k for k in range(len(ha)) if ha[k] != hb[k]]
        if len(diff) == 1:
            k = diff[0]
            return "trans_H", [np.sqrt(ext[a]["H"][k][0] * ext[b]["H"][k][0])]
    if ha == hb and len(sa) == len(sb) and abs(len(ma) - len(mb)) == 1:
        diff = [k for k in range(len(sa)) if sa[k] != sb[k]]
        if len(diff) == 1:
            k = diff[0]
            return "trans_S", [np.sqrt(ext[a]["S"][k][0] * ext[b]["S"][k][0])]
    # one extremum more on one side, the outermost, and nothing else changes: it crosses the cutoff circle
    if sa == sb and ma == mb and abs(len(ha) - len(hb)) == 1 and min(ha, hb, key=len) == max(ha, hb, key=len)[:-1]:
        return "cut_H", []
    if ha == hb and ma == mb and abs(len(sa) - len(sb)) == 1 and min(sa, sb, key=len) == max(sa, sb, key=len)[:-1]:
        return "cut_S", []
    if ha == hb and sa == sb and abs(len(ma) - len(mb)) == 1:
        more = a if len(ma) > len(mb) else b
        roots = ext[more]["M"]
        for outer in sorted(range(len(roots)), key=lambda k: -(roots[k][0] + roots[k][1])):
            rest = tuple(sorted(t for k, (_, _, t) in enumerate(roots) if k != outer))
            if rest == (mb if more == a else ma):
                x, y, _ = roots[outer]
                return "cut_M", [np.arctan2(np.sqrt(y), np.sqrt(x))]
    return None, None


# ---------------------------------------------------------------------------------------------------------------------
# The conditions, as residuals in (unknown field coordinates..., s)
# ---------------------------------------------------------------------------------------------------------------------

def _point(kind, zc, xb, yb):
    """(x, y) of an extremum of the given kind from its log coordinates zc (columns)."""
    zero = np.zeros(xb.shape)
    if kind == "0":
        return zero, zero
    if kind == "H":
        return xb * np.exp(zc[:, 0]), zero
    if kind == "S":
        return zero, yb * np.exp(zc[:, 0])
    return xb * np.exp(zc[:, 0]), yb * np.exp(zc[:, 1])


def _stationarity(kind, x, y, P, xb, yb, vs):
    if kind == "0":
        return []
    gx, gy = L.gradient(x, y, P)
    if kind == "H":
        return [xb * gx / vs]
    if kind == "S":
        return [yb * gy / vs]
    return [xb * gx / vs, yb * gy / vs]


def _residual(event, kinds=None, cutoff=None):
    def F(z, P, xb, yb, vs):
        if event == "origin_H":
            gx, _ = L.gradient(0.0 * xb, 0.0 * xb, P)
            return (2.0 * gx / np.abs(P.muH2))[:, None]
        if event == "origin_S":
            _, gy = L.gradient(0.0 * xb, 0.0 * xb, P)
            return (2.0 * gy / np.abs(P.muS2))[:, None]
        if event in ("trans_H", "fold_H"):
            x = xb * np.exp(z[:, 0])
            gx, gy = L.gradient(x, 0.0 * x, P)
            other = gy if event == "trans_H" else x * L.d2V_dx2_H_axis(x, P)
            return np.column_stack([xb * gx / vs, xb * other / vs])
        if event in ("trans_S", "fold_S"):
            y = yb * np.exp(z[:, 0])
            gx, gy = L.gradient(0.0 * y, y, P)
            other = gx if event == "trans_S" else y * L.d2V_dy2_S_axis(y, P)
            return np.column_stack([yb * gy / vs, yb * other / vs])
        if event == "fold_M":
            x, y = xb * np.exp(z[:, 0]), yb * np.exp(z[:, 1])
            gx, gy = L.gradient(x, y, P)
            Hxx, Hxy, Hyy = L.hessian_xy(x, y, P)
            det = (xb * xb * Hxx) * (yb * yb * Hyy) - (xb * yb * Hxy) ** 2
            return np.column_stack([xb * gx / vs, yb * gy / vs, det / vs**2])
        # fopt
        ka, kb = kinds
        na = _NCOORD[ka]
        xa, ya = _point(ka, z[:, :na], xb, yb)
        xb_, yb_ = _point(kb, z[:, na:na + _NCOORD[kb]], xb, yb)
        cols = (_stationarity(ka, xa, ya, P, xb, yb, vs) + _stationarity(kb, xb_, yb_, P, xb, yb, vs)
                + [(L.potential(xa, ya, P) - L.potential(xb_, yb_, P)) / vs])
        return np.column_stack(cols)
    return F


# Convergence tolerance on the scaled residuals. The second-derivative conditions (folds) are computed by finite
# differences of the gradient, whose noise floor with RG improvement is ~1e-8; a residual of 1e-6 there moves the
# fold by ~1e-6 relative, far below anything visible.
_TOL = dict(origin_H=1e-9, origin_S=1e-9, trans_H=1e-9, trans_S=1e-9, fold_H=1e-6, fold_S=1e-6, fold_M=1e-6,
            fopt=1e-9, swap=1e-9)


def _is_min(kind, x, y, P):
    """Whether the extremum of the given kind at (x, y) is a minimum (both curvatures positive)."""
    gx, gy = L.gradient(x, y, P)
    if kind == "0":
        return (gx > 0) & (gy > 0)
    if kind == "H":
        return (L.d2V_dx2_H_axis(x, P) > 0) & (gy > 0)
    if kind == "S":
        return (L.d2V_dy2_S_axis(y, P) > 0) & (gx > 0)
    Hxx, Hxy, Hyy = L.hessian_xy(x, y, P)
    return (Hxx * Hyy - Hxy**2 > 0) & (Hxx + Hyy > 0)


def _solve_batch(P, scales, A, B, event, seeds, kinds=None, cutoff=None, chunk=1500, debug=None,
                 s_starts=(0.5, 0.05, 0.95, 0.25, 0.75)):
    """Solve one condition on a batch of edges (a, b). Returns s (fraction along the edge; nan if not converged).
    Newton starts at s = s_starts[0]; edges where it fails are started again from the others in turn (e.g. a fold
    near one end of the edge, where from the middle the pair no longer exists)."""
    xbar, ybar, Vs = scales
    F0 = _residual(event, kinds, cutoff)
    s_out = np.full(A.size, np.nan)
    for lo in range(0, A.size, chunk):
        idx_all = np.arange(lo, min(lo + chunk, A.size))
        todo = idx_all
        for s0 in s_starts:
            if todo.size == 0:
                break
            a, b = A[todo], B[todo]
            base = P.take(a)
            lH, lS, lK = P.lamH[a], P.lamS[a], P.kappa[a]
            dH, dS, dK = P.lamH[b] - lH, P.lamS[b] - lS, P.kappa[b] - lK
            xb, yb, vs = xbar[a], ybar[a], Vs[a]
            z0 = (np.column_stack([seeds[todo], np.full(a.size, s0)]) if seeds.shape[1]
                  else np.full((a.size, 1), s0))

            def F(z, idx):
                s = z[:, -1]
                Pz = model_with_lambdas(base.take(idx), lH[idx] + s * dH[idx], lS[idx] + s * dS[idx],
                                        lK[idx] + s * dK[idx])
                return F0(z, Pz, xb[idx], yb[idx], vs[idx])

            with np.errstate(all="ignore"):
                z, f = L._newton_fd(F, z0, n_iter=40, max_step=0.5)
                s = z[:, -1]
                ok = (np.isfinite(f).all(axis=1) & (np.abs(f).max(axis=1) < _TOL[event])
                      & (s > -_S_SLACK) & (s < 1 + _S_SLACK))
                if event in _PAIR_EVENTS and ok.any():
                    # both solved points must still be distinct minima
                    ka, kb = kinds
                    na = _NCOORD[ka]
                    Pz = model_with_lambdas(base, lH + s * dH, lS + s * dS, lK + s * dK)
                    xa, ya = _point(ka, z[:, :na], xb, yb)
                    xq, yq = _point(kb, z[:, na:na + _NCOORD[kb]], xb, yb)
                    distinct = (ka != kb) | (np.abs(np.log((xa + ya) / (xq + yq))) > 1e-3)
                    ok &= distinct & _is_min(ka, xa, ya, Pz) & _is_min(kb, xq, yq, Pz)
            s_out[todo[ok]] = np.clip(s[ok], 0.0, 1.0)
            if debug is not None:
                debug.append((z, f, ok))
            todo = todo[~ok]
    return s_out


def _tangential_root(P, cutoff, seed, n_scan=91, n_bisect=40):
    """The direction theta on the cutoff circle nearest to seed where V is stationary along the circle (dV/dx = dV/dy
    there, theta strictly between the axes); nan where there is none."""
    t = np.linspace(0.0, 0.5 * np.pi, n_scan)[1:-1]
    c2, s2 = np.cos(t)[None, :]**2, np.sin(t)[None, :]**2
    gx, gy = L.gradient(cutoff**2 * c2, cutoff**2 * s2, P.expand(1))
    T = gy - gx
    change = np.sign(T[:, :-1]) != np.sign(T[:, 1:])
    mid = 0.5 * (t[:-1] + t[1:])
    dist = np.where(change, np.abs(mid[None, :] - seed[:, None]), np.inf)
    k = np.argmin(dist, axis=1)
    found = np.isfinite(dist[np.arange(seed.size), k])
    lo, hi = t[k], t[k + 1]

    def T_at(th):
        gx, gy = L.gradient(cutoff**2 * np.cos(th)**2, cutoff**2 * np.sin(th)**2, P)
        return gy - gx

    T_lo = T_at(lo)
    for _ in range(n_bisect):
        m = 0.5 * (lo + hi)
        T_m = T_at(m)
        same = np.sign(T_m) == np.sign(T_lo)
        lo, T_lo = np.where(same, m, lo), np.where(same, T_m, T_lo)
        hi = np.where(same, hi, m)
    return np.where(found, 0.5 * (lo + hi), np.nan)


def _bracket_value(event, P, cutoff, seed=None):
    """The function of the couplings whose zero is the condition (it changes sign across the edge):
      ngs    the smallest dV/d(rho^2) over the local minima of V along the cutoff circle (it can jump where they
             appear or disappear; the bisection on its sign allows that);
      np     t_np - t_cut - dt, the scale where a coupling reaches 4 pi minus the largest scale used and one RG step;
      cut_H  dV/dx on the H axis at the cutoff (an H-axis extremum is on the circle where it vanishes);
      cut_S  the mirror on the S axis;
      cut_M  dV/d(rho^2) at the direction (nearest seed) where V is stationary along the circle: both components of
             the gradient vanish there, i.e. an off-axis extremum is on the circle, where this vanishes."""
    if event == "ngs":
        return L.boundary_minimum_slope(P, cutoff)[0]
    if event == "np":
        t_lim = 0.5 * np.log1p(cutoff**2 / P.rg.mu02) + P.rg.dt     # as in loop_phase_map: one RG step of margin
        return np.minimum(P.rg.t_np[P.pidx], t_lim + 1.0) - t_lim
    n = P.muH2.size
    if event == "cut_H":
        return L.gradient(np.full(n, cutoff**2), np.zeros(n), P)[0]
    if event == "cut_S":
        return L.gradient(np.zeros(n), np.full(n, cutoff**2), P)[1]
    th = _tangential_root(P, cutoff, seed)
    return L._radial_slope(np.where(np.isfinite(th), th, 0.0), P, cutoff) + np.where(np.isfinite(th), 0.0, np.nan)


def _solve_bracket(P, A, B, event, cutoff, seeds=None, n_iter=45):
    """Solve one of _BRACKET_EVENTS on a batch of edges (a, b) by bisection in s. Returns s (nan where the value does
    not change sign across the edge). seeds: the direction of the crossing extremum, for cut_M."""
    lH, lS, lK = P.lamH[A], P.lamS[A], P.kappa[A]
    dH, dS, dK = P.lamH[B] - lH, P.lamS[B] - lS, P.kappa[B] - lK
    base = P.take(A)
    g = lambda s: _bracket_value(event, model_with_lambdas(base, lH + s * dH, lS + s * dS, lK + s * dK), cutoff,
                                 seeds)
    with np.errstate(all="ignore"):
        lo, hi = np.zeros(A.size), np.ones(A.size)
        g_lo, g_hi = g(lo), g(hi)
        ok = np.isfinite(g_lo) & np.isfinite(g_hi) & (np.sign(g_lo) != np.sign(g_hi))
        for _ in range(n_iter):
            mid = 0.5 * (lo + hi)
            g_mid = g(mid)
            same = np.sign(g_mid) == np.sign(g_lo)
            lo, g_lo = np.where(same, mid, lo), np.where(same, g_mid, g_lo)
            hi = np.where(same, hi, mid)
    return np.where(ok, 0.5 * (lo + hi), np.nan)


# ---------------------------------------------------------------------------------------------------------------------
# All edges of a grid
# ---------------------------------------------------------------------------------------------------------------------

def _edge_list(shape):
    """Flat indices (a, b) of every edge: first across columns (row-major over (ny, nx-1)), then across rows."""
    ny, nx = shape
    idx = np.arange(ny * nx).reshape(shape)
    a = np.concatenate([idx[:, :-1].ravel(), idx[:-1, :].ravel()])
    b = np.concatenate([idx[:, 1:].ravel(), idx[1:, :].ravel()])
    return a, b


def solve_edges(res, verbose=False, map_kwargs=None, rows=True, max_depth=3, _depth=0):
    """The solved crossing point of every edge of a loop_phase_map grid where one condition is identified.

    With map_kwargs (the keyword arguments loop_phase_map was called with), edges where no single condition holds
    across the whole edge -- two are crossed within it, next to where lines meet -- are subdivided and solved part by
    part (_refine), recursively up to max_depth times (1/8^max_depth of an edge). rows=False solves only the edges
    across columns.

    Returns dict(pos_c, pos_r, event_c, event_r): pos_c (ny, nx-1) is Omega_H on the edges across columns, pos_r
    (ny-1, nx) Omega_S on the edges across rows (nan where nothing was solved); event_* the index into EVENTS, NONE
    FAILED or UNCHANGED."""
    shape = res["gs_kind"].shape
    ny, nx = shape
    P = res["model"]
    scales = L._scales(P)
    jc, jr = L.ground_state_jumps(res)
    a_all, b_all = _edge_list(shape)
    jump = np.concatenate([jc.ravel(), jr.ravel()])
    st = res["structure"].ravel()
    gk = res["gs_kind"].ravel()
    od = res["order"].ravel()
    run, nonp = res["runaway"].ravel(), res["nonperturbative"].ravel()
    changed = np.array([st[a] != st[b] or gk[a] != gk[b] or od[a] != od[b] for a, b in zip(a_all, b_all)],
                       dtype=bool) | jump | (run[a_all] != run[b_all]) | (nonp[a_all] != nonp[b_all])
    cutoff = res["params"].get("cutoff")
    nc = ny * (nx - 1)
    if not rows:
        changed[nc:] = False

    event = np.where(changed, NONE, UNCHANGED)
    s = np.full(a_all.size, np.nan)
    groups = {}
    for e in np.flatnonzero(changed):
        ev, seed = _classify(res, a_all[e], b_all[e], jump[e], scales[:2])
        if ev is None:
            continue
        if ev in _PAIR_EVENTS:
            ka, ca, kb, cb = seed
            key = (ev, ka, kb)
            seed = list(ca) + list(cb)
        else:
            key = (ev,)
        groups.setdefault(key, []).append((e, seed))

    for key, items in groups.items():
        ev = key[0]
        E = np.array([e for e, _ in items])
        A, B = a_all[E], b_all[E]
        xbar, ybar, _ = scales
        if ev in _BRACKET_EVENTS or ev.startswith("cut_"):
            if cutoff is None:
                continue
        if ev in _BRACKET_EVENTS:
            seeds = (np.array([sd[0] for _, sd in items], dtype=float) if ev == "cut_M" else None)
            sol = _solve_bracket(P, A, B, ev, cutoff, seeds)
        elif ev in _PAIR_EVENTS:
            ka, kb = key[1], key[2]
            raw = np.array([sd for _, sd in items], dtype=float).reshape(E.size, -1)
            cols, k = [], 0
            for kind in (ka, kb):
                if kind in ("H", "M"):
                    cols.append(np.log(raw[:, k] / xbar[A]))
                    k += 1
                if kind in ("S", "M"):
                    cols.append(np.log(raw[:, k] / ybar[A]))
                    k += 1
            seeds = np.column_stack(cols) if cols else np.zeros((E.size, 0))
            sol = _solve_batch(P, scales, A, B, ev, seeds, kinds=(ka, kb))
        else:
            raw = np.array([sd for _, sd in items], dtype=float).reshape(E.size, -1)
            if ev in ("trans_H", "fold_H"):
                seeds = np.log(raw[:, :1] / xbar[A, None])
            elif ev in ("trans_S", "fold_S"):
                seeds = np.log(raw[:, :1] / ybar[A, None])
            elif ev == "fold_M":
                seeds = np.column_stack([np.log(raw[:, 0] / xbar[A]), np.log(raw[:, 1] / ybar[A])])
            else:
                seeds = np.zeros((E.size, 0))
            sol = _solve_batch(P, scales, A, B, ev, seeds, cutoff=cutoff)
        s[E] = sol
        event[E] = np.where(np.isfinite(sol), EVENTS.index(ev), FAILED)
        if verbose:
            print(f"  {'/'.join(key)}: {np.isfinite(sol).sum()} of {E.size} edges solved")

    if map_kwargs is not None and _depth < max_depth:
        E = np.flatnonzero((event == NONE) | (event == FAILED))
        if E.size:
            s[E], event[E] = _refine(res, a_all[E], b_all[E], map_kwargs, max_depth, _depth + 1)
            if verbose:
                print(f"  subdivided: {np.isfinite(s[E]).sum()} of {E.size} edges solved")

    # positions in Omega along each edge
    OH, OS = res["OmegaH"].ravel(), res["OmegaS"].ravel()
    pos = np.where(np.arange(a_all.size) < nc,
                   OH[a_all] + s * (OH[b_all] - OH[a_all]),
                   OS[a_all] + s * (OS[b_all] - OS[a_all]))
    return dict(pos_c=pos[:nc].reshape(ny, nx - 1), pos_r=pos[nc:].reshape(ny - 1, nx),
                event_c=event[:nc].reshape(ny, nx - 1), event_r=event[nc:].reshape(ny - 1, nx))


def _refine(res, A, B, map_kwargs, max_depth, depth, n_sub=8):
    """(s, event) on the edges (A, B) where no single condition holds across the whole edge. Each such edge is
    divided into n_sub parts by running loop_phase_map in between, and every part is solved as an edge of its own.
    Of the crossings found, the one kept is the one that changes what the figure draws across the whole edge: the
    quantum phase (ground state, NGS, NP) if that changes, otherwise the first change of the extrema."""
    P, prm = res["model"], res["params"]
    t = np.linspace(0.0, 1.0, n_sub + 1)
    lamH = P.lamH[A][:, None] + t * (P.lamH[B] - P.lamH[A])[:, None]
    lamS = P.lamS[A][:, None] + t * (P.lamS[B] - P.lamS[A])[:, None]
    kap = P.kappa[A][:, None] + t * (P.kappa[B] - P.kappa[A])[:, None]
    sub = L.loop_phase_map(prm["r"], kap, lamH, lamS, prm["mu"], **map_kwargs)
    sub["OmegaH"] = np.broadcast_to(t, lamH.shape).copy()      # so that the solved positions are fractions s
    e = solve_edges(sub, rows=False, map_kwargs=map_kwargs, max_depth=max_depth, _depth=depth)
    ev, pos = e["event_c"], e["pos_c"]
    gk, run, nonp = sub["gs_kind"], sub["runaway"], sub["nonperturbative"]
    quantum = lambda k, i, j: gk[k, i] != gk[k, j] or run[k, i] != run[k, j] or nonp[k, i] != nonp[k, j]
    s_out, ev_out = np.full(A.size, np.nan), np.full(A.size, FAILED)
    for k in range(A.size):
        whole = quantum(k, 0, n_sub)
        for j in range(n_sub):
            if ev[k, j] >= 0 and (not whole or quantum(k, j, j + 1)):
                s_out[k], ev_out[k] = pos[k, j], ev[k, j]
                break
    return s_out, ev_out


# ---------------------------------------------------------------------------------------------------------------------
# Curves from edge points
# ---------------------------------------------------------------------------------------------------------------------

def edge_points(pos_c, pos_r, hc, sc):
    """Coordinates of every edge's crossing point: the solved one, or the edge midpoint where there is none."""
    ny, nx = sc.size, hc.size
    Xc = np.where(np.isfinite(pos_c), pos_c, np.broadcast_to(0.5 * (hc[:-1] + hc[1:]), (ny, nx - 1)))
    Yc = np.broadcast_to(sc[:, None], (ny, nx - 1))
    Xr = np.broadcast_to(hc[None, :], (ny - 1, nx))
    Yr = np.where(np.isfinite(pos_r), pos_r, np.broadcast_to(0.5 * (sc[:-1] + sc[1:])[:, None], (ny - 1, nx)))
    return (Xc, Yc), (Xr, Yr)


def polylines(active_c, active_r, all_c, all_r, points, min_free_points=4, loose=None, short=6):
    """Curves through the crossing points of the active edges (marching squares on edges).

    active_* mark the edges of one kind of line; all_* the edges of every kind of line, so that where lines of several
    kinds meet (a square with other than two boundary edges) they all end at the same junction point, the mean of
    that square's crossing points. Isolated curves (no junction at either end) with fewer than min_free_points points
    are dropped as grid noise. Returns a list of (N, 2) arrays. If loose is a list, the edges of short pieces of
    curve (at most short points) and of dropped curves are appended to it as (orientation, i, j): where a line breaks up
    because the grid does not resolve what lies between (see auto_refine in loop_phase_grid.py)."""
    (Xc, Yc), (Xr, Yr) = points

    def xy(node):
        t, i, j = node
        if t == "c":
            return Xc[i, j], Yc[i, j]
        if t == "r":
            return Xr[i, j], Yr[i, j]
        return junction[(i, j)]

    sq = active_c[:-1, :] | active_c[1:, :] | active_r[:, :-1] | active_r[:, 1:]
    adj, junction = {}, {}

    def link(u, v):
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)

    for i, j in zip(*np.nonzero(sq)):
        sides = [("c", i, j), ("c", i + 1, j), ("r", i, j), ("r", i, j + 1)]
        flags = [(active_c if t == "c" else active_r)[a, b] for t, a, b in sides]
        every = [(all_c if t == "c" else all_r)[a, b] for t, a, b in sides]
        mine = [sd for sd, f in zip(sides, flags) if f]
        alls = [sd for sd, f in zip(sides, every) if f]
        if len(alls) == 2 and len(mine) == 2:
            link(mine[0], mine[1])
            continue
        pts = np.array([xy(sd) for sd in alls])
        junction[(i, j)] = tuple(pts.mean(axis=0))
        for sd in mine:
            link(sd, ("q", i, j))

    # walk the graph into chains: start at nodes of degree != 2, then close the remaining loops
    seen = set()
    out = []

    def walk(start, nxt):
        chain = [start, nxt]
        seen.add(frozenset((start, nxt)))
        prev, cur = start, nxt
        while len(adj[cur]) == 2:
            nn = adj[cur][0] if adj[cur][1] == prev else adj[cur][1]
            e = frozenset((cur, nn))
            if e in seen:
                break
            seen.add(e)
            chain.append(nn)
            prev, cur = cur, nn
        return chain

    for node in list(adj):
        if len(adj[node]) != 2:
            for nb in adj[node]:
                if frozenset((node, nb)) not in seen:
                    out.append(walk(node, nb))
    for node in list(adj):
        for nb in adj[node]:
            if frozenset((node, nb)) not in seen:
                out.append(walk(node, nb))

    lines = []
    for chain in out:
        free = chain[0][0] != "q" and chain[-1][0] != "q"
        if free and len(chain) < min_free_points and chain[0] != chain[-1]:
            if loose is not None:
                loose.extend(n for n in chain if n[0] != "q")
            continue
        if loose is not None and chain[0] != chain[-1] and len(chain) <= short:
            loose.extend(n for n in chain if n[0] != "q")
        lines.append(np.array([xy(n) for n in chain]))
    return lines


def snap_paths(paths, hc, sc, pos_c, pos_r, use_c, use_r):
    """The paths of a filled contour of a 0/1 field on the cell centres (hc, sc), with every vertex that sits at the
    midpoint of an edge marked in use_* moved to that edge's solved crossing point."""
    dx, dy = hc[1] - hc[0], sc[1] - sc[0]
    out = []
    for path in paths:
        v = path.vertices.copy()
        if v.size:
            fi, fj = (v[:, 1] - sc[0]) / dy, (v[:, 0] - hc[0]) / dx
            ri, rj = np.rint(fi), np.rint(fj)
            on_row, on_col = np.abs(fi - ri) < 1e-6, np.abs(fj - rj) < 1e-6
            mid_c = on_row & (np.abs(fj - np.floor(fj) - 0.5) < 1e-6)
            mid_r = on_col & (np.abs(fi - np.floor(fi) - 0.5) < 1e-6)
            for k in np.flatnonzero(mid_c):
                i, j = int(ri[k]), int(np.floor(fj[k]))
                if 0 <= i < use_c.shape[0] and 0 <= j < use_c.shape[1] and use_c[i, j] and np.isfinite(pos_c[i, j]):
                    v[k, 0] = pos_c[i, j]
            for k in np.flatnonzero(mid_r):
                i, j = int(np.floor(fi[k])), int(rj[k])
                if 0 <= i < use_r.shape[0] and 0 <= j < use_r.shape[1] and use_r[i, j] and np.isfinite(pos_r[i, j]):
                    v[k, 1] = pos_r[i, j]
        out.append(type(path)(v, path.codes))
    return out
