"""Checks of the one-loop potential (nearcrit/singlet_loop.py):
  - the gradient and the axis second derivatives agree with finite differences of V;
  - with the loop switched off the classification reproduces the tree-level phase map in all four mu^2 quadrants;
  - with a tiny loop the loop-level Omegas are close to the tree ones;
  - the RG-improved gradient agrees with finite differences, the beta functions are those of V_1, and near mu0 the
    RG-improved potential and the origin's curvatures are the fixed-scale ones.
The tolerances are an order of magnitude or more above the values found (printed with -v). Run
`python tests/test_singlet_loop.py` (about 20 s), or with pytest.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the nearcrit package

import numpy as np

from nearcrit import singlet_tree as tq
from nearcrit.singlet_loop import (
    LoopModel, _potential_fixed, _rg_fixed_model, attach_rg, d2V_dx2_H_axis, d2V_dy2_S_axis, gradient,
    lambdas_from_tree_Omegas, loop_mu2s, loop_phase_map, potential, sm_species,
)

VERBOSE = "-v" in sys.argv


def run_checks(ngrid=61, verbose=True):
    """1. gradient and the axis second derivatives agree with finite differences of V;
    2. with the loop switched off (eps = 0) the classification reproduces
       tree_potential_phase_map_Omega in all four mu^2 quadrants;
    3. with a tiny loop the Omega^(1) are close to the tree Omegas."""
    rng = np.random.default_rng(3)
    n = 400
    P = LoopModel(muH2=rng.uniform(-2, 2, n), muS2=rng.uniform(-2, 2, n),
                  lamH=rng.uniform(0.05, 2, n), lamS=rng.uniform(0.05, 2, n),
                  kappa=rng.uniform(-2, 2, n), mu2=1.3, eps=1.0, n_goldstone=3,
                  species=sm_species(), n_goldstone_S=1)
    x, y = rng.uniform(0.05, 4, n), rng.uniform(0.05, 4, n)
    h = 1e-6
    gx, gy = gradient(x, y, P)
    fx = (potential(x + h, y, P) - potential(x - h, y, P)) / (2 * h)
    fy = (potential(x, y + h, P) - potential(x, y - h, P)) / (2 * h)
    err_g = max(np.max(np.abs(gx - fx)), np.max(np.abs(gy - fy)))
    a2 = d2V_dx2_H_axis(x, P)
    f2 = (gradient(x + h, 0.0, P)[0] - gradient(x - h, 0.0, P)[0]) / (2 * h)
    b2 = d2V_dy2_S_axis(y, P)
    g2 = (gradient(0.0, y + h, P)[1] - gradient(0.0, y - h, P)[1]) / (2 * h)
    err_2 = max(np.max(np.abs(a2 - f2)), np.max(np.abs(b2 - g2)))
    if verbose:
        print(f"1. max |grad - FD| = {err_g:.2e},  max |axis d2 - FD| = {err_2:.2e}")

    worst = 0.0
    for r, kappa, muH2 in [(1.0, 1.0, 1.0), (-1.0, -1.0, 1.0), (-1.0, 1.0, -1.0),
                           (1.0, 1.0, -1.0), (1.0, -1.0, 1.0)]:
        sgn = np.sign(r * kappa)
        g = sgn * np.linspace(0.05, 3.0, ngrid)
        GH, GS = np.meshgrid(g, g)
        lamH, lamS = lambdas_from_tree_Omegas(r, kappa, GH, GS)
        res = loop_phase_map(r, kappa, lamH, lamS, 1.0, muH2=muH2, eps=0.0,
                             mask_tree_unbounded=True)
        tree = tq.tree_potential_phase_map_Omega(GH, GS, muH2, r * muH2)
        # tree == 0 inside the bounded region only on exact ties (e.g. grid points with
        # OmegaH == OmegaS, where its strict inequalities assign no phase)
        bounded = res['tree_bounded'] & (tree != 0)
        bad = (res['phase'] != tree) & bounded
        frac = bad.sum() / max(bounded.sum(), 1)
        worst = max(worst, frac)
        if verbose:
            print(f"2. eps=0, r={r:+g}, kappa={kappa:+g}, muH2={muH2:+g}: "
                  f"{bad.sum()} / {bounded.sum()} bounded points differ from the tree map")

    g = np.linspace(0.1, 2.5, 25)
    GH, GS = np.meshgrid(g, g)
    lamH, lamS = lambdas_from_tree_Omegas(1.0, 1.0, GH, GS)
    res = loop_phase_map(1.0, 1.0, lamH, lamS, 1.0, eps=1e-4)
    dH = np.nanmax(np.abs(res['OmegaH1'] - GH))
    dS = np.nanmax(np.abs(res['OmegaS1'] - GS))
    if verbose:
        print(f"3. eps=1e-4: max |OmegaH1 - OmegaH| = {dH:.2e}, "
              f"max |OmegaS1 - OmegaS| = {dS:.2e}")
    return dict(grad=err_g, axis_d2=err_2, tree_mismatch=worst, omega_shift=(dH, dS))


def run_rg_checks(verbose=True):
    """1. the RG-improved gradient agrees with finite differences of V_RG;
    2. the beta functions are those of V_1: shifting the scale of the fixed-scale
       potential while running the couplings (and rescaling H by xi) leaves a change of
       V(H,S) - V(0) that is one order higher in the couplings than the frozen-coupling
       change, i.e. it shrinks tenfold when all couplings shrink tenfold;
    3. near mu0 the RG-improved potential is the fixed-scale one;
    4. so are the origin's curvatures (this tests the running vacuum energy Omega)."""
    rng = np.random.default_rng(5)
    n = 300
    sm = dict(g=0.64779, gp=0.35830, g3=1.1666, yt=0.93690)
    P = LoopModel(muH2=rng.uniform(-1, 1, n), muS2=rng.uniform(-1, 1, n),
                  lamH=rng.uniform(0.05, 0.6, n), lamS=rng.uniform(0.05, 0.6, n),
                  kappa=rng.uniform(-0.5, 1.0, n), mu2=1.96**2, n_goldstone=3,
                  n_goldstone_S=1)
    Prg = attach_rg(P, sm, 1e6)
    x, y = 10**rng.uniform(-1, 4, n), 10**rng.uniform(-1, 4, n)
    gx, gy = gradient(x, y, Prg)
    hx, hy = 1e-6 * x, 1e-6 * y
    fx = (potential(x + hx, y, Prg) - potential(x - hx, y, Prg)) / (2 * hx)
    fy = (potential(x, y + hy, Prg) - potential(x, y - hy, Prg)) / (2 * hy)
    err_g = max(np.max(np.abs(gx - fx) / (np.abs(fx) + 1e-3)),
                np.max(np.abs(gy - fy) / (np.abs(fy) + 1e-3)))

    # 2: shift the scale by t1 with the couplings run (and H rescaled) and compare with
    # the frozen-coupling shift. The leftover is one order higher in the couplings, so it
    # must shrink in proportion when all couplings are scaled down by eps (at fixed masses).
    t1 = 0.2
    x2, y2 = 10**rng.uniform(-1, 2, n), 10**rng.uniform(-1, 2, n)
    ratios = []
    for eps in (1.0, 0.1):
        sm_e = {k: v * np.sqrt(eps) for k, v in sm.items()}
        Pe = P._replace(lamH=P.lamH * eps, lamS=P.lamS * eps, kappa=P.kappa * eps)
        Pe_rg = attach_rg(Pe, sm_e, 1e6)
        Pfix = Pe._replace(species=sm_species(sm_e['g'], sm_e['gp'], sm_e['yt']))
        X, Y = x2 / eps, y2 / eps
        dV = lambda Q, xx, yy: _potential_fixed(xx, yy, Q) - _potential_fixed(0.0, 0.0, Q)
        V0 = dV(Pfix, X, Y)
        Pf, xi2, _ = _rg_fixed_model(Pe_rg, np.full(n, t1))
        V1 = dV(Pf, xi2 * X, Y)
        Vfz = dV(Pfix._replace(mu2=Pfix.mu2 * np.exp(2 * t1)), X, Y)
        ratios.append(np.median(np.abs(V1 - V0) / np.abs(Vfz - V0)))
    ratio = ratios[1] / ratios[0]

    # 3: at small fields (t -> 0) V_RG -> fixed-scale V
    xs, ys = rng.uniform(0, 0.02, n), rng.uniform(0, 0.02, n)
    Pfix = Prg._replace(rg=None, species=sm_species(sm['g'], sm['gp'], sm['yt']))
    dV = np.max(np.abs((potential(xs, ys, Prg) - potential(0.0, 0.0, Prg))
                       - (_potential_fixed(xs, ys, Pfix) - _potential_fixed(0.0, 0.0, Pfix))))
    scale = np.max(np.abs(_potential_fixed(xs, ys, Pfix) - _potential_fixed(0.0, 0.0, Pfix)))
    if verbose:
        print(f"1. RG gradient vs FD: max relative error {err_g:.1e}")
        print(f"2. scale shift t = {t1}: |V change with running| / |V change frozen| = "
              f"{ratios[0]:.3f}, and {ratios[1]:.4f} with couplings x 0.1 "
              f"(ratio {ratio:.3f}; 0.1 if the beta functions are those of V_1)")
        print(f"3. small fields: |V_RG - V_fixed| / |V| = {dV / scale:.1e}")
    # 4: the origin's curvatures match the fixed-scale ones at mu0 (needs the running vacuum
    # energy Omega: without it they were off by ~ (N muH2^2 + M muS2^2) / (32 pi^2 mu0^2))
    mrg, srg = loop_mu2s(Prg)
    mfx, sfx = loop_mu2s(Pfix)
    size = np.abs(Pfix.muH2) + np.abs(Pfix.muS2)
    origin = max(np.max(np.abs(mrg - mfx) / size), np.max(np.abs(srg - sfx) / size))
    if verbose:
        print(f"4. origin: |mu2^(1)_RG - mu2^(1)_fixed| / (|muH2| + |muS2|) = {origin:.1e}")
    return dict(grad=err_g, rg_invariance=ratio, small_field=dV / scale, origin=origin)


def test_fixed_scale():
    c = run_checks(verbose=VERBOSE)
    assert c["grad"] < 1e-7 and c["axis_d2"] < 1e-7, c                   # found ~3e-9, 1e-9
    assert c["tree_mismatch"] == 0, c
    assert max(c["omega_shift"]) < 1e-3, c                                # found 1.3e-5


def test_rg_improved():
    c = run_rg_checks(verbose=VERBOSE)
    assert c["grad"] < 1e-2, c                                            # found 1.3e-4
    assert abs(c["rg_invariance"] - 0.1) < 0.02, c                        # 0.1 if the beta functions are right (0.102)
    assert c["small_field"] < 1e-2 and c["origin"] < 1e-2, c              # found 6e-4, 4e-4


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"{name}: passed")
