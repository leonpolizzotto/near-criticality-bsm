"""One-loop (Coleman-Weinberg) potential phases of the two-scalar model.

Conventions follow nearcrit/singlet_tree.py:

    V_tree = -1/2 muH2 H^2 - 1/2 muS2 S^2 + 1/4 lambdaH H^4 + 1/4 lambdaS S^4
             + 1/4 kappa H^2 S^2

    OmegaH = 2 lambdaH muS2 / (kappa muH2),    OmegaS = 2 lambdaS muH2 / (kappa muS2)

The one-loop term is the MS-bar Coleman-Weinberg potential (real part, so tachyonic
masses enter through ln|m^2|):

    V_1 = 1/(64 pi^2) sum_i n_i m_i^4 (ln|m_i^2|/mu^2 - c_i)

By default the loop contains only the two scalars themselves (the h-s mass matrix). Pass
n_goldstone=3 for a Higgs doublet, n_goldstone_S=1 for S a real doublet (an O(2)
multiplet, with one Goldstone), and species=sm_species() for W, Z and top. The Goldstone
masses are (1/H) dV/dH = -muH2 + lambdaH H^2 + kappa S^2/2 and
(1/S) dV/dS = -muS2 + lambdaS S^2 + kappa H^2/2.

Inputs are (r, kappa, lambdaH, lambdaS, mu), with r = muS2/muH2. Masses are measured in
units of |muH2| = 1, so muH2 = +1 or -1 and muS2 = r muH2. The plots are drawn against
the tree-level Omegas,

    OmegaH = 2 r lambdaH / kappa,    OmegaS = 2 lambdaS / (r kappa),

so every panel can be laid directly over the tree-level atlas.

How the phase is found
----------------------
Everything is done in the squared fields x = H^2, y = S^2. At tree level V is quadratic
in (x, y); that is what makes the tree analysis analytic.

* Origin. Its curvatures are -muH2^(1) and -muS2^(1), the loop-corrected mass
  parameters muH2^(1) = -2 dV/dx(0,0) and muS2^(1) = -2 dV/dy(0,0).

* H axis (S = 0). The extrema are the roots of dV/dx(x, 0) = 0. They are found by a
  log-spaced scan and bisection, so uniqueness is CHECKED rather than assumed. At each
  root the S-direction mass is m_S^2 = 2 dV/dy(x, 0), and

      OmegaH^(1) = muS2 / (muS2 + m_S^2(P1^(1))),

  which reduces to the tree OmegaH and satisfies m_S^2 = muS2 (1/OmegaH^(1) - 1), the
  tree-level relation. The transverse type of P1 is therefore read off OmegaH^(1)
  exactly as at tree level. The S axis is the same with H <-> S.

* Off the axes. The extrema are the roots of dV/dx = dV/dy = 0 with x, y > 0, found by a
  scan plus Newton's method. At those points the Hessian in (x, y) has the same signature
  as the one in (H, S), so it classifies them. Interior extrema are created or destroyed
  where, in addition, det Hess = 0 (a fold).

Boundedness is decided at one loop (runaway_mask): a point gets no phase if V is still
decreasing at the edge of the searched region. That edge acts as a cutoff, so where a
barrier sits near it (e.g. the top-induced instability) the grey region depends on
axis_decades / interior_decades.

Each tree potential phase is one pattern of extremum types plus one ground state (see
TREE_SIGNATURES). A one-loop pattern matching a tree pattern gets the tree label and
color. Any other pattern, e.g. three extrema on the H axis, is a new, radiatively
generated phase. It is registered on first sight as "<quantum phase>*<k>" and keeps that
code for the rest of the session; NEW_PHASES lists what each one is.
"""

from typing import NamedTuple

import numpy as np

from nearcrit import singlet_tree as tq
# ---------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------

LOOP_FACTOR = 1.0 / (64.0 * np.pi**2)
_SCALAR_C = 1.5
_TINY = 1e-300


class Species(NamedTuple):
    """An extra field in the loop with m^2 = a x + b y (x = H^2, y = S^2).

    n counts degrees of freedom and is negative for fermions. c is the MS-bar constant:
    3/2 for scalars and fermions, 5/6 for gauge bosons."""
    n: float
    c: float
    a: float
    b: float = 0.0


def sm_species(g=0.64, gp=0.34, yt=0.94):
    """W, Z and top, with m_W^2 = g^2 H^2/4, m_Z^2 = (g^2+g'^2) H^2/4, m_t^2 = yt^2 H^2/2."""
    return (Species(6, 5.0 / 6.0, g**2 / 4.0),
            Species(3, 5.0 / 6.0, (g**2 + gp**2) / 4.0),
            Species(-12, 1.5, yt**2 / 2.0))


class LoopModel(NamedTuple):
    """Parameters of V_tree + eps * V_1. The five couplings may be arrays (one entry per
    parameter point); mu2, eps, the Goldstone counts and species are shared by all."""
    muH2: np.ndarray
    muS2: np.ndarray
    lamH: np.ndarray
    lamS: np.ndarray
    kappa: np.ndarray
    mu2: float = 1.0
    eps: float = 1.0
    n_goldstone: int = 0
    species: tuple = ()
    n_goldstone_S: int = 0
    rg: object = None          # RGTable: RG-improved potential (see attach_rg), else None
    pidx: np.ndarray = None    # with rg: each point's row in the RG table

    def expand(self, k):
        """Append k singleton axes to the coupling arrays, to broadcast against fields."""
        f = lambda a: np.asarray(a, dtype=float)[(...,) + (None,) * k]
        out = self._replace(muH2=f(self.muH2), muS2=f(self.muS2), lamH=f(self.lamH),
                            lamS=f(self.lamS), kappa=f(self.kappa))
        if self.pidx is not None:
            out = out._replace(pidx=np.asarray(self.pidx)[(...,) + (None,) * k])
        return out

    def take(self, idx):
        """The parameter points idx (coupling arrays must be 1D)."""
        f = lambda a: np.asarray(a, dtype=float)[idx]
        out = self._replace(muH2=f(self.muH2), muS2=f(self.muS2), lamH=f(self.lamH),
                            lamS=f(self.lamS), kappa=f(self.kappa))
        if self.pidx is not None:
            out = out._replace(pidx=np.asarray(self.pidx)[idx])
        return out


def make_model(r, kappa, lambdaH, lambdaS, mu, muH2=1.0, eps=1.0, n_goldstone=0,
               species=(), n_goldstone_S=0):
    """LoopModel from the (r, kappa, lambdaH, lambdaS, mu) inputs, flattened to 1D.

    Units: |muH2| = 1, so mu is the renormalization scale in units of |muH|."""
    lamH, lamS, kap = np.broadcast_arrays(np.asarray(lambdaH, dtype=float),
                                          np.asarray(lambdaS, dtype=float), np.asarray(kappa, dtype=float))
    n = lamH.size
    return LoopModel(muH2=np.full(n, float(muH2)), muS2=np.full(n, float(r * muH2)),
                     lamH=lamH.ravel().copy(), lamS=lamS.ravel().copy(),
                     kappa=kap.ravel().copy(), mu2=float(mu)**2, eps=float(eps),
                     n_goldstone=int(n_goldstone), species=tuple(species),
                     n_goldstone_S=int(n_goldstone_S))


def tree_Omegas(r, kappa, lambdaH, lambdaS):
    """The tree-level (OmegaH, OmegaS) of an (r, kappa, lambdaH, lambdaS) point."""
    return 2.0 * r * np.asarray(lambdaH) / kappa, 2.0 * np.asarray(lambdaS) / (r * kappa)


def lambdas_from_tree_Omegas(r, kappa, OmegaH, OmegaS):
    """Inverse of tree_Omegas: lambdaH = kappa OmegaH / (2 r), lambdaS = kappa r OmegaS / 2."""
    return kappa * np.asarray(OmegaH) / (2.0 * r), kappa * r * np.asarray(OmegaS) / 2.0


# ---------------------------------------------------------------------------------------
# The potential and its derivatives in (x, y) = (H^2, S^2)
# ---------------------------------------------------------------------------------------

def _logabs(u, mu2):
    return np.log(np.maximum(np.abs(u), _TINY) / mu2)


def _f(u, mu2, c):
    """u^2 (ln|u|/mu^2 - c): one degree of freedom's m^4 (ln m^2 - c)."""
    return u * u * (_logabs(u, mu2) - c)


def _fp(u, mu2, c):
    """d/du of _f. It goes to zero as u -> 0, so massless states need no special case."""
    return 2.0 * u * (_logabs(u, mu2) - c + 0.5)


def _fpp(u, mu2, c):
    """d^2/du^2 of _f. Diverges like 2 ln|u| as u -> 0 (the one-loop IR log)."""
    return 2.0 * _logabs(u, mu2) - 2.0 * c + 3.0


def _scalar_block(x, y, P):
    """Entries of the h-s mass matrix [[A, B], [B, C]], with B^2 = Q, and its eigenvalues
    s +/- R. A, C and Q are polynomials in (x, y), so everything below is smooth in them."""
    A = -P.muH2 + 3.0 * P.lamH * x + 0.5 * P.kappa * y
    C = -P.muS2 + 3.0 * P.lamS * y + 0.5 * P.kappa * x
    Q = P.kappa**2 * x * y
    s = 0.5 * (A + C)
    d = 0.5 * (A - C)
    R = np.sqrt(d * d + Q)
    return A, C, Q, s, d, R


def _goldstone_mass(x, y, P):
    return -P.muH2 + P.lamH * x + 0.5 * P.kappa * y


def _goldstone_mass_S(x, y, P):
    return -P.muS2 + P.lamS * y + 0.5 * P.kappa * x


def _potential_fixed(x, y, P):
    """V_tree + eps V_1 at (x, y) = (H^2, S^2), all couplings at the one scale P.mu2."""
    V = (-0.5 * P.muH2 * x - 0.5 * P.muS2 * y + 0.25 * P.lamH * x * x
         + 0.25 * P.lamS * y * y + 0.25 * P.kappa * x * y)
    if P.eps == 0:
        return V
    _, _, _, s, _, R = _scalar_block(x, y, P)
    loop = _f(s + R, P.mu2, _SCALAR_C) + _f(s - R, P.mu2, _SCALAR_C)
    if P.n_goldstone:
        loop = loop + P.n_goldstone * _f(_goldstone_mass(x, y, P), P.mu2, _SCALAR_C)
    if P.n_goldstone_S:
        loop = loop + P.n_goldstone_S * _f(_goldstone_mass_S(x, y, P), P.mu2, _SCALAR_C)
    for sp in P.species:
        loop = loop + sp.n * _f(sp.a * x + sp.b * y, P.mu2, sp.c)
    return V + P.eps * LOOP_FACTOR * loop


def _gradient_fixed(x, y, P):
    """(dV/dx, dV/dy) of _potential_fixed. Valid everywhere, including on the axes.

    The scalar loop sum_i f(m_i^2) depends on (A, C, Q) only through the two eigenvalues.
    Its derivatives involve the divided difference DD = [f'(m+) - f'(m-)] / (m+ - m-),
    which tends to f''(s) where the eigenvalues meet. That is why no eigenvalue
    derivatives, and no separate massless cases, are needed."""
    gx = -0.5 * P.muH2 + 0.5 * P.lamH * x + 0.25 * P.kappa * y
    gy = -0.5 * P.muS2 + 0.5 * P.lamS * y + 0.25 * P.kappa * x
    if P.eps == 0:
        return gx, gy

    _, _, _, s, d, R = _scalar_block(x, y, P)
    fp_p = _fp(s + R, P.mu2, _SCALAR_C)
    fp_m = _fp(s - R, P.mu2, _SCALAR_C)
    split = R > 1e-7 * (np.abs(s) + _TINY)
    with np.errstate(divide='ignore', invalid='ignore'):
        DD = np.where(split, (fp_p - fp_m) / np.where(split, 2.0 * R, 1.0),
                      _fpp(s, P.mu2, _SCALAR_C))
    avg = 0.5 * (fp_p + fp_m)
    FA = avg + d * DD
    FC = avg - d * DD
    lx = 3.0 * P.lamH * FA + 0.5 * P.kappa * FC + P.kappa**2 * y * DD
    ly = 0.5 * P.kappa * FA + 3.0 * P.lamS * FC + P.kappa**2 * x * DD

    if P.n_goldstone:
        fG = P.n_goldstone * _fp(_goldstone_mass(x, y, P), P.mu2, _SCALAR_C)
        lx = lx + P.lamH * fG
        ly = ly + 0.5 * P.kappa * fG
    if P.n_goldstone_S:
        fGS = P.n_goldstone_S * _fp(_goldstone_mass_S(x, y, P), P.mu2, _SCALAR_C)
        lx = lx + 0.5 * P.kappa * fGS
        ly = ly + P.lamS * fGS
    for sp in P.species:
        fs = sp.n * _fp(sp.a * x + sp.b * y, P.mu2, sp.c)
        lx = lx + sp.a * fs
        ly = ly + sp.b * fs

    k = P.eps * LOOP_FACTOR
    return gx + k * lx, gy + k * ly


# ---------------------------------------------------------------------------------------
# RG improvement
# ---------------------------------------------------------------------------------------
#
# The RG-improved potential is the same one-loop potential evaluated at the
# field-dependent scale mu^2 = mu0^2 + H^2 + S^2, with every coupling run from mu0 to that
# scale and H rescaled by xi = exp(-int gamma_H dt):
#
#     V_RG(H, S) = V_tree + V_1 at (xi H, S), couplings(t), scale mu0 e^t,
#     t = (1/2) ln(1 + (H^2 + S^2) / mu0^2).
#
# The logs ln(m^2/mu^2) then stay of order one at every field value, and solving the
# RGEs resums the leading logs. Near the electroweak vacuum t is small and V_RG is the
# fixed-scale potential. The running is one loop; the scalar beta functions are derived
# from the log coefficient of the same V_1 (so the two are consistent by construction,
# which run_checks verifies). With H an O(N) multiplet (N = n_goldstone + 1) and S an
# O(M) one (M = n_goldstone_S + 1), in units of 1/(16 pi^2), with
# gamma_H = 3 yt^2 - 9/4 g^2 - 3/4 g'^2:
#
#   lambda_H : 2(N+8) lH^2 + M/2 k^2 - 6 yt^4 + 3/8 (2 g^4 + (g^2+g'^2)^2) + 4 gamma_H lH
#   lambda_S : 2(M+8) lS^2 + N/2 k^2
#   kappa    : 2(N+2) lH k + 2(M+2) lS k + 4 k^2 + 2 gamma_H k
#   muH2     : 2(N+2) lH muH2 + M k muS2 + 2 gamma_H muH2
#   muS2     : 2(M+2) lS muS2 + N k muH2
#   g, g', g3: -19/6 g^3, 41/6 g'^3, -7 g3^3
#   yt       : yt (9/2 yt^2 - 8 g3^2 - 9/4 g^2 - 17/12 g'^2)
#   Omega    : (N muH2^2 + M muS2^2) / 2
#
# For the SM (N = 4, M = 0) these are the standard one-loop SM equations. The anomalous
# dimension of S vanishes at one loop. Landau gauge, as V_1.
#
# Omega is the field-independent vacuum energy, V_RG = V_tree + V_1 + Omega(t). Its
# running cancels the explicit scale dependence of V_1 at the origin, where the masses
# are -muH2 (N states) and -muS2 (M states). With a field-dependent scale it must be
# included: otherwise that scale dependence turns into spurious field dependence (it
# shifted muH2^(1) by ~ (2 muH2^2 + muS2^2) / (16 pi^2 M_t^2), about 0.5%). It is set to
# zero at mu0; only its change with t matters.

_RG_SCALARS = ('lamH', 'lamS', 'kappa', 'muH2', 'muS2', 'Omega')
_RG_SHARED = ('g', 'gp', 'g3', 'yt', 'lnxi')


def rg_beta(Y, N, M):
    """d Y / d ln mu for Y = (lamH, lamS, kappa, muH2, muS2, g, gp, g3, yt, lnxi, Omega) on
    the last axis (one loop)."""
    lH, lS, k, mH, mS, g, gp, g3, yt = (Y[..., i] for i in range(9))
    gH = 3.0 * yt**2 - 2.25 * g**2 - 0.75 * gp**2
    b = np.stack([
        2 * (N + 8) * lH**2 + 0.5 * M * k**2 - 6 * yt**4
        + 0.375 * (2 * g**4 + (g**2 + gp**2)**2) + 4 * gH * lH,
        2 * (M + 8) * lS**2 + 0.5 * N * k**2,
        2 * (N + 2) * lH * k + 2 * (M + 2) * lS * k + 4 * k**2 + 2 * gH * k,
        2 * (N + 2) * lH * mH + M * k * mS + 2 * gH * mH,
        2 * (M + 2) * lS * mS + N * k * mH,
        -19.0 / 6.0 * g**3,
        41.0 / 6.0 * gp**3,
        -7.0 * g3**3,
        yt * (4.5 * yt**2 - 8 * g3**2 - 2.25 * g**2 - 17.0 / 12.0 * gp**2),
        -gH,
        0.5 * (N * mH**2 + M * mS**2),
    ], axis=-1)
    return b / (16.0 * np.pi**2)


class RGTable:
    """The running couplings of every parameter point on a grid in t = ln(mu/mu0),
    interpolated with cubic Hermite polynomials (value and beta function at the nodes),
    so the potential stays smooth in the fields.

    The gauge and Yukawa couplings and xi do not depend on the scalar couplings at one
    loop, so they are stored once; only the five scalar parameters are stored per point.

    t_np[p] is the scale where point p becomes non-perturbative: the first t at which
    |lambda_H|, |lambda_S| or |kappa| exceeds coupling_max (4 pi), i.e. near its Landau
    pole (inf if never), interpolated within the RG step (linearly in 1/|coupling|), so that
    it is continuous in the couplings at mu0. From there on the point's couplings are frozen, so nothing
    overflows; the potential beyond t_np is not meaningful and loop_phase_map gives such
    points no phase."""

    def __init__(self, P, sm, t_max, dt=0.05, coupling_max=4.0 * np.pi):
        # the inputs, so that the running can be redone for other couplings
        # (loop_boundaries.model_with_lambdas varies lambda_H, lambda_S continuously)
        self.sm, self.t_max_arg, self.coupling_max = dict(sm), t_max, coupling_max
        self.mu02 = float(P.mu2)
        self.N, self.M = P.n_goldstone + 1, P.n_goldstone_S + 1
        self.dt = dt
        nt = int(np.ceil(t_max / dt)) + 2
        self.t_max = (nt - 1) * dt
        n = P.muH2.size
        Y = np.empty((n, 11))
        Y[:, 0], Y[:, 1], Y[:, 2] = P.lamH, P.lamS, P.kappa
        Y[:, 3], Y[:, 4] = P.muH2, P.muS2
        Y[:, 5], Y[:, 6], Y[:, 7], Y[:, 8] = sm['g'], sm['gp'], sm['g3'], sm['yt']
        Y[:, 9] = 0.0
        Y[:, 10] = 0.0                       # Omega(mu0) = 0
        f = lambda Z: rg_beta(Z, self.N, self.M)
        own = [0, 1, 2, 3, 4, 10]            # per-point columns of Y: the five scalars and Omega
        self.S = np.empty((n, nt, 6))
        self.dS = np.empty((n, nt, 6))
        self.G = np.empty((nt, 5))
        self.dG = np.empty((nt, 5))
        self.t_np = np.full(n, np.inf)
        frozen = np.zeros(n, dtype=bool)
        for i in range(nt):
            d = f(Y)
            d[np.ix_(frozen, own)] = 0.0
            self.S[:, i], self.dS[:, i] = Y[:, own], d[:, own]
            self.G[i], self.dG[i] = Y[0, 5:10], d[0, 5:10]
            if i == nt - 1:
                break
            k1 = d
            k2 = f(Y + 0.5 * dt * k1)
            k3 = f(Y + 0.5 * dt * k2)
            k4 = f(Y + dt * k3)
            Y_new = Y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            with np.errstate(invalid='ignore', divide='ignore'):
                blown = ~frozen & ~(np.abs(Y_new[:, :3]) <= coupling_max).all(axis=1)
                # where in the step |coupling| = coupling_max: near a Landau pole 1/|coupling| is close to linear
                # in t, so interpolate that (this makes t_np continuous in the couplings at M_t)
                inv0, inv1 = 1.0 / np.abs(Y[blown, :3]), 1.0 / np.abs(Y_new[blown, :3])
                u = (inv0 - 1.0 / coupling_max) / (inv0 - inv1)
                u = np.where((np.abs(Y_new[blown, :3]) > coupling_max) & np.isfinite(u), np.clip(u, 0.0, 1.0), np.inf)
            if blown.any():
                um = np.min(u, axis=1)
                self.t_np[blown] = (i + np.where(np.isfinite(um), um, 1.0)) * dt
            frozen |= blown
            Y_new[np.ix_(frozen, own)] = Y[np.ix_(frozen, own)]
            Y = Y_new

    def at(self, pidx, t):
        """(scalars, shared): the running couplings at points pidx and scales t (any
        broadcastable shapes), on a new last axis: the 6 per-point scalars (lamH, lamS,
        kappa, muH2, muS2, Omega) and the 5 shared ones (g, gp, g3, yt, lnxi)."""
        s = np.asarray(t, dtype=float) / self.dt
        s = np.where(np.isfinite(s), s, (self.S.shape[1] - 1))   # guard: never index with nan
        k = np.clip(np.floor(s), 0, self.S.shape[1] - 2).astype(int)
        u = (s - k)[..., None]
        h00, h10 = 2 * u**3 - 3 * u**2 + 1, u**3 - 2 * u**2 + u
        h01, h11 = -2 * u**3 + 3 * u**2, u**3 - u**2
        pidx = np.asarray(pidx)
        sc = (h00 * self.S[pidx, k] + h10 * self.dt * self.dS[pidx, k]
              + h01 * self.S[pidx, k + 1] + h11 * self.dt * self.dS[pidx, k + 1])
        sh = (h00 * self.G[k] + h10 * self.dt * self.dG[k]
              + h01 * self.G[k + 1] + h11 * self.dt * self.dG[k + 1])
        return sc, sh


def attach_rg(P, sm, field2_max):
    """P made RG-improved: couplings given at mu0 = sqrt(P.mu2) and run with the SM
    couplings sm = dict(g, gp, g3, yt) (also at mu0), up to fields H^2 + S^2 <=
    field2_max. The W, Z and top are then included through sm, so P.species must be
    empty."""
    if P.species:
        raise ValueError("with RG improvement the W, Z and top come from sm; pass no species")
    t_max = 0.5 * np.log1p(field2_max / P.mu2) + 0.25
    return P._replace(rg=RGTable(P, sm, t_max), pidx=np.arange(P.muH2.size))


def _rg_t(x, y, P):
    return 0.5 * np.log1p((x + y) / P.rg.mu02)


def _rg_fixed_model(P, t):
    """The fixed-scale model at scale mu0 e^t, xi^2 there, and the vacuum energy Omega."""
    sc, sh = P.rg.at(P.pidx, t)
    Pf = P._replace(lamH=sc[..., 0], lamS=sc[..., 1], kappa=sc[..., 2], muH2=sc[..., 3],
                    muS2=sc[..., 4], mu2=P.rg.mu02 * np.exp(2.0 * t),
                    species=sm_species(sh[..., 0], sh[..., 1], sh[..., 3]), rg=None)
    return Pf, np.exp(2.0 * sh[..., 4]), sc[..., 5]


def _rg_V_at(x, y, P, t):
    Pf, xi2, omega = _rg_fixed_model(P, t)
    return _potential_fixed(xi2 * x, y, Pf) + omega


def potential(x, y, P):
    """V at (x, y) = (H^2, S^2): fixed-scale, or RG-improved if P.rg is set."""
    if P.rg is None:
        return _potential_fixed(x, y, P)
    return _rg_V_at(x, y, P, _rg_t(x, y, P))


def gradient(x, y, P):
    """(dV/dx, dV/dy): fixed-scale (analytic), or RG-improved if P.rg is set.

    For the RG-improved V this is its exact derivative: the fixed-scale gradient at the
    local scale, plus dV/dt dt/dx for the field dependence of the scale, with dV/dt by a
    central difference in t (the Hermite-interpolated couplings are smooth in t)."""
    if P.rg is None:
        return _gradient_fixed(x, y, P)
    x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    t = _rg_t(x, y, P)
    Pf, xi2, _ = _rg_fixed_model(P, t)
    gx, gy = _gradient_fixed(xi2 * x, y, Pf)
    dt = 1e-3
    dVdt = (_rg_V_at(x, y, P, t + dt) - _rg_V_at(x, y, P, t - dt)) / (2.0 * dt)
    dtdx = 0.5 / (P.rg.mu02 + x + y)
    return xi2 * gx + dVdt * dtdx, gy + dVdt * dtdx


def d2V_dx2_H_axis(x, P):
    """d^2V/dx^2 on the H axis (y = 0). There the mass matrix is diagonal, so this is
    exact and analytic (by central differences of the gradient for the RG-improved V)."""
    if P.rg is not None:
        h = 1e-5 * np.maximum(x, 1e-12)
        return (gradient(x + h, 0.0, P)[0] - gradient(x - h, 0.0, P)[0]) / (2.0 * h)
    out = 0.5 * P.lamH
    if P.eps == 0:
        return out + 0.0 * x
    A = -P.muH2 + 3.0 * P.lamH * x
    C = -P.muS2 + 0.5 * P.kappa * x
    loop = (9.0 * P.lamH**2 * _fpp(A, P.mu2, _SCALAR_C)
            + 0.25 * P.kappa**2 * _fpp(C, P.mu2, _SCALAR_C))
    if P.n_goldstone:
        loop = loop + P.n_goldstone * P.lamH**2 * _fpp(-P.muH2 + P.lamH * x, P.mu2,
                                                      _SCALAR_C)
    if P.n_goldstone_S:
        loop = loop + P.n_goldstone_S * 0.25 * P.kappa**2 * _fpp(C, P.mu2, _SCALAR_C)
    for sp in P.species:
        loop = loop + sp.n * sp.a**2 * _fpp(sp.a * x, P.mu2, sp.c)
    return out + P.eps * LOOP_FACTOR * loop


def d2V_dy2_S_axis(y, P):
    """d^2V/dy^2 on the S axis (x = 0); the mirror of d2V_dx2_H_axis."""
    if P.rg is not None:
        h = 1e-5 * np.maximum(y, 1e-12)
        return (gradient(0.0, y + h, P)[1] - gradient(0.0, y - h, P)[1]) / (2.0 * h)
    out = 0.5 * P.lamS
    if P.eps == 0:
        return out + 0.0 * y
    A = -P.muH2 + 0.5 * P.kappa * y
    C = -P.muS2 + 3.0 * P.lamS * y
    loop = (0.25 * P.kappa**2 * _fpp(A, P.mu2, _SCALAR_C)
            + 9.0 * P.lamS**2 * _fpp(C, P.mu2, _SCALAR_C))
    if P.n_goldstone:
        loop = loop + P.n_goldstone * 0.25 * P.kappa**2 * _fpp(A, P.mu2, _SCALAR_C)
    if P.n_goldstone_S:
        loop = loop + P.n_goldstone_S * P.lamS**2 * _fpp(-P.muS2 + P.lamS * y, P.mu2,
                                                        _SCALAR_C)
    for sp in P.species:
        loop = loop + sp.n * sp.b**2 * _fpp(sp.b * y, P.mu2, sp.c)
    return out + P.eps * LOOP_FACTOR * loop


def hessian_xy(x, y, P, rel=1e-5):
    """(Hxx, Hxy, Hyy) by central differences of the analytic gradient (x, y > 0)."""
    hx, hy = rel * x, rel * y
    gxp, gyp = gradient(x + hx, y, P)
    gxm, gym = gradient(x - hx, y, P)
    gxq, gyq = gradient(x, y + hy, P)
    gxn, gyn = gradient(x, y - hy, P)
    Hxx = (gxp - gxm) / (2.0 * hx)
    Hyy = (gyq - gyn) / (2.0 * hy)
    Hxy = 0.5 * ((gyp - gym) / (2.0 * hx) + (gxq - gxn) / (2.0 * hy))
    return Hxx, Hxy, Hyy


def loop_mu2s(P):
    """(muH2^(1), muS2^(1)) = -2 grad V at the origin: minus the origin's curvatures."""
    gx, gy = gradient(0.0, 0.0, P)
    return -2.0 * gx, -2.0 * gy


def OmegaH1_at(x1, P):
    """OmegaH^(1) at an H-axis extremum x1: muS2 / (muS2 + m_S^2(x1))."""
    mS2 = 2.0 * gradient(x1, 0.0, P)[1]
    with np.errstate(divide='ignore', invalid='ignore'):
        return P.muS2 / (P.muS2 + mS2)


def OmegaS1_at(y2, P):
    """OmegaS^(1) at an S-axis extremum y2: muH2 / (muH2 + m_H^2(y2))."""
    mH2 = 2.0 * gradient(0.0, y2, P)[0]
    with np.errstate(divide='ignore', invalid='ignore'):
        return P.muH2 / (P.muH2 + mH2)


def _transverse_positive(Omega1, mu2_transverse):
    """Whether the transverse mass at an axis extremum is positive, read off Omega^(1)
    with the tree-level relation m^2 = mu^2 (1/Omega - 1)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return mu2_transverse * (1.0 / Omega1 - 1.0) > 0


# ---------------------------------------------------------------------------------------
# Root finding
# ---------------------------------------------------------------------------------------

def _scales(P):
    """Natural field scales xbar = |muH2|/lamH and ybar = |muS2|/lamS (the tree VEVs),
    and the matching energy scale."""
    xbar = np.where(P.muH2 != 0, np.abs(P.muH2), 1.0) / np.abs(P.lamH)
    ybar = np.where(P.muS2 != 0, np.abs(P.muS2), 1.0) / np.abs(P.lamS)
    Vs = np.abs(P.muH2) * xbar + np.abs(P.muS2) * ybar
    return xbar, ybar, Vs


def _chunks(n, size):
    for i in range(0, n, size):
        yield np.arange(i, min(i + size, n))


def _bisect(fun, lo, hi, n_iter=60):
    """Vectorized bisection of fun on the brackets [lo, hi] (fun changes sign on each)."""
    flo = fun(lo, np.arange(lo.size)) > 0
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        fm = fun(mid, np.arange(lo.size)) > 0
        same = fm == flo
        lo = np.where(same, mid, lo)
        hi = np.where(same, hi, mid)
    return 0.5 * (lo + hi)


# With a cutoff, the searched region is the disk H^2 + S^2 <= cutoff^2. Stored in the figure
# scripts' cache keys, so that caches made with the earlier square search are not reused.
SEARCH_REGION = "disk"


def _log_window(P, decades, cutoff):
    """Per-point search window in u = ln(x/xbar) and v = ln(y/ybar): (lo, hi_x, hi_y).

    The lower end is always 10^-decades of the tree scale. The upper end is 10^+decades
    of it, or, with an absolute field cutoff, H^2 = cutoff^2 (resp. S^2 = cutoff^2). With a
    cutoff this square only bounds the scan: interior_extrema keeps the disk
    H^2 + S^2 <= cutoff^2 inside it."""
    xbar, ybar, _ = _scales(P)
    U = decades * np.log(10.0)
    lo = np.full(xbar.shape, -U)
    if cutoff is None:
        return lo, np.full(xbar.shape, U), np.full(xbar.shape, U)
    hx = np.maximum(np.log(cutoff**2 / xbar), lo + 1.0)
    hy = np.maximum(np.log(cutoff**2 / ybar), lo + 1.0)
    return lo, hx, hy


def axis_extrema(P, axis='H', decades=5.0, n_scan=800, chunk=1500, cutoff=None,
                 max_roots=12):
    """All extrema on one axis, for every parameter point in P (1D arrays).

    Scans dV/dx(x, 0) (or dV/dy(0, y)) on a log grid of x/xbar in
    [10^-decades, 10^+decades] (or up to x = cutoff^2), then bisects every sign change. Returns flat arrays
    (pix, field2, along, Omega1, V): pix is the parameter index, field2 the squared VEV,
    along the curvature along the axis, Omega1 = OmegaH^(1) (H axis) or OmegaS^(1)
    (S axis), and V the potential there.

    Uniqueness of the axis extremum is checked by counting sign changes, not by testing
    d^2V/dx^2 > 0 along the whole axis. The latter always fails in a tiny window where a
    tree mass crosses zero (the ln|m^2| spike), without producing any extra roots."""
    xbar, ybar, _ = _scales(P)
    ref = xbar if axis == 'H' else ybar
    lo_w, hx_w, hy_w = _log_window(P, decades, cutoff)
    hi_w = hx_w if axis == 'H' else hy_w
    t = np.linspace(0.0, 1.0, n_scan)

    def g_on_axis(q, Pq):
        if axis == 'H':
            return gradient(q, 0.0, Pq)[0]
        return gradient(0.0, q, Pq)[1]

    pix_all, lo_all, hi_all = [], [], []
    for idx in _chunks(P.muH2.size, chunk):
        Pc = P.take(idx).expand(1)
        u = lo_w[idx][:, None] + (hi_w - lo_w)[idx][:, None] * t[None, :]
        q = ref[idx][:, None] * np.exp(u)
        pos = g_on_axis(q, Pc) > 0
        ii, kk = np.nonzero(pos[:, :-1] != pos[:, 1:])
        pix_all.append(idx[ii])
        lo_all.append(u[ii, kk])
        hi_all.append(u[ii, kk + 1])
    pix = np.concatenate(pix_all)
    if pix.size == 0:
        e = np.array([])
        return pix, e, e, e, e
    lo, hi = np.concatenate(lo_all), np.concatenate(hi_all)
    # at most max_roots per point: more means V is pathological there (e.g. NaN)
    order = np.argsort(pix, kind='stable')
    pix, lo, hi = pix[order], lo[order], hi[order]
    rank = np.arange(pix.size) - np.searchsorted(pix, pix)
    keep = rank < max_roots
    pix, lo, hi = pix[keep], lo[keep], hi[keep]
    uroot = np.empty(pix.size)
    for idx in _chunks(pix.size, 20000):
        Pt, rt = P.take(pix[idx]), ref[pix[idx]]
        uroot[idx] = _bisect(lambda uu, k: g_on_axis(rt * np.exp(uu), Pt), lo[idx], hi[idx])
    Pt = P.take(pix)
    rt = ref[pix]
    q = rt * np.exp(uroot)

    if axis == 'H':
        along = d2V_dx2_H_axis(q, Pt)
        Omega1 = OmegaH1_at(q, Pt)
        V = potential(q, 0.0, Pt)
    else:
        along = d2V_dy2_S_axis(q, Pt)
        Omega1 = OmegaS1_at(q, Pt)
        V = potential(0.0, q, Pt)
    return pix, q, along, Omega1, V


def _newton_fd(F, z0, n_iter=40, max_step=0.5, fd_rel=1e-6):
    """Vectorized Newton on F: (N, k) -> (N, k), with a finite-difference Jacobian and
    the step clipped to max_step in every component."""
    z = np.array(z0, dtype=float)
    N, k = z.shape
    active = np.arange(N)
    for _ in range(n_iter):
        if active.size == 0:
            break
        Fa = lambda zz: F(zz, active)
        za = z[active]
        f = Fa(za)
        # freeze points that have converged to machine precision
        done = np.isfinite(f).all(axis=1) & (np.abs(f).max(axis=1) < 1e-14)
        active, za, f = active[~done], za[~done], f[~done]
        if active.size == 0:
            break
        J = np.empty((active.size, k, k))
        for j in range(k):
            h = fd_rel * np.maximum(1.0, np.abs(za[:, j]))
            zp, zm = za.copy(), za.copy()
            zp[:, j] += h
            zm[:, j] -= h
            J[:, :, j] = (Fa(zp) - Fa(zm)) / (2.0 * h[:, None])
        bad = ~np.isfinite(J).all(axis=(1, 2)) | ~np.isfinite(f).all(axis=1)
        bad |= np.abs(np.linalg.det(np.where(bad[:, None, None], np.eye(k), J))) < 1e-300
        J[bad] = np.eye(k)
        f = np.where(bad[:, None], 0.0, f)
        dz = -np.linalg.solve(J, f[..., None])[..., 0]
        z[active] = za + np.clip(dz, -max_step, max_step)
    return z, F(z, np.arange(N))


def interior_extrema(P, decades=4.0, n_scan=40, chunk=400, tol=1e-9, cutoff=None,
                     max_seeds=48, newton_chunk=20000):
    """All stationary points with x, y > 0, for every parameter point in P.

    Scans an n_scan x n_scan log grid of (x/xbar, y/ybar) over +/- decades (or, with a
    cutoff, over the disk H^2 + S^2 <= cutoff^2: cells wholly outside it are skipped and
    roots outside it dropped). Every cell in
    which both gradient components change sign seeds Newton's method in (ln x, ln y).
    Converged roots are deduplicated. Returns flat arrays (pix, x, y, kind, V), with kind
    one of 'min', 'max', 'saddle' from the (x, y) Hessian.

    Roots outside the scanned window are not found. Two roots closer together than one
    grid cell can show up as one, which only happens right at a fold. At most max_seeds
    cells per point seed Newton (far more than any regular potential needs; a flood of
    seeds means V is pathological there), and Newton runs in chunks of newton_chunk
    seeds, which bounds the memory used."""
    xbar, ybar, Vs = _scales(P)
    lo_w, hx_w, hy_w = _log_window(P, decades, cutoff)
    t = np.linspace(0.0, 1.0, n_scan)

    pix_s, u_s, v_s = [], [], []
    for idx in _chunks(P.muH2.size, chunk):
        Pc = P.take(idx).expand(2)
        u = lo_w[idx][:, None] + (hx_w - lo_w)[idx][:, None] * t[None, :]
        v = lo_w[idx][:, None] + (hy_w - lo_w)[idx][:, None] * t[None, :]
        X = xbar[idx][:, None, None] * np.exp(u)[:, :, None]
        Y = ybar[idx][:, None, None] * np.exp(v)[:, None, :]
        gx, gy = gradient(X, Y, Pc)
        cx = _cell_changes(gx > 0)
        cy = _cell_changes(gy > 0)
        pp, ii, jj = np.nonzero(cx & cy)
        if cutoff is not None:
            # skip cells wholly outside the disk: their lower corner is already outside
            near = (xbar[idx][pp] * np.exp(u[pp, ii]) + ybar[idx][pp] * np.exp(v[pp, jj])
                    <= cutoff**2)
            pp, ii, jj = pp[near], ii[near], jj[near]
        pix_s.append(idx[pp])
        u_s.append(0.5 * (u[pp, ii] + u[pp, ii + 1]))
        v_s.append(0.5 * (v[pp, jj] + v[pp, jj + 1]))
    pix = np.concatenate(pix_s)
    if pix.size == 0:
        e = np.array([])
        return pix, e, e, np.array([], dtype=object), e

    z0 = np.column_stack([np.concatenate(u_s), np.concatenate(v_s)])
    # When a point has more than max_seeds candidate cells -- e.g. along a nearly flat
    # valley, where dV/dx = 0 and dV/dy = 0 run almost on top of each other -- keep the
    # ones where the gradient is smallest, i.e. closest to a root, not the first found.
    resid = np.empty(pix.size)
    for sel in _chunks(pix.size, newton_chunk):
        x0 = xbar[pix[sel]] * np.exp(z0[sel, 0])
        y0 = ybar[pix[sel]] * np.exp(z0[sel, 1])
        gx, gy = gradient(x0, y0, P.take(pix[sel]))
        resid[sel] = np.hypot(xbar[pix[sel]] * gx, ybar[pix[sel]] * gy) / Vs[pix[sel]]
    resid = np.where(np.isfinite(resid), resid, np.inf)
    # Spread the kept seeds over field radius: rank the cells within each half decade of H^2 + S^2 and take the
    # best of every half decade before the second best of any. Ranking by resid alone favours small fields, where
    # the gradient in tree units is small anyway, and can crowd out a root at large field.
    rho2 = xbar[pix] * np.exp(z0[:, 0]) + ybar[pix] * np.exp(z0[:, 1])
    rbin = np.floor(2.0 * np.log10(rho2)).astype(int)
    by_bin = np.lexsort((resid, rbin, pix))
    starts = np.r_[0, np.flatnonzero((np.diff(pix[by_bin]) != 0) | (np.diff(rbin[by_bin]) != 0)) + 1]
    rank = np.arange(by_bin.size) - np.repeat(starts, np.diff(np.r_[starts, by_bin.size]))
    rank_of = np.empty(pix.size, dtype=int)
    rank_of[by_bin] = rank
    order = np.lexsort((resid, rank_of, pix))
    pix, z0 = pix[order], z0[order]
    keep = (np.arange(pix.size) - np.searchsorted(pix, pix)) < max_seeds
    pix, z0 = pix[keep], z0[keep]

    # residual scaled by the FIXED tree scales: scaling by x, y themselves would make
    # x -> 0 a spurious solution, since x dV/dx vanishes there trivially
    z, res = np.empty_like(z0), np.empty_like(z0)
    for sel in _chunks(pix.size, newton_chunk):
        Pt = P.take(pix[sel])
        xr, yr, Vr = xbar[pix[sel]], ybar[pix[sel]], Vs[pix[sel]]

        def F(zz, idx):
            xi, yi, Vi = xr[idx], yr[idx], Vr[idx]
            x, y = xi * np.exp(zz[:, 0]), yi * np.exp(zz[:, 1])
            gx, gy = gradient(x, y, Pt.take(idx))
            return np.column_stack([xi * gx / Vi, yi * gy / Vi])

        z[sel], res[sel] = _newton_fd(F, z0[sel])
    ok = (np.isfinite(res).all(axis=1) & (np.abs(res).max(axis=1) < tol)
          & (z[:, 0] >= lo_w[pix] - 2.0) & (z[:, 1] >= lo_w[pix] - 2.0)
          & (z[:, 0] <= hx_w[pix] + (0.0 if cutoff is not None else 2.0))
          & (z[:, 1] <= hy_w[pix] + (0.0 if cutoff is not None else 2.0)))
    if cutoff is not None:
        with np.errstate(over='ignore', invalid='ignore'):
            ok &= (xbar[pix] * np.exp(z[:, 0]) + ybar[pix] * np.exp(z[:, 1])
                   <= cutoff**2 * (1.0 + 1e-9))
    pix, z = pix[ok], z[ok]

    # deduplicate per parameter point
    key = np.column_stack([pix, np.round(z[:, 0] * 1e4), np.round(z[:, 1] * 1e4)])
    _, first = np.unique(key, axis=0, return_index=True)
    pix, z = pix[first], z[first]

    Pt = P.take(pix)
    x, y = xbar[pix] * np.exp(z[:, 0]), ybar[pix] * np.exp(z[:, 1])
    Hxx, Hxy, Hyy = hessian_xy(x, y, Pt)
    det, tr = Hxx * Hyy - Hxy**2, Hxx + Hyy
    kind = np.where(det < 0, 'saddle', np.where(tr > 0, 'min', 'max')).astype(object)
    return pix, x, y, kind, potential(x, y, Pt)


def _cell_changes(b):
    """For a boolean (N, n, n) corner array, whether each of the (n-1)^2 cells has corners
    of both signs."""
    s = (b[:, :-1, :-1].astype(int) + b[:, 1:, :-1] + b[:, :-1, 1:] + b[:, 1:, 1:])
    return (s > 0) & (s < 4)


# ---------------------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------------------

def _type_from_curvatures(a, b):
    if a > 0 and b > 0:
        return 'min'
    if a < 0 and b < 0:
        return 'max'
    return 'saddle'


# Each tree potential phase as (origin, H-axis extrema, S-axis extrema, interior extrema,
# ground state). Derived from tree_potential_phase_map_Omega; run_checks() confirms that
# this classification reproduces the tree map when the loop is switched off (eps = 0).
TREE_SIGNATURES = {
    ('max', ('min',), ('min',), ('saddle',), 'H'): 'Ia',
    ('max', ('min',), ('min',), ('saddle',), 'S'): 'IIa',
    ('max', ('min',), ('saddle',), (), 'H'): 'Ib',
    ('max', ('saddle',), ('min',), (), 'S'): 'IIb',
    ('max', ('saddle',), ('saddle',), ('min',), 'M'): 'IIIa',
    ('saddle', ('min',), (), (), 'H'): 'Ic',
    ('saddle', ('saddle',), (), ('min',), 'M'): 'IIIc',
    ('saddle', (), ('min',), (), 'S'): 'IIc',
    ('saddle', (), ('saddle',), ('min',), 'M'): 'IIIb',
    ('min', (), (), (), '0'): '0',
}

# ground-state kind -> tree quantum phase label, and -> branch number used by
# tq.TRANSITION_ORDER (0 = origin, 1 = H axis, 2 = S axis, 3 = mixed)
KIND_TO_QUANTUM_LABEL = {'0': '0', 'H': 'I', 'S': 'II', 'M': 'III'}
KIND_TO_BRANCH = {'0': 0, 'H': 1, 'S': 2, 'M': 3}

# phases with no tree counterpart: signature -> (code, label, description); filled in as
# they are met, so a phase keeps its code and color across every panel of a session
NEW_PHASES = {}
_FIRST_NEW_CODE = max(tq.PHASE_LABELS) + 1


def _describe(sig):
    p0, h, s, m, gs = sig
    fmt = lambda t: '[' + ', '.join(t) + ']' if t else 'none'
    return (f"origin {p0}; H axis {fmt(h)}; S axis {fmt(s)}; mixed {fmt(m)}; "
            f"ground state {dict(H='H axis', S='S axis', M='mixed', **{'0': 'origin'})[gs]}")


def _code_of_signature(sig):
    label = TREE_SIGNATURES.get(sig)
    if label is not None:
        return tq._LABEL_TO_POTENTIAL_CODE[label]
    if sig not in NEW_PHASES:
        q = KIND_TO_QUANTUM_LABEL[sig[4]]
        k = 1 + sum(1 for v in NEW_PHASES.values() if v[1].startswith(q + '*'))
        NEW_PHASES[sig] = (_FIRST_NEW_CODE + len(NEW_PHASES), f"{q}*{k}", _describe(sig))
    return NEW_PHASES[sig][0]


def _group(pix, n, *arrays):
    """Split flat per-root arrays into per-parameter-point lists."""
    order = np.argsort(pix, kind='stable')
    bounds = np.searchsorted(pix[order], np.arange(n + 1))
    # sort each array ONCE: indexing inside the loop would copy it per point and keep
    # every copy alive through its slice (memory ~ points x roots)
    arrays = [np.asarray(a)[order] for a in arrays]
    return [[a[bounds[i]:bounds[i + 1]].copy() for a in arrays] for i in range(n)]


_MORSE_SIGN = {'min': 1, 'max': 1, 'saddle': -1}


def index_sum(structure):
    """Sum over all stationary points of the full (H, S) plane of (-1)^(Morse index),
    from the extremum structure of one quadrant: the origin once, axis points twice,
    interior points four times. Where V rises at the edge of the searched region this
    must be 1 (Poincare-Hopf on a disk), so any other value means extrema were missed."""
    p0, h, s, m = structure
    return (_MORSE_SIGN[p0] + 2 * sum(_MORSE_SIGN[t] for t in h)
            + 2 * sum(_MORSE_SIGN[t] for t in s) + 4 * sum(_MORSE_SIGN[t] for t in m))


def _radial_slope(theta, P, cutoff):
    """D(theta) = dV/d(rho^2) on the circle H^2 + S^2 = cutoff^2 at (H, S) = cutoff (cos theta, sin theta):
    gx cos^2 theta + gy sin^2 theta, in units of the tree scale Vs / cutoff^2."""
    c2, s2 = np.cos(theta)**2, np.sin(theta)**2
    gx, gy = gradient(cutoff**2 * c2, cutoff**2 * s2, P)
    return (gx * c2 + gy * s2) * cutoff**2 / _scales(P)[2]


def boundary_minimum_slope(P, cutoff, n_theta=1025, n_bisect=50, chunk=4000):
    """(Dmin, theta_min): the smallest radial slope D = dV/d(rho^2) (see _radial_slope) over the local minima of V
    along the cutoff circle, 0 <= theta <= pi/2, and the direction where it is taken.

    There is no ground state below the cutoff exactly when Dmin < 0: then V restricted to the closed disk
    H^2 + S^2 <= cutoff^2 has a local minimum on its edge, a valley of V crossing the circle with V still falling
    outward along it, so V can be lowered by leaving the disk. (V merely decreasing outward along some ray is not
    enough: along the wall of a narrow valley, e.g. near a flat direction for kappa < 0, a ray can slope outward and
    down into the valley while V along the valley floor rises again past a minimum just inside the cutoff.) At a
    local minimum along the circle the gradient is radial, so D there is the slope along the valley floor.

    The minima along the circle are the sign changes (- to +) of dV/dtheta = cutoff^2 sin(2 theta) (gy - gx) on
    n_theta directions, refined by bisection, plus an axis where V rises away from it. A valley narrower than the
    spacing is still found, since the sign of gy - gx on either side is set by its walls. Dmin is +inf only where
    there is no minimum along the circle, which cannot happen (V has one there), so it is always finite. Where the set
    of minima along the circle changes, Dmin can jump; the NGS edge is solved by bisection on its sign, which allows
    that."""
    n = P.muH2.size
    Dmin, thmin = np.full(n, np.inf), np.full(n, np.nan)
    th = np.linspace(0.0, 0.5 * np.pi, n_theta)

    def T_at(t, Q):
        gx, gy = gradient(cutoff**2 * np.cos(t)**2, cutoff**2 * np.sin(t)**2, Q)
        return gy - gx

    for s0 in range(0, n, chunk):
        idx = np.arange(s0, min(s0 + chunk, n))
        Q = P.take(idx)
        T = np.stack([T_at(t, Q) for t in th], axis=-1)                  # (m, n_theta)
        pt, k = np.nonzero((T[:, :-1] < 0) & (T[:, 1:] > 0))            # V along the circle falls, then rises
        lo, hi = th[k], th[k + 1]
        if pt.size:
            Qc = Q.take(pt)
            for _ in range(n_bisect):
                mid = 0.5 * (lo + hi)
                neg = T_at(mid, Qc) < 0
                lo, hi = np.where(neg, mid, lo), np.where(neg, hi, mid)
        # the axes, where V rises away from them along the circle
        a0 = np.flatnonzero(T[:, 0] > 0)
        a1 = np.flatnonzero(T[:, -1] < 0)
        cand_pt = np.concatenate([pt, a0, a1])
        cand_th = np.concatenate([0.5 * (lo + hi), np.zeros(a0.size), np.full(a1.size, 0.5 * np.pi)])
        if not cand_pt.size:
            continue
        D = _radial_slope(cand_th, Q.take(cand_pt), cutoff)
        dm, tm = np.full(idx.size, np.inf), np.full(idx.size, np.nan)
        np.minimum.at(dm, cand_pt, D)
        lowest = D == dm[cand_pt]
        tm[cand_pt[lowest]] = cand_th[lowest]
        Dmin[idx], thmin[idx] = dm, tm
    return Dmin, thmin


def runaway_mask(P, decades, n_rays=33, cutoff=None):
    """Whether V is still decreasing outward at the edge of the searched region: along any
    of n_rays directions in the (x/xbar, y/ybar) quadrant at radius 10^decades, or, with a
    cutoff, along a valley of V crossing the circle H^2 + S^2 = cutoff^2 in field space
    (boundary_minimum_slope: V on the disk has a local minimum on its edge).

    If it is, the lowest minimum found is at best metastable: V keeps falling beyond the
    region searched (e.g. past the barrier that the top loop creates on the H axis). The
    ground state is then not known, so such points get no phase."""
    xbar, ybar, _ = _scales(P)
    run = np.zeros(P.muH2.size, dtype=bool)
    if cutoff is not None:
        return boundary_minimum_slope(P, cutoff)[0] < 0
    R = 10.0**decades
    for t in np.linspace(0.0, 0.5 * np.pi, n_rays):
        cx, cy = np.cos(t), np.sin(t)
        gx, gy = gradient(xbar * R * cx, ybar * R * cy, P)
        run |= (gx * xbar * cx + gy * ybar * cy) < 0
    return run


def loop_phase_map(r, kappa, lambdaH, lambdaS, mu, muH2=1.0, eps=1.0, n_goldstone=0,
                   species=(), axis_decades=5.0, interior_decades=6.0, interior_n=48,
                   mask_tree_unbounded=False, n_goldstone_S=0, cutoff=None, rg=None,
                   verbose=False):
    """One-loop potential and quantum phases at every (lambdaH, lambdaS) point.

    With rg = dict(g=..., gp=..., g3=..., yt=...) the potential is RG-improved (see
    attach_rg): all couplings, including these SM ones, are inputs at the scale mu and
    run to mu^2 = mu^2 + H^2 + S^2. W, Z and the top are then included automatically, so
    species must be empty. The fold solvers do not support this mode.

    r and mu are scalars. lambdaH and lambdaS (and kappa, which may also be a scalar) are
    arrays of any common shape (e.g. a meshgrid). Extrema are searched from 10^-decades of the tree field scale up to
    10^+decades of it, or over the disk H^2 + S^2 <= cutoff^2 (cutoff in units of |muH|)
    if one is given.
    Returns a dict of arrays of that shape:

        phase              potential-phase code (tree codes 1-10, new ones from 11 on, 0
                           where V runs away beyond the searched region, no minimum
                           was found, or -- with mask_tree_unbounded=True -- the tree
                           potential is unbounded below)
        runaway            V still decreasing at the edge of the searched region
                           (see runaway_mask)
        tree_bounded       the tree-level conditions of is_bounded_below_lambda
        quantum            tree quantum-phase code of the ground state
        gs_kind            '0', 'H', 'S' or 'M': where the ground state is
        gs_Hbar, gs_Sbar   ground-state fields in tree-normalized units, H/sqrt(xbar) etc.
        OmegaH, OmegaS     tree Omegas (the plot coordinates)
        OmegaH1, OmegaS1   loop Omegas at the axis extremum (nan unless it is unique)
        muH2_1, muS2_1     loop-corrected mass parameters at the origin
        nH, nS, nM         number of extrema on the H axis, on the S axis and off the axes
        axis_unique        nH <= 1 and nS <= 1
        extrema            per point: dict of the H-axis, S-axis and interior roots
        structure          per point: (origin, H-axis types, S-axis types, interior types)
                           -- the extremum structure without the ground state, defined
                           also where there is no ground state (runaway points); None
                           at non-perturbative points
        order              per point: the minima from deepest to shallowest, each as
                           (kind, index): kind '0', 'H', 'S' or 'M' and its index in that
                           list of `extrema` (also where there is no ground state)
        nonperturbative    with rg: a coupling passes 4 pi (a Landau pole) below the
                           largest field searched, or within one RG step (dt) above it;
                           such points are not searched
        index_ok           the extrema pass the Poincare-Hopf check (see index_sum);
                           checked where V does not run away, True elsewhere

    Boundedness is decided at one loop by default, through runaway: loops can stabilize a
    direction that is flat or slightly unbounded at tree level (kappa < 0 just past
    OmegaH OmegaS = 1, the Gildener-Weinberg situation), and then there is a minimum
    the tree analysis does not see. Pass mask_tree_unbounded=True to blank the
    tree-unbounded region instead, as the tree atlas does.
    """
    shape = np.broadcast(np.asarray(lambdaH), np.asarray(lambdaS), np.asarray(kappa)).shape
    P = make_model(r, kappa, lambdaH, lambdaS, mu, muH2=muH2, eps=eps,
                   n_goldstone=n_goldstone, species=species, n_goldstone_S=n_goldstone_S)
    n = P.muH2.size
    xbar, ybar, _ = _scales(P)
    nonpert = np.zeros(n, dtype=bool)
    if rg is not None:
        if cutoff is not None:
            field2 = np.full(n, float(cutoff)**2)     # largest H^2 + S^2 on the disk
        else:
            _, hx_w, hy_w = _log_window(P, max(axis_decades, interior_decades), cutoff)
            field2 = xbar * np.exp(hx_w) + ybar * np.exp(hy_w)
        P = attach_rg(P, rg, np.max(field2))
        # couplings past 4 pi somewhere in the searched region, or within one step of the RG table beyond it (the
        # couplings are frozen at the Landau pole, and the interpolation over that step bends V just inside the
        # cutoff, giving spurious extrema there): no phase, not searched
        nonpert = P.rg.t_np <= 0.5 * np.log1p(field2 / P.rg.mu02) + P.rg.dt

    tree_bounded = np.broadcast_to(
        np.asarray(tq.is_bounded_below_lambda(P.lamH, P.lamS, P.kappa)), (n,)).ravel()
    # every search runs on the perturbative points only, then is mapped back
    ok = np.flatnonzero(~nonpert)
    Pok = P.take(ok)
    runaway = np.zeros(n, dtype=bool)
    runaway[ok] = runaway_mask(Pok, max(axis_decades, interior_decades), cutoff=cutoff)
    allowed = ~runaway & ~nonpert & (tree_bounded if mask_tree_unbounded else True)
    mH1, mS1 = loop_mu2s(P)
    V0 = potential(0.0, 0.0, P)

    hp, hx, halong, hOm, hV = axis_extrema(Pok, 'H', decades=axis_decades, cutoff=cutoff,
                                           chunk=1500 if rg is None else 300)
    sp, sy, salong, sOm, sV = axis_extrema(Pok, 'S', decades=axis_decades, cutoff=cutoff,
                                           chunk=1500 if rg is None else 300)
    mp, mx, my, mkind, mV = interior_extrema(Pok, decades=interior_decades,
                                             n_scan=interior_n, cutoff=cutoff,
                                             chunk=400 if rg is None else 80)
    hp, sp, mp = ok[hp.astype(int)], ok[sp.astype(int)], ok[mp.astype(int)]

    Hroots = _group(hp, n, hx, halong, hOm, hV)
    Sroots = _group(sp, n, sy, salong, sOm, sV)
    Mroots = _group(mp, n, mx, my, mkind, mV)

    # Missed extrema show up as a failed index check (Poincare-Hopf). Typically an extremum that has just left an
    # axis (or the origin) still sits below the bottom of the search window, e.g. at x ~ 1e-6 xbar just past a
    # supercritical Omega^(1) = 1. Such points are searched again with a window reaching 7 decades further down.
    def quick_index(i):
        p0 = _type_from_curvatures(-mH1[i], -mS1[i])
        h = tuple(_type_from_curvatures(al, 1.0 if _transverse_positive(Om, P.muS2[i]) else -1.0)
                  for _, al, Om, _ in zip(*Hroots[i]))
        s = tuple(_type_from_curvatures(al, 1.0 if _transverse_positive(Om, P.muH2[i]) else -1.0)
                  for _, al, Om, _ in zip(*Sroots[i]))
        return index_sum((p0, h, s, tuple(Mroots[i][2])))
    # Two retries: (1) the window reaching 7 decades further down, with a finer scan; (2) a dense scan with no cap on
    # the number of Newton seeds. The cap keeps the cells where the gradient is smallest in tree units, which along a
    # nearly flat valley (kappa < 0 near OmegaH OmegaS = 1, where dV/dx = 0 and dV/dy = 0 run almost parallel) can
    # crowd out a root at large field, e.g. a minimum at H^2 + S^2 ~ 0.6 cutoff^2.
    redo = np.array([i for i in ok if not runaway[i] and quick_index(i) != 1], dtype=int)
    n_redo, n_fixed = redo.size, 0
    for attempt in (1, 2):
        if redo.size == 0:
            break
        Pr = P.take(redo)
        wide = max(axis_decades, interior_decades) + 7.0
        if attempt == 1:
            for axis, roots in (('H', Hroots), ('S', Sroots)):
                rp, *vals = axis_extrema(Pr, axis, decades=wide, n_scan=1600, cutoff=cutoff, chunk=300)
                for k, g in enumerate(_group(rp.astype(int), redo.size, *vals)):
                    roots[redo[k]] = g
            rp, *vals = interior_extrema(Pr, decades=wide, n_scan=2 * interior_n, cutoff=cutoff, chunk=80)
        else:
            rp, *vals = interior_extrema(Pr, decades=wide, n_scan=5 * interior_n, cutoff=cutoff, chunk=20,
                                         max_seeds=100000)
        for k, g in enumerate(_group(rp.astype(int), redo.size, *vals)):
            Mroots[redo[k]] = g
        still = np.array([quick_index(i) != 1 for i in redo], dtype=bool)
        n_fixed += int((~still).sum())
        redo = redo[still]
    if verbose and n_redo:
        print(f"  index check failed at {n_redo} points; {n_fixed} fixed by a wider or denser search")

    phase = np.zeros(n, dtype=int)
    quantum = np.zeros(n, dtype=int)
    gs_kind = np.full(n, '', dtype=object)
    gs_H = np.full(n, np.nan)
    gs_S = np.full(n, np.nan)
    OmH1 = np.full(n, np.nan)
    OmS1 = np.full(n, np.nan)
    nH = np.zeros(n, dtype=int)
    nS = np.zeros(n, dtype=int)
    nM = np.zeros(n, dtype=int)
    extrema = []
    structure = []
    order = []

    for i in range(n):
        p0 = _type_from_curvatures(-mH1[i], -mS1[i])
        candidates = [(V0[i], '0', 0.0, 0.0)] if p0 == 'min' else []
        # each minimum's identity: its kind and its index among that kind's extrema (in
        # the order they are listed in `extrema`), comparable between neighbouring points
        ids = [('0', 0)] if p0 == 'min' else []

        h_types = []
        for k, (x, along, Om, V) in enumerate(zip(*Hroots[i])):
            t = _type_from_curvatures(along, 1.0 if _transverse_positive(Om, P.muS2[i])
                                      else -1.0)
            h_types.append(t)
            if t == 'min':
                candidates.append((V, 'H', x, 0.0))
                ids.append(('H', k))
        s_types = []
        for k, (y, along, Om, V) in enumerate(zip(*Sroots[i])):
            t = _type_from_curvatures(along, 1.0 if _transverse_positive(Om, P.muH2[i])
                                      else -1.0)
            s_types.append(t)
            if t == 'min':
                candidates.append((V, 'S', 0.0, y))
                ids.append(('S', k))
        m_types = []
        for k, (x, y, kind, V) in enumerate(zip(*Mroots[i])):
            m_types.append(kind)
            if kind == 'min':
                candidates.append((V, 'M', x, y))
                ids.append(('M', k))
        # the minima from deepest to shallowest
        order.append(tuple(ids[c] for c in np.argsort([c[0] for c in candidates],
                                                      kind='stable')))

        nH[i], nS[i], nM[i] = len(h_types), len(s_types), len(m_types)
        if nH[i] == 1:
            OmH1[i] = Hroots[i][2][0]
        if nS[i] == 1:
            OmS1[i] = Sroots[i][2][0]
        extrema.append(dict(origin=p0, H=list(zip(Hroots[i][0], h_types)),
                            minima_bar=[(np.sqrt(c[2] / xbar[i]), np.sqrt(c[3] / ybar[i]))
                                        for c in candidates],
                            S=list(zip(Sroots[i][0], s_types)),
                            M=list(zip(Mroots[i][0], Mroots[i][1], m_types))))

        structure.append(None if nonpert[i] else
                         (p0, tuple(h_types), tuple(s_types), tuple(sorted(m_types))))
        if not allowed[i] or not candidates:
            continue
        V, kind, x, y = min(candidates, key=lambda c: c[0])
        sig = (p0, tuple(h_types), tuple(s_types), tuple(sorted(m_types)), kind)
        phase[i] = _code_of_signature(sig)
        quantum[i] = tq._QUANTUM_LABEL_TO_CODE[KIND_TO_QUANTUM_LABEL[kind]]
        gs_kind[i] = kind
        gs_H[i], gs_S[i] = np.sqrt(x / xbar[i]), np.sqrt(y / ybar[i])

    with np.errstate(divide='ignore', invalid='ignore'):
        OmegaH, OmegaS = tree_Omegas(r, P.kappa, P.lamH, P.lamS)
    out = dict(phase=phase, quantum=quantum, gs_kind=gs_kind, gs_Hbar=gs_H, gs_Sbar=gs_S,
               OmegaH=OmegaH, OmegaS=OmegaS, OmegaH1=OmH1, OmegaS1=OmS1,
               muH2_1=mH1, muS2_1=mS1, nH=nH, nS=nS, nM=nM,
               axis_unique=(nH <= 1) & (nS <= 1), tree_bounded=tree_bounded,
               runaway=runaway, nonperturbative=nonpert,
               index_ok=np.array([st is None or runaway[i] or index_sum(st) == 1
                                  for i, st in enumerate(structure)]))
    out = {k: np.asarray(v).reshape(shape) for k, v in out.items()}
    out['extrema'] = np.array(extrema, dtype=object).reshape(shape)
    out['structure'] = np.empty(n, dtype=object)
    out['structure'][:] = structure
    out['structure'] = out['structure'].reshape(shape)
    out['order'] = np.empty(n, dtype=object)
    out['order'][:] = order
    out['order'] = out['order'].reshape(shape)
    out['model'] = P
    out['params'] = dict(r=r, kappa=kappa, mu=mu, muH2=muH2, eps=eps,
                         n_goldstone=n_goldstone, species=species,
                         n_goldstone_S=n_goldstone_S, cutoff=cutoff, rg=rg)

    if verbose:
        frac = 1.0 - out['axis_unique'].mean()
        frac_run = out['runaway'].mean()
        lam_max = max(np.max(P.lamH), np.max(P.lamS), np.max(np.abs(P.kappa)))
        kap = f"{kappa:g}" if np.ndim(kappa) == 0 else f"[{np.min(P.kappa):g}, {np.max(P.kappa):g}]"
        print(f"r={r:g}, kappa={kap}: axis extrema non-unique at {100 * frac:.2f}%, "
              f"runaway at {100 * frac_run:.2f}% of points; "
              f"largest coupling {lam_max:.3g}; "
              f"extrema failing the index check: {int((~out['index_ok']).sum())}")
    return out


# ---------------------------------------------------------------------------------------
# First-order jumps of the ground state
# ---------------------------------------------------------------------------------------

def ground_state_jumps(res, jump_tol=0.3):
    """Neighbor pairs where the ground state jumps between two distinct minima, as two
    boolean edge arrays (across columns, across rows).

    A jump is flagged when the two ground states are far apart (relative distance in
    tree-normalized fields above jump_tol) AND the ground state of one side still exists
    as a local minimum on the other side. That is what a first-order transition is: two
    coexisting minima exchanging the lead. A ground state that merely moves fast (near
    OmegaH OmegaS = 1 it grows like 1/(OmegaH OmegaS - 1)) has no such partner and is
    not flagged. Neither is a continuous P1 -> P3 transition: on the P3 side, P1 is a
    saddle, not a minimum."""
    kind = res['gs_kind']
    Hb, Sb = res['gs_Hbar'], res['gs_Sbar']
    ext = res['extrema']

    def first_order(ia, ib):
        if kind[ia] == '' or kind[ib] == '':
            return False
        ga, gb = np.array([Hb[ia], Sb[ia]]), np.array([Hb[ib], Sb[ib]])
        if np.hypot(*(ga - gb)) <= jump_tol * max(np.hypot(*ga), np.hypot(*gb), 1e-12):
            return False
        for g_self, other in ((ga, ib), (gb, ia)):
            scale = 0.5 * jump_tol * max(np.hypot(*g_self), 1e-12)
            if any(np.hypot(*(np.asarray(m) - g_self)) < scale
                   for m in ext[other]['minima_bar']):
                return True
        return False

    ny, nx = kind.shape
    across_cols = np.zeros((ny, nx - 1), dtype=bool)
    across_rows = np.zeros((ny - 1, nx), dtype=bool)
    for i in range(ny):
        for j in range(nx - 1):
            across_cols[i, j] = first_order((i, j), (i, j + 1))
    for i in range(ny - 1):
        for j in range(nx):
            across_rows[i, j] = first_order((i, j), (i + 1, j))
    return across_cols, across_rows
