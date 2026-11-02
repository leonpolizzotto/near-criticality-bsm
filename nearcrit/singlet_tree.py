"""The tree-level Higgs-singlet potential: its extrema, potential and quantum phases, and the minimum V_min with its
derivatives, in the Omega and lambda parametrizations of Sec. 5.

    V = -mu_H^2 |H|^2 / 2 - mu_S^2 |S|^2 / 2 + lambda_H |H|^4 / 4 + lambda_S |S|^4 / 4 + kappa |H|^2 |S|^2 / 4,

with Omega_H = 2 lambda_H mu_S^2 / (kappa mu_H^2) and Omega_S = 2 lambda_S mu_H^2 / (kappa mu_S^2). The extrema are
P0 (origin), P1 (H axis), P2 (S axis) and P3 (mixed). 
"""
import numpy as np
from scipy import ndimage


def Omegas_from_lambdas(lambdaH, lambdaS, muH2, muS2, kappa):
    """(OmegaH, OmegaS, omega_prod) induced by the physical parameters.

    OmegaH = 2 muS2 lambdaH / (muH2 kappa) and OmegaS = 2 muH2 lambdaS / (muS2 kappa),
    so both blow up to +/-inf on the mu axes. The product, by contrast, is the constant
    4 lambdaH lambdaS / kappa^2 over the whole plane -- it does not depend on the mus at
    all -- but evaluating it as OmegaH * OmegaS gives 0 * inf = nan on those axes, so it
    is returned separately and computed directly."""
    muH2 = np.asarray(muH2, dtype=float)
    muS2 = np.asarray(muS2, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        OmegaH = 2.0 * muS2 * lambdaH / (muH2 * kappa)
        OmegaS = 2.0 * muH2 * lambdaS / (muS2 * kappa)
    # kappa joins the broadcast so that an ARRAY kappa with scalar mus -- the kappa
    # sweep -- comes out path-shaped rather than raising; with a scalar kappa, as both
    # two-dimensional sweeps pass, the shape is unchanged
    omega_prod = np.broadcast_to(4.0 * lambdaH * lambdaS / kappa**2,
                                 np.broadcast(muH2, muS2,
                                              np.asarray(kappa, dtype=float)).shape)
    return OmegaH, OmegaS, omega_prod


def lambdas_from_Omegas(OmegaH, OmegaS, muH2, muS2, kappa):
    """(lambdaH, lambdaS) induced by the physical parameters.

    Inverting the Omegas_from_lambdas mapping. Note the product is fixed by the Omegas
    alone, lambdaH lambdaS = kappa^2 OmegaH OmegaS / 4, independently of the mus."""
    with np.errstate(divide='ignore', invalid='ignore'):
        lambdaH = muH2 * kappa * OmegaH / (2.0 * muS2)
        lambdaS = muS2 * kappa * OmegaS / (2.0 * muH2)
    return lambdaH, lambdaS


def is_bounded_below_lambda(lambdaH, lambdaS, kappa):
    """Eqs. (16)-(17). With (lambdaH, lambdaS, kappa) as the independent parameters this
    is a single global check, not a per-point mask: either the whole (muH2, muS2) plane
    is bounded from below or none of it is."""
    lambdaH = np.asarray(lambdaH, dtype=float)
    lambdaS = np.asarray(lambdaS, dtype=float)
    with np.errstate(invalid='ignore'):
        return ((lambdaH > 0) & (lambdaS > 0)
                & ((kappa >= 0) | (lambdaH * lambdaS > kappa**2 / 4.0)))


def is_bounded_below_Omega(OmegaH, OmegaS, muH2, muS2, kappa):
    """Eqs. (16)-(17) as a mask over the (OmegaH, OmegaS) plane at fixed (mus, kappa).

    Here it genuinely is a per-point condition, and it is what carves Fig. 1 into its
    accessible and inaccessible regions: lambdaH > 0 forces sign(OmegaH) =
    sign(muH2 muS2 kappa) and likewise for OmegaS, which empties two quadrants, while
    lambdaH lambdaS > kappa^2/4 for kappa < 0 reduces to OmegaH OmegaS > 1."""
    
    lambdaH, lambdaS = lambdas_from_Omegas(OmegaH, OmegaS, muH2, muS2, kappa)
    return is_bounded_below_lambda(lambdaH, lambdaS, kappa)


def Vbar_Omega(Hbar, Sbar, OmegaH, OmegaS, muH2, muS2, kappa):
    """The tree-level potential in the normalized fields (Hbar, Sbar), in the Omega parametrization."""
    return (muH2*muS2/kappa) * ((-Hbar**2 + Hbar**4/2)/OmegaH +
                                (-Sbar**2 + Sbar**4/2)/OmegaS +
                                Hbar**2 * Sbar**2/(OmegaH * OmegaS))




# The mixed extremum, in normalized variables on both sides. Both return the SQUARED
# normalized fields carrying the sign[muH2]/sign[muS2] of Eq. (14),
#
#     Hbar3 = sign[muH2] * Hbar^2 = 2 lambdaH |H|^2 / muH2 ,
#
# so they solve dVbar/dHbar = dVbar/dSbar = 0 for Hbar^2, Sbar^2 rather than for Hbar,
# Sbar, and must not be fed to Vbar_* directly. The two routes return the same numbers;
# the bare VEVs, if wanted, are |H|^2 = muH2 Hbar3 / (2 lambdaH) and
# |S|^2 = muS2 Sbar3 / (2 lambdaS).
#
# The third return value is each side's natural common denominator; they differ only by
# the constant factor D = kappa^2 * denom.


def P3_vevs_Omega(OmegaH, OmegaS, muH2, muS2, kappa):
    """(Hbar3, Sbar3, denom) at the mixed extremum P3, with denom = OmegaH OmegaS - 1."""
    OmegaH = np.asarray(OmegaH, dtype=float)
    OmegaS = np.asarray(OmegaS, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = OmegaH * OmegaS - 1.0
        Hbar3 = OmegaH * (OmegaS - 1.0) / denom
        Sbar3 = OmegaS * (OmegaH - 1.0) / denom
    return Hbar3, Sbar3, denom


def P3_exists_Omega(OmegaH, OmegaS, muH2, muS2, kappa):
    """P3 exists where both squared VEVs come out positive."""
    Hbar3, Sbar3, _ = P3_vevs_Omega(OmegaH, OmegaS, muH2, muS2, kappa)
    with np.errstate(invalid='ignore'):
        return (muH2 * Hbar3 > 0) & (muS2 * Sbar3 > 0)


def Vmin_branches_Omega(OmegaH, OmegaS, muH2, muS2, kappa):
    """The four candidate extremum values [V_P0, V_P1, V_P2, V_P3], with branches that do
    not exist zero-filled."""
    OmegaH = np.asarray(OmegaH, dtype=float)
    OmegaS = np.asarray(OmegaS, dtype=float)
    shape = np.broadcast(OmegaH, OmegaS, np.asarray(muH2, dtype=float),
                         np.asarray(muS2, dtype=float)).shape
    _, _, D = P3_vevs_Omega(OmegaH, OmegaS, muH2, muS2, kappa)

    with np.errstate(divide='ignore', invalid='ignore'):
        VP0 = np.zeros(shape)
        VP1 = np.where(muH2 > 0, Vbar_Omega(1.0, 0.0, OmegaH, OmegaS, muH2, muS2, kappa), 0.0)
        VP2 = np.where(muS2 > 0, Vbar_Omega(0.0, 1.0, OmegaH, OmegaS, muH2, muS2, kappa), 0.0)
        # Vbar_Omega substituted at (Hbar^2, Sbar^2) = (Hbar3, Sbar3) and simplified; this
        # is the V3 that the dVmin_Omega*/ddVmin_Omega* branch formulae differentiate
        VP3 = np.where(P3_exists_Omega(OmegaH, OmegaS, muH2, muS2, kappa),
                       (muH2 * muS2 / kappa) * (2.0 - OmegaH - OmegaS) / (2.0 * D), 0.0)

    return list(np.broadcast_arrays(VP0, VP1, VP2, VP3))


def _argmin_branches(branches, return_P):
    """P labels which of the four candidate extrema (0, 1, 2, 3) realizes the minimum at
    each point; ties (e.g. a disallowed P3, which is zero-filled and so degenerate with
    P0) resolve to the lowest index."""
    stacked = np.stack(branches)
    if return_P:
        return np.min(stacked, axis=0), np.argmin(stacked, axis=0)
    return np.min(stacked, axis=0)


def Vmin_Omega(OmegaH, OmegaS, muH2, muS2, kappa, return_P=False):
    """Vmin over an (OmegaH, OmegaS) grid at fixed (muH2, muS2, kappa)."""
    return _argmin_branches(
        Vmin_branches_Omega(OmegaH, OmegaS, muH2, muS2, kappa), return_P)


def select_branch(P, branches):
    """Pick branch P at each point, where branches is a list of the four (broadcast)
    per-extremum arrays [P0, P1, P2, P3]. P may be a scalar or an array of the same
    shape as the branches, e.g. the argmin returned by Vmin_*(..., return_P=True)."""
    stacked = np.stack(np.broadcast_arrays(*branches))
    P = np.broadcast_to(np.asarray(P, dtype=int), stacked.shape[1:])
    return np.take_along_axis(stacked, P[np.newaxis, ...], axis=0)[0]


# Derivatives with respect to the swept coordinates of the Omega parametrization, i.e.
# at fixed (muH2, muS2, kappa). Every branch is (muH2 muS2 / kappa) times a function of
# the Omegas alone, so the mus enter only through that prefactor.


def dVmin_OmegaH(P, OmegaH, OmegaS, muH2, muS2, kappa):
    """dV_min/dOmega_H on branch P (0-3), at fixed (muH2, muS2, kappa)."""
    OmegaH = np.asarray(OmegaH, dtype=float)
    OmegaS = np.asarray(OmegaS, dtype=float)
    zero = np.zeros(np.broadcast(OmegaH, OmegaS).shape)
    with np.errstate(divide='ignore', invalid='ignore'):
        branches = [
            zero,
            muH2 * muS2 / (2 * kappa * OmegaH**2),
            zero,
            (muH2 * muS2 * (OmegaS - 1.0)**2) / (2 * kappa * (OmegaH * OmegaS - 1.0)**2),
        ]
    return select_branch(P, branches)


def dVmin_OmegaS(P, OmegaH, OmegaS, muH2, muS2, kappa):
    """dV_min/dOmega_S on branch P (0-3), at fixed (muH2, muS2, kappa)."""
    OmegaH = np.asarray(OmegaH, dtype=float)
    OmegaS = np.asarray(OmegaS, dtype=float)
    zero = np.zeros(np.broadcast(OmegaH, OmegaS).shape)
    with np.errstate(divide='ignore', invalid='ignore'):
        branches = [
            zero,
            zero,
            muH2 * muS2 / (2 * kappa * OmegaS**2),
            (muH2 * muS2 * (OmegaH - 1.0)**2) / (2 * kappa * (OmegaH * OmegaS - 1.0)**2),
        ]
    return select_branch(P, branches)


def ddVmin_OmegaH(P, OmegaH, OmegaS, muH2, muS2, kappa):
    """d^2V_min/dOmega_H^2 on branch P (0-3), at fixed (muH2, muS2, kappa)."""
    OmegaH = np.asarray(OmegaH, dtype=float)
    OmegaS = np.asarray(OmegaS, dtype=float)
    zero = np.zeros(np.broadcast(OmegaH, OmegaS).shape)
    with np.errstate(divide='ignore', invalid='ignore'):
        branches = [
            zero,
            - muH2 * muS2 / (kappa * OmegaH**3),
            zero,
            - (muH2 * muS2 * (OmegaS - 1.0)**2 * OmegaS) / (kappa * (OmegaH * OmegaS - 1.0)**3),
        ]
    return select_branch(P, branches)


def ddVmin_OmegaS(P, OmegaH, OmegaS, muH2, muS2, kappa):
    """d^2V_min/dOmega_S^2 on branch P (0-3), at fixed (muH2, muS2, kappa)."""
    OmegaH = np.asarray(OmegaH, dtype=float)
    OmegaS = np.asarray(OmegaS, dtype=float)
    zero = np.zeros(np.broadcast(OmegaH, OmegaS).shape)
    with np.errstate(divide='ignore', invalid='ignore'):
        branches = [
            zero,
            zero,
            - muH2 * muS2 / (kappa * OmegaS**3),
            - (muH2 * muS2 * (OmegaH - 1.0)**2 * OmegaH) / (kappa * (OmegaH * OmegaS - 1.0)**3),
        ]
    return select_branch(P, branches)


def ddVmin_OmegaH_OmegaS(P, OmegaH, OmegaS, muH2, muS2, kappa):
    """Mixed derivative. P1 and P2 depend on only one Omega each, so it is carried
    entirely by P3, where it is symmetric under H <-> S."""
    OmegaH = np.asarray(OmegaH, dtype=float)
    OmegaS = np.asarray(OmegaS, dtype=float)
    zero = np.zeros(np.broadcast(OmegaH, OmegaS).shape)
    with np.errstate(divide='ignore', invalid='ignore'):
        branches = [
            zero,
            zero,
            zero,
            (muH2 * muS2 * (OmegaH - 1.0) * (OmegaS - 1.0)
             / (kappa * (OmegaH * OmegaS - 1.0)**3)),
        ]
    return select_branch(P, branches)




# Derivatives PROJECTED ALONG THE CONTOUR, built from the per-coordinate ones above (or
# from their lambda-side counterparts below). The two-dimensional sweeps walk an ellipse
# through the swept plane, and the direction of that walk is a perfectly good direction to
# differentiate in -- it is, in fact, the direction the 1D panels are already plotted
# against. Projecting onto it collapses the four per-coordinate panels into two, one per
# derivative order, which is the natural Ehrenfest readout: V_min is continuous at a
# change of global minimum, so a jump in the FIRST projected derivative is a first-order
# transition and a jump in only the SECOND is a second-order one.


def _ellipse_unit_tangent(c0, c1, x0, y0, a, b):
    """Unit tangent of the ellipse (c0 - x0)^2/a^2 + (c1 - y0)^2/b^2 = 1 at a point ON it,
    pointing in the direction of increasing theta (counterclockwise).

    Taken from the coordinates rather than from theta: on the contour c0 = x0 + a cos th
    and c1 = y0 + b sin th, so dc0/dth = -a sin th = -a (c1 - y0)/b and dc1/dth =
    b cos th = b (c0 - x0)/a, with no theta left in either. That is what lets the
    projected derivatives keep the ordinary fn(P, ...) signature of every other derivative
    here, and hence pass through _report_switches -- which locates a crossing by bisecting
    in theta but then evaluates the derivatives at COORDINATES -- with no changes at all.

    Only the direction survives the normalization, so the (arbitrary) overall scale of the
    two components above is irrelevant; a and b are not, since they set their ratio."""
    t0 = -a * (np.asarray(c1, dtype=float) - y0) / b
    t1 = b * (np.asarray(c0, dtype=float) - x0) / a
    norm = np.hypot(t0, t1)
    with np.errstate(divide='ignore', invalid='ignore'):
        return t0 / norm, t1 / norm


def _directional_deriv_fns(grad_fns, hess_fns, ellipse, coord_idx=(0, 1)):
    """The first and second derivatives of V_min along the contour tangent, returned as a
    [(fn, name), ...] list matching the deriv_fns contract of _report_switches.

        grad_fns = (dV/dc0, dV/dc1)
        hess_fns = (d2V/dc0^2, d2V/dc0 dc1, d2V/dc1^2)
        ellipse  = (x0, y0, a, b), i.e. ctx['ellipse']

    coord_idx says which two positional arguments AFTER P are the swept coordinates:
    (0, 1) for the Omega family, whose signature is (P, OmegaH, OmegaS, muH2, muS2,
    kappa), and (2, 3) for the lambda family, (P, lambdaH, lambdaS, muH2, muS2, kappa).
    Everything else is passed straight through, so the returned functions have exactly the
    signature of the ones they are built from.

    These are PURE PROJECTIONS -- t.grad V and t^T Hess V t with t the UNIT tangent --
    not d/dtheta or d/d(arc length) along the curve. The distinction is only in the second
    one, and it matters: the true second derivative of V restricted to the curve carries
    an extra curvature term, curv * (grad V . n), which mixes the FIRST derivatives into
    the second-derivative panel. A first-order transition would then jump in BOTH panels
    and the Ehrenfest reading described above would be lost. The projected Hessian sees
    only second derivatives, so each order of transition shows up in exactly one panel.

    The mixed second derivative is genuinely needed for the second one and is exact on
    every branch (ddVmin_OmegaH_OmegaS, ddVmin_muH2_muS2), so nothing here is finite
    differenced."""
    x0, y0, a, b = ellipse
    i, j = coord_idx
    d0, d1 = grad_fns
    h00, h01, h11 = hess_fns

    def _tangent(args):
        return _ellipse_unit_tangent(args[i], args[j], x0, y0, a, b)

    def dVmin_tangent(P, *args):
        t0, t1 = _tangent(args)
        with np.errstate(divide='ignore', invalid='ignore'):
            return t0 * d0(P, *args) + t1 * d1(P, *args)

    def ddVmin_tangent(P, *args):
        t0, t1 = _tangent(args)
        with np.errstate(divide='ignore', invalid='ignore'):
            return (t0 * t0 * h00(P, *args)
                    + 2.0 * t0 * t1 * h01(P, *args)
                    + t1 * t1 * h11(P, *args))

    return [(dVmin_tangent, "dV/dtangent"), (ddVmin_tangent, "d2V/dtangent^2")]


def tree_potential_phase_map_Omega(OmegaH, OmegaS, muH2, muS2, omega_prod=None):
    """Potential phase, vectorized in all four arguments.

    omega_prod supplies OmegaH*OmegaS; the lambda-side caller passes it explicitly because
    there it is a constant that the naive product evaluates as nan on the mu axes."""
    OmegaH, OmegaS, muH2, muS2 = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (OmegaH, OmegaS, muH2, muS2)))
    if omega_prod is None:
        with np.errstate(invalid='ignore'):
            omega_prod = OmegaH * OmegaS
    omega_prod = np.broadcast_to(np.asarray(omega_prod, dtype=float), OmegaH.shape)

    qPP = (muH2 > 0) & (muS2 > 0)
    qPN = (muH2 > 0) & (muS2 < 0)
    qNP = (muH2 < 0) & (muS2 > 0)
    qNN = (muH2 < 0) & (muS2 < 0)

    phaseIa   = qPP & (0 < OmegaH) & (OmegaH < OmegaS) & (OmegaS < 1)
    phaseIb   = qPP & (0 < OmegaH) & (OmegaH < 1) & (OmegaS > 1)
    phaseIIa  = qPP & (0 < OmegaS) & (OmegaS < OmegaH) & (OmegaH < 1)
    phaseIIb  = qPP & (OmegaH > 1) & (0 < OmegaS) & (OmegaS < 1)
    phaseIIIa = qPP & (((OmegaH > 1) & (OmegaS > 1))
                       | ((OmegaH < 0) & (OmegaS < 0) & (omega_prod > 1)))
    
    
    phaseIc   = qPN & (((OmegaH > 1) & (0 < OmegaS) & (omega_prod > 1))
                       | ((OmegaH < 0) & (OmegaS < 0)))
    phaseIIIc = qPN & (0 < OmegaH) & (OmegaH < 1) & (0 < OmegaS) & (omega_prod > 1)
        

    phaseIIc  = qNP & (((0 < OmegaH) & (OmegaS > 1) & (omega_prod > 1))
                       | ((OmegaH < 0) & (OmegaS < 0)))
    phaseIIIb = qNP & (0 < OmegaH) & (0 < OmegaS) & (OmegaS < 1) & (omega_prod > 1)

    # for muH2, muS2 < 0 the origin is the only extremum (P1/P2 need a positive mu^2, and
    # P3 would need both squared VEVs negative), so the whole allowed region is phase V
    phase0 = qNN & (((0 > OmegaH) & (0 > OmegaS) & (omega_prod > 1))
                    | ((OmegaH > 0) & (OmegaS > 0)))

    out = np.zeros(OmegaH.shape, dtype=int)
    out[phaseIa]   = 1
    out[phaseIb]   = 2
    out[phaseIc]   = 3
    out[phaseIIa]  = 4
    out[phaseIIb]  = 5
    out[phaseIIc]  = 6
    out[phaseIIIa] = 7
    out[phaseIIIb] = 8
    out[phaseIIIc] = 9
    out[phase0]   = 10

    return out


def tree_potential_phase_map_lambda(lambdaH, lambdaS, muH2, muS2, kappa):
    """Potential phase over a (muH2, muS2) grid at fixed (lambdaH, lambdaS, kappa).

    The Omegas depend on the mus only through the ratio muS2/muH2 while the phase map also
    branches on each sign separately, so the phase is constant on every open ray from the
    origin: the plane is a fan of wedges bounded by the four half-axes and the rays where
    OmegaH = 1 or OmegaS = 1. The radius only sets scale."""
    OmegaH, OmegaS, omega_prod = Omegas_from_lambdas(lambdaH, lambdaS, muH2, muS2, kappa)
    return tree_potential_phase_map_Omega(OmegaH, OmegaS, muH2, muS2, omega_prod=omega_prod)


PHASE_COLORS = ['#ffffff', #none
                '#66D6C4', #phase Ia
                "#009988", #phase Ib
                "#005E53", #phase Ic
                '#F8A7B0', #phase IIa
                "#EE6677", #phase IIb
                "#A3323F", #phase IIc
                "#8FB4D8", #phase IIIa
                "#4477AA", #phase IIIb
                "#24486E", #phase IIIc
                "#AA3377", #phase 0
                ]


# Opacity the phase colors are drawn at, in the 2D maps and in the background shading of
# the 1D panels alike. The colorbar key uses it too, so the swatches match the figure.
PHASE_ALPHA = 0.55

# phase codes as assigned by tree_potential_phase_map_Omega
PHASE_LABELS = {
    1:  "Ia",
    2:  "Ib",
    3:  "Ic",
    4:  "IIa",
    5:  "IIb",
    6:  "IIc",
    7:  "IIIa",
    8:  "IIIb",
    9:  "IIIc",
    10: "0",
}


# Quantum phase = a further relabeling of the potential phase: several potential phases
# can turn out to realize the same quantum phase, so this maps potential-phase LABELS (as
# in PHASE_LABELS above, e.g. "Ia", "III") to a quantum-phase label of your choosing.
# Giving two potential-phase labels the same quantum-phase label merges them into one
# quantum phase; anything else keeps its own. This dict is the only thing to edit -- the
# numeric codes, the lookup table below and the plot hatching are all derived from it
# automatically, so there is no bookkeeping of quantum-phase numbers to get wrong (no risk
# of two edits accidentally colliding on the same integer, or of a merge needing you to
# renumber everything downstream).
QUANTUM_PHASE_OF_POTENTIAL_LABEL = {label: label for label in PHASE_LABELS.values()}

QUANTUM_PHASE_OF_POTENTIAL_LABEL["Ia"]   = "I"  
QUANTUM_PHASE_OF_POTENTIAL_LABEL["Ib"]   = "I"
QUANTUM_PHASE_OF_POTENTIAL_LABEL["Ic"]   = "I"

QUANTUM_PHASE_OF_POTENTIAL_LABEL["IIa"]  = "II" 
QUANTUM_PHASE_OF_POTENTIAL_LABEL["IIb"]  = "II"
QUANTUM_PHASE_OF_POTENTIAL_LABEL["IIc"]  = "II"

QUANTUM_PHASE_OF_POTENTIAL_LABEL["IIIa"] = "III"
QUANTUM_PHASE_OF_POTENTIAL_LABEL["IIIb"] = "III"
QUANTUM_PHASE_OF_POTENTIAL_LABEL["IIIc"] = "III"

QUANTUM_PHASE_OF_POTENTIAL_LABEL["0"]   = "0"

_LABEL_TO_POTENTIAL_CODE = {label: code for code, label in PHASE_LABELS.items()}
_quantum_labels_in_order = list(dict.fromkeys(QUANTUM_PHASE_OF_POTENTIAL_LABEL.values()))
_QUANTUM_LABEL_TO_CODE = {label: i + 1 for i, label in enumerate(_quantum_labels_in_order)}
QUANTUM_PHASE_LABELS = {code: label for label, code in _QUANTUM_LABEL_TO_CODE.items()}

QUANTUM_PHASE_OF_POTENTIAL = {0: 0}
for _potential_label, _quantum_label in QUANTUM_PHASE_OF_POTENTIAL_LABEL.items():
    QUANTUM_PHASE_OF_POTENTIAL[_LABEL_TO_POTENTIAL_CODE[_potential_label]] = \
        _QUANTUM_LABEL_TO_CODE[_quantum_label]

_QUANTUM_LUT = np.zeros(max(QUANTUM_PHASE_OF_POTENTIAL) + 1, dtype=int)
for _p, _q in QUANTUM_PHASE_OF_POTENTIAL.items():
    _QUANTUM_LUT[_p] = _q




# Which extremum of the potential realizes the minimum in each potential phase: P0 is the
# origin, P1 the H-only vacuum, P2 the S-only one and P3 the mixed one. (Checked against
# the argmin of Vmin_branches_* over both parametrizations, everywhere the potential is
# bounded from below.)
#
# This is what fixes the ORDER of every phase boundary, and it is why the plots below can
# look the order up in a table instead of differentiating V_min at each boundary every
# time: on either side of a boundary between two phases that share an extremum, V_min is
# the same analytic function of the parameters, so such a boundary is no transition at all,
# while a boundary between two different extrema is one -- of an order that depends only on
# WHICH two, never on the parameter values the plot happens to use.
BRANCH_OF_POTENTIAL_LABEL = {
    "Ia": 1, "Ib": 1, "Ic": 1,        # P2: <S> only
    "IIa": 2, "IIb": 2, "IIc": 2,     # P1: <H> only
    "IIIa": 3, "IIIb": 3, "IIIc": 3,  # P3: both
    "0": 0,                          # P0: the symmetric vacuum
}

# Ehrenfest order of a transition, keyed by the unordered pair of extrema that exchange the
# global minimum across it. Six entries cover every boundary that can occur:
#
#   {P1, P2}            both VEVs jump -- <H> switches off as <S> switches on   1st order
#   {P0, P1}, {P0, P2}  the single VEV vanishes continuously at mu^2 = 0        2nd order
#   {P1, P3}, {P2, P3}  the second VEV of P3 turns on continuously, at
#                       Omega_S = 1 (resp. Omega_H = 1), which is exactly where
#                       Hbar3 = OmegaH (OmegaS - 1)/(OmegaH OmegaS - 1) (resp.
#                       Sbar3) passes through zero                              2nd order
#   {P0, P3}            the mixed VEV would have to jump to zero, so first order; listed
#                       for completeness only, since P3 needs at least one mu^2 > 0 and so
#                       never borders P0
#
# A pair absent from the dict -- i.e. the same extremum on both sides, as at the Ia|Ib
# boundary Omega_H = 1 -- is not a transition and draws no line: the a/b/c distinctions are
# about which OTHER stationary points exist, not about the minimum.
TRANSITION_ORDER = {
    frozenset((1, 2)): 'first',
    frozenset((0, 1)): 'second',
    frozenset((0, 2)): 'second',
    frozenset((1, 3)): 'second',
    frozenset((2, 3)): 'second',
    frozenset((0, 3)): 'first',
}
TRANSITION_LINESTYLE = {'first': '-', 'second': '--'}


_BRANCH_LUT = np.zeros(max(PHASE_LABELS) + 1, dtype=int)
for _code, _label in PHASE_LABELS.items():
    _BRANCH_LUT[_code] = BRANCH_OF_POTENTIAL_LABEL[_label]


def _transition_order(p1, p2):
    """Order of the boundary between potential phases p1 and p2, or None where there is no
    quantum phase transition to draw -- either because the two share a quantum phase, or
    because they share an extremum and V_min is analytic across the boundary."""
    if p1 == 0 or p2 == 0:
        return None
    if (QUANTUM_PHASE_OF_POTENTIAL_LABEL[PHASE_LABELS[p1]]
            == QUANTUM_PHASE_OF_POTENTIAL_LABEL[PHASE_LABELS[p2]]):
        return None
    return TRANSITION_ORDER.get(frozenset((int(_BRANCH_LUT[p1]), int(_BRANCH_LUT[p2]))))


def _fill_phase_seams(PH, max_gap=1):
    """Close the hairline seams of "no phase" that the phase map leaves along its own
    boundaries.

    Every condition in tree_potential_phase_map_Omega is a strict inequality, so a sample
    landing exactly ON a boundary belongs to neither side and comes back as code 0. Which
    samples do is an accident of where the grid falls -- but not a rare one: the
    first-order Omega_H = Omega_S boundary is the grid diagonal whenever lambdaH = lambdaS,
    and the mu^2 half-axes are grid lines whenever the grid straddles zero symmetrically.
    Left alone, such a seam separates the two phases it lies between and the border between
    them is never drawn at all.

    Only thin components are filled -- those whose every cell lies within max_gap of a real
    phase -- so genuinely undefined REGIONS (the quadrants Eqs. (16)-(17) empty, say) are
    left alone rather than eroded by a cell."""
    holes = (PH == 0)
    if not holes.any():
        return PH
    dist, idx = ndimage.distance_transform_edt(holes, return_indices=True)
    labels, n = ndimage.label(holes)
    if n == 0:
        return PH
    widest = np.asarray(ndimage.maximum(dist, labels, range(1, n + 1)))
    seam = np.isin(labels, 1 + np.flatnonzero(widest <= max_gap))
    filled = PH.copy()
    filled[seam] = PH[idx[0][seam], idx[1][seam]]
    return filled
