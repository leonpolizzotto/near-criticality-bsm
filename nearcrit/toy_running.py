"""The running-coupling toy potentials of Sec. 4 and the helpers shared by Figs. 12-20.

  - V4p8(x, a/c) = -(a/c) x^2 + x^4 ln x (one-loop dominated, eq. (4.8)); its extrema are where F(x) = a/c.
  - V4p17(y, a, b) = -a y^2 + (b + ln^2 y) y^4 (two-loop dominated, eq. (4.19)); its extrema are where G(y, b) = a,
    and those of G where H(y) = b.

Importing this module computes the phase colours on a 600 x 600 grid over the (a, b) plane (PHASE_IMAGE) and the
first-order lines, which takes a few seconds.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.optimize import brentq
from scipy.special import lambertw

from nearcrit.style import set_xlabel_right, set_ylabel_top
def F(x):
    return (4*np.log(x) + 1) * x**2 / 2

def V4p8(x,a_over_c, sign_c=1):
    if sign_c != 1 and sign_c != -1:
        raise ValueError("sign_c must be +1 or -1")
    return (- a_over_c * x**2 + x**4 * np.log(x)) * sign_c

a_over_c_CRIT_1 = 0.0
a_over_c_CRIT_2 = -np.exp(-1.5)

def G(y,b):
    return (2*b + np.log(y)*(2*np.log(y) + 1)) * y**2

def H(y):
    return -0.25 - 1.5*np.log(y) - np.log(y)**2

def a_plus_minus(b, pm):
    if pm != 1 and pm != -1:
        raise ValueError("pm must be +1 or -1")
    return 0.5 * np.exp(-1.5 - pm * 0.5*np.sqrt(5 - 16*b)) * (2 + pm * np.sqrt(5 - 16*b))

def V4p17(y,a,b):  # V4p17' = 2y (G(y, b) - a)
    return - a * y**2 + (b + np.log(y)**2) * y**4

b_CRIT = 5/16
b0     = 1/16


def extrema_positions(a_over_c):
    """Positions x > 0 of the extrema of V4p8, i.e. the solutions of F(x) = a_over_c (V4p8' = 2x (F - a_over_c)).

    With x = exp((u - 1/2)/2), F(x) = a_over_c becomes u e^u = a_over_c sqrt(e), so u is a branch of the Lambert W
    function: W_0 gives the root above the minimum of F at x = e^{-3/4}, W_{-1} the one below it (real only for
    a_over_c_CRIT_2 <= a_over_c < 0).
    """
    z = a_over_c * np.exp(0.5)
    if np.isclose(z, -np.exp(-1), rtol=0, atol=1e-12):
        return np.array([np.exp(-0.75)])  # branch point: the double root at the minimum of F (rounding gives nan)
    if z < -np.exp(-1):
        return np.array([])
    branches = (0, -1) if z < 0 else (0,)
    return np.sort([np.exp((lambertw(z, k).real - 0.5) / 2) for k in branches])


def plot_extrema(ax, x, y, color):
    """Mark the extrema at x: minima (F increasing, so V4p8'' > 0) filled, maxima and inflection points open."""
    is_min = 4*np.log(x) + 3 > 1e-9  # F'(x) = x (4 ln x + 3)
    ax.plot(x[is_min], y[is_min], "o", color=color, ms=5, zorder=3)
    ax.plot(x[~is_min], y[~is_min], "o", mfc="white", color=color, ms=5, zorder=3)


a_plus = lambda b: a_plus_minus(b, 1)
a_minus = lambda b: a_plus_minus(b, -1)


def n_minima_region(a, b):
    """Which of the regions above (a, b) lies in, numbered 0-9 in the order of n_minima_regions in shade_n_minima,
    and its number of extrema."""
    a, b = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    b_clip = np.minimum(b, b_CRIT)
    upper, middle = b >= b_CRIT, (b >= b0) & (b < b_CRIT)
    a_m, a_p = a_minus(b_clip), a_plus(b_clip)
    region = np.select(
        [upper & (a < 0), upper,
         middle & (a < 0), middle & (a < a_m), middle & (a < a_p), middle,
         a < a_m, a < 0, a < a_p],
        [0, 1, 2, 3, 4, 5, 6, 7, 8], default=9)
    return region, np.array([0, 1, 0, 1, 3, 1, 0, 2, 3, 1])[region]


def n_extrema_bands(a_lim, b_lim):
    """The regions of n_minima_region as bands (b, a from, a to, number of extrema), for fill_betweenx."""
    b_upper = np.array([b_CRIT, b_lim[1]])
    b_middle = np.linspace(b0, b_CRIT, 400)
    b_lower = np.linspace(b_lim[0], b0, 400)
    return [
        (b_upper, a_lim[0], 0, 0),
        (b_upper, 0, a_lim[1], 1),
        (b_middle, a_lim[0], 0, 0),
        (b_middle, 0, a_minus(b_middle), 1),  # one extremum, below the maximum of G (a_- > 0 here)
        (b_middle, a_minus(b_middle), a_plus(b_middle), 3),
        (b_middle, a_plus(b_middle), a_lim[1], 1),
        (b_lower, a_lim[0], a_minus(b_lower), 0),
        (b_lower, a_minus(b_lower), 0, 2),
        (b_lower, 0, a_plus(b_lower), 3),
        (b_lower, a_plus(b_lower), a_lim[1], 1),
    ]


def format_phase_diagram(ax, a_lim, b_lim, tick_b0=True):
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xticks([0, np.exp(-1.5)], [r"$0$", r"$a_\text{crit}$"])
    if tick_b0:
        ax.set_yticks([0, b0, b_CRIT], [r"$0$", r"$b_0$", r"$b_\text{crit}$"])
    else:  # too close to 0 to label when b spans a wide range
        ax.set_yticks([0, b_CRIT], [r"$0$", r"$b_\text{crit}$"])
    ax.set_xlim(a_lim)  # after the ticks, which would otherwise widen the limits to include 0 on a zoomed diagram
    ax.set_ylim(b_lim)
    set_xlabel_right(ax, r"$a$")
    set_ylabel_top(ax, r"$b$")


def draw_extrema_count_lines(ax, b_lim, a_pm_style, zero_style):
    """The region boundaries across which the number of extrema changes: a_+-(b) below b_CRIT (in a_pm_style), and
    a = 0 (in zero_style). b = b_CRIT and b = b0 are not: there the count is the same on both sides."""
    b_below_crit = np.linspace(b_lim[0], b_CRIT, 800)
    for a in (a_plus(b_below_crit), a_minus(b_below_crit)):
        ax.plot(a, b_below_crit, **a_pm_style)
    ax.plot([0, 0], b_lim, **zero_style)


# Ground state figure along an ellipse in the (a, b) plane, a = a_c + A cos(theta), b = b_c + B sin(theta): the
# contour on the phase diagram, the ground state energy E0 along it, and its first and second derivatives along the
# contour tangent.
#
# V4p17' = 2y (G(y, b) - a), so the minima at y > 0 are where G(., b) crosses a from below. G rises from 0 at y = 0;
# for b < b_CRIT it has a maximum a_+ at y = e^{(-3 - s)/4} and a minimum a_- at y = e^{(-3 + s)/4}, s = sqrt(5 - 16b),
# so there is a minimum P1 below the maximum of G when 0 < a < a_+, and one, P2, above the minimum of G when a > a_-.
# For b >= b_CRIT G is increasing and the single minimum (for a > 0) is called P2. y = 0 (P0, E = 0) is a minimum for
# a <= 0. Checked against brute-force minimisation of V4p17 to 1e-8.
#
# Derivatives: E0(a, b) = V4p17(y(a, b), a, b), with y(a, b) the position of the minimum. By the chain rule
# dE0/da = dV/da + (dV/dy)(dy/da), and dV/dy = 0 at the minimum, so dE0/da = dV/da = -y^2 and dE0/db = y^4 (envelope
# theorem). The second derivatives do need dy/da and dy/db, from differentiating G(y(a, b), b) = a: with
# dG/dy = 4y (b - H) and dG/db = 2y^2, dy/da = 1 / (4y (b - H)) and dy/db = -y / (2 (b - H)), so
# d2E0/da2 = -2y dy/da = -1 / (2 (b - H)), d2E0/da db = -2y dy/db = y^2 / (b - H), d2E0/db2 = 4y^3 dy/db
# = -2y^4 / (b - H). All of them vanish on P0. Checked against finite differences of E0.
def _solve_G(a, b, u_lo, u_hi, n_iter=100):
    """y with G(y, b) = a, for G increasing in u = ln y on [u_lo, u_hi] (bisection, vectorised)."""
    lo, hi = np.broadcast_arrays(np.asarray(u_lo, dtype=float), np.asarray(u_hi, dtype=float))
    lo, hi = lo.copy(), hi.copy()
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        above = G(np.exp(mid), b) > a
        hi = np.where(above, mid, hi)
        lo = np.where(above, lo, mid)
    return np.exp(0.5 * (lo + hi))


def local_minima(a, b):
    """Positions and energies of the local minima P0 (y = 0, for a <= 0), P1 and P2 at each (a, b), stacked along the
    first axis; nan where that minimum does not exist."""
    a, b = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    below = b < b_CRIT
    b_clip = np.minimum(b, b_CRIT)
    s = np.sqrt(5 - 16 * b_clip)
    has_P1 = below & (a > 0) & (a < a_plus(b_clip))
    has_P2 = np.where(below, a > a_minus(b_clip), a > 0)
    with np.errstate(invalid="ignore"):
        y1 = np.where(has_P1, _solve_G(a, b, -60.0, (-3 - s) / 4), np.nan)
        y2 = np.where(has_P2, _solve_G(a, b, np.where(below, (-3 + s) / 4, -60.0), 5.0), np.nan)
        Y = np.stack([np.where(a <= 0, 0.0, np.nan), y1, y2])
        E = np.stack([np.where(a <= 0, 0.0, np.nan), V4p17(y1, a, b), V4p17(y2, a, b)])
    return Y, E


def nth_lowest_minimum(a, b, n):
    """Energy, position and label (0, 1, 2) of the n-th lowest local minimum (n = 0: the ground state), and the
    energies of all three; energy and position are nan, and the label -1, where there are fewer than n + 1 minima."""
    Y, E = local_minima(a, b)
    label = np.argsort(np.where(np.isnan(E), np.inf, E), axis=0)[n]
    E_n = np.take_along_axis(E, label[None], axis=0)[0]
    y_n = np.take_along_axis(Y, label[None], axis=0)[0]
    return E_n, y_n, np.where(np.isnan(E_n), -1, label), E


def ground_state(a, b):
    """E0, the position y of the global minimum, its label P (0, 1, 2), and the energies of P0, P1, P2 (nan where that
    minimum does not exist) at each (a, b)."""
    return nth_lowest_minimum(a, b, 0)


def minimum_derivatives(a, b, n=0):
    """Energy E_n of the n-th lowest minimum, its position y, its gradient (dE/da, dE/db) and Hessian (d2E/da2,
    d2E/da db, d2E/db2), and its label. The formulas hold at any local minimum, not only the global one; all of
    them vanish on P0 and are nan where the minimum does not exist."""
    E_n, y, label, E = nth_lowest_minimum(a, b, n)
    on = label > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        b_minus_H = b - H(np.where(on, y, 1.0))
        grad = (np.where(on, -y**2, 0.0), np.where(on, y**4, 0.0))
        hess = (np.where(on, -1 / (2 * b_minus_H), 0.0), np.where(on, y**2 / b_minus_H, 0.0),
                np.where(on, -2 * y**4 / b_minus_H, 0.0))
    exists = label >= 0
    grad = tuple(np.where(exists, g, np.nan) for g in grad)
    hess = tuple(np.where(exists, h, np.nan) for h in hess)
    return E_n, y, grad, hess, label


def ground_state_derivatives(a, b):
    """E0, its gradient (dE0/da, dE0/db) and Hessian (d2E0/da2, d2E0/da db, d2E0/db2), P, and the energies of all
    three minima."""
    E0, y, grad, hess, P = minimum_derivatives(a, b, 0)
    return E0, grad, hess, P, local_minima(a, b)[1]


CONTOUR_COLOR = "C3"
TRANSITION_STYLE = {"first": dict(color="k", ls="-", lw=1.0), "continuous": dict(color="k", ls="--", lw=1.0)}
EXTREMA_LINE_STYLE = dict(color="0.4", lw=0.6)  # where the number of extrema changes at a = 0
# where it changes at a_+-(b) (a metastable minimum and a maximum appear or disappear): thicker than the axes and the
# hatching, and in a colour that stands out against all three phase colours, so the caption can refer to them
A_PM_STYLE = dict(color="#8e24aa", lw=1.1)
a_lim_gs, b_lim_gs = (-0.65, 0.65), (-0.65, 0.65)

# Widom line above the critical point (b_CRIT < b < 9/16), where the correlation length xi = 1 / sqrt(V'') at the
# minimum peaks along a at fixed b. V''(y_min) = 2y dG/dy = 8 y^2 (b - H) = 8 e^{2L} (b + 1/4 + 3L/2 + L^2), with
# L = ln y, has d/dL = 8 e^{2L} (2L^2 + 5L + 2b + 2), so xi peaks at L = (-5 + sqrt(9 - 16b)) / 4 (the larger root; xi
# also diverges towards the continuous transition at a = 0). At b_CRIT this is L = -3/4, the critical point, with slope
# da/db = dG/db = 2 e^{-3/2}, the Clausius-Clapeyron slope y1^2 + y2^2 of the first-order line, which it continues. At
# b = 9/16 the peak merges with the minimum of xi and the line ends. (The maximum of the susceptibility d(y^2)/da =
# 4 y^2 xi^2 is instead at L = -3/4 for all b; the two agree only near the critical point.)
B_WIDOM_END = 9 / 16


def ln_y_widom(b):
    return (-5 + np.sqrt(9 - 16 * b)) / 4


# Ground state phases. P0 (y = 0) is separated from the y > 0 minima everywhere by a transition, so it gets its own
# colour. P1 and P2 are only told apart by the size of y: they are distinct across the first-order line, but connected
# around the critical point (like liquid and gas), so their colours blend in ln y, about the Widom line above b_CRIT
# and about ln y = -3/4 (the critical point) below. They then jump across the first-order line, by less towards the
# critical point, and fade into each other beyond it.
PHASE_COLORS = {P: np.array(plt.matplotlib.colors.to_rgb(c)) for P, c in ((0, "C0"), (1, "C1"), (2, "C2"))}
PHASE_ALPHA = 0.55  # strength of the colours over white
WIDOM_WIDTH = 0.4  # width of the P1/P2 blend in ln y
# Number of extrema as hatching, one line direction per extremum
N_EXTREMA_HATCHES = ["", "//", "xx", "xxxx----"]
HATCH_COLOR = "0.5"


def phase_color(y, P, b, alpha=PHASE_ALPHA):
    """RGB colour of the ground state with position y and label P at b (arrays of the same shape); alpha = 1 gives the
    full-strength colour, for lines."""
    ln_y_centre = ln_y_widom(np.clip(b, b_CRIT, B_WIDOM_END))  # -3/4 below b_CRIT
    with np.errstate(divide="ignore"):
        s = 0.5 * (1 + np.tanh((np.log(np.where(P > 0, y, 1.0)) - ln_y_centre) / WIDOM_WIDTH))[..., None]
    rgb = np.where((P == 0)[..., None], PHASE_COLORS[0], (1 - s) * PHASE_COLORS[1] + s * PHASE_COLORS[2])
    return 1 - alpha * (1 - rgb)


PHASE_NAMES = {0: "0", 1: "I", 2: "II"}  # quantum phases: P0 is 0, P1 is I, P2 is II


def phase_legend_handles():
    return [Patch(color=1 - PHASE_ALPHA * (1 - PHASE_COLORS[P]), label=PHASE_NAMES[P]) for P in range(3)]


def hatch_legend_handles():
    return [Patch(facecolor="white", edgecolor=HATCH_COLOR, hatch=N_EXTREMA_HATCHES[n], lw=0.5, label=rf"${n}$")
            for n in range(4)]


# The phase colours on a grid over the phase diagram, for the ground state figure (pixel centres on the grid)
_n_pixels = 600
_a_pixels = np.linspace(*a_lim_gs, _n_pixels + 1)
_b_pixels = np.linspace(*b_lim_gs, _n_pixels + 1)
_A_pixels, _B_pixels = np.meshgrid(0.5 * (_a_pixels[1:] + _a_pixels[:-1]), 0.5 * (_b_pixels[1:] + _b_pixels[:-1]))
_, _y_pixels, _P_pixels, _ = ground_state(_A_pixels, _B_pixels)
PHASE_IMAGE = phase_color(_y_pixels, _P_pixels, _B_pixels)

# Ground state transitions in the (a, b) plane. First order: where the global minimum jumps between two coexisting
# minima, P0 <-> P2 for a < 0 and P1 <-> P2 for a > 0. They meet at the origin, where y jumps from 0 to 1, and
# P1 <-> P2 ends at the critical point (e^{-3/2}, b_CRIT), where the jump in y goes to zero. Continuous: a = 0, b > 0,
# where the minimum moves out of y = 0. This line ends at the origin, on the first-order line, where P0 and P1
# coincide (y = 0) and coexist with P2: a critical end point. The first-order line goes smoothly through it, with
# slope da/db = y^2 = 1 on both sides.
#
# P0 <-> P2 (a < 0) is exact. At any extremum a = G(y, b), so V = -y^4 (b + L^2 + L) with L = ln y. Degeneracy with
# P0 (V = 0) gives b = -L (L + 1), and then a = G = -L e^{2L}. a < 0 needs L > 0 (y > 1); there b - H = 1/4 + L/2 > 0,
# so it is a minimum. L -> 0 is the origin, and the line leaves it along b = a.
L_first_order = np.linspace(0, 1.5, 400)
a_first_order_P0_P2 = -L_first_order * np.exp(2 * L_first_order)
b_first_order_P0_P2 = -L_first_order * (L_first_order + 1)
# P1 <-> P2 (a > 0), 0 < b < b_CRIT: at each b, the a where E_P1 = E_P2. Both minima exist for max(a_-, 0) < a < a_+
# (y = 0 is not a minimum for a > 0, so they are the only competitors), and d(E_P1 - E_P2)/da = y2^2 - y1^2 > 0 there
# (envelope theorem), so E_P1 - E_P2 has exactly one root in that interval, found by Brent's method. The line runs from
# the critical end point (0, 0) to the critical point (e^{-3/2}, b_CRIT), where a_- = a_+.
def first_order_a_P1_P2(b):
    a_lo, a_hi = max(a_minus(b), 0.0), a_plus(b)
    margin = 1e-9 * (a_hi - a_lo)  # the minima are only defined strictly inside the interval
    def energy_difference(a):
        E = local_minima(a, b)[1]
        return float(E[1] - E[2])  # E_P1 - E_P2

    return brentq(energy_difference, a_lo + margin, a_hi - margin, xtol=1e-15)


# b spaced more finely towards both ends, where the line turns (it leaves the origin along b = a)
s_first_order = np.linspace(0, 1, 301)[1:-1]
b_first_order_P1_P2 = b_CRIT * (3 * s_first_order**2 - 2 * s_first_order**3)
a_first_order_P1_P2 = np.array([first_order_a_P1_P2(b) for b in b_first_order_P1_P2])
a_first_order_P1_P2 = np.concatenate(([0.0], a_first_order_P1_P2, [np.exp(-1.5)]))
b_first_order_P1_P2 = np.concatenate(([0.0], b_first_order_P1_P2, [b_CRIT]))


def transitions_along(a, b, P, E):
    """Indices along a path where the ground state changes, with "first" (it jumps between two minima that both exist
    on either side) or "continuous" (to or from y = 0). A change of label where the outgoing minimum disappears without
    involving y = 0 (P1 -> P2 across b_CRIT, above the critical end point) is not a transition."""
    exists = np.stack([a <= 0, ~np.isnan(E[1]), ~np.isnan(E[2])])
    transitions = []
    for i in np.flatnonzero(np.diff(P)) + 1:
        if exists[[P[i - 1], P[i]]][:, [i - 1, i]].all():
            transitions.append((i, "first"))
        elif 0 in (P[i - 1], P[i]):
            transitions.append((i, "continuous"))
    return transitions


def draw_ground_state_phase_diagram(ax, a_lim=a_lim_gs, b_lim=b_lim_gs, lw=1.5, ms=4):
    """The (a, b) plane coloured by the ground state phase and hatched by the number of extrema, with the extrema count
    lines, the ground state transitions, the critical point and the triple point."""
    ax.imshow(PHASE_IMAGE, origin="lower", extent=(*a_lim_gs, *b_lim_gs), aspect="auto", interpolation="nearest",
              zorder=0)
    for b_band, a_from, a_to, n in n_extrema_bands(a_lim_gs, b_lim_gs):
        ax.fill_betweenx(b_band, a_from, a_to, facecolor="none", edgecolor=HATCH_COLOR, hatch=N_EXTREMA_HATCHES[n],
                         lw=0, zorder=0.5)
    format_phase_diagram(ax, a_lim, b_lim, tick_b0=False)
    draw_extrema_count_lines(ax, b_lim_gs, A_PM_STYLE, EXTREMA_LINE_STYLE)
    ax.plot(a_first_order_P1_P2, b_first_order_P1_P2, color="k", lw=lw)
    ax.plot(a_first_order_P0_P2, b_first_order_P0_P2, color="k", lw=lw)
    ax.plot([0, 0], [0, b_lim_gs[1]], color="k", ls="--", lw=lw)
    ax.plot(np.exp(-1.5), b_CRIT, "o", color="k", ms=ms)  # critical point
    ax.plot(0, 0, "o", color="k", mfc="white", ms=ms, zorder=3)  # triple point (critical end point, see above)


# Third derivatives of E0, differentiating the Hessian above with D = b - H(y), H'(y) = -h / y, h = 3/2 + 2 ln y:
# dD/da = h / (4 y^2 D) and dD/db = 1 - h / (2D), so
# E_aaa = h / (8 y^2 D^3), E_aab = (1 - h / (2D)) / (2 D^2), E_abb = -y^2 (2 - h / (2D)) / D^2, E_bbb = y^4 (6 - h / D) / D^2.
# All vanish on P0. Checked against finite differences of the Hessian.
def ground_state_third_derivatives(a, b):
    """(E_aaa, E_aab, E_abb, E_bbb) of the ground state energy."""
    _, y, P, _ = ground_state(a, b)
    on = P > 0
    y = np.where(on, y, 1.0)
    D = b - H(y)
    h = 1.5 + 2 * np.log(y)
    third = (h / (8 * y**2 * D**3), (1 - h / (2 * D)) / (2 * D**2), -y**2 * (2 - h / (2 * D)) / D**2,
             y**4 * (6 - h / D) / D**2)
    return tuple(np.where(on, d, 0.0) for d in third)


def draw_contour(ax, a_c, b_c, A, B, color, ls, n_theta=4001):
    """The ellipse a = a_c + A cos(theta), b = b_c + B sin(theta) on a phase diagram, with a dot at its start (theta = 0)
    and an arrow showing its direction."""
    theta = np.linspace(0, 2 * np.pi, n_theta)
    a, b = a_c + A * np.cos(theta), b_c + B * np.sin(theta)
    ax.plot(a, b, color=color, ls=ls, lw=1.5)
    ax.plot(a[0], b[0], "o", color=color, ms=3.5, zorder=4)
    j = n_theta // 16
    ax.annotate("", xy=(a[j], b[j]), xytext=(a[j - 1], b[j - 1]),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5, mutation_scale=10))


THETA_TICK_LABELS = {-1: r"$-\pi/2$", 0: r"$0$", 1: r"$\pi/2$", 2: r"$\pi$", 3: r"$3\pi/2$"}


def decorate_along_contour(ax, theta, a, b, transitions):
    """Background of a panel against theta along a contour: hatched by the number of extrema at each point, with
    lines where it changes (purple at a_+-(b), grey at a = 0) and at the ground state transitions (black: solid first
    order, dashed continuous), a zero line, and the theta axis. The phase colours are added by shade_along_contour once
    the limits are set."""
    region, n_minima = n_minima_region(a, b)
    region_crossings = np.flatnonzero(np.diff(region)) + 1
    count_changes = np.flatnonzero(np.diff(n_minima)) + 1
    edges = np.concatenate(([theta[0]], 0.5 * (theta[1:] + theta[:-1]), [theta[-1]]))
    theta_between = lambda i: 0.5 * (theta[i - 1] + theta[i])
    for start, end in zip(np.concatenate(([0], region_crossings)), np.concatenate((region_crossings, [len(theta)]))):
        ax.axvspan(edges[start], edges[end], facecolor="none", edgecolor=HATCH_COLOR,
                   hatch=N_EXTREMA_HATCHES[n_minima[start]], lw=0, zorder=0.5)
    for i in count_changes:  # crossing a = 0 or a_+-(b)
        ax.axvline(theta_between(i), zorder=1, **(EXTREMA_LINE_STYLE if a[i - 1] * a[i] <= 0 else A_PM_STYLE))
    for i, kind in transitions:
        ax.axvline(theta_between(i), zorder=2, **TRANSITION_STYLE[kind])
    ax.axhline(0, color="gray", lw=0.5, zorder=1)
    ax.set_xlim(theta[0], theta[-1])
    # ticks every pi/2, but none at the end of the panel, where a label would collide with the axis label
    quarters = np.arange(round(theta[0] / (np.pi / 2)), round(theta[-1] / (np.pi / 2)))
    ax.set_xticks(quarters * np.pi / 2, [THETA_TICK_LABELS[k] for k in quarters])
    ax.set_yticks([0], [r"$0$"])
    set_xlabel_right(ax, r"$\theta$")


def shade_along_contour(ax, theta, contour_colors):
    """Phase colours over the full height of a panel against theta (y in axes coordinates). imshow resets the limits
    to its extent, so this comes after they are set."""
    limits = ax.get_xlim(), ax.get_ylim()
    ax.imshow(contour_colors[None], extent=(theta[0], theta[-1], 0, 1), transform=ax.get_xaxis_transform(),
              aspect="auto", interpolation="nearest", zorder=0)
    ax.set_xlim(limits[0])
    ax.set_ylim(limits[1])


GS_HEIGHT_IN = 7.2  # short enough to fit on an A4 page (1 in margins) with its caption
PEAK_INSET_LS = (0, (0.01, 2.2))  # dotted (with round caps) border of the peak inset and its box (dashed lines are
# continuous transitions)
FAR_INSET_BOUNDS = [0.63, 0.5, 0.24, 0.45]  # the continuous transition inset when there is a peak inset too (axes fractions)


# Ground state figure for the single log potential of the first figure, V = c V4p8(x, a/c) = -a x^2 + c x^4 ln x, as the
# circle figure: the (a, c) phase diagram, and the ground state along a/c at fixed c > 0 (position, E0 and dE0/da; the
# only quantum transition is first order, so higher derivatives add nothing). For c < 0 V -> -inf at large x: no ground
# state. For c > 0, E0 = c e0(a/c), so the (a, c) plane is cut into wedges by rays through the origin, and along any line
# of fixed c, dE0/da = e0'(a/c).
#
# Write r = a/c. The extrema at x > 0 are where F(x) = r (first figure): the maximum x1 < e^{-3/4} and the minimum
# x2 > e^{-3/4} for -e^{-3/2} < r < 0, and only x2 for r >= 0. x = 0 is a minimum for r < 0 (V ~ -r x^2) and a maximum
# for r >= 0 (at r = 0, x1 merges into it). At an extremum V = -x^4 (ln x + 1/2), so E(x2) < 0 = E(0) exactly when
# ln x2 > -1/2, i.e. r > F(e^{-1/2}) = -1/(2e). So:
# - r = -1/(2e): first-order transition between x = 0 (phase 0) and x2 = e^{-1/2} (phase II): x jumps, and dE0/da = -x^2
#   jumps from 0 to -1/e.
# - r = -e^{-3/2} (the minimum of F): x2 and x1 are born together as a metastable minimum and a maximum (a spinodal).
#   The ground state is x = 0 on both sides, so E0 = 0 is analytic there: a potential phase transition, not a quantum one.
#   (The metastable minimum's second derivative -2 / (4 ln x2 + 3) does diverge there, like (r + e^{-3/2})^{-1/2}.)
# - r = 0: x = 0 turns from a (metastable) minimum to a maximum. x2 is analytic through it (it is the W_0 branch), so
#   again only a potential phase transition.
# On x2, e0' = -x2^2 (envelope theorem). Checked against brute-force minimisation and finite differences.
r_FO_single_log = -1 / (2 * np.e)
r_SP_single_log = a_over_c_CRIT_2  # -e^{-3/2}
PHASE_COLOR_NGS = np.array(plt.matplotlib.colors.to_rgb("gray"))


def single_log_x2(r):
    """The minimum x2 > e^{-3/4} of V4p8(., r) (nan for r < -e^{-3/2}, where it does not exist)."""
    r = np.asarray(r, dtype=float)
    with np.errstate(invalid="ignore"):
        x2 = np.exp((lambertw(np.maximum(r, r_SP_single_log) * np.exp(0.5), 0).real - 0.5) / 2)
    return np.where(r >= r_SP_single_log, x2, np.nan)


def single_log_branch(x2, exists):
    """Position, energy and derivative in r (e0') of a minimum: x2 where on_x2, else x = 0 (all zero), and nan where
    the minimum does not exist."""
    def branch(on_x2):
        x, L = np.where(on_x2, x2, 1.0), np.log(np.where(on_x2, x2, 1.0))
        values = (x, -x**4 * (L + 0.5), -x**2)
        return tuple(np.where(exists, np.where(on_x2, v, 0.0), np.nan) for v in values)
    return branch


def single_log_minima(r):
    """(x, e0, e0') of the ground state and of the metastable minimum (nan where there is none), and
    whether the ground state is x2 (phase II)."""
    x2 = single_log_x2(r)
    in_II = r > r_FO_single_log
    with np.errstate(divide="ignore", invalid="ignore"):
        ground = single_log_branch(x2, np.ones_like(r, bool))(in_II)
        # metastable: x = 0 for -1/(2e) < r < 0, x2 for -e^{-3/2} <= r < -1/(2e)
        meta = single_log_branch(x2, (r >= r_SP_single_log) & (r < 0))(~in_II)
    return ground, meta, in_II


def single_log_n_extrema(r):
    """Number of extrema at x > 0."""
    return np.select([r > 0, r >= r_SP_single_log], [1, 2], default=0)


def single_log_phase_rgb(in_II, alpha=PHASE_ALPHA):
    return 1 - alpha * (1 - np.where(np.asarray(in_II)[..., None], PHASE_COLORS[2], PHASE_COLORS[0]))




# Extrema of V4p17 at y > 0, for the potential figures below. V4p17' = 2y (G(y, b) - a), so they are the solutions of
# G = a. G is monotonic in L = ln y between its extrema at H(y) = b, L = (-3 -+ s) / 4 (b < b_CRIT), so each piece has
# at most one root, found by Brent's method. A root is a minimum where G is increasing, i.e. b > H.
def V_extrema(a, b):
    """Positions y > 0 of the extrema of V4p17(., a, b), increasing, and whether each is a minimum."""
    cuts = [-60.0, 5.0]
    if b < b_CRIT:
        s = np.sqrt(5 - 16 * b)
        cuts = [-60.0, (-3 - s) / 4, (-3 + s) / 4, 5.0]
    f = lambda L: G(np.exp(L), b) - a
    ys = [np.exp(brentq(f, lo, hi, xtol=1e-14)) for lo, hi in zip(cuts[:-1], cuts[1:]) if f(lo) * f(hi) < 0]
    ys = np.array(ys)
    return ys, b > H(ys)


def all_extrema(a, b):
    """Every extremum of V4p17(., a, b) at y >= 0 (y = 0 is a minimum for a < 0 and a maximum for a > 0), as
    (y, V, kind) with kind "ground" (the global minimum), "metastable" (another minimum) or "max"."""
    ys, is_min = V_extrema(a, b)
    ys, is_min = np.concatenate(([0.0], ys)), np.concatenate(([a < 0], is_min))
    with np.errstate(divide="ignore", invalid="ignore"):
        Vs = np.where(ys > 0, V4p17(ys, a, b), 0.0)
    ground = np.argmin(np.where(is_min, Vs, np.inf))
    kinds = ["ground" if i == ground else "metastable" if m else "max" for i, m in enumerate(is_min)]
    return ys, Vs, kinds


EXTREMUM_MARKERS = {"ground": dict(color="C3", ms=5), "metastable": dict(color="C3", mfc="white", ms=5),
                    "max": dict(color="k", mfc="white", ms=3.5)}


def V_curve(y, a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(y > 0, V4p17(y, a, b), 0.0)



def V_limits(y_plot, V_plot, Vs):
    """y limits for a potential sketch: all extrema (Vs) with room around them, and at least some of the rise of
    the curve when the extrema span next to no range (a single minimum at y = 0)."""
    lo, hi = Vs.min(), Vs.max()
    if hi - lo < 1e-12:  # one extremum: show the curve rising to the top of the panel
        return lo - 0.4 * (V_plot.max() - lo), V_plot.max()
    return lo - 0.3 * (hi - lo), hi + 0.6 * (hi - lo)
