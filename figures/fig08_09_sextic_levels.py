"""Figs. 8 and 9: potential phase transitions of a sextic potential that leave the ground state alone.

Fig. 8: two minima swap depth order at delta_c (E1 has a kink, E0 is smooth).
Fig. 9: a minimum splits in two as mu2 goes through 0 (the second derivative of E1 jumps).

Writes output/1d_example_6_energy_levels_delta.pdf and output/1d_example_6_energy_levels_mu2.pdf.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the nearcrit package

import numpy as np
import matplotlib.pyplot as plt

from nearcrit import style
from nearcrit.style import LINEWIDTH_IN, set_xlabel_right

style.use()
def V(phi, mu2, delta, lambda_, c5, c6):
    return mu2*phi**2/2 + delta*phi**3/3 + lambda_*phi**4/4 + c5*phi**5/5 + c6*phi**6/6

def dV(phi, mu2, delta, lambda_, c5, c6):
    return mu2*phi + delta*phi**2 + lambda_*phi**3 + c5*phi**4 + c6*phi**5

def ddV(phi, mu2, delta, lambda_, c5, c6):
    return mu2 + 2*delta*phi + 3*lambda_*phi**2 + 4*c5*phi**3 + 5*c6*phi**4


def bounded_below(mu2, delta, lambda_, c5, c6):
    """Whether V is bounded below: its highest nonvanishing term has an even power and a positive coefficient."""
    for power, c in zip((6, 5, 4, 3, 2), (c6, c5, lambda_, delta, mu2)):
        if c != 0:
            return power % 2 == 0 and c > 0
    return False  # V = 0 has no isolated minimum


def minima_positions(mu2, delta, lambda_, c5, c6, tol=1e-7):
    """Positions of the local minima of V, in increasing order.

    The critical points are the real roots of dV. A critical point is a minimum if dV goes from negative
    to positive across it, which also catches minima with ddV = 0 (e.g. phi = 0 at mu2 = 0, lambda > 0)
    and rejects inflection points (double roots of dV).
    """
    roots = np.roots([c6, c5, lambda_, delta, mu2, 0.0])
    roots = np.sort(roots[np.abs(roots.imag) < tol].real)
    # Merge numerically coincident roots (a double root comes out as two nearby ones).
    roots = roots[np.concatenate([[True], np.diff(roots) > tol])]
    # dV between consecutive critical points, and beyond the outermost ones.
    probes = np.concatenate([[roots[0] - 1], (roots[:-1] + roots[1:]) / 2, [roots[-1] + 1]])
    sign = np.sign(dV(probes, mu2, delta, lambda_, c5, c6))
    return roots[(sign[:-1] < 0) & (sign[1:] > 0)]


def level_positions(mu2, delta, lambda_, c5, c6, atol=1e-9):
    """Positions phi0, phi1 of the minima that give the ground state energy E0 and the first level E1.

    Degenerate minima (V equal within atol, e.g. the minima at +-phi of a symmetric potential) are one
    level, and the leftmost of them is returned. phi1 is nan when there is no second level.
    """
    if not bounded_below(mu2, delta, lambda_, c5, c6):
        raise ValueError(
            "V is not bounded below: mu2={}, delta={}, lambda={}, c5={}, c6={}".format(mu2, delta, lambda_, c5, c6)
        )
    pm = minima_positions(mu2, delta, lambda_, c5, c6)
    Vm = V(pm, mu2, delta, lambda_, c5, c6)
    order = np.argsort(Vm, kind="stable")
    pm, Vm = pm[order], Vm[order]
    pm = pm[np.concatenate([[True], np.diff(Vm) > atol])]
    return pm[0], (pm[1] if len(pm) > 1 else np.nan)


def energy_levels(mu2, delta, lambda_, c5, c6, atol=1e-9):
    """Ground state energy E0 and first energy level E1 (the second lowest local minimum of V).

    Degenerate minima are one level (see level_positions). E1 is nan when there is no second level.
    """
    phi0, phi1 = level_positions(mu2, delta, lambda_, c5, c6, atol)
    return V(phi0, mu2, delta, lambda_, c5, c6), V(phi1, mu2, delta, lambda_, c5, c6)


def energy_levels_dparam(param, mu2, delta, lambda_, c5, c6, atol=1e-9):
    """First and second derivatives of E0 and E1 with respect to param ("mu2" or "delta"), as
    ((dE0, d2E0), (dE1, d2E1)).

    V depends on the parameter p through a term p g(phi), with g = phi^2/2 for mu2 and phi^3/3 for delta. At a
    minimum dV = 0, so dE/dp = dV/dp = g(phi). Differentiating dV(phi(p), p) = 0 gives dphi/dp = -g'(phi)/ddV,
    so d2E/dp2 = g'(phi) dphi/dp = -g'(phi)^2/ddV.
    """
    g, dg = {
        "mu2": (lambda phi: phi**2 / 2, lambda phi: phi),
        "delta": (lambda phi: phi**3 / 3, lambda phi: phi**2),
    }[param]
    return tuple(
        (g(phi), -dg(phi)**2 / ddV(phi, mu2, delta, lambda_, c5, c6))
        for phi in level_positions(mu2, delta, lambda_, c5, c6, atol)
    )


# Symmetric sextic potential with lambda < 0 < c6: as mu2 grows, the minimum at phi = 0 appears at



# Local potential phase transition: two nearby minima, at phi = 0 and phi > 0, swap which is lower at
# delta_c, while the ground state at phi < 0 is unaffected. E1 has a kink at delta_c (its first
# derivative jumps) but E0 is smooth there.
# Chosen so that E0 is not much deeper than E1, so that both are visible on the same axes.
mu2, lambda_, c5, c6 = 6.4, -22.7, 8.7, 14.9
# The swap is where V(phi_c) = V(0) = 0 and dV(phi_c) = 0 for some phi_c > 0. Taking 4 V(phi)/phi^2 - dV(phi)/phi
# gives mu2 + delta phi/3 - c5 phi^3/5 - c6 phi^4/3 = 0, so delta = 3 (c5 phi^3/5 + c6 phi^4/3 - mu2)/phi, and
# putting that back into dV(phi)/phi = 0 leaves 2 c6 phi^4 + 8 c5 phi^3/5 + lambda phi^2 - 2 mu2 = 0.
phi_c = [r.real for r in np.roots([2 * c6, 8 * c5 / 5, lambda_, 0, -2 * mu2]) if abs(r.imag) < 1e-12 and r.real > 0]
delta_c = [3 * (c5 * x**3 / 5 + c6 * x**4 / 3 - mu2) / x for x in phi_c]
# Keep the positive root at which phi_c is a minimum (a maximum at V = 0 is not a swap).
phi_c, delta_c = next((x, d) for x, d in zip(phi_c, delta_c) if ddV(x, mu2, d, lambda_, c5, c6) > 0)

# The potentials in the top row are at delta_c and delta_c +- delta_offset. The energies are plotted over
# delta_c +- 2 delta_offset, so that the lines marking delta_c +- delta_offset are half way to the ends.
delta_offset = 0.6
# Each side of delta_c is evaluated separately, so that no line is drawn across the jumps.
eps = 1e-9
delta_sides = (np.linspace(delta_c - 2 * delta_offset, delta_c - eps, 400),
               np.linspace(delta_c + eps, delta_c + 2 * delta_offset, 400))


def level_curves(param, values, params):
    """E0 and its first two derivatives with respect to param, and the same for E1, as arrays over values.

    params holds the parameters of V (mu2, delta, lambda_, c5, c6); param is set to each of values in turn.
    """
    rows = []
    for x in values:
        p = {**params, param: x}
        (E0, E1), ((dE0, d2E0), (dE1, d2E1)) = energy_levels(**p), energy_levels_dparam(param, **p)
        rows.append([[E0, dE0, d2E0], [E1, dE1, d2E1]])
    return np.moveaxis(np.array(rows), 0, -1)  # indexed [level, derivative, value]


curves = [level_curves("delta", deltas, dict(mu2=mu2, delta=delta_c, lambda_=lambda_, c5=c5, c6=c6))
          for deltas in delta_sides]

# Top row: the potential just below, at and just above delta_c. Bottom row: E0 and E1 share each panel;
# the derivatives of E1 jump at delta_c, those of E0 do not.
# (delta, title, colour) for each potential; the colour is also that of the line in the bottom row
# marking where the potential sits in delta. Red and orange are kept for E0 and E1.
potential_panels = [
    (delta_c - delta_offset, r"$\delta<\delta_c$", "C0"),
    (delta_c, r"$\delta=\delta_c$", "C4"),
    (delta_c + delta_offset, r"$\delta>\delta_c$", "C2"),
]
# (colour, line style) of E0 and E1, shared by the curves in the bottom row and the minima in the top row.
level_styles = [("C3", "-"), ("C1", "--")]
phi = np.linspace(-1.75, 1.35, 2001)
energy_labels = [
    r"$E_n$",
    r"$\displaystyle\frac{\partial E_n}{\partial\delta}$",
    r"$\displaystyle\frac{\partial^2 E_n}{\partial\delta^2}$",
]


def level_index(Vm, atol=1e-9):
    """Energy level n of each minimum with value Vm (degenerate minima share a level, as in energy_levels)."""
    levels = np.sort(Vm)
    levels = levels[np.concatenate([[True], np.diff(levels) > atol])]
    return np.array([np.sum(levels < v - atol) for v in Vm])


fig, axes = plt.subplots(2, 3, figsize=(LINEWIDTH_IN, 4.9))
for ax, (delta, title, color) in zip(axes[0], potential_panels):
    ax.plot(phi, V(phi, mu2, delta, lambda_, c5, c6), color=color, lw=1.5)
    # A filled marker in the colour of E0 at the ground state, an open one in the colour of E1 at E1 (both
    # metastable minima are E1 at delta_c, where they are degenerate) and an open one in the colour of the
    # curve at E2.
    pm = minima_positions(mu2, delta, lambda_, c5, c6)
    Vm = V(pm, mu2, delta, lambda_, c5, c6)
    for x, v, n in zip(pm, Vm, level_index(Vm)):
        if n == 0:
            ax.plot(x, v, "o", color=level_styles[0][0], ms=6, zorder=3)
        else:
            ax.plot(x, v, "o", mfc="white", color=level_styles[1][0] if n == 1 else color, ms=6, zorder=3)
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_title(title)
    ax.set_xlabel(r"$\phi$")
    ax.set_xlim(phi[0], phi[-1])
    ax.set_ylim(-2.6, 1.2)
    ax.set_xticks([])
    ax.set_yticks([])
axes[0, 0].set_ylabel(r"$V(\phi)$")

for k, ax in enumerate(axes[1]):
    for n, (color, ls) in enumerate(level_styles):
        for side, (deltas, c) in enumerate(zip(delta_sides, curves)):
            ax.plot(deltas, c[n, k], color=color, ls=ls, lw=1.5, label=rf"$E_{n}$" if side == 0 else None)
    ax.axhline(0, color="gray", lw=0.5, zorder=1)  # under the curves, so E1 = 0 shows on top of it
    # Where the potentials in the top row are, in their colours.
    delta_lines = [ax.axvline(delta, color=color, lw=0.9, zorder=1, label=title)
                   for delta, title, color in potential_panels]
    if k > 0:
        # Mark the jumps in the derivatives of E1 at the transition.
        for value in (curves[0][1, k, -1], curves[1][1, k, 0]):
            ax.plot(delta_c, value, "o", mfc="white", color=level_styles[1][0], ms=6, zorder=3)
    ax.set_ylabel(energy_labels[k], rotation=0, ha="right", va="center")
    set_xlabel_right(ax, r"$\delta$")
    ax.set_xlim(delta_sides[0][0], delta_sides[1][-1])
    ax.set_xticks([delta_c])
    ax.set_xticklabels([r"$\delta_c$"])
    ax.set_yticks([])
# Frosted boxes behind the legends, so that the lines they cover do not run through the text.
legend_frame = dict(frameon=True, facecolor="white", edgecolor="none", framealpha=0.85, fancybox=True, fontsize=9)
axes[1, 0].legend(handles=axes[1, 0].get_lines()[:4:2], loc="lower left", handlelength=1.6, **legend_frame).set_zorder(5)
# On the right, between E1 (= 0 there) and E0, above the jump marker at delta_c.
axes[1, 2].legend(handles=delta_lines, loc="center right", bbox_to_anchor=(1, 0.7), handlelength=0.9,
                  handletextpad=0.4, borderaxespad=0.2, **legend_frame).set_zorder(5)

fig.tight_layout()
style.savefig(fig, "1d_example_6_energy_levels_delta.pdf")


# Local second order transition: with delta = 0, the minimum at phi = 0 splits continuously into two as mu2
# goes through 0 (V''(0) = mu2 and V'''(0) = 2 delta = 0), while the ground state at phi < 0 is unaffected. E1
# and its first derivative are continuous at mu2 = 0, but its second derivative jumps. Varying delta instead
# does not give this: away from phi = 0 it tilts the potential, and at phi = 0 it does not change V''.
# Because E0 changes as dE0/dmu2 = phi0^2/2, the ground state has to be much deeper than the split minima to
# stay the ground state over the range of mu2, so the local structure is tiny on the scale of the full
# potential, so it is shown in insets around phi = 0.
energy_labels_mu2 = [
    r"$E_n$",
    r"$\displaystyle\frac{\partial E_n}{\partial\mu^2}$",
    r"$\displaystyle\frac{\partial^2 E_n}{\partial(\mu^2)^2}$",
]


def plot_level_minima(ax, params, color, levels=(0, 1, 2)):
    """Mark the minima of V on ax: E0 filled in its colour, E1 open in its colour, E2 open in color."""
    pm = minima_positions(**params)
    Vm = V(pm, **params)
    for x, v, n in zip(pm, Vm, level_index(Vm)):
        if n not in levels:
            continue
        if n == 0:
            ax.plot(x, v, "o", color=level_styles[0][0], ms=6, zorder=3)
        else:
            ax.plot(x, v, "o", mfc="white", color=level_styles[1][0] if n == 1 else color, ms=6, zorder=3)


def set_band(ax, values, band):
    """Set the y limits of ax so that values fill the band (bottom, top) of the axes height."""
    lo, hi = band
    vmin, vmax = np.nanmin(values), np.nanmax(values)
    span = (vmax - vmin) / (hi - lo)
    ax.set_ylim(vmin - lo * span, vmin + (1 - lo) * span)


def plot_second_order(params, mu2_offset, phi, filename, phi_zoom):
    """Potentials at mu2 = -mu2_offset, 0, mu2_offset (top row) and E0, E1 and their derivatives over
    |mu2| < 2 mu2_offset (bottom row) for the local second order transition, with the other parameters in params.
    The top row shows the full potentials, with insets around phi = 0 over the phi range phi_zoom.
    """
    eps = 1e-6 * mu2_offset
    mu2_sides = (np.linspace(-2 * mu2_offset, -eps, 400), np.linspace(eps, 2 * mu2_offset, 400))
    curves = [level_curves("mu2", mu2s, {**params, "mu2": 0.0}) for mu2s in mu2_sides]
    potential_panels = [
        (-mu2_offset, r"$\mu^2<0$", "C0"),
        (0.0, r"$\mu^2=0$", "C4"),
        (mu2_offset, r"$\mu^2>0$", "C2"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(LINEWIDTH_IN, 4.9))
    split = {**params, "mu2": -mu2_offset}  # the potential with the split minima
    V_ground = V(level_positions(**split)[0], **split)
    ylim = (1.3 * V_ground, -0.9 * V_ground)
    zoom_depth = mu2_offset**2 / (4 * params["lambda_"])  # depth of the split minima at mu2 = -mu2_offset
    for ax, (mu2, title, color) in zip(axes[0], potential_panels):
        p = {**params, "mu2": mu2}
        ax.plot(phi, V(phi, **p), color=color, lw=1.5)
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        ax.set_title(title)
        ax.set_xlabel(r"$\phi$")
        ax.set_xlim(phi[0], phi[-1])
        ax.set_ylim(*ylim)
        ax.set_xticks([])
        ax.set_yticks([])
        plot_level_minima(ax, p, color, levels=(0,))
        inset = ax.inset_axes([0.47, 0.06, 0.5, 0.42])
        inset.plot(phi_zoom, V(phi_zoom, **p), color=color, lw=1.2)
        plot_level_minima(inset, p, color, levels=(1, 2))
        inset.axhline(0, color="gray", lw=0.5)
        inset.axvline(0, color="gray", lw=0.5)
        inset.set_xlim(phi_zoom[0], phi_zoom[-1])
        inset.set_ylim(-2.0 * zoom_depth, 2.5 * zoom_depth)
        inset.set_xticks([])
        inset.set_yticks([])
        for spine in inset.spines.values():
            spine.set_linewidth(0.6)
        # Box around the zoomed region, joined to the corners of the inset. The region is far too flat to
        # see at the scale of the panel, so the box is given a minimum height (a fraction of the panel).
        box_height = max(np.diff(inset.get_ylim())[0], 0.05 * np.diff(ylim)[0])
        box = (phi_zoom[0], np.mean(inset.get_ylim()) - box_height / 2, np.ptp(phi_zoom), box_height)
        indicator = ax.indicate_inset(box, inset_ax=inset, edgecolor="0.3", lw=0.6, alpha=1)
        for connector in indicator.connectors:  # matplotlib shows the two that do not cross the inset
            connector.set(lw=0.6, color="0.3")
    axes[0, 0].set_ylabel(r"$V(\phi)$")

    # E0 and E1 share each panel but have their own vertical scales (E1 changes far less than E0), each in its
    # own band of the panel, with the larger of the two on top so that they are in the right order.
    lower_band, upper_band = (0.07, 0.43), (0.57, 0.93)
    for k, ax in enumerate(axes[1]):
        ax1 = ax.twinx()  # E1's scale; drawn over ax
        for n, (axn, (color, ls)) in enumerate(zip((ax, ax1), level_styles)):
            for side, (mu2s, c) in enumerate(zip(mu2_sides, curves)):
                axn.plot(mu2s, c[n, k], color=color, ls=ls, lw=1.5, label=rf"$E_{n}$" if side == 0 else None)
        E0_values, E1_values = (np.concatenate([c[n, k] for c in curves]) for n in range(2))
        E1_on_top = E1_values.mean() > E0_values.mean()
        set_band(ax, E0_values, lower_band if E1_on_top else upper_band)
        set_band(ax1, E1_values, upper_band if E1_on_top else lower_band)
        ax1.axhline(0, color="gray", lw=0.5, zorder=1)  # E1's zero line (E1 = 0 for mu2 > 0)
        # Where the potentials in the top row are, in their colours.
        mu2_lines = [ax.axvline(mu2, color=color, lw=0.9, zorder=1, label=title)
                     for mu2, title, color in potential_panels]
        if k == 2:
            # Mark the jump in the second derivative of E1 at the transition.
            for value in (curves[0][1, k, -1], curves[1][1, k, 0]):
                ax1.plot(0, value, "o", mfc="white", color=level_styles[1][0], ms=6, zorder=3)
        ax.set_ylabel(energy_labels_mu2[k], rotation=0, ha="right", va="center")
        set_xlabel_right(ax, r"$\mu^2$")
        ax.set_xlim(mu2_sides[0][0], mu2_sides[1][-1])
        ax.set_xticks([0])
        ax.set_xticklabels(["0"])
        for a in (ax, ax1):
            a.set_yticks([])
        if k == 0:
            ax1.legend(handles=[ax.get_lines()[0], ax1.get_lines()[0]], loc="lower right", handlelength=1.6,
                       **legend_frame).set_zorder(5)
        if k == 2:
            ax1.legend(handles=mu2_lines, loc="center right", handlelength=0.9, handletextpad=0.4,
                       borderaxespad=0.2, **legend_frame).set_zorder(5)

    fig.tight_layout()
    style.savefig(fig, filename)


# Insets: the splitting is only symmetric enough for both new minima to exist while
# sqrt(-mu2/lambda) << lambda/c5, so the range of mu2 is small.
plot_second_order(dict(delta=0.0, lambda_=0.48, c5=1.6, c6=1.0), 0.002, np.linspace(-1.5, 0.9, 2001),
                  "1d_example_6_energy_levels_mu2.pdf", phi_zoom=np.linspace(-0.12, 0.12, 801))
