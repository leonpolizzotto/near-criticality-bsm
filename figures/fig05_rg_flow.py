"""Fig. 5: the one-loop RG flow towards the IR on the phase diagram, in (mu2, lambda) and in (t, lambda).

Writes output/1d_example_phase_diagram_flow_t.pdf."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the nearcrit package

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import FancyArrowPatch

from nearcrit import style
from nearcrit.style import LINEWIDTH_IN
from nearcrit.toy_quartic import beta_function_sol, beta_function_sol_t, plot_phase_diagram

style.use()
def flow_curve_off_axis(mu20, lambda0, mu0=1.0, n=400, lambda_end=4.0):
    """Flow line from (mu20, lambda0), with lambda0 slightly off lambda = 0, running off the plot.

    dlambda/dln(mu) = 9 lambda^2 / (8 pi^2) is positive, so lambda > 0 only reaches the plotted region
    from larger mu (it runs to the Landau pole going up), and lambda < 0 runs away from 0 towards the
    IR. So run lambda > 0 towards the UV and lambda < 0 towards the IR, until |lambda| = lambda_end.
    The curve is returned ordered along the IR flow (towards smaller mu). Uses beta_function_sol, with
    mu found from lambda: ln(mu / mu0) = 8 pi^2 / 9 * (1 / lambda0 - 1 / lambda).
    """
    # lambda in geometric steps, since it changes quickly far from lambda0
    lam = lambda0 * (lambda_end / abs(lambda0)) ** np.linspace(0, 1, n)
    mu = mu0 * np.exp(8 * np.pi**2 / 9 * (1 / lambda0 - 1 / lam))
    mu2, lam = beta_function_sol(mu, mu0, mu20, lambda0)
    return (mu2[::-1], lam[::-1]) if lambda0 > 0 else (mu2, lam)


class HandlerArrow(HandlerPatch):
    """Legend handler that draws a FancyArrowPatch as a horizontal arrow across the handle box."""

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        y = height / 2 - ydescent
        arrow = FancyArrowPatch(
            (-xdescent, y), (width - xdescent, y), arrowstyle=orig_handle.get_arrowstyle(),
            mutation_scale=orig_handle.get_mutation_scale(), color=orig_handle.get_edgecolor(),
            lw=orig_handle.get_linewidth(), transform=trans,
        )
        return [arrow]


def plot_flows(ax, start_points, color="C4", flow=flow_curve_off_axis, flow_is_ir=True, arrow_fractions=None):
    """Draw RG flow lines from each (mu2, lambda) start point on the phase diagram, with arrows towards the IR.

    flow(mu20, lambda0) returns the flow curve (mu2, lambda) from the start point. If flow_is_ir the curve is
    ordered along the IR flow and the arrows follow it; otherwise it runs towards increasing mu and the
    arrows point backwards along it. arrow_fractions are the positions of the arrow along the visible
    curve to try, as a fraction of its length, in order (the first one not under a label is used).
    """
    if arrow_fractions is None:
        arrow_fractions = (0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8, 0.1, 0.9)
    ax.figure.draw_without_rendering()  # so that the text sizes are known
    label_boxes = [text.get_window_extent().transformed(ax.transData.inverted()).expanded(1.1, 1.1)
                   for text in ax.texts]
    for mu20, lambda0 in start_points:
        mu2, lam = flow(mu20, lambda0)
        inside = (np.abs(mu2) <= max(ax.get_xlim())) & (np.abs(lam) <= max(ax.get_ylim()))  # the flow can leave the plotted region
        mu2, lam = mu2[inside], lam[inside]
        ax.plot(mu2, lam, color=color, lw=0.9, zorder=3)
        # Arrow as close to half way along the visible curve (arc length in units of the axes size) as
        # possible without sitting under one of the labels. Short stubs at the edge get no arrow.
        xspan, yspan = np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim())
        length = np.concatenate([[0], np.cumsum(np.hypot(np.diff(mu2) / xspan, np.diff(lam) / yspan))])
        if length[-1] < 0.12:
            continue
        for fraction in arrow_fractions:
            i = min(np.searchsorted(length, fraction * length[-1]), len(mu2) - 2)
            if not any(box.contains(mu2[i], lam[i]) for box in label_boxes):
                break
        head, tail = (i + 1, i) if flow_is_ir else (i, i + 1)
        ax.annotate("", xy=(mu2[head], lam[head]), xytext=(mu2[tail], lam[tail]), zorder=3,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=0.9, mutation_scale=8, shrinkA=0, shrinkB=0))


def make_flow_starts(edge=0.95, n_mu2=13, lambda_off=0.05):
    """Start points (mu2, lambda) just above and below lambda = 0, evenly spaced in mu2 (and mirrored)."""
    return [(mu20, sign * lambda_off) for sign in (-1, 1) for mu20 in np.linspace(-edge, edge, n_mu2)]


late_arrows = (0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25)
flow_arrow = FancyArrowPatch((0, 0), (1, 0), arrowstyle="-|>", mutation_scale=8, color="C4", lw=0.9)


def flow_curve_t_ir(t0, lambda0, mu0=1.0, n=200, t_max=1.2):
    """Run (t0, lambda0) from mu0 down towards the IR until |t| reaches t_max.

    t = t0 (mu0 / mu)^2 grows exponentially in ln(mu0 / mu), while lambda barely moves over that range.
    """
    s_max = 0.5 * np.log(t_max / abs(t0))
    mu = mu0 * np.exp(-np.linspace(0, s_max, n))
    return beta_function_sol_t(mu, mu0, t0, lambda0)


# The flow in the original parameterization (mu2) next to the one with the scaled variable t, which
# runs as t ~ 1/mu^2 (beta_function_sol_t). In t the flow leaves t = 0 very quickly towards the IR
# (it grows exponentially while lambda hardly changes), so on the right start just off t = 0 at a
# range of lambda and run to the IR.
lambda_range = 3.0  # both panels cover the same, larger range of lambda (the flow lines bend more there)
fig, axes = plt.subplots(1, 2, figsize=(LINEWIDTH_IN, 2.9))
plot_phase_diagram(axes[0], label_bbox=True, lambda_max=lambda_range)
plot_phase_diagram(axes[1], label_bbox=True, xlabel=r"$t$", lambda_max=lambda_range)
axes[1].set_ylabel("")
axes[1].set_yticks([])  # lambda is shared with the left panel, so no 0 tick or label
axes[1].legend(
    [flow_arrow], ["RG Flow into the IR"], loc="lower right", bbox_to_anchor=(1, 1), frameon=False, fontsize=9,
    handler_map={FancyArrowPatch: HandlerArrow()},
)
fig.tight_layout()

# Start just off lambda = 0 by 0.05 lambda_range (like the single flow plot, whose range is 1), so the
# lines look the same on the page as in that plot: mu2 depends on lambda / lambda0 only.
plot_flows(axes[0], make_flow_starts(n_mu2=17, lambda_off=0.05 * lambda_range), arrow_fractions=late_arrows)
# Start points for the t panel: two columns just off t = 0, evenly spaced in lambda on the page, and
# lines that come in through the top border. The latter start just off t = 0 too, but at lambda above
# the plotted range, chosen so that they cross the top border at evenly spaced values of t. The
# spacing d in t is even, apart from the central pair at +-t_start which straddles t = 0 (no flow
# starts exactly on t = 0). The columns have the same spacing on the page as the top border, adjusted
# so that they are even through lambda = 0.
t_start, t_edge, n_half = 0.03, 0.95, 4
d = (t_edge - t_start) / n_half
t_cross = t_start + d * np.arange(1, n_half + 1)  # where the extra lines cross the top border
lambda_top = 0.95 * lambda_range
axes_width, axes_height = axes[1].get_position().size * fig.get_size_inches()
d_lambda_page = d * axes_width / axes_height * lambda_range  # lambda spacing that looks like d in t
m = max(1, round(lambda_top / d_lambda_page - 0.5))
d_lambda = lambda_top / (m + 0.5)
column_lambda = d_lambda * (np.arange(-m - 1, m + 1) + 0.5)  # from -lambda_top to lambda_top
# Running from t_start at lambda0 down to the top border (lambda = lambda_range) takes
# s = ln(t / t_start) / 2 in ln(mu0/mu), and 1/lambda = 1/lambda0 + 9 s / (8 pi^2).
lambda_off_grid = 1 / (1 / lambda_range - 9 * np.log(t_cross / t_start) / (16 * np.pi**2))
t_flow_starts = (
    [(sign * t_start, lam0) for sign in (-1, 1) for lam0 in column_lambda]
    + [(sign * t_start, lam0) for sign in (-1, 1) for lam0 in lambda_off_grid]
)
plot_flows(axes[1], t_flow_starts, flow=flow_curve_t_ir, flow_is_ir=True)
style.savefig(fig, "1d_example_phase_diagram_flow_t.pdf")
