"""The quartic toy model V = mu2 phi^2 / 2 + lambda phi^4 / 4 (+ delta phi^3 / 3 when tilted): its ground state
energy and derivatives, its one-loop RG flow, and the drawing helpers shared by Figs. 1-5, 10 and 11."""
import numpy as np
from matplotlib.colors import to_rgb

from nearcrit.style import set_xlabel_right


def V1D_example(phi, mu2, lambda_):
    """1D example potential function."""
    return 0.5 * mu2 * phi**2 + 0.25 * lambda_ * phi**4

def V1D_example_tilted(phi, mu2, lambda_,delta):
    """1D example potential function."""
    return 0.5 * mu2 * phi**2 + 0.25 * lambda_ * phi**4 + delta * phi**3 / 3


def E0(mu2, lambda_):
    """Ground state energy of the 1D example potential."""
    if lambda_ <= 0:
        raise ValueError("Invalid parameters: lambda must be positive.")

    if mu2 > 0:
        return 0.0
    elif mu2 < 0:
        return -mu2**2 / (4 * lambda_)
    else:
        raise ValueError("Invalid parameters: mu2={}, lambda={}".format(mu2, lambda_))


def dE0_dmu2(mu2, lambda_):
    """Derivative of the ground state energy with respect to mu2."""
    if lambda_ <= 0:
        raise ValueError("Invalid parameters: lambda must be positive.")

    if mu2 > 0:
        return 0.0
    elif mu2 < 0:
        return -mu2 / (2 * lambda_)
    else:
        raise ValueError("Invalid parameters: mu2={}, lambda={}".format(mu2, lambda_))


def d2E0_dmu22(mu2, lambda_):
    """Second derivative of the ground state energy with respect to mu2."""
    if lambda_ <= 0:
        raise ValueError("Invalid parameters: lambda must be positive.")

    if mu2 > 0:
        return 0.0
    elif mu2 < 0:
        return -1 / (2 * lambda_)
    else:
        raise ValueError("Invalid parameters: mu2={}, lambda={}".format(mu2, lambda_))


def E0_tilted(mu2, lambda_, delta):
    """Ground state energy of the 1D example potential."""
    if lambda_ <= 0:
        raise ValueError("Invalid parameters: lambda must be positive.")

    sqrt_arg = delta**2 - 4 * lambda_ * mu2

    if sqrt_arg < 0:
        return 0.0

    else:
        sqrt = np.sqrt(sqrt_arg)
        cond = abs(delta) * (abs(delta) + sqrt) > 6 * lambda_ * mu2

        if not cond:
            return 0.0
        else:
            if delta > 0:
                return (delta + sqrt)**2 * (6 * lambda_ * mu2 - delta * (delta + sqrt)) / (96 * lambda_**3)
            else:
                return (delta - sqrt)**2 * (6 * lambda_ * mu2 - delta * (delta - sqrt)) / (96 * lambda_**3)


def dE0_ddelta(mu2, lambda_, delta):
    """Ground state energy of the 1D example potential."""
    if lambda_ <= 0:
        raise ValueError("Invalid parameters: lambda must be positive.")

    sqrt_arg = delta**2 - 4 * lambda_ * mu2

    if sqrt_arg < 0:
        return 0.0

    else:
        sqrt = np.sqrt(sqrt_arg)
        cond = abs(delta) * (abs(delta) + sqrt) > 6 * lambda_ * mu2

        if not cond:
            return 0.0
        else:
            if delta > 0:
                return (-delta**3 + 3*delta*lambda_*mu2 - sqrt*(delta**2 - lambda_*mu2)) / (6 * lambda_**3)
            else:
                return (-delta**3 + 3*delta*lambda_*mu2 + sqrt*(delta**2 - lambda_*mu2)) / (6 * lambda_**3)


def d2E0_ddelta2(mu2, lambda_, delta):
    """Ground state energy of the 1D example potential."""
    if lambda_ <= 0:
        raise ValueError("Invalid parameters: lambda must be positive.")

    sqrt_arg = delta**2 - 4 * lambda_ * mu2

    if sqrt_arg < 0:
        return 0.0

    else:
        sqrt = np.sqrt(sqrt_arg)
        cond = abs(delta) * (abs(delta) + sqrt) > 6 * lambda_ * mu2

        if not cond:
            return 0.0
        else:
            if delta > 0:
                return (mu2 * lambda_ - delta**2 - delta * (delta**2 - 3 * lambda_ * mu2) / sqrt) / (2 * lambda_**3)
            else:
                return (mu2 * lambda_ - delta**2 + delta * (delta**2 - 3 * lambda_ * mu2) / sqrt) / (2 * lambda_**3)


def local_minima(phi, V):
    """Grid points that are interior local minima of V (V is sampled on phi)."""
    idx = np.where((V[1:-1] < V[:-2]) & (V[1:-1] < V[2:]))[0] + 1
    return phi[idx], V[idx]


def plot_minima(ax, phi, V, lambda_, color="C3"):
    """Mark the local minima of V: filled where there is a ground state (lambda > 0 or tilted ground
    states), open for the local minima of potentials unbounded below (lambda < 0)."""
    pm, Vm = local_minima(phi, V)
    kw = dict(mfc="white") if lambda_ < 0 else {}
    ax.plot(pm, Vm, "o", color=color, ms=6, zorder=3, **kw)



def mark_critical_point(ax, label="0", value=0.0):
    """Tick the x axis at the critical value only, labelled (0 unless the critical value is not 0)."""
    ax.set_xticks([value])
    ax.set_xticklabels([label])


def beta_function_sol(mu, mu0, mu20, lambda0):

    lambda_sol = 8 * np.pi**2 * lambda0 / (8 * np.pi**2 - 9 * lambda0 * np.log(mu/mu0))
    mu2_sol = mu20 * (lambda_sol / lambda0)**(1/3)

    return mu2_sol, lambda_sol

def beta_function_sol_t(mu, mu0, t0, lambda0):

    lambda_sol = 8 * np.pi**2 * lambda0 / (8 * np.pi**2 - 9 * lambda0 * np.log(mu/mu0))
    t_sol = t0 * (mu0 / mu)**2

    return t_sol, lambda_sol

def plot_phase_diagram(ax, label_bbox=False, xlabel=r"$\mu^2$", opaque_labels=False, lambda_max=1.0):
    """Phase diagram of the untilted potential in the (mu2, lambda) plane.

    label_bbox puts a translucent white box behind the phase labels, for drawing over other curves.
    xlabel is the label of the horizontal axis (mu2, or t in the other parameterization).
    opaque_labels puts a box in the color of the region behind each phase label, hiding the mu2 = 0 line.
    lambda_max sets the plotted range of lambda, -lambda_max to lambda_max.
    """
    bbox = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75) if label_bbox else None
    # (polygon x, polygon y, label position, label) for each quantum phase.
    L = lambda_max
    regions = [
        ([0, 1, 1, 0], [0, 0, L, L], (0.5, L / 2), "Quantum\nPhase I", "C0"),
        ([-1, 0, 0, -1], [0, 0, L, L], (-0.5, L / 2), "Quantum\nPhase II", "C1"),
        ([-1, 1, 1, -1], [-L, -L, 0, 0], (0.0, -L / 2), "No Ground State\n(NGS)", "gray"),
    ]
    for x, y, pos, label, color in regions:
        ax.fill(x, y, color=color, alpha=0.3, lw=0)
        if opaque_labels:
            # The region color is the fill color at alpha 0.3 over white.
            region_color = 0.3 * np.array(to_rgb(color)) + 0.7
            bbox = dict(boxstyle="round,pad=0.15", fc=region_color, ec="none")
        ax.text(*pos, label, ha="center", va="center", bbox=bbox, zorder=5)
    # Phase boundaries: I-II at mu2 = 0 (lambda > 0), and lambda = 0 (the I/II-NGS transitions).
    ax.plot([0, 0], [0, L], color="k", lw=1.5)
    ax.plot([0, 0], [-L, 0], color="0.35", lw=0.8)  # mu2 = 0 inside the NGS region, where it is not a transition
    ax.plot([-1, 1], [0, 0], color="k", lw=1.5)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-L, L)
    # The axis labels sit at the ends of the axes (right and top) so they do not crowd the 0 tick labels.
    set_xlabel_right(ax, xlabel)
    ax.set_ylabel(r"$\lambda$", rotation=0, ha="right", va="top")
    ax.yaxis.set_label_coords(-0.03, 1.0)
    # Only the zero axes are ticked and labelled: unlike in the potential and energy sketches, it is not
    # obvious where the axes are drawn.
    ax.set_xticks([0])
    ax.set_yticks([0])
    ax.set_xticklabels(["0"])
    ax.set_yticklabels(["0"])

