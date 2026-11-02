"""Data for Fig. 7: the SM phase (stable, metastable, unstable, no EW vacuum) on two grids of weak-scale couplings,
u = lambda_0 - lambda_c(y0_best) and y_0 = y_t(M_t), with lambda_0 = lambda(M_t).

For every grid point (lambda_0, y_0) the other couplings are set to their measured values at M_t, eqs. (58)-(60) of
arXiv:1307.3536, all couplings run up to M_Pl with the 3-loop RGEs, and the point is classified (nearcrit/sm_rge.py).

Writes data/sm_weak_grid.npz (about 3 s).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the nearcrit package

import numpy as np

from nearcrit.sm_rge import LAM, LAMC_BEST, N_STEPS, T0, T_PL, YT, classify, run_rge, weak_scale_couplings
from nearcrit.style import DATA_DIR

OUTPUT = DATA_DIR / "sm_weak_grid.npz"

# Panel ranges in (u, y_0).
PANELS = {
    "wide": dict(u=(-0.2, 0.2), y0=(0.01, 1.25), n=(161, 161)),
    "zoom": dict(u=(-0.08, 0.08), y0=(0.86, 1.02), n=(161, 161)),
}


def phase_grid(u_grid, y0_grid):
    """Run every (u, y_0) of the grid up from M_t to M_Pl and classify it (see `classify`)."""
    U, Y = np.meshgrid(u_grid, y0_grid)
    x = np.repeat(weak_scale_couplings()[:, None], U.size, axis=1)
    x[LAM], x[YT] = U.ravel() + LAMC_BEST, Y.ravel() ** 2
    with np.errstate(all="ignore"):
        xP, _, frozen, run = run_rge(x, T0, T_PL, Gamma_start=np.zeros(U.size))  # Gamma(M_t) = 0, eq. (105)
        fields = classify(run)
    fields["lam_P"] = np.where(np.isfinite(xP[LAM]), xP[LAM], np.inf)  # lambda(M_Pl)
    return {k: v.reshape(U.shape) for k, v in fields.items()}


def compute():
    """Grids for both panels, saved to OUTPUT with a key recording the settings."""
    key = repr(PANELS) + repr(N_STEPS)
    out, flat = {}, {"key": key}
    for name, p in PANELS.items():
        print(f"Computing the {name} panel ({p['n'][0]} x {p['n'][1]} points)...")
        u_grid = np.linspace(*p["u"], p["n"][0])
        y0_grid = np.linspace(*p["y0"], p["n"][1])
        out[name] = dict(u=u_grid, y0=y0_grid, **phase_grid(u_grid, y0_grid))
        flat.update({f"{name}_{k}": v for k, v in out[name].items()})
    DATA_DIR.mkdir(exist_ok=True)
    np.savez(OUTPUT, **flat)
    return out


if __name__ == "__main__":
    compute()
    print("Saved", OUTPUT)
