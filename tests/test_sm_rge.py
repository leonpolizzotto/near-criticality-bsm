"""Check of the SM running (nearcrit/sm_rge.py) against arXiv:1307.3536: the couplings at M_Pl (eqs. (61a)-(61e)),
the scales of sec. 4.2 and eq. (67), and the stability bound on M_t (eq. (64)). Run `python tests/test_sm_rge.py`, or
with pytest.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the nearcrit package

import numpy as np

from nearcrit.sm_rge import (
    AS0, DAS, G1, G2, G3, LAM, MH0, MT0, T0, T_PL, YT, lowest_zero, run_rge, weak_scale_couplings,
)
def planck_scale_fit(Mh=MH0, Mt=MT0, alpha3=AS0):
    """lambda(M_Pl) and y_t(M_Pl) from the linear fits eqs. (61e) and (61d)."""
    dMh, dMt, das = Mh - MH0, Mt - MT0, (alpha3 - AS0) / DAS
    lam = -0.0143 - 0.0066 * dMt + 0.0018 * das / 0.0007 + 0.0029 * dMh   # eq. (61e)
    yt = 0.3825 + 0.0051 * dMt - 0.0021 * das / 0.0007                    # eq. (61d)
    return lam, yt


def check_against_paper():
    """Print our running at the central inputs next to the paper's numbers; returns ours and the paper's."""
    print("Check against the paper (central inputs, running up from M_t):")
    x0 = weak_scale_couplings()
    x, _, _, run = run_rge(x0, T0, T_PL, Gamma_start=0.0)
    lam_P, yt_P = planck_scale_fit()
    rows = [("g1(M_Pl)", np.sqrt(x[G1]), 0.6154, "(61a)"), ("g2(M_Pl)", np.sqrt(x[G2]), 0.5055, "(61b)"),
            ("g3(M_Pl)", np.sqrt(x[G3]), 0.4873, "(61c)"), ("yt(M_Pl)", np.sqrt(x[YT]), yt_P, "(61d)"),
            ("lam(M_Pl)", x[LAM], lam_P, "(61e)")]
    for name, ours, paper, eq in rows:
        print(f"  {name:10s} ours {ours:8.4f}   paper {paper:8.4f}  eq. {eq}")

    t, lam_eff = run["t"], run["lam_eff"].astype(float)
    LI = lowest_zero(t, lam_eff)
    Llam = lowest_zero(t, run["lam"].astype(float))
    below = t < LI
    LV = np.max(np.log(lam_eff[below] / 4) / 4 + t[below])  # Lambda_V = (max V_eff)^(1/4), eq. (63)
    print(f"  log10 Lambda_V/GeV ours {LV / np.log(10):.2f}   paper 9.5  eq. (67)")
    print(f"  Lambda_lambda/Lambda_V ours {np.exp(Llam - LV):.1f}   paper ~2;   "
          f"Lambda_I/Lambda_V ours {np.exp(LI - LV):.1f}   paper ~13  (sec. 4.2)")

    # Stability bound: the M_t at which min lambda_eff (up to M_Pl) crosses zero, by bisection.
    def min_lam_eff(Mt):
        _, _, _, r = run_rge(weak_scale_couplings(Mt=Mt), np.log(Mt), T_PL, Gamma_start=0.0)
        return r["lam_eff"].min()

    lo, hi = 169.0, 174.0
    for _ in range(30):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if min_lam_eff(mid) > 0 else (lo, mid)
    print(f"  stability bound M_t < {lo:.2f} GeV;  paper: eq. (64) gives "
          f"{MT0 + (MH0 - 129.6) / 2.0:.2f}, eq. (66) quotes 171.53 +- 0.15 th")
    return dict(rows=rows, LV=LV / np.log(10), stability_bound=lo)


def test_against_paper():
    c = check_against_paper()
    for name, ours, paper, eq in c["rows"]:
        assert abs(ours - paper) < 5e-4, (name, ours, paper)             # found agreement to 1e-4
    assert abs(c["LV"] - 9.5) < 0.05, c["LV"]
    assert abs(c["stability_bound"] - (MT0 + (MH0 - 129.6) / 2.0)) < 0.05, c["stability_bound"]   # eq. (64): 171.12


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"{name}: passed")
