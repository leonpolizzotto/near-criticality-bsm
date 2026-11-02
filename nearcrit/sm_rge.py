"""SM running and vacuum-stability classification, following Buttazzo, Degrassi, Giardino, Giudice, Sala, Salvio,
Strumia, "Investigating the near-criticality of the Higgs boson", arXiv:1307.3536v4 (JHEP 1312 (2013) 089).
Equation numbers refer to that paper.

  - The MS-bar couplings at M_t from the measured inputs, eqs. (55)-(60).
  - The 3-loop RGEs, eqs. (96)-(103), integrated with RK4.
  - The 2-loop effective quartic lambda_eff(h), eqs. (104)-(110), and the vacuum-decay probability in our past light
    cone, eqs. (81)-(83).
  - Classification of a run (sec. 4.2, 4.4, 6.3): no EW vacuum if lambda(M_t) < 0; stable if the running lambda stays
    >= 0 up to M_Pl; unstable if the decay probability p0 > 1; metastable otherwise.

Note on the stability criterion. Sec. 4.2 defines stability with lambda_eff, and with that criterion the paper's
stability bound, eq. (64), is recovered to 0.01 GeV (see tests/test_sm_rge.py). In its fig. 4, however, the stability
edge lies exactly at lambda(M_Pl) = 0 for y_t(M_Pl) > 0.37, which is the running-lambda criterion; lambda_eff would
move it to lambda(M_Pl) ~ -0.015 at y_t(M_Pl) = 0.7, because the top loop in eq. (106) grows with y_t. The two
criteria agree below y_t(M_Pl) ~ 0.4 and differ only inside the Planck-scale dominated region, where that paper
considers the assessment unreliable. The running lambda is used, as in its fig. 4; set STABILITY_FROM_LAMBDA_EFF = True
for the sec. 4.2 criterion.
"""
import numpy as np
from scipy.special import spence
# ---------------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------------

# Table 2: central values and errors of the measured inputs (GeV).
MH0, DMH = 125.15, 0.24
MT0, DMT = 173.34, np.hypot(0.76, 0.3)  # experimental and non-perturbative errors in quadrature
AS0, DAS = 0.1184, 0.0007
MW0, DMW = 80.384, 0.014
V_HIGGS = 246.21971

M_PL = 1.22e19  # fig. 2 caption: M_Pl ~ 1.2e19 GeV
H0 = 67.4 / 3.0857e19 * 6.5821e-25  # eq. (82): H0 ~ 67.4 km/s/Mpc, in GeV
T0, T_PL = np.log(MT0), np.log(M_PL)  # RG time t = ln(mu / GeV); the weak scale is mu = M_t
N_STEPS = 400  # RK4 steps between M_t and M_Pl (dt ~ 0.1)
STABILITY_FROM_LAMBDA_EFF = False  # see the note in the module docstring

K1 = 1 / (4 * np.pi) ** 2  # loop factor 1/(4 pi)^2

# Index of each coupling in the state vector x = (g1^2, g2^2, g3^2, yt^2, yb^2, ytau^2, lambda).
G1, G2, G3, YT, YB, YL, LAM = range(7)


def weak_scale_couplings(Mh=MH0, Mt=MT0, alpha3=AS0, MW=MW0):
    """MS-bar couplings at mu = M_t from the measured inputs, as the state vector x.

    lambda: eq. (55); y_t: eq. (57); g2, gY: eqs. (58), (59); g3: eq. (60).
    g1 is in GUT normalisation, g1^2 = 5 gY^2 / 3 (sec. 2).
    """
    dMh, dMt = Mh - MH0, Mt - MT0
    das, dmw = (alpha3 - AS0) / DAS, (MW - MW0) / DMW
    lam = 0.12604 + 0.00206 * dMh - 0.00004 * dMt                     # eq. (55)
    yt = 0.93690 + 0.00556 * dMt - 0.00042 * das / 0.0007             # eq. (57)
    g2 = 0.64779 + 0.00004 * dMt + 0.00011 * dmw / 0.014              # eq. (58)
    gY = 0.35830 + 0.00011 * dMt - 0.00020 * dmw / 0.014              # eq. (59)
    g3 = 1.1666 + 0.00314 * das - 0.00046 * dMt                       # eq. (60)
    # Not given as fits in the paper (sec. 3.5 only quotes mb(mb) = 4.2 GeV and M_tau = 1.777 GeV):
    # tree-level y = sqrt(2) m / v with m_b(M_t) ~ 2.75 GeV. They barely affect the running.
    yb = np.sqrt(2) * 2.75 / V_HIGGS
    ytau = np.sqrt(2) * 1.777 / V_HIGGS
    return np.array([5 * gY**2 / 3, g2**2, g3**2, yt**2, yb**2, ytau**2, lam])



# ---------------------------------------------------------------------------------------------------
# Renormalisation group equations (appendix B)
# ---------------------------------------------------------------------------------------------------


def beta(x):
    """dx/d ln(mu) for x = (g1^2, g2^2, g3^2, yt^2, yb^2, ytau^2, lambda); eqs. (96)-(99), (101)-(103).

    The paper writes the RGEs as d/d ln(mu^2), so each right-hand side is multiplied by 2 at the end.
    The Higgs mass RGE, eq. (100), is not needed. Variables ending in `s` are squared couplings.
    """
    g1s, g2s, g3s, yts, ybs, yls, lam = x
    k1, k2, k3, k4 = K1, K1**2, K1**3, K1**4

    # eq. (96): hypercharge gauge coupling (GUT normalisation)
    dg1 = g1s**2 * (
        k1 * 41 / 10
        + k2 * (44 * g3s / 5 + 27 * g2s / 10 + 199 * g1s / 50 - 17 * yts / 10 - ybs / 2 - 3 * yls / 2)
        + k3 * (yts * (189 * yts / 16 - 29 * g3s / 5 - 471 * g2s / 32 - 2827 * g1s / 800)
                + lam * (-9 * lam / 5 + 9 * g2s / 10 + 27 * g1s / 50)
                + 297 * g3s**2 / 5 + 789 * g2s**2 / 64 - 388613 * g1s**2 / 24000
                - 3 * g3s * g2s / 5 - 137 * g3s * g1s / 75 + 123 * g2s * g1s / 160))

    # eq. (97): SU(2)_L gauge coupling
    dg2 = g2s**2 * (
        k1 * (-19 / 6)
        + k2 * (12 * g3s + 35 * g2s / 6 + 9 * g1s / 10 - 3 * yts / 2 - 3 * ybs / 2 - yls / 2)
        + k3 * (yts * (147 * yts / 16 - 7 * g3s - 729 * g2s / 32 - 593 * g1s / 160)
                + lam * (-3 * lam + 3 * g2s / 2 + 3 * g1s / 10)
                + 81 * g3s**2 + 324953 * g2s**2 / 1728 - 5597 * g1s**2 / 1600
                + 39 * g3s * g2s - g3s * g1s / 5 + 873 * g2s * g1s / 160))

    # eq. (98): strong gauge coupling, with the 4-loop pure QCD term
    dg3 = g3s**2 * (
        k1 * (-7)
        + k2 * (-26 * g3s + 9 * g2s / 2 + 11 * g1s / 10 - 2 * yts - 2 * ybs)
        + k3 * (yts * (15 * yts - 40 * g3s - 93 * g2s / 8 - 101 * g1s / 40)
                + 65 * g3s**2 / 2 + 109 * g2s**2 / 8 - 523 * g1s**2 / 120
                + 21 * g3s * g2s + 77 * g3s * g1s / 15 - 3 * g2s * g1s / 40)
        + k4 * g3s**3 * (-2472.28))

    # eq. (99): Higgs quartic coupling
    dlam = (
        k1 * (lam * (12 * lam + 6 * yts + 6 * ybs + 2 * yls - 9 * g2s / 2 - 9 * g1s / 10)
              - 3 * yts**2 - 3 * ybs**2 - yls**2
              + 9 * g2s**2 / 16 + 27 * g1s**2 / 400 + 9 * g2s * g1s / 40)
        + k2 * (lam**2 * (-156 * lam - 72 * yts - 72 * ybs - 24 * yls + 54 * g2s + 54 * g1s / 5)
                + lam * yts * (-3 * yts / 2 - 21 * ybs + 40 * g3s + 45 * g2s / 4 + 17 * g1s / 4)
                + lam * ybs * (-3 * ybs / 2 + 40 * g3s + 45 * g2s / 4 + 5 * g1s / 4)
                + lam * yls * (-yls / 2 + 15 * g2s / 4 + 15 * g1s / 4)
                + lam * (-73 * g2s**2 / 16 + 1887 * g1s**2 / 400 + 117 * g2s * g1s / 40)
                + yts**2 * (15 * yts - 3 * ybs - 16 * g3s - 4 * g1s / 5)
                + yts * (-9 * g2s**2 / 8 - 171 * g1s**2 / 200 + 63 * g2s * g1s / 20)
                + ybs**2 * (-3 * yts + 15 * ybs - 16 * g3s + 2 * g1s / 5)
                + ybs * (-9 * g2s**2 / 8 + 9 * g1s**2 / 40 + 27 * g2s * g1s / 20)
                + yls**2 * (5 * yls - 6 * g1s / 5)
                + yls * (-3 * g2s**2 / 8 - 9 * g1s**2 / 8 + 33 * g2s * g1s / 20)
                + 305 * g2s**3 / 32 - 3411 * g1s**3 / 4000
                - 289 * g2s**2 * g1s / 160 - 1677 * g2s * g1s**2 / 800)
        + k3 * (lam**3 * (6011.35 * lam + 873 * yts - 387.452 * g2s - 77.490 * g1s)
                + lam**2 * yts * (1768.26 * yts + 160.77 * g3s - 359.539 * g2s - 63.869 * g1s)
                + lam**2 * (-790.28 * g2s**2 - 185.532 * g1s**2 - 316.64 * g2s * g1s)
                + lam * yts**2 * (-223.382 * yts - 662.866 * g3s - 5.470 * g2s - 21.015 * g1s)
                + lam * yts * (356.968 * g3s**2 - 319.664 * g2s**2 - 74.8599 * g1s**2
                               + 15.1443 * g3s * g2s + 17.454 * g3s * g1s + 5.615 * g2s * g1s)
                + lam * g2s**2 * (-57.144 * g3s + 865.483 * g2s + 79.638 * g1s)
                + lam * g1s**2 * (-8.381 * g3s + 61.753 * g2s + 28.168 * g1s)
                + yts**3 * (-243.149 * yts + 250.494 * g3s + 74.138 * g2s + 33.930 * g1s)
                + yts**2 * (-50.201 * g3s**2 + 15.884 * g2s**2 + 15.948 * g1s**2
                            + 13.349 * g3s * g2s + 17.570 * g3s * g1s - 70.356 * g2s * g1s)
                + yts * g3s * (16.464 * g2s**2 + 1.016 * g1s**2 + 11.386 * g2s * g1s)
                + yts * g2s**2 * (62.500 * g2s + 13.041 * g1s)
                + yts * g1s**2 * (10.627 * g2s + 11.117 * g1s)
                + g3s * (7.536 * g2s**3 + 0.663 * g1s**3 + 1.507 * g2s**2 * g1s + 1.105 * g2s * g1s**2)
                - 114.091 * g2s**4 - 1.508 * g1s**4 - 37.889 * g2s**3 * g1s
                + 6.500 * g2s**2 * g1s**2 - 1.543 * g2s * g1s**3))

    # eq. (101): top Yukawa coupling
    dyt = yts * (
        k1 * (9 * yts / 2 + 3 * ybs / 2 + yls - 8 * g3s - 9 * g2s / 4 - 17 * g1s / 20)
        + k2 * (yts * (-12 * yts - 11 * ybs / 4 - 9 * yls / 4 - 12 * lam + 36 * g3s
                       + 225 * g2s / 16 + 393 * g1s / 80)
                + ybs * (-ybs / 4 + 5 * yls / 4 + 4 * g3s + 99 * g2s / 16 + 7 * g1s / 80)
                + yls * (-9 * yls / 4 + 15 * g2s / 8 + 15 * g1s / 8)
                + 6 * lam**2 - 108 * g3s**2 - 23 * g2s**2 / 4 + 1187 * g1s**2 / 600
                + 9 * g3s * g2s + 19 * g3s * g1s / 15 - 9 * g2s * g1s / 20)
        + k3 * (yts**2 * (58.6028 * yts + 198 * lam - 157 * g3s - 1593 * g2s / 16 - 2437 * g1s / 80)
                + lam * yts * (15 * lam / 4 + 16 * g3s - 135 * g2s / 2 - 127 * g1s / 10)
                + yts * (363.764 * g3s**2 + 16.990 * g2s**2 - 24.422 * g1s**2
                         + 48.370 * g3s * g2s + 18.074 * g3s * g1s + 34.829 * g2s * g1s)
                + lam**2 * (-36 * lam + 45 * g2s + 9 * g1s)
                + lam * (-171 * g2s**2 / 16 - 1089 * g1s**2 / 400 + 117 * g2s * g1s / 40)
                - 619.35 * g3s**3 + 169.829 * g2s**3 + 16.099 * g1s**3
                + 73.654 * g3s**2 * g2s - 15.096 * g3s**2 * g1s - 21.072 * g3s * g2s**2
                - 22.319 * g3s * g1s**2 - 321 * g3s * g2s * g1s / 20
                - 4.743 * g2s**2 * g1s - 4.442 * g2s * g1s**2))

    # eq. (102): bottom Yukawa coupling (two loops)
    dyb = ybs * (
        k1 * (3 * yts / 2 + 9 * ybs / 2 + yls - 8 * g3s - 9 * g2s / 4 - g1s / 4)
        + k2 * (yts * (-yts / 4 - 11 * ybs / 4 + 5 * yls / 4 + 4 * g3s + 99 * g2s / 16 + 91 * g1s / 80)
                + ybs * (-12 * ybs - 9 * yls / 4 - 12 * lam + 36 * g3s + 225 * g2s / 16 + 237 * g1s / 80)
                + yls * (-9 * yls / 4 + 15 * g2s / 8 + 15 * g1s / 8)
                + 6 * lam**2 - 108 * g3s**2 - 23 * g2s**2 / 4 - 127 * g1s**2 / 600
                + 9 * g3s * g2s + 31 * g3s * g1s / 15 - 27 * g2s * g1s / 20))

    # eq. (103): tau Yukawa coupling (two loops)
    dyl = yls * (
        k1 * (3 * yts + 3 * ybs + 5 * yls / 2 - 9 * g2s / 4 - 9 * g1s / 4)
        + k2 * (6 * lam**2 - 23 * g2s**2 / 4 + 1371 * g1s**2 / 200 + 27 * g2s * g1s / 20
                + yts * (-27 * yts / 4 + 3 * ybs / 2 - 27 * yls / 4 + 20 * g3s + 45 * g2s / 8 + 17 * g1s / 8)
                + ybs * (-27 * ybs / 4 - 27 * yls / 4 + 20 * g3s + 45 * g2s / 8 + 5 * g1s / 8)
                + yls * (-3 * yls - 12 * lam + 165 * g2s / 16 + 537 * g1s / 80)))

    return 2 * np.array([dg1, dg2, dg3, dyt, dyb, dyl, dlam])


def gamma_higgs(x):
    """Higgs field anomalous dimension gamma, eq. (105); Gamma(h) = int_{M_t}^h gamma d ln(mu)."""
    g1s, g2s, g3s, yts, ybs, yls, lam = x
    k1, k2, k3 = K1, K1**2, K1**3
    return (
        k1 * (9 * g2s / 4 + 9 * g1s / 20 - 3 * yts - 3 * ybs - yls)
        + k2 * (yts * (-3 * ybs / 2 - 17 * g1s / 8 - 45 * g2s / 8 - 20 * g3s + 27 * yts / 4)
                - yls * (15 * g1s / 8 + 15 * g2s / 8 - 9 * yls / 4)
                + ybs * (-5 * g1s / 8 - 45 * g2s / 8 - 20 * g3s + 27 * ybs / 4)
                - 1293 * g1s**2 / 800 - 27 * g2s * g1s / 80 + 271 * g2s**2 / 32 - 6 * lam**2)
        + k3 * (-9 * g1s * lam**2 - 45 * g2s * lam**2 + 1.07 * g1s**2 * lam
                + 3.57 * g2s * g1s * lam + 8.92 * g2s**2 * lam
                + 14.99 * g1s**2 * yts + 14.13 * g1s * yts**2 - 13.21 * g2s * g1s * yts
                - 8.73 * g3s * g1s * yts + 40.11 * g2s * yts**2 + 79.05 * g3s * yts**2
                + 23.40 * g2s**2 * yts - 178.48 * g3s**2 * yts - 7.57 * g2s * g3s * yts
                - 5.26 * g1s**3 + 1.93 * g2s * g1s**2 + 4.19 * g3s * g1s**2
                + 1.81 * g2s**2 * g1s - 158.51 * g2s**3 + 28.57 * g2s**2 * g3s
                + 36 * lam**3 + 135 * lam**2 * yts / 2 - 45.00 * lam * yts**2 - 60.13 * yts**3))


# ---------------------------------------------------------------------------------------------------
# Effective potential (appendix C)
# ---------------------------------------------------------------------------------------------------


def li2(z):
    """Dilogarithm Li_2(z) for complex z (scipy's spence(z) is Li_2(1 - z))."""
    return spence(1 - np.asarray(z, dtype=complex))


def xi(z):
    """The function xi(z) of eq. (108).

    For 0 < z < 4 the square root is imaginary and so is the bracket; their product is real.
    """
    z = np.asarray(z, dtype=complex)
    s = np.sqrt(z * z - 4 * z)
    w = (z - s) / (2 * z)
    return (s * (2 * np.log(w) ** 2 - np.log(z) ** 2 - 4 * li2(w) + np.pi**2 / 3)).real


def lambda_eff(x, Gamma):
    """Effective quartic lambda_eff(h) at h = mu, eq. (104), with V_eff = lambda_eff h^4 / 4, eq. (63).

    x holds the couplings at mu = h and Gamma = Gamma(h). One loop: eq. (106); two loops: eq. (107)
    (given in the paper for lambda = 0), with the logs r_W, r_Z, r_t, r_tW of eqs. (109), (110) and
    xi of eq. (108). Logs of negative arguments (lambda < 0 in eq. 106, y_t^2/2 < g2^2/4 in eq. 110) and
    Li_2 above its branch point are replaced by their real parts.
    """
    g1s, g2s, g3s, yts, _, _, lam = x
    gYs = 3 * g1s / 5
    pi2 = np.pi**2

    # eq. (106): one-loop correction
    lam_log = np.where(lam != 0, lam**2 * np.log(np.abs(lam) + 1e-300), 0.0)
    l1 = K1 * (3 * g2s**2 / 8 * (np.log(g2s / 4) - 5 / 6 + 2 * Gamma)
               + 3 / 16 * (g2s + gYs) ** 2 * (np.log((g2s + gYs) / 4) - 5 / 6 + 2 * Gamma)
               - 3 * yts**2 * (np.log(yts / 2) - 3 / 2 + 2 * Gamma)
               + 3 * (4 * lam_log + lam**2 * (-6 + 3 * np.log(3) + 8 * Gamma)))

    # eqs. (109), (110)
    rW = np.log(g2s / 4) + 2 * Gamma
    rZ = np.log((g2s + gYs) / 4) + 2 * Gamma
    rt = np.log(yts / 2) + 2 * Gamma
    rtW = (rt - rW) * (np.log(np.abs(yts / 2 - g2s / 4) + 1e-300) + 2 * Gamma)

    # eq. (107): two-loop correction (lambda = 0)
    l2 = K1**2 * (
        8 * g3s * yts**2 * (3 * rt**2 - 8 * rt + 9)
        + yts**3 / 2 * (-6 * rt * rW - 3 * rt**2 + 48 * rt - 6 * rtW - 69 - pi2)
        + 3 * yts * g2s**2 / 16 * (8 * rW + 4 * rZ - 3 * rt**2 - 6 * rt * rZ - 12 * rt + 12 * rtW
                                   + 15 + 2 * pi2)
        + yts * gYs**2 / 48 * (27 * rt**2 - 54 * rt * rZ - 68 * rt - 28 * rZ + 189)
        + yts * g2s * gYs / 8 * (9 * rt**2 - 18 * rt * rZ + 4 * rt + 44 * rZ - 57)
        + g2s**3 / 192 * (36 * rt * rZ + 54 * rt**2 - 414 * rW * rZ + 69 * rW**2 + 1264 * rW
                          + 156 * rZ**2 + 632 * rZ - 144 * rtW - 2067 + 90 * pi2)
        + g2s**2 * gYs / 192 * (12 * rt * rZ - 6 * rt**2 - 6 * rW * (53 * rZ + 50) + 213 * rW**2
                                + 4 * rZ * (57 * rZ - 91) + 817 + 46 * pi2)
        + g2s * gYs**2 / 576 * (132 * rt * rZ - 66 * rt**2 + 306 * rW * rZ - 153 * rW**2 - 36 * rW
                                + 924 * rZ**2 - 4080 * rZ + 4359 + 218 * pi2)
        + gYs**3 / 576 * (6 * rZ * (34 * rt + 3 * rW - 470) - 102 * rt**2 - 9 * rW**2
                          + 708 * rZ**2 + 2883 + 206 * pi2)
        + yts**2 / 6 * (4 * gYs * (3 * rt**2 - 8 * rt + 9) - 9 * g2s * (rt - rW + 1))
        + 3 / 4 * (g2s**3 - 3 * g2s**2 * yts + 4 * yts**3) * li2(g2s / (2 * yts)).real
        + yts / 48 * xi((g2s + gYs) / (2 * yts))
        * (9 * g2s**2 - 6 * g2s * gYs + 17 * gYs**2 + 2 * yts * (7 * gYs - 73 * g2s + 64 * g2s**2 / (gYs + g2s)))
        + g2s / 64 * xi((g2s + gYs) / g2s)
        * (18 * g2s * gYs + gYs**2 - 51 * g2s**2 - 48 * g2s**3 / (gYs + g2s)))

    return np.exp(4 * Gamma) * (lam + l1 + l2)  # eq. (104)


def log_decay_probability(t, lam):
    """ln of the vacuum-decay probability p0 = 0.15 Lambda_B^4 / H0^4 exp(-S), eqs. (81) and (83).

    Evaluated with Lambda_B = mu = exp(t) and S = 8 pi^2 / (3 |lambda(Lambda_B)|) (eq. 81); maximising
    over t gives the bounce scale (sec. 6.3). Returns -inf where lambda >= 0.
    """
    with np.errstate(divide="ignore"):
        S = np.where(lam < 0, 8 * np.pi**2 / (3 * np.abs(lam)), np.inf)
    return np.log(0.15) + 4 * (t - np.log(H0)) - S


# ---------------------------------------------------------------------------------------------------
# Running and classification
# ---------------------------------------------------------------------------------------------------


def run_rge(x, t_start, t_end, n_steps=N_STEPS, Gamma_start=None):
    """Run the couplings x (shape (7,) or (7, N)) from ln(mu) = t_start to t_end with RK4.

    Points whose couplings blow up (|lambda| > 20 or y_t^2 > 50) are frozen there. Returns x at t_end,
    A = int_{t_start}^{t_end} gamma d ln(mu) and the mask of frozen points. If Gamma_start = Gamma(t_start) is given, also returns a
    dict with the scales `t`, and lambda and lambda_eff at each of them (shape (n_steps + 1, N)).
    """
    x = np.array(x, dtype=float)
    dt = (t_end - t_start) / n_steps
    frozen = np.zeros(x.shape[1:], dtype=bool)
    A = np.zeros(x.shape[1:])
    analyse = Gamma_start is not None
    if analyse:
        ts = t_start + dt * np.arange(n_steps + 1)
        lams = np.empty((n_steps + 1,) + x.shape[1:], dtype=np.float32)
        lam_effs = np.empty_like(lams)

    gam = gamma_higgs(x)
    for i in range(n_steps + 1):
        if analyse:
            lams[i] = x[LAM]
            lam_effs[i] = lambda_eff(x, Gamma_start + A)
        if i == n_steps:
            break
        k1 = beta(x)
        k2 = beta(x + dt / 2 * k1)
        k3 = beta(x + dt / 2 * k2)
        k4 = beta(x + dt * k3)
        x_new = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        frozen |= ~np.isfinite(x_new).all(axis=0) | (np.abs(x_new[LAM]) > 20) | (x_new[YT] > 50)
        x = np.where(frozen, x, x_new)
        gam_new = gamma_higgs(x)
        A = A + np.where(frozen, 0.0, dt / 2 * (gam + gam_new))  # trapezoid rule for Gamma
        gam = gam_new

    if analyse:
        return x, A, frozen, {"t": ts, "lam": lams, "lam_eff": lam_effs}
    return x, A, frozen


def lowest_zero(t, f):
    """Lowest t at which f(t) turns negative, interpolated linearly; NaN if f >= 0 everywhere.

    Used for the instability scale Lambda_I (lambda_eff = 0, sec. 4.2) and Lambda_lambda (lambda = 0).
    """
    order = np.argsort(t)
    t, f = t[order], f[order].astype(float)
    neg = f < 0
    i = np.argmax(neg, axis=0)
    i_prev = np.maximum(i - 1, 0)
    f0 = np.take_along_axis(f, i_prev[None], 0)[0]
    f1 = np.take_along_axis(f, i[None], 0)[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        tz = np.where(i > 0, t[i_prev] + (t[i] - t[i_prev]) * f0 / (f0 - f1), t[0])
    return np.where(neg.any(axis=0), tz, np.nan)


def classify(run):
    """Phase-diagram fields from the output of `run_rge(..., Gamma_start=...)`.

    min_lam_eff: smallest lambda_eff between M_t and M_Pl (stability if >= 0, sec. 4.2)
    min_lam:     smallest running MS-bar lambda between M_t and M_Pl
    log10_LI:    log10(Lambda_I / GeV), where lambda_eff first turns negative (sec. 4.2)
    log10_p0:    log10 of the maximal decay probability, eq. (83) (instability if > 0)
    """
    t = run["t"]
    log_p = log_decay_probability(t.reshape((-1,) + (1,) * (run["lam"].ndim - 1)), run["lam"])
    return {
        "min_lam_eff": run["lam_eff"].min(axis=0).astype(float),
        "min_lam": run["lam"].min(axis=0).astype(float),
        "log10_LI": lowest_zero(t, run["lam_eff"]) / np.log(10),
        "log10_p0": np.maximum(log_p.max(axis=0), -1e4) / np.log(10),
    }



# ---------------------------------------------------------------------------------------------------
# The stability boundary near the SM, in the weak-scale couplings (eq. (2.42) of the paper)
# ---------------------------------------------------------------------------------------------------


def lambda_c(y0):
    """The linearised stability bound lambda_c(y_0), eq. (2.42) of the paper."""
    return 0.13521 + 0.734 * (y0 - 0.93690)


def u(lambda0, y0):
    """The distance u = lambda_0 - lambda_c(y_0) from the stability bound."""
    return lambda0 - lambda_c(y0)



# Best fits and errors of the weak-scale couplings, from the linear fits and table 2.
LAM0_BEST = 0.12604                                    # eq. (55)
DLAM0 = np.hypot(0.00206 * DMH, 0.00004 * DMT)         # eq. (55)
Y0_BEST = 0.93690                                      # eq. (57)
DY0 = np.hypot(0.00556 * DMT, 0.00042)                 # eq. (57); the alpha_3 term is per 1 sigma
LAMC_BEST = lambda_c(Y0_BEST)                          # x = 0
DLAMC = 0.734 * DY0                                    # error of lambda_c(y0_best) from that of y_0
U_BEST = u(LAM0_BEST, Y0_BEST)                         # best-fit lambda_0 on the x axis
