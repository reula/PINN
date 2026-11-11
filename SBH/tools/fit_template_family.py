"""Can a rescaled vacuum puncture represent the hole of the scalar reference near r = 0? (2026-11-11)

    JAX_PLATFORMS=cpu python tools/fit_template_family.py

Least-squares fit, at each time and in r = 0.05-0.225 (the exact core) and 0.05-0.45 (to the horizon), of chi, Khat
and alpha of reference_1d/ref_scalar_m0.5_A0.03_dr0.005_t8.npz by the vacuum table
reference_1d/ref_single_m0.5_dr0.005_t8_snap0025.npz rescaled with (mu, s, k, tau):

    chi = chi_T(tau, s r / lam) / s^2,   Khat = Khat_T(tau, s r / lam) / lam,   alpha = k alpha_T(tau, s r / lam),

lam = mu / 0.5. s = (1 + c)^2 is the constant conformal factor (a dilation of the coordinates), k the lapse factor;
s = k = 1 is the --tpl_mu family. Three fits: nothing free (the fixed template), (mu, tau), (mu, s, k, tau).
Prints the rms misfit of the three fields and the fitted parameters. tau is limited to the table (0-8).
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.optimize import least_squares

V = np.load("reference_1d/ref_single_m0.5_dr0.005_t8_snap0025.npz")
Z = np.load("reference_1d/ref_scalar_m0.5_A0.03_dr0.005_t8.npz")
rv = V["r"]
m = rv < 3
I = {i: RGI((V["t"], rv[m]), V["U"][:, i, m], bounds_error=False, fill_value=None) for i in (4, 5, 8)}


def model(p, r):
    mu, s, k, tau = p
    lam = mu / 0.5
    x = np.clip(s * r / lam, rv[0], 2.9)
    q = np.stack([np.full_like(x, np.clip(tau, 0, 8)), x], -1)
    return I[4](q) / s ** 2, I[5](q) / lam, k * I[8](q)


def fit(T, lo, hi, free):
    j = int(np.argmin(abs(Z["t"] - T)))
    sel = (Z["r"] >= lo) & (Z["r"] <= hi)
    r, u = Z["r"][sel], Z["U"][j][:, sel]
    p0 = np.array([0.5, 1.0, 1.0, T])

    def res(q):
        p = p0.copy()
        p[free] = q
        a = model(p, r)
        return np.concatenate([a[0] - u[4], a[1] - u[5], a[2] - u[8]])

    p = p0.copy()
    if free:
        p[free] = least_squares(res, p0[free], x_scale=0.01).x
    a = model(p, r)
    return p, [np.sqrt(((a[i] - u[f]) ** 2).mean()) for i, f in enumerate((4, 5, 8))]


for lo, hi in ((0.05, 0.225), (0.05, 0.45)):
    print(f"\nfit region r = {lo}-{hi}: rms misfit of chi, Khat, alpha")
    for T in (0.25, 1, 2, 3, 4, 5, 6, 8):
        out = []
        for nm, free in (("fixed", []), ("mu,tau", [0, 3]), ("mu,s,k,tau", [0, 1, 2, 3])):
            p, e = fit(T, lo, hi, free)
            out.append(f"{nm}: " + " ".join(f"{x:.0e}" for x in e)
                       + (f" [mu {p[0]:.4f} s {p[1]:.4f} k {p[2]:.4f} tau {p[3]:.2f}]" if free else ""))
        print(f" t={T}: " + " | ".join(out))
