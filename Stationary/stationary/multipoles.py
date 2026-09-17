"""Spherical-harmonic decomposition of a scalar field on a sphere, and figures.

Provides the multipole content of lambda on any coordinate sphere, plus the two
figures requested for the new problem: lambda on the inner sphere (the imposed
boundary data against what the network produces) and the angular dependence of the
first three multipoles of lambda on the outer sphere.
"""
from __future__ import annotations

import os

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

# --------------------------------------------------------------- real harmonics
def real_sph_harm(l: int, m: int, mu, phi):
    """Orthonormal real spherical harmonics, |m| <= l <= 3."""
    pi = jnp.pi
    s = jnp.sqrt(1.0 - mu**2)
    if l == 0:
        return 1.0 / jnp.sqrt(4 * pi) * jnp.ones_like(mu)
    if l == 1:
        if m == 0:
            return jnp.sqrt(3 / (4 * pi)) * mu
        if m == 1:
            return jnp.sqrt(3 / (4 * pi)) * s * jnp.cos(phi)
        return jnp.sqrt(3 / (4 * pi)) * s * jnp.sin(phi)
    if l == 2:
        if m == 0:
            return jnp.sqrt(5 / (16 * pi)) * (3 * mu**2 - 1)
        if m == 1:
            return jnp.sqrt(15 / (4 * pi)) * mu * s * jnp.cos(phi)
        if m == -1:
            return jnp.sqrt(15 / (4 * pi)) * mu * s * jnp.sin(phi)
        if m == 2:
            return jnp.sqrt(15 / (16 * pi)) * (1 - mu**2) * jnp.cos(2 * phi)
        return jnp.sqrt(15 / (16 * pi)) * (1 - mu**2) * jnp.sin(2 * phi)
    if l == 3:
        if m == 0:
            return jnp.sqrt(7 / (16 * pi)) * (5 * mu**3 - 3 * mu)
        if m == 1:
            return jnp.sqrt(21 / (32 * pi)) * (5 * mu**2 - 1) * s * jnp.cos(phi)
        if m == -1:
            return jnp.sqrt(21 / (32 * pi)) * (5 * mu**2 - 1) * s * jnp.sin(phi)
        if m == 2:
            return jnp.sqrt(105 / (16 * pi)) * mu * (1 - mu**2) * jnp.cos(2 * phi)
        if m == -2:
            return jnp.sqrt(105 / (16 * pi)) * mu * (1 - mu**2) * jnp.sin(2 * phi)
        if m == 3:
            return jnp.sqrt(35 / (32 * pi)) * s**3 * jnp.cos(3 * phi)
        return jnp.sqrt(35 / (32 * pi)) * s**3 * jnp.sin(3 * phi)
    raise ValueError("l <= 3 supported")


def sphere_grid(n_mu: int = 48, n_phi: int = 32):
    """Gauss-Legendre (mu) x uniform (phi) grid with quadrature weights summing to 4 pi."""
    mu, w = np.polynomial.legendre.leggauss(n_mu)
    phi = 2 * np.pi * np.arange(n_phi) / n_phi
    MU, PHI = np.meshgrid(mu, phi, indexing="ij")
    W = np.outer(w, np.full(n_phi, 2 * np.pi / n_phi))
    return jnp.asarray(MU), jnp.asarray(PHI), jnp.asarray(W)


def directions(mu, phi):
    s = jnp.sqrt(1.0 - mu**2)
    return jnp.stack([s * jnp.cos(phi), s * jnp.sin(phi), mu], axis=-1)


def decompose(field_on_sphere, mu, phi, weights, lmax: int = 3):
    """Coefficients a_lm of the expansion field = sum a_lm Y_lm."""
    out = {}
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            Y = real_sph_harm(l, m, mu, phi)
            out[(l, m)] = float(jnp.sum(weights * field_on_sphere * Y))
    return out


def lambda_multipoles(point_fields, rho: float, lmax: int = 3, n_mu: int = 48,
                      n_phi: int = 32):
    """Multipole content of lambda on the coordinate sphere |x| = rho."""
    mu, phi, w = sphere_grid(n_mu, n_phi)
    xs = rho * directions(mu, phi)
    vals = jax.vmap(lambda y: point_fields(y).lam)(xs.reshape(-1, 3)).reshape(mu.shape)
    coef = decompose(vals, mu, phi, w, lmax)
    power = {l: float(sum(coef[(l, m)] ** 2 for m in range(-l, l + 1))) for l in range(lmax + 1)}
    return coef, power, (mu, phi, vals)


def angular_profiles(coef: dict, lmax: int = 3, n_theta: int = 181):
    """lambda_l(theta) = sum_m a_lm Y_lm(theta, phi=0) for each l."""
    th = jnp.linspace(0.0, jnp.pi, n_theta)
    mu = jnp.cos(th)
    phi0 = jnp.zeros_like(th)
    prof = {}
    for l in range(lmax + 1):
        prof[l] = sum(coef[(l, m)] * real_sph_harm(l, m, mu, phi0)
                      for m in range(-l, l + 1))
    return jnp.asarray(th), prof


# ------------------------------------------------------------------- figures
def make_figures(point_fields, cfg, outdir: str, lmax: int = 3):
    """Write the two requested figures; returns the numbers behind them."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    report = {}

    # ---------------- lambda on the inner sphere: boundary data vs network
    n_mu, n_phi = 97, 64
    mu, phi, w = sphere_grid(n_mu, n_phi)
    xs_in = cfg.rho_in * directions(mu, phi)
    lam_net = jax.vmap(lambda y: point_fields(y).lam)(xs_in.reshape(-1, 3)).reshape(mu.shape)

    from .problem import lam_inner_bc
    lam_bc = jax.vmap(lambda y: lam_inner_bc(y, cfg))(xs_in.reshape(-1, 3)).reshape(mu.shape)
    report["inner_bc_max_err"] = float(jnp.max(jnp.abs(lam_net - lam_bc)))
    report["inner_bc_rms_err"] = float(jnp.sqrt(jnp.mean((lam_net - lam_bc) ** 2)))

    th = np.arccos(np.asarray(mu))[:, 0]
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    ax[0, 0].plot(th, np.asarray(lam_bc)[:, 0], "k--", label="imposed $\\lambda$ at $\\rho_{in}$")
    ax[0, 0].plot(th, np.asarray(lam_net)[:, 0], "r-", label="network")
    ax[0, 0].set_title(f"$\\lambda$ on the inner sphere ($\\rho={cfg.rho_in:g}$), $\\varphi=0$")
    ax[0, 0].set_xlabel("$\\theta$"); ax[0, 0].set_ylabel("$\\lambda$"); ax[0, 0].legend()
    ax[0, 1].semilogy(th, np.abs(np.asarray(lam_net - lam_bc))[:, 0] + 1e-18)
    ax[0, 1].set_title("$|\\lambda_{net}-\\lambda_{BC}|$ on the inner sphere")
    ax[0, 1].set_xlabel("$\\theta$")
    im = ax[1, 0].pcolormesh(np.asarray(phi)[0], th, np.asarray(lam_bc), shading="auto")
    ax[1, 0].set_title("imposed $\\lambda_{BC}(\\theta,\\varphi)$"); ax[1, 0].set_xlabel("$\\varphi$"); ax[1, 0].set_ylabel("$\\theta$")
    plt.colorbar(im, ax=ax[1, 0])
    d = np.asarray(lam_net - lam_bc)
    im = ax[1, 1].pcolormesh(np.asarray(phi)[0], th, d, shading="auto")
    ax[1, 1].set_title("$\\lambda_{net}-\\lambda_{BC}$"); ax[1, 1].set_xlabel("$\\varphi$"); ax[1, 1].set_ylabel("$\\theta$")
    plt.colorbar(im, ax=ax[1, 1])
    for a in ax.ravel():
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "lambda_inner.png"), dpi=120)
    plt.close(fig)

    # ---------------- multipoles of lambda on the outer sphere
    coef, power, (mu_o, phi_o, vals_o) = lambda_multipoles(point_fields, cfg.rho_out, lmax)
    th_o, prof = angular_profiles(coef, lmax)
    report["outer_coef"] = {f"l{l}_m{m}": v for (l, m), v in coef.items()}
    report["outer_power"] = power
    report["lambda_outer_mean"] = float(jnp.mean(vals_o))

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    for l in range(min(3, lmax + 1)):
        ax[0].plot(np.asarray(th_o), np.asarray(prof[l]), label=f"$l={l}$ (amp {np.sqrt(power[l]):.3e})")
    ax[0].set_title(f"$\\lambda$ multipole angular dependence at $\\rho={cfg.rho_out:g}$")
    ax[0].set_xlabel("$\\theta$"); ax[0].set_ylabel("$\\lambda_l(\\theta)$"); ax[0].legend()
    ls = list(range(min(3, lmax + 1)))
    ax[1].bar([str(l) for l in ls], [np.sqrt(power[l]) for l in ls])
    ax[1].set_title("multipole amplitudes $\\sqrt{\\sum_m a_{lm}^2}$"); ax[1].set_xlabel("$l$")
    th_mu = np.arccos(np.asarray(mu_o)[:, 0])
    ax[2].plot(th_mu, np.asarray(vals_o[:, 0]), label="$\\lambda(\\theta,\\varphi=0)$")
    ax[2].plot(np.asarray(th_o), np.asarray(sum(prof[l] for l in ls)), "--", label="$l\\leq2$ sum")
    ax[2].set_title("reconstructed vs full $\\lambda$ at the outer sphere")
    ax[2].set_xlabel("$\\theta$"); ax[2].legend()
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "lambda_multipoles_outer.png"), dpi=120)
    plt.close(fig)

    # ---------------- how each multipole decays with rho
    rhos = np.geomspace(max(2.0, cfg.rho_in * 1.5), cfg.rho_out, 14)
    prof_r = multipole_radial_profile(point_fields, rhos, lmax, lam_inf=getattr(cfg, "lam_inf", None) or 1.0)
    report["multipole_decay"] = {l: {"fitted_power": prof_r[l]["fitted_power"],
                                     "expected_power": prof_r[l]["expected_power"],
                                     "amplitudes": prof_r[l]["amplitudes"]}
                                 for l in prof_r}
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for l in range(min(3, lmax + 1)):
        a = np.asarray(prof_r[l]["amplitudes"])
        m = a > 1e-15
        ax[0].loglog(np.asarray(rhos)[m], a[m], "o-",
                     label=f"$l={l}$: power {prof_r[l]['fitted_power']:.2f} (expect {prof_r[l]['expected_power']})")
    ax[0].set_title("multipole amplitudes of $\\lambda$ vs $\\rho$")
    ax[0].set_xlabel("$\\rho$"); ax[0].set_ylabel("$\\sqrt{\\sum_m a_{lm}^2}$"); ax[0].legend()
    for l in range(min(3, lmax + 1)):
        a = np.asarray(prof_r[l]["amplitudes"])
        m = a > 1e-15
        ax[1].plot(np.asarray(rhos)[m], a[m] * np.asarray(rhos)[m] ** (l + 1), "o-", label=f"$l={l}$")
    ax[1].set_title("$a_l(\\rho)\\,\\rho^{l+1}$ (flat = correct decay)")
    ax[1].set_xlabel("$\\rho$"); ax[1].legend()
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "lambda_multipole_decay.png"), dpi=120)
    plt.close(fig)

    return report


def multipole_radial_profile(point_fields, rhos, lmax: int = 3, n_mu: int = 40,
                             n_phi: int = 24, lam_inf: float = 0.0):
    """Amplitude sqrt(sum_m a_lm^2) of each multipole of lambda as a function of rho.

    For a decaying harmonic field the l-th multipole of lambda should fall off as
    rho^-(l+1); this is the direct check that the higher-order Robin condition lets the
    higher multipoles through with the correct power instead of forcing every field
    onto its leading decay law.
    """
    amps = {l: [] for l in range(lmax + 1)}
    mu, phi, w = sphere_grid(n_mu, n_phi)
    dirs = directions(mu, phi)
    for rho in rhos:
        xs = float(rho) * dirs
        vals = jax.vmap(lambda y: point_fields(y).lam - lam_inf)(xs.reshape(-1, 3))
        vals = vals.reshape(mu.shape)
        coef = decompose(vals, mu, phi, w, lmax)
        for l in range(lmax + 1):
            amps[l].append(float(np.sqrt(sum(coef[(l, m)] ** 2 for m in range(-l, l + 1)))))
    out = {}
    for l in range(lmax + 1):
        a = np.asarray(amps[l])
        m = a > 1e-14
        if m.sum() >= 3:
            rr = np.asarray(rhos, dtype=float)[m]
            slope = float(np.polyfit(np.log(rr), np.log(a[m]), 1)[0])
        else:
            slope = float("nan")
        out[l] = {"amplitudes": [float(v) for v in a], "fitted_power": slope,
                  "expected_power": -(l + 1)}
    return out
