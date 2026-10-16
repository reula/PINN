"""Spherical-harmonic decomposition of a scalar field on a sphere, and figures.

Provides the multipole content of a scalar on any coordinate sphere, plus the figures:
lambda on the inner sphere (the imposed boundary data against what the network produces),
and the multipoles of the Geroch-Hansen potential P = (1 - lambda^2)/(4 lambda) on the outer
sphere and against rho.  The multipole figures use P only, never lambda: P is the potential
the moments are defined from (MULTIPOLES_GH_THORNE.md section 2), the one the outer scalar
condition acts on (losses.gh_potential), and lambda's own coefficients differ from it from
l = 2 on (lambda - 1 = 2U + 2U^2 + ..., P = -U - (2/3)U^3 - ...), so mixing the two compares
different numbers.  The coefficients of P on coordinate spheres are still not the GH moments
themselves beyond linear order (section 6.1 there); they are the tail coefficients of P.
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


class _Scalar:
    """Stands in for point_fields(y) where only `.lam` is read, so the estimators below run on P."""
    __slots__ = ("lam",)

    def __init__(self, lam):
        self.lam = lam


def gh_fields(point_fields):
    """point_fields with `.lam` replaced by the Geroch-Hansen potential P = (1 - lam^2)/(4 lam)."""
    from .losses import gh_potential
    return lambda y: _Scalar(gh_potential(point_fields(y).lam))


def gh_inner_constants(cfg, lmax: int = 3, n_mu: int = 48, n_phi: int = 32, factor=None,
                       lam_inf: float = 1.0):
    """S_lm of P read off the IMPOSED inner data, in the units of `multipole_constants`.

    P of the data lambda = lam_inner_bc, expanded about P_inf = P(lam_inf) and multiplied by
    rho_in_phys^(l+1): the dashed lines the outer P coefficients are compared with.
    """
    from .losses import gh_potential
    from .problem import lam_inner_bc, physical_factor
    factor = physical_factor(cfg) if factor is None else float(factor)
    c = multipole_constants(lambda q: _Scalar(gh_potential(lam_inner_bc(q, cfg))),
                            [float(cfg.rho_in)], lmax, n_mu, n_phi,
                            lam_inf=float(gh_potential(lam_inf)), factor=factor)
    return {k: float(v[0]) for k, v in c["S"].items()}


def multipole_constants(point_fields, rhos, lmax: int = 3, n_mu: int = 48, n_phi: int = 32,
                        lam_inf: float = 0.0, factor: float = 1.0):
    """S_lm(rho) = a_lm(rho) * rho_phys^(l+1): the constant in front of Y_lm/rho^(l+1).

    A tail is  lambda - lambda_inf = sum_lm S_lm Y_lm / rho^(l+1)  with S_lm CONSTANT, so this
    is the number to report and to compare with the inner data.  The amplitude
    sqrt(sum_m a_lm^2) is not: it carries the rho^-(l+1) of whatever radius it was measured at,
    and at the inner sphere the data are imposed with the OTHER power (1/rho^l), so the two
    numbers were never comparable.

    `rhos` are CHART radii; the coefficient comes back in the physical chart, i.e. in front of
    Y_lm / rho_phys^(l+1) with rho_phys = factor * rho_chart.
    """
    mu, phi, w = sphere_grid(n_mu, n_phi)
    dirs = directions(mu, phi)
    keys = [(l, m) for l in range(lmax + 1) for m in range(-l, l + 1)]
    S = {k: [] for k in keys}
    for rho in rhos:
        y = float(rho) * dirs
        vals = jax.vmap(lambda q: point_fields(q).lam - lam_inf)(y.reshape(-1, 3))
        coef = decompose(vals.reshape(mu.shape), mu, phi, w, lmax)
        r_phys = float(rho) * float(factor)
        for (l, m) in keys:
            S[(l, m)].append(float(coef[(l, m)]) * r_phys ** (l + 1))
    return {"r_phys": [float(r) * float(factor) for r in rhos], "S": S}


def inner_constants(cfg, lmax: int = 3, n_mu: int = 48, n_phi: int = 32, factor=None):
    """The same S_lm read off the IMPOSED inner data, in the same units.

    The data are  lambda = lam0 + S1 z/rho_in + S2 (z^2-(x^2+y^2)/2)/rho_in^2, i.e. powers
    1/rho^l rather than 1/rho^(l+1).  Reading them as the tail means multiplying the angular
    coefficient by rho_in_phys^(l+1) -- no factor at all when the run is at the physical inner
    radius, which is the convention (rho_in_phys = 1).

    The expansion is taken from lambda_INF, not from lam0, so the MONOPOLE is included: its
    constant is sqrt(4 pi)(lam0 - lam_inf) rho_in_phys, which is the number the outer monopole
    has to be compared with.  Subtracting lam0 instead left l=0 at machine zero and made the
    ratio column meaningless for it (pq_c200_vac's S_00 is -2.363 at rho_in = 1, and the exact
    reference gives the same).

    Equivalently: the numbers reproduce the data pointwise as sum_lm S_lm Y_lm/rho_in_phys^(l+1)
    (see `reconstruct` and tests/test_multipole_constants.py).
    """
    from .problem import lam_inner_bc, physical_factor
    factor = physical_factor(cfg) if factor is None else float(factor)
    mu, phi, w = sphere_grid(n_mu, n_phi)
    dirs = directions(mu, phi)
    x = float(cfg.rho_in) * dirs
    lam_inf = getattr(cfg, "lam_inf", None)
    if lam_inf is None:
        lam_inf = getattr(cfg, "lam_inf_init", 1.0)
    vals = jax.vmap(lambda q: lam_inner_bc(q, cfg) - lam_inf)(x.reshape(-1, 3))
    coef = decompose(vals.reshape(mu.shape), mu, phi, w, lmax)
    r_phys = float(cfg.rho_in) * factor
    return {(l, m): float(coef[(l, m)]) * r_phys ** (l + 1)
            for l in range(lmax + 1) for m in range(-l, l + 1)}


def multipole_table(const, inner, exact=None, lmax: int = 3):
    """The S_lm table as text lines: one row per (l, m), the inner value and ratio beside it.

    Kept out of report.py so that it can be tested: the number that made the old table
    unreadable was a ratio against a machine-zero inner value (4e15), which says nothing except
    that both numbers are round-off.  A ratio is printed only when the inner value is within
    1e-6 of the largest inner value present -- i.e. when the inner data actually impose that
    multipole -- and `--` otherwise.
    """
    keys = sorted(const["S"])
    biggest = max((max(abs(float(x)) for x in const["S"][k]) for k in keys), default=0.0)
    inner_big = max((abs(float(inner[k])) for k in keys), default=0.0)
    lines = []
    header = "      l  m      S_lm at those rho"
    header += " " * max(1, 44 - len(header)) + "drift"
    if exact is not None:
        header += "     exact S_lm"
    header += "    inner data    ratio"
    lines.append(header)
    for (l, m) in keys:
        v = [float(x) for x in const["S"][(l, m)]]
        iv = float(inner[(l, m)])
        if max(max(abs(x) for x in v), abs(iv)) < 1e-6 * max(biggest, inner_big):
            continue                          # named in the note below, not given a row
        cells = " ".join(f"{x: .3e}" for x in v)
        drift = (max(v) - min(v)) / max(max(abs(x) for x in v), 1e-300)
        row = f"      {l}  {m:+d}  {cells}   {drift:6.1%}"
        if exact is not None:
            ev = [float(x) for x in exact["S"][(l, m)]]
            row += f"   {ev[-1]: .3e}"
        row += f"   {iv: .3e}"
        if inner_big > 0 and abs(iv) > 1e-6 * inner_big:
            row += f"   {max(abs(x) for x in v) / abs(iv):7.2f}"
        else:
            row += "        --"
        lines.append(row)
    if biggest > 0:
        floor = 1e-6 * max(biggest, inner_big)
        tiny = [f"l={l},m={m:+d}" for (l, m) in keys
                if max(max(abs(float(x)) for x in const["S"][(l, m)]),
                       abs(float(inner[(l, m)]))) < floor]
        if tiny:
            lines.append("      (not listed above: " + ", ".join(tiny) + " -- below 1e-6 of the"
                         " largest S_lm, i.e. the network's noise floor)")
    lines.append("      drift = (max-min)/max over these radii: 0 is a correct tail."
                 "  ratio = max|S_lm| / the inner data's value (only when the data"
                 " impose it).")
    return lines


def reconstruct(constants, r_phys, mu, phi, lmax: int = 3):
    """sum_lm S_lm Y_lm / r_phys^(l+1): the field that `constants` stands for at that radius."""
    out = 0.0
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            out = out + constants[(l, m)] * real_sph_harm(l, m, mu, phi) / float(r_phys) ** (l + 1)
    return out


# ------------------------------------------------------------------- figures
def make_figures(point_fields, cfg, outdir: str, lmax: int = 3, exact_fields=None):
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

    if cfg.inner_bc == "reference" and exact_fields is not None:
        # In reference mode the imposed inner data is the reference's lambda, not the
        # round-sphere polynomial: against the latter the figure would show the gap between
        # two different solutions (0.21 for the Weyl run) and call it the network's error.
        lam_bc = jax.vmap(lambda y: exact_fields(y).lam)(
            xs_in.reshape(-1, 3)).reshape(mu.shape)
        bc_label = "reference $\\lambda$ at $\\rho_{in}$"
    else:
        from .problem import lam_inner_bc
        lam_bc = jax.vmap(lambda y: lam_inner_bc(y, cfg))(
            xs_in.reshape(-1, 3)).reshape(mu.shape)
        bc_label = "imposed $\\lambda$ at $\\rho_{in}$"
    report["inner_bc_max_err"] = float(jnp.max(jnp.abs(lam_net - lam_bc)))
    report["inner_bc_rms_err"] = float(jnp.sqrt(jnp.mean((lam_net - lam_bc) ** 2)))

    th = np.arccos(np.asarray(mu))[:, 0]
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    ax[0, 0].plot(th, np.asarray(lam_bc)[:, 0], "k--", label=bc_label)
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
    ax[0, 0].set_ylabel(r"$\lambda$")
    ax[0, 1].set_ylabel(r"$|\lambda_{net}-\lambda_{BC}|$")
    ax[1, 0].set_ylabel(r"$\theta$"); ax[1, 0].set_xlabel(r"$\varphi$")
    ax[1, 1].set_ylabel(r"$\theta$"); ax[1, 1].set_xlabel(r"$\varphi$")
    for a in ax.ravel():
        a.grid(alpha=0.3)
    fig.suptitle(f"$\\lambda$ on the inner sphere $\\rho={cfg.rho_in:g}$  =  "
                 f"$\\lambda_0 + S_1 z/\\rho_{{in}} + S_2 (z^2-(x^2+y^2)/2)/\\rho_{{in}}^2$"
                 f"   with  $\\lambda_0={cfg.lam0:g}$,  "
                 f"$\\bf S_1={cfg.lam_bc_S1:g}$,  S_2={cfg.lam_bc_S2:g}$"
                 f"\narch={cfg.arch}   |   max |network - imposed| = {report['inner_bc_max_err']:.2e}",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(outdir, "lambda_inner.png"), dpi=120)
    plt.close(fig)

    # ---------------- multipoles of the GH potential P on the outer sphere (P only, not lambda)
    from .losses import gh_potential
    pf_P = gh_fields(point_fields)
    lam_inf = getattr(cfg, "lam_inf", None) or 1.0
    P_inf = float(gh_potential(lam_inf))
    coef, power, (mu_o, phi_o, vals_o) = lambda_multipoles(pf_P, cfg.rho_out, lmax)
    th_o, prof = angular_profiles(coef, lmax)
    report["outer_coef_P"] = {f"l{l}_m{m}": v for (l, m), v in coef.items()}
    report["outer_power_P"] = power
    report["P_outer_mean"] = float(jnp.mean(vals_o))

    outer_figure(pf_P, cfg, outdir, exact_fields, report)

    # ---------------- how each multipole decays with rho
    from .problem import physical_factor
    factor = physical_factor(cfg)
    # SAMPLE THE WHOLE SHELL, from the inner sphere out.  It used to be
    # `max(2.0, rho_in*1.5)`, which for a run whose chart is the short scale -- pq_c200_vac is
    # [0.005, 1] -- starts ABOVE rho_out and fits the decay where the network only extrapolates.
    # Starting at rho_in also makes the drift printed here the same number as the table in
    # report.txt, which is where the inner-data comparison lives.
    rhos = np.geomspace(float(cfg.rho_in), float(cfg.rho_out), 14)
    prof_r = multipole_radial_profile(pf_P, rhos, lmax, lam_inf=P_inf)
    const = multipole_constants(pf_P, rhos, lmax, lam_inf=P_inf, factor=factor)
    const["inner"] = gh_inner_constants(cfg, lmax=lmax, factor=factor, lam_inf=lam_inf)
    report["multipole_decay_P"] = {l: {"fitted_power": prof_r[l]["fitted_power"],
                                     "expected_power": prof_r[l]["expected_power"],
                                     "amplitudes": prof_r[l]["amplitudes"]}
                                 for l in prof_r}
    report["multipole_constants_P"] = {
        "physical_factor": factor,
        "r_phys": const["r_phys"],
        "S_lm": {f"{l},{m}": v for (l, m), v in const["S"].items()},
        "inner_S_lm": {f"{l},{m}": v for (l, m), v in const["inner"].items()},
    }
    r_phys = np.asarray(const["r_phys"])
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for l in range(min(3, lmax + 1)):
        a = np.asarray(prof_r[l]["amplitudes"])
        m = a > 1e-15
        ax[0].loglog(np.asarray(rhos)[m] * factor, a[m], "o-",
                     label=f"$l={l}$: power {prof_r[l]['fitted_power']:.2f} (expect {prof_r[l]['expected_power']})")
    ax[0].set_title("multipole amplitudes of $P$ vs $\\rho$ (slope = decay power)")
    ax[0].set_xlabel(r"$\rho$ (physical)"); ax[0].set_ylabel(r"amplitude"); ax[0].legend()
    # THE CONSTANT in front of Y_lm/rho^(l+1), per (l, m).  Flat = the tail has the expected
    # power; the dashed lines are the values the inner data impose, which they should equal.
    keys = sorted(const["S"])
    biggest = max((max(abs(float(x)) for x in const["S"][k]) for k in keys), default=0.0)
    n_drawn = 0
    for (l, m) in keys:
        v = np.asarray(const["S"][(l, m)])
        iv = float(const["inner"][(l, m)])
        # RELATIVE threshold: the off-axisymmetric components sit at the network's noise level
        # (1e-16..1e-13) and plotting them buries the three that carry the physics.
        if max(float(np.max(np.abs(v))), abs(iv)) < 1e-6 * biggest:
            continue
        colour = f"C{n_drawn % 10}"   # counted over the DRAWN curves: by key index, l=1 and
        n_drawn += 1                  # l=3 (m=0) both landed on C2
        drift = float(np.ptp(v)) / max(float(np.max(np.abs(v))), 1e-300)
        ax[1].plot(r_phys, v, "o-", ms=3, color=colour,
                   label=rf"$S_{{{l},{m}}}$" + (f"  drift {drift:.1%}" if len(v) > 1 else ""))
        ax[1].axhline(iv, ls="--", lw=1, color=colour)
    ax[1].set_xscale("log")
    # symlog: a coefficient whose tail has the wrong power grows like rho^k and on a linear axis
    # flattens every correct (flat) curve onto zero
    inner_big = max((abs(float(x)) for x in const["inner"].values()), default=0.0)
    ax[1].set_yscale("symlog", linthresh=max(1e-3 * inner_big, 1e-12))
    ax[1].set_title(r"$S_{lm}$ in $P-P_\infty=\sum S_{lm}Y_{lm}/\rho^{l+1}$"
                    "\n(flat $=$ correct power; dashed $=$ $P$ of the inner data)")
    ax[1].set_xlabel(r"$\rho$ (physical)"); ax[1].set_ylabel(r"$S_{lm}$")
    ax[1].legend(fontsize=7)
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle(f"decay of each multipole of $P=(1-\\lambda^2)/(4\\lambda)$   |   "
                 f"inner $\\lambda$ data: $S_1={cfg.lam_bc_S1:g}$, "
                 f"$S_2={cfg.lam_bc_S2:g}$, $\\lambda_0={cfg.lam0:g}$   |   arch={cfg.arch}, "
                 f"$\\rho\\in[{cfg.rho_in * factor:g},{cfg.rho_out * factor:g}]$ physical "
                 f"(chart x {factor:g})", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(outdir, "P_multipole_decay.png"), dpi=120)
    plt.close(fig)

    return report


def legendre_spectrum(point_fields, rho, lmax: int = 8, n_mu: int = 64, n_phi: int = 32):
    """Axisymmetric Legendre coefficients c_l of `.lam` on the sphere |x| = rho, l = 0..lmax.

    field(theta, phi) = sum_l c_l P_l(cos theta) + (non-axisymmetric part).  The phi-average is
    expanded; the non-axisymmetric part is returned as one number, its rms over the sphere.  Also
    returns the phi-averaged profile on the Gauss nodes, for plotting against theta.
    """
    mu, w = np.polynomial.legendre.leggauss(n_mu)
    phi = 2 * np.pi * np.arange(n_phi) / n_phi
    MU, PHI = np.meshgrid(mu, phi, indexing="ij")
    s = np.sqrt(1 - MU ** 2)
    xs = jnp.asarray(np.stack([rho * s * np.cos(PHI), rho * s * np.sin(PHI), rho * MU], -1))
    vals = np.asarray(jax.vmap(lambda y: point_fields(y).lam)(xs.reshape(-1, 3))).reshape(MU.shape)
    avg = vals.mean(axis=1)
    c = [(2 * l + 1) / 2 * float(np.sum(w * avg * np.polynomial.legendre.Legendre.basis(l)(mu)))
         for l in range(lmax + 1)]
    nonaxi = float(np.sqrt(np.sum(w[:, None] * (vals - avg[:, None]) ** 2) / (2 * n_phi)))
    return np.asarray(c), nonaxi, mu, avg


def outer_figure(pf_P, cfg, outdir, exact_fields=None, report=None, lmax: int = 8):
    """P_multipoles_outer.png: what the far field of P is made of, at rho_out.

    (a) the Legendre spectrum |c_l| of P, l = 0..lmax, log scale, network against the nonlinear
        reference when the run has one: which multipoles are physical and which are noise (with
        S1 = 0 the odd l vanish by reflection symmetry, so they are the network's noise floor);
    (b) the anisotropy P(theta) - c_0, network and reference, and the reference's l = 2 term;
    (c) what is left after l <= 2, network and reference.
    The old version drew l = 0..2 on linear axes, where the monopole (50x the rest) flattened
    everything else, and a bar chart in which only l = 0 was visible.
    """
    import matplotlib.pyplot as plt
    from .losses import gh_potential
    BLUE, ORANGE, GREEN, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#898781"
    rho = float(cfg.rho_out)
    c_net, nonaxi, mu, avg_net = legendre_spectrum(pf_P, rho, lmax)
    ref, why = None, None
    if exact_fields is not None and (cfg.lam_bc_S1 or cfg.lam_bc_S2):
        try:
            from .specref import load_reference
            ref, why = load_reference(outdir, cfg, exact_fields)
        except Exception as exc:                     # a figure must not fail on the reference
            ref, why = None, str(exc)
    if ref is not None:
        c_ref, _, _, avg_ref = legendre_spectrum(gh_fields(ref), rho, lmax)
    if report is not None:
        report["outer_legendre_P"] = [float(v) for v in c_net]
        report["outer_nonaxisymmetric_rms_P"] = nonaxi
        if ref is not None:
            report["outer_legendre_P_reference"] = [float(v) for v in c_ref]

    th = np.arccos(mu)
    ls = np.arange(lmax + 1)
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
    a = ax[0]
    off = 0.18 if ref is not None else 0.0
    a.bar(ls - off, np.maximum(np.abs(c_net), 1e-18), width=0.36 if ref is not None else 0.6,
          color=BLUE, label="network")
    if ref is not None:
        a.bar(ls + off, np.maximum(np.abs(c_ref), 1e-18), width=0.36, color=ORANGE,
              label="nonlinear reference")
    # the axis starts just below the network's smallest coefficient: the reference's odd l are
    # machine zeros (1e-20) and would otherwise stretch the axis over 14 empty decades
    floor = max(float(np.min(np.abs(c_net[np.abs(c_net) > 0]))) / 30.0, 1e-16)
    if nonaxi >= floor:
        a.axhline(nonaxi, color=MUTED, ls=":", lw=1.2,
                  label=f"non-axisymmetric part (rms {nonaxi:.1e})")
    else:
        a.plot([], [], " ", label=f"non-axisymmetric part: rms {nonaxi:.0e}")
    if ref is not None:
        for l in ls:
            if abs(c_ref[l]) < floor:
                a.text(l + off, floor * 1.6, "$\\approx0$", ha="center", fontsize=8, color=ORANGE)
    a.set_yscale("log"); a.set_ylim(bottom=floor); a.set_xticks(ls); a.set_xlabel("$l$")
    a.set_ylabel(r"$|c_l|$,  $P=\sum_l c_l P_l(\cos\theta)$")
    odd = ("\n(odd $l$ vanish by symmetry, so the network's odd $l$ are its noise floor)"
           if not cfg.lam_bc_S1 else "")
    a.set_title(f"spectrum of $P$ at $\\rho={rho:g}${odd}", fontsize=10)
    a.legend(fontsize=8)

    a = ax[1]
    a.plot(th, avg_net - c_net[0], color=BLUE, lw=2, label="network")
    if ref is not None:
        a.plot(th, avg_ref - c_ref[0], "--", color=ORANGE, lw=2, label="nonlinear reference")
        a.plot(th, c_ref[2] * np.polynomial.legendre.Legendre.basis(2)(mu), ":", color=GREEN,
               lw=1.5, label="reference, $l=2$ term only")
    a.set_xlabel(r"$\theta$"); a.set_ylabel(r"$P(\theta)-c_0$  ($\varphi$-averaged)")
    a.set_title("the anisotropy: everything but the monopole", fontsize=10)
    a.legend(fontsize=8)

    a = ax[2]
    low = lambda c: sum(c[l] * np.polynomial.legendre.Legendre.basis(l)(mu) for l in range(3))
    a.plot(th, avg_net - low(c_net), color=BLUE, lw=2, label="network")
    if ref is not None:
        a.plot(th, avg_ref - low(c_ref), "--", color=ORANGE, lw=2, label="nonlinear reference")
    a.set_xlabel(r"$\theta$"); a.set_ylabel(r"$P(\theta)-\sum_{l\leq2}c_lP_l$")
    a.set_title("what is left after $l\\leq2$ (higher multipoles, and the error)", fontsize=10)
    a.legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.3)
    refline = (f"reference: {os.path.basename(why['path'])}" if ref is not None
               else f"no nonlinear reference ({why})" if why else "no nonlinear reference")
    fig.suptitle(f"the Geroch-Hansen potential $P=(1-\\lambda^2)/(4\\lambda)$ at the outer sphere "
                 f"$\\rho={rho:g}$   |   inner $\\lambda$ data $S_1={cfg.lam_bc_S1:g}$, "
                 f"$S_2={cfg.lam_bc_S2:g}$, $\\lambda_0={cfg.lam0:g}$   |   {refline}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(outdir, "P_multipoles_outer.png"), dpi=120)
    plt.close(fig)


def multipole_radial_profile(point_fields, rhos, lmax: int = 3, n_mu: int = 40,
                             n_phi: int = 24, lam_inf: float = 0.0):
    """Amplitude sqrt(sum_m a_lm^2) of each multipole of `.lam` as a function of rho.

    Called with `gh_fields(...)` it is the multipoles of P.  For a decaying harmonic field the
    l-th multipole should fall off as
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
