"""Diagnostics: residual norms, boundary geometry and comparison with the exact solution."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .geometry import residuals_batch
from .problem import Config, sample_shell, sample_sphere

I3 = jnp.eye(3)


def residual_report(point_fields, cfg: Config, n: int = 2048, seed: int = 12345) -> dict:
    """RMS and max of every equation group on a validation sample."""
    key = jax.random.PRNGKey(seed)
    xs = sample_shell(key, n, cfg)
    r = residuals_batch(point_fields, xs)
    out = {}
    for k, v in r.items():
        a = jnp.abs(v)
        out[f"res_{k}_rms"] = float(jnp.sqrt(jnp.mean(v ** 2)))
        out[f"res_{k}_max"] = float(jnp.max(a))
    # inner/outer boundary residuals are checked separately
    return out


def inner_boundary_report(point_fields, cfg: Config, n: int = 512, seed: int = 54321) -> dict:
    """lambda and the areal radius of the inner sphere as measured by the metric.

    For the round metric of areal radius r_a in angles (theta, phi):
        sqrt(det G_2) / sin(theta) = r_a^2 .
    """
    key = jax.random.PRNGKey(seed)
    xs = sample_sphere(key, n, cfg.rho_in)

    def one(x):
        f = point_fields(x)
        # tangent basis through the standard angles, in the (theta, phi) COORDINATE basis
        rho = jnp.linalg.norm(x)
        th = jnp.arccos(jnp.clip(x[2] / rho, -1.0, 1.0))
        ph = jnp.arctan2(x[1], x[0])
        eth = rho * jnp.array([jnp.cos(th) * jnp.cos(ph), jnp.cos(th) * jnp.sin(ph),
                               -jnp.sin(th)])
        eph = rho * jnp.sin(th) * jnp.array([-jnp.sin(ph), jnp.cos(ph), 0.0])
        G2 = jnp.array([[eth @ f.h @ eth, eth @ f.h @ eph],
                        [eph @ f.h @ eth, eph @ f.h @ eph]])
        area_ratio = jnp.sqrt(jnp.linalg.det(G2)) / jnp.sin(th)
        return f.lam, area_ratio, f.h

    lam_v, area_ratio, _ = jax.vmap(one)(xs)
    return {
        "lam_mean": float(jnp.mean(lam_v)),
        "lam_std": float(jnp.std(lam_v)),
        "areal_radius2_mean": float(jnp.mean(area_ratio)),
        "areal_radius2_std": float(jnp.std(area_ratio)),
    }


def exact_comparison(point_fields, exact_fields, cfg: Config, n: int = 4096,
                     seed: int = 999) -> dict:
    """Pointwise differences from the exact solution (milestone 1 acceptance test)."""
    key = jax.random.PRNGKey(seed)
    xs = sample_shell(key, n, cfg)

    def one(x):
        f = point_fields(x)
        e = exact_fields(x)
        return (jnp.max(jnp.abs(f.h - e.h)), jnp.max(jnp.abs(f.G - e.G)),
                jnp.abs(f.lam - e.lam))

    dh, dG, dl = jax.vmap(one)(xs)
    return {
        "max_dh": float(jnp.max(dh)),
        "max_dG": float(jnp.max(dG)),
        "max_dlam": float(jnp.max(dl)),
        "rms_dh": float(jnp.sqrt(jnp.mean(dh**2))),
        "rms_dG": float(jnp.sqrt(jnp.mean(dG**2))),
        "rms_dlam": float(jnp.sqrt(jnp.mean(dl**2))),
    }


def radial_profiles(point_fields, cfg: Config, nrho: int = 40, nth: int = 1) -> dict:
    """Fields along a radial ray (theta = pi/2) -- handy for plots and sanity checks."""
    rhos = jnp.linspace(cfg.rho_in, cfg.rho_out, nrho)
    xs = jnp.stack([rhos, jnp.zeros_like(rhos), jnp.zeros_like(rhos)], axis=-1)

    def one(x):
        f = point_fields(x)
        return f.h, f.G, f.lam

    h, G, lam = jax.vmap(one)(xs)
    return {"rho": [float(v) for v in rhos],
            "h": h, "G": G, "lam": lam}
