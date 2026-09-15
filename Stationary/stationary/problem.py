"""Problem configuration and sampling for the shell  rho_in <= |x| <= rho_out."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from .exact import rho_in as rho_in_of_R0


@dataclass
class Config:
    # ---------------------------------------------------------------- physics
    R0: float = 1.0                 # exact-solution parameter used for milestone 1
    rho_out: float = 20.0
    lam0: float = 1.0               # lambda on the inner sphere
    rho_in: float | None = None     # None -> sqrt(4 + R0^2): the areal-radius-2 sphere

    # ------------------------------------------------------- outer boundary
    outer_bc: str = "dirichlet_exact"   # "dirichlet_exact" (milestone 1) | "robin"
    lam_inf: float | None = None        # asymptotic lambda; None -> learnable (robin)

    # ---------------------------------------------------------------- model
    arch: str = "sym"               # "sym" (spherically symmetric ansatz) | "3d"
    width: int = 64
    depth: int = 4
    fourier: int = 8

    # ------------------------------------------------------------- sampling
    n_coll: int = 4096
    n_bnd: int = 256
    resample_every: int = 500
    radial: str = "log"              # "uniform" | "log"

    # --------------------------------------------------- weights / scaling
    eq_weights: dict = field(default_factory=lambda: dict(
        compat=1.0, ricci=1.0, gauge=1.0, lam_eq=1.0))
    scale_exps: dict = field(default_factory=lambda: dict(
        compat=1.0, ricci=2.0, gauge=1.0, lam_eq=2.0))
    scale_ref: float | None = None   # None -> use local rho; else a fixed length
    w_inner: float = 10.0
    w_outer: float = 10.0
    pde_ramp_steps: int = 0     # ramp the PDE weights in over this many steps (0 = off)
    reweight_every: int = 2000  # gradient-norm adaptive reweighting period (0 = off)

    # ---------------------------------------------------------- optimisation
    steps: int = 20000
    lr: float = 1e-3
    lbfgs_steps: int = 300
    log_every: int = 200
    seed: int = 0
    outdir: str = "runs/m1"
    init_from: str | None = None

    def __post_init__(self):
        if self.rho_in is None:
            self.rho_in = rho_in_of_R0(self.R0)
        self.rho_in = float(self.rho_in)


def sphere_directions(key, n: int) -> jnp.ndarray:
    """Uniform directions on S^2."""
    u = jax.random.normal(key, (n, 3))
    return u / jnp.linalg.norm(u, axis=-1, keepdims=True)


def sample_shell(key, n: int, cfg: Config) -> jnp.ndarray:
    """Collocation points in the shell (uniform in rho or in log rho)."""
    k1, k2 = jax.random.split(key)
    if cfg.radial == "log":
        rho = jnp.exp(jax.random.uniform(k1, (n,), minval=math.log(cfg.rho_in),
                                         maxval=math.log(cfg.rho_out)))
    else:
        rho = jax.random.uniform(k1, (n,), minval=cfg.rho_in, maxval=cfg.rho_out)
    return rho[:, None] * sphere_directions(k2, n)


def sample_sphere(key, n: int, radius: float) -> jnp.ndarray:
    return radius * sphere_directions(key, n)
