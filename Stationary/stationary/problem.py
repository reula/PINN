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
    lam0: float = 1.0               # constant part of lambda on the inner sphere
    inner_radius: float | None = None   # areal radius required on the inner sphere
                                        # None -> rho_in (round sphere of radius rho_in)
    lam_bc_S1: float = 0.0          # dipole amplitude    S1 * z / rho_in
    lam_bc_S2: float = 0.0          # quadrupole amplitude S2 * (z^2-(x^2+y^2)/2)/rho_in^2
    inner_h_rr: float | None = 1.0  # None -> do NOT constrain h_rr on the inner sphere
    rho_in: float | None = None     # None -> sqrt(4 + R0^2): the areal-radius-2 sphere

    # ------------------------------------------------------- outer boundary
    outer_bc: str = "dirichlet_exact"   # "dirichlet_exact" (milestone 1) | "robin"
    ref_solution: bool = False          # build the exact reference for diagnostics only
    robin_source: bool = False           # inhomogeneous Robin source (manufactured test)
    ref_asymptotic: float | None = None  # if set, reference has lambda -> this value
    lam_inf: float | None = None        # asymptotic lambda; None -> learnable (robin)
    lam_inf_init: float = 1.0           # initial value when lam_inf is learnable
    # decay exponents of the Robin condition  n^i d_i f = -p (f - f_inf)/rho.
    # The system forces h ~ 1/rho^2, Gamma ~ 1/rho^3, lambda ~ 1/rho (measured for
    # the exact solution), so a single exponent p = 1 for every field would impose
    # the wrong asymptotics and create a boundary layer at rho_out.
    robin_exps: dict = field(default_factory=lambda: dict(h=2.0, G=3.0, lam=1.0))
    # Higher-multipole Robin: instead of forcing each field onto its single leading
    # decay power, annihilate the first `robin_order` powers with the Euler operators
    #     prod_{i=0}^{n-1} (rho d_rho + (base+i)) (field - field_inf) = 0 .
    # These operators commute and annihilate exactly rho^-(base+i), so n=1 reproduces
    # the first-order condition above, n=2 allows the next multipole through, etc.
    # Example (lambda, base 1, n=2):  rho^2 lam'' + 4 rho lam' + 2 (lam-1) = 0.
    robin_order: int = 2
    robin_orders: dict | None = None     # per-field override, e.g. dict(h=4, G=1, lam=4)
    robin_include_G: bool = True         # impose the Robin condition on Gamma too.
                                         # In the metric-only (hybrid) schemes Gamma is
                                         # derived from h, so its condition is redundant
                                         # and is what forces fifth derivatives of h.

    # ---------------------------------------------------------------- model
    arch: str = "sym"               # "sym" (spherically symmetric ansatz) | "3d"
    width: int = 64
    depth: int = 4
    fourier: int = 8
    decay_feature: bool = False      # add rho_in/rho to the network features

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
    reweight_max_ratio_inv: float = 0.5   # per-update cap: weights move by at most 2x

    # ---------------------------------------------------------- optimisation
    steps: int = 20000
    lr: float = 1e-3
    lbfgs_steps: int = 300
    log_every: int = 200
    seed: int = 0
    outdir: str = "runs/m1"
    init_from: str | None = None
    ckpt_every: int = 0             # write a resumable checkpoint every N Adam steps (0 = off)
    resume: str | None = None       # checkpoint path, or "auto" for <outdir>/ckpt.pkl
    make_figures: bool = True       # lambda at the inner sphere + outer multipoles

    def __post_init__(self):
        if self.rho_in is None:
            self.rho_in = rho_in_of_R0(self.R0)
        self.rho_in = float(self.rho_in)
        if self.inner_radius is None:
            # The inner sphere is the round sphere of AREAL RADIUS 2 -- the problem
            # statement -- and `rho_in_of_R0(R0) = sqrt(4+R0^2)` is by construction the
            # coordinate radius at which the canonical-chart solution has areal radius 2.
            #
            # It must NOT default to rho_in: the coordinate sphere |x| = rho_in has
            # tangential metric (rho_in/rho_in)^2 = 1, i.e. flat, while the exact solution
            # has h_tan = 1 - R0^2/rho_in^2 = 4/(4+R0^2) there (0.8 for R0 = 1). Asking for
            # the former makes the inner data contradict the exact solution used for
            # `dirichlet_exact` (and for the Robin source), so no metric can satisfy both:
            # the run then converges to a hybrid that is 25% off near the inner sphere.
            # This default was rho_in between commits 67a802a and this one; the runs made
            # before it (m1_sym, m1_3d) used the areal radius 2.
            self.inner_radius = 2.0


def lam_inner_bc(x, cfg):
    """lambda prescribed on the inner sphere:

        lambda = lam0 + S1 z/rho_in + S2 (z^2 - (x^2+y^2)/2)/rho_in^2

    written in the harmonic coordinates (x, y, z); on |x| = rho_in it reduces to
    lam0 + S1 n_z + S2 (3 n_z^2 - 1)/2.
    """
    z = x[..., 2]
    r2 = jnp.sum(x * x, axis=-1)
    return (cfg.lam0 + cfg.lam_bc_S1 * z / cfg.rho_in
            + cfg.lam_bc_S2 * (z * z - 0.5 * (r2 - z * z)) / cfg.rho_in**2)


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
