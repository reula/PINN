"""The inner metric condition is on the INDUCED metric only -- never h_rr, never h_rA.

Until 2026-10-15 `inner_bc_raw` imposed `h - h_rr n n = c (I - n n)`, whose mixed part set
`h_rA = 0` on the inner sphere too, and the hard blend imposed `h_rr = 1` as well.  That is one
condition too many for every l >= 1 mode: the spectral reference (stationary/specref.py) stalled at
a residual floor and kept a non-decaying l = 2 metric part, and with the induced metric alone it
converged (loss 1.8e-5 -> 3.0e-9) with the l = 2 tails decaying as rho^-3.  These tests make sure
neither version can impose h_rr or h_rA again.
"""
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from stationary.losses import inner_bc_raw
from stationary.model import Fields, _blend_inner
from stationary.problem import Config, lam_inner_bc, sample_sphere

I3 = jnp.eye(3)


def _cfg(**kw):
    return Config(R0=0.5773502691896258, rho_in=1.0, rho_out=100.0, inner_radius=1.0,
                  lam0=1.0 / 3.0, lam_bc_S2=-1.0 / 12.0, **kw)


def _round_with_free_normal_parts(cfg, a_rr=0.37, a_ra=0.21):
    """h = round tangential block + ARBITRARY h_rr and h_rA (angle-dependent), lambda = datum."""
    c = (cfg.inner_radius / cfg.rho_in) ** 2

    def fields(x):
        rho = jnp.linalg.norm(x)
        n = x / rho
        P = I3 - jnp.outer(n, n)
        t = P @ jnp.array([1.0, 0.5, -0.3])          # some tangential vector field
        h = (c * (cfg.rho_in / rho) ** 2 * P + (1.0 + a_rr * n[2]) * jnp.outer(n, n)
             + a_ra * (jnp.outer(n, t) + jnp.outer(t, n)))
        return Fields(h, jnp.zeros((3, 3, 3)), lam_inner_bc(x, cfg))
    return fields


def test_the_soft_condition_ignores_h_rr_and_h_rA():
    cfg = _cfg()
    xs = sample_sphere(jax.random.PRNGKey(0), 256, cfg.rho_in)
    out = inner_bc_raw(_round_with_free_normal_parts(cfg), xs, cfg)
    for k, v in out.items():
        assert float(jnp.max(jnp.abs(v))) < 1e-13, (k, float(jnp.max(jnp.abs(v))))


def test_the_soft_condition_still_sees_a_non_round_induced_metric():
    cfg = _cfg()
    xs = sample_sphere(jax.random.PRNGKey(1), 256, cfg.rho_in)
    good = _round_with_free_normal_parts(cfg)

    def bad(x):
        f = good(x)
        n = x / jnp.linalg.norm(x)
        P = I3 - jnp.outer(n, n)
        return Fields(f.h + 0.05 * n[2] ** 2 * P, f.G, f.lam)   # an l = 2 squash of the sphere
    assert float(jnp.max(jnp.abs(inner_bc_raw(bad, xs, cfg)["h_tan"]))) > 1e-3


def test_the_hard_blend_keeps_the_networks_h_rr_and_h_rA():
    """On the inner sphere: P h P is the datum, the normal parts are the network's own."""
    rho_in = 1.0
    x = sample_sphere(jax.random.PRNGKey(2), 64, rho_in)
    n = x / jnp.linalg.norm(x, axis=-1, keepdims=True)
    nn = n[:, :, None] * n[:, None, :]
    P = I3[None] - nn
    h_net = I3[None] + 0.3 * jax.random.normal(jax.random.PRNGKey(3), (64, 3, 3))
    h_net = 0.5 * (h_net + jnp.swapaxes(h_net, 1, 2))
    datum = lambda X: (nn + 0.8 * P, jnp.full((X.shape[0],), 0.5))
    h, lam = _blend_inner(x, h_net, jnp.ones(64), rho_in, datum)
    assert float(jnp.max(jnp.abs(P @ h @ P - 0.8 * P))) < 1e-13            # induced metric = datum
    assert float(jnp.max(jnp.abs(h - P @ h @ P - (h_net - P @ h_net @ P)))) < 1e-13  # rr, rA free
    assert float(jnp.max(jnp.abs(lam - 0.5))) < 1e-13
