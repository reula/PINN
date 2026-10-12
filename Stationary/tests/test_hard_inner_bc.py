"""The hard inner BC holds to roundoff; the soft one is a penalty and does not.

`hard_inner_bc` replaces "penalise the violation of the inner data" with "the fields ARE the
datum on the inner sphere" (a directional blend with weight 1 - rho_in/rho).  The two things this
file pins are (a) that the inner residuals then come out at roundoff rather than at the size of the
network's initial output, and (b) that turning it on changes nothing else about how a run is
loaded -- `make_model` is also what `evaluate.load_run` calls, so a hard-BC run is reloaded with
the same constrained ansatz.
"""
import jax
import jax.numpy as jnp

from stationary.geometry import apply_config
from stationary.losses import inner_bc_terms
from stationary.model import point_fields
from stationary.problem import Config, sample_sphere
from stationary.train import make_model

jax.config.update("jax_enable_x64", True)


def test_the_datum_uses_the_inner_radius_not_rho_in():
    """`inner_radius` and `rho_in` are different settings and the round-sphere factor is
    (inner_radius/rho_in)^2.  A datum that hardcodes 1 is right only when they agree, and the
    residual then reads 3|1 - c|: 1.73 for the default inner_radius = 2 with rho_in = 1, which is
    exactly what a tiny end-to-end run reported before this was fixed.
    """
    cfg = Config(arch="sym_hybrid", width=6, depth=2, fourier=0, rho_in=1.0, rho_out=10.0,
                 inner_radius=2.0, lam0=1.0 / 3.0, lam_bc_S2=-1.0 / 12.0, hard_inner_bc=True)
    apply_config(cfg)
    model = make_model(cfg)
    params = model.init(jax.random.PRNGKey(0), jnp.ones((4, 3)))
    pf = point_fields(model, params)
    xs = sample_sphere(jax.random.PRNGKey(1), 64, cfg.rho_in)
    res = {k: float(v) for k, v in inner_bc_terms(pf, xs, cfg).items()}
    for key, value in res.items():
        assert value < 1e-20, (key, value)


def _cfg(hard: bool) -> Config:
    return Config(arch="axisym_hybrid", width=8, depth=2, fourier=0, rho_in=1.0, rho_out=10.0,
                  inner_radius=1.0, lam0=1.0 / 3.0, lam_bc_S2=-1.0 / 12.0,
                  hard_inner_bc=hard)


def _inner_residuals(hard: bool):
    cfg = _cfg(hard)
    apply_config(cfg)
    model = make_model(cfg)
    params = model.init(jax.random.PRNGKey(0), jnp.ones((4, 3)))
    pf = point_fields(model, params)
    xs = sample_sphere(jax.random.PRNGKey(1), 64, cfg.rho_in)
    return {k: float(v) for k, v in inner_bc_terms(pf, xs, cfg).items()}


def test_the_hard_inner_bc_holds_to_roundoff():
    res = _inner_residuals(hard=True)
    assert res, "no inner residuals were produced"
    for key, value in res.items():
        assert value < 1e-20, (key, value)


def test_the_soft_inner_bc_is_a_real_penalty():
    """At initialisation the unconstrained net violates the datum by O(out_std) -- which is the
    term the hard version removes from the loss rather than merely shrinking."""
    res = _inner_residuals(hard=False)
    assert max(res.values()) > 1e-8, res


def test_the_flag_reaches_the_config():
    from stationary.train import parse_args
    cfg = parse_args(["--outdir", "/tmp/x", "--hard-inner-bc"])
    assert cfg.hard_inner_bc is True
    assert parse_args(["--outdir", "/tmp/x"]).hard_inner_bc is False
