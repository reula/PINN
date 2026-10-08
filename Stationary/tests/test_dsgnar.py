"""The residual-vector form of the loss, and the DSGNAR (Gauss-Newton) phase it feeds.

`total_loss` is a sum of weighted MEANS of squares, so it is also the squared norm of a
residual vector.  That second form is what a least-squares optimiser needs, and it is the only
reason `losses.residual_vector` exists: same weights, same points, same terms, reduction
deferred.  The binding constraint is therefore an equality, not a tolerance --

    jnp.sum(residual_vector(state, batch, cfg, model, ...)**2) == total_loss(state, batch, ...)[0]

-- and it has to hold for every combination of the pin reductions (averaged pins square a MEAN,
value pins average squares), which is where a residual vector is easy to get wrong.

The DSGNAR phase is then checked end to end on a problem small enough to run in seconds: it
must lower the objective it was handed, return a flattened parameter vector of the right size,
and report the iteration count it actually used.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

from flax import linen as nn

from stationary import exact
from stationary.losses import residual_vector, total_loss
from stationary.model import Fields
from stationary.problem import Config
from stationary.train import build, make_batch, parse_args

BASE = ["--arch", "axisym_hybrid", "--n-coll", "64", "--n-bnd", "24",
        "--width", "8", "--depth", "2", "--no-figures",
        "--R0", "0.5773502691896258", "--rho-in", "1.0", "--rho-out", "100.0",
        "--inner-radius", "1.0", "--lam0", "0.33333333333333337", "--lam-inf", "1.0",
        "--no-robin-G"]


def _cfg(tmp_path, extra=()):
    return parse_args(BASE + ["--outdir", str(tmp_path / "run")] + list(extra))


def _squared_norm(state, batch, cfg, model, exact_fields, **kw):
    r = residual_vector(state, batch, cfg, model, exact_fields, **kw)
    return float(jnp.sum(r ** 2)), r


@pytest.mark.parametrize("extra", [
    (),                                                        # plain Robin, no pins
    ("--pin-lam-robin",),                                      # an AVERAGED pin
    ("--ref-solution", "--ref-asymptotic", "1.0", "--pin-lam",
     "--pin-h-tan", "--pin-h-rr"),                             # VALUE pins
    ("--ref-solution", "--ref-asymptotic", "1.0", "--pin-h-robin"),
    ("--ref-solution", "--ref-asymptotic", "1.0", "--pin-lam-robin", "--pin-h-robin"),
    ("--w-lam-eq-radial", "1.0"),                              # the radial-derivative term
    ("--robin-source", "--ref-solution", "--ref-asymptotic", "1.0"),   # manufactured source
])
def test_the_residual_vector_squares_to_the_total_loss(tmp_path, extra):
    cfg = _cfg(tmp_path, extra)
    model, state, exact_fields = build(cfg)
    batch = make_batch(jax.random.PRNGKey(0), cfg)
    loss, _ = total_loss(state, batch, cfg, model, exact_fields)
    norm2, r = _squared_norm(state, batch, cfg, model, exact_fields)
    assert r.ndim == 1 and r.size > 0
    assert norm2 == pytest.approx(float(loss), rel=1e-11, abs=1e-30)


class ExactModel(nn.Module):
    """A network-shaped module whose output IS the exact solution (no real parameters)."""

    R0: float = 1.0

    @nn.compact
    def __call__(self, x):
        x = jnp.atleast_2d(x)
        dummy = self.param("dummy", nn.initializers.zeros, (1,))
        f = jax.vmap(exact.exact_fields(self.R0, 1.0))(x)
        d = dummy.sum() * 0.0
        return Fields(f.h + d, f.G + d, f.lam + d)


def test_the_residual_vector_is_zero_on_the_exact_solution():
    """The exact solution has zero loss, so the residual vector must vanish identically.

    The configuration is the one tests/test_pipeline.py uses for the same statement: R0 = 1
    with the round inner sphere of areal radius 2 (which sits at rho = sqrt 5), which is the
    branch whose inner data the exact solution actually carries.
    """
    from stationary.problem import sample_shell, sample_sphere
    cfg = Config(R0=1.0, lam0=exact.lambda0_from_k(1.0, 1.0, 2.0), inner_radius=2.0,
                 robin_order=1)
    model = ExactModel()
    state = {"net": model.init(jax.random.PRNGKey(2), jnp.ones((1, 3)))}
    key = jax.random.PRNGKey(3)
    batch = {"coll": sample_shell(key, 256, cfg),
             "inner": sample_sphere(key, 64, cfg.rho_in),
             "outer": sample_sphere(key, 64, cfg.rho_out)}
    ef = lambda x: exact.exact_fields(1.0, 1.0)(x)
    loss, _ = total_loss(state, batch, cfg, model, exact_fields=ef)
    norm2, _ = _squared_norm(state, batch, cfg, model, ef)
    assert float(loss) < 1e-16, float(loss)
    assert norm2 < 1e-16, norm2


def test_the_weights_are_in_the_vector_and_not_in_the_reduction(tmp_path):
    """A group weight must move the residual by sqrt(w): that is what makes the vector's norm
    the loss rather than the loss times an arbitrary constant."""
    cfg = _cfg(tmp_path)
    model, state, exact_fields = build(cfg)
    batch = make_batch(jax.random.PRNGKey(0), cfg)
    w = dict(ricci=4.0, gauge=1.0, lam_eq=1.0, inner=10.0, outer=10.0)
    norm2, _ = _squared_norm(state, batch, cfg, model, exact_fields, weights=w)
    loss, _ = total_loss(state, batch, cfg, model, exact_fields, weights=w)
    assert norm2 == pytest.approx(float(loss), rel=1e-11, abs=1e-30)


def test_the_dsgnar_objective_and_phase_lower_the_loss(tmp_path):
    """End to end: wrap the residual vector for the optimiser, run a few iterations."""
    from stationary.dsgnar import dsgnar_phase
    from stationary.flat_objective import FlatObjective

    cfg = _cfg(tmp_path)
    cfg.dsgnar_steps = 4
    cfg.dsgnar_sketch = 8
    cfg.resample_every = 0
    model, state, exact_fields = build(cfg)
    batch = make_batch(jax.random.PRNGKey(0), cfg)
    obj = FlatObjective(state, batch, cfg, model, exact_fields)
    loss0, _ = total_loss(state, batch, cfg, model, exact_fields)
    assert float(obj._loss(obj.flat0)) == pytest.approx(float(loss0), rel=1e-11)
    assert obj.n == obj.flat0.size and obj.dtype == obj.flat0.dtype

    flat, history, info = dsgnar_phase(obj, obj.flat0, cfg, verbose=False)
    assert flat.size == obj.flat0.size
    assert len(history) == info["iterations"] + 1        # the opening entry plus one per step
    assert info["iterations"] >= 1
    assert history[-1]["loss"] < history[0]["loss"], (history[0]["loss"], history[-1]["loss"])
    assert info["n_residuals"] == obj._residual(obj.flat0).size
