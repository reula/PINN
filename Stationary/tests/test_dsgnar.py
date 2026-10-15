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
     "--pin-h-tan", "--pin-h-rr", "--no-pin-h-robin"),         # VALUE pins (and no averaged one:
                                                               # --pin-h-robin is ON by default now)
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


def test_the_chunked_sketch_is_the_unchunked_sketch(tmp_path):
    """`--dsgnar-chunk` only changes how the tangents are scheduled, never the arithmetic.

    The reference implementation vmapped all s tangents in one call; on this problem that is a
    single 4.78 GiB allocation at s = 128 and the GPU refused it (runs/pq_c100_vacF_dsgnar_probe_s128).
    Chunking is the fix, so it has to give the same columns: this compares the two paths
    elementwise, including an s that is NOT a multiple of the chunk.
    """
    from stationary.dsgnar import jvp_columns

    n, m = 14, 7

    def res(flat):
        return jnp.tanh(flat[:m] * 1.7) + 0.3 * flat[m:2 * m] ** 2 + flat[:m] @ flat[:m]

    flat = jnp.linspace(-0.4, 0.6, n)
    for s, chunk in ((8, 0), (8, 4), (8, 3), (8, 8), (6, 4), (5, 2)):
        basis = jnp.asarray(jax.random.normal(jax.random.PRNGKey(s + chunk), (s, n)))
        want = jvp_columns(res, flat, basis, 0)
        got = jvp_columns(res, flat, basis, chunk)
        assert got.shape == want.shape == (s, m)
        assert jnp.allclose(got, want, rtol=1e-12, atol=1e-14), (s, chunk)


def test_the_row_chunked_sketch_is_the_same_sketch():
    """`--dsgnar-row-chunk` changes the workspace, not the result.

    CountSketch builds an `rows x s x K` temporary.  At rows ~ 4e5 and s = 128 that is several
    GiB and the GPU refused it (runs/pq_c100_vacF_dsgnar_probe_s128: 8.87 GiB, `jit__where`),
    so the rows are summed in blocks.  Blocking must be exactly transparent: the hash draws are
    untouched and a sum over rows does not care how it is bracketed.
    """
    from stationary.dsgnar import count_sketch

    rows, s, k = 24, 5, 4
    key = jax.random.PRNGKey(0)
    mat = jax.random.normal(jax.random.PRNGKey(1), (rows, 7))
    vec = jax.random.normal(jax.random.PRNGKey(2), (rows,))
    want_m, want_v = count_sketch(mat, vec, key, s, k_hashes=k, row_chunk=0)
    for chunk in (2, 3, 4, 6, 12):
        got_m, got_v = count_sketch(mat, vec, key, s, k_hashes=k, row_chunk=chunk)
        assert jnp.allclose(got_m, want_m, rtol=1e-12, atol=1e-14), chunk
        assert jnp.allclose(got_v, want_v, rtol=1e-12, atol=1e-14), chunk
    # a vector-only sketch (mat=None) takes the same path
    _, v0 = count_sketch(None, vec, key, s, k_hashes=k, row_chunk=0)
    _, v1 = count_sketch(None, vec, key, s, k_hashes=k, row_chunk=4)
    assert jnp.allclose(v0, v1, rtol=1e-12, atol=1e-14)


def test_the_probe_losses_are_evaluated_sequentially(tmp_path):
    """The trust-region probes must NOT be one big vmap of the loss.

    `vmap(loss_fn)` over the 25 probes materialises the whole residual computation 25 times
    over, and that is what the GPU refused at production size (8.87 GiB, surfacing at
    `rho_host = np.asarray(rhos)`).  This pins both halves: the values agree, and the sequential
    form needs a small fraction of the workspace -- measured with XLA's own analysis, so a
    regression that puts the vmap back fails here and not on the hub.
    """
    from stationary.flat_objective import FlatObjective

    cfg = _cfg(tmp_path)
    model, state, exact_fields = build(cfg)
    batch = make_batch(jax.random.PRNGKey(0), cfg)
    obj = FlatObjective(state, batch, cfg, model, exact_fields)
    flats = obj.flat0[None, :] + jnp.linspace(-1e-3, 1e-3, 25)[:, None] * jnp.ones_like(obj.flat0)

    seq = jax.jit(lambda F: jax.lax.map(obj._loss, F))
    par = jax.jit(lambda F: jax.vmap(obj._loss)(F))
    assert jnp.allclose(seq(flats), par(flats), rtol=1e-12)
    t_seq = seq.lower(flats).compile().memory_analysis().temp_size_in_bytes
    t_par = par.lower(flats).compile().memory_analysis().temp_size_in_bytes
    assert t_seq * 5 < t_par, (t_seq, t_par)


def test_the_radius_floor_anchors_to_the_high_water_mark():
    """A phase whose every step is rejected must still reach its budget.

    This is the A/B that the first floor failed: with the floor, `radius` may not fall below
    `floor * delta0` no matter how many rejections arrive, so the `radius below threshold` stop
    cannot fire and the phase spends its budget.  The objective here is deliberately uphill
    (`loss = 1 + |flat|^2`), which makes EVERY step rejected, and `J = I` so the sketch is exact
    and the iterations are meaningful rather than degenerate.
    """
    import jax.numpy as jnp
    from stationary.dsgnar import dsgnar_phase
    from stationary.problem import Config

    class Uphill:
        n = 4
        dtype = jnp.float64
        flat0 = jnp.zeros(4)

        # the phase reads the residual through `_residual` (its jitted scalar->residual map) and
        # the reduced loss through `_loss`
        _residual = residual = lambda self, flat: jnp.asarray(flat, dtype=jnp.float64)
        _loss = loss = None            # replaced below (lambda fields cannot reference siblings)



        def unflatten(self, flat):
            return flat

    def _res(flat):
        return jnp.asarray(flat, dtype=jnp.float64)

    def _lss(flat):
        return 1.0 + jnp.sum(jnp.asarray(flat, dtype=jnp.float64) ** 2)

    Uphill._residual = staticmethod(jax.jit(_res))   # staticmethod: a plain function
    Uphill._loss = staticmethod(jax.jit(_lss))       # would bind self on attribute access
    Uphill.residual = lambda self, flat: _res(flat)
    Uphill.loss = lambda self, flat: _lss(flat)

    def run(floor):
        # 40 rejections is past 3^-28 ~ 1e-13, which is where a floorless phase stops
        cfg = Config(dsgnar_steps=40, dsgnar_sketch=4, dsgnar_delta0=1.0, dsgnar_radius_floor=floor,
                     seed=0, n_coll=8, n_bnd=4)
        obj = Uphill()
        _, hist, info = dsgnar_phase(obj, obj.flat0, cfg, verbose=False)
        return hist, info

    hist, info = run(0.0)
    assert info["stopped"] == "radius below threshold", info          # the old behaviour
    assert info["iterations"] < 40

    hist, info = run(1.0e-3)
    assert info["stopped"] == "budget", info                          # the floor holds it up
    assert info["iterations"] == 40
    assert min(h["radius"] for h in hist) >= 1.0e-3 * 1.0 - 1e-15


def test_the_model_decrease_is_in_the_units_of_the_objective(monkeypatch):
    """With an EXACT model the accepted step's ratio is 1, not 1/2.

    The objective handed to the phase is `0.5 * mean(r**2)` (`FlatObjective`), so the model
    decrease must carry `1/(2M)`.  It carried `1/M`, which over-predicted every decrease by
    exactly two and halved every rho.  Here the residual is linear (`r = A x - b`), the
    parameter sketch keeps every direction (`s = n`, and the SRCT is orthogonal) and the row
    sketch is replaced by the identity (`M = s`), so the Gauss-Newton model IS the objective
    and the measured ratio has to be 1 to round-off.
    """
    from stationary import dsgnar
    from stationary.problem import Config

    n = 4
    A = jnp.array([[3.0, 1.0, 0.0, 0.5],
                   [0.0, 2.0, 1.0, 0.0],
                   [1.0, 0.0, 4.0, 1.0],
                   [0.0, 0.5, 0.0, 1.5]])
    b = jnp.array([1.0, -2.0, 0.5, 3.0])

    def _res(flat):
        return A @ jnp.asarray(flat, dtype=jnp.float64) - b

    def _lss(flat):
        return 0.5 * jnp.mean(_res(flat) ** 2)

    class Linear:
        dtype = jnp.float64
        _residual = staticmethod(jax.jit(_res))
        _loss = staticmethod(jax.jit(_lss))

        def residual(self, flat):
            return _res(flat)

        def loss(self, flat):
            return float(_lss(flat))

    Linear.n = n
    # one hash, sign +1, row i -> bucket i: with M = s that IS the identity sketch
    monkeypatch.setattr(dsgnar, "count_sketch_draws",
                        lambda key, rows, s, k, dtype: (jnp.ones((1, rows), dtype=dtype),
                                                        jnp.arange(rows)[None, :]))
    cfg = Config(dsgnar_steps=1, dsgnar_sketch=n, dsgnar_delta0=0.1, dsgnar_calibrate=False,
                 seed=0, n_coll=8, n_bnd=4)
    obj = Linear()
    _, hist, info = dsgnar.dsgnar_phase(obj, jnp.zeros(n), cfg, verbose=False)
    assert info["accepted"] == 1, info
    assert hist[1]["frac"] == 1.0, hist[1]
    assert abs(hist[1]["rho"] - 1.0) < 1e-8, hist[1]["rho"]


@pytest.mark.parametrize("s_cols, chunk", [(10, 0), (10, 4), (7, 3)])
def test_the_fused_sketch_is_the_two_step_sketch(s_cols, chunk):
    """`sketched_jvp_columns` == `count_sketch(jvp_columns(...).T, ...)` with the same key.

    The fused path pushes each block of tangents through the JVP and sketches it at once, so
    the M x s column matrix is never held (it is what capped s at d/18 on this problem).  It
    must be the SAME sketch: same hash draws, same columns, only the grouping of sums differs.
    """
    from stationary.dsgnar import (apply_count_sketch, count_sketch, count_sketch_draws,
                                   jvp_columns, sketched_jvp_columns)
    n, M, s, K = 9, 37, s_cols, 3
    A = jax.random.normal(jax.random.PRNGKey(1), (M, n))

    def res_fn(flat):
        return jnp.tanh(A @ flat) + 0.1 * (A @ flat) ** 2

    flat = 0.3 * jax.random.normal(jax.random.PRNGKey(2), (n,))
    basis = jax.random.normal(jax.random.PRNGKey(3), (s_cols, n))
    key = jax.random.PRNGKey(4)
    r = res_fn(flat)

    want_j, want_r = count_sketch(jvp_columns(res_fn, flat, basis, 0).T, r, key, s, k_hashes=K)
    signs, buckets = count_sketch_draws(key, M, s, K, r.dtype)
    got_j = sketched_jvp_columns(res_fn, flat, basis, signs, buckets, s, chunk)
    got_r = apply_count_sketch(r[:, None], signs, buckets, s)[:, 0]
    assert got_j.shape == (s, s_cols)
    assert float(jnp.max(jnp.abs(got_j - want_j))) < 1e-12
    assert float(jnp.max(jnp.abs(got_r - want_r))) < 1e-12


def _linear_objective(A, b, blocks=None):
    """A duck-typed objective with residual A x - b (exact Gauss-Newton model)."""
    def _res(flat):
        return A @ jnp.asarray(flat, dtype=jnp.float64) - b

    def _lss(flat):
        return 0.5 * jnp.mean(_res(flat) ** 2)

    class Linear:
        dtype = jnp.float64
        _residual = staticmethod(jax.jit(_res))
        _loss = staticmethod(jax.jit(_lss))

        def residual(self, flat):
            return _res(flat)

        def loss(self, flat):
            return float(_lss(flat))

    Linear.n = int(A.shape[1])
    obj = Linear()
    obj.blocks = blocks
    return obj


def _identity_row_sketch(monkeypatch):
    from stationary import dsgnar
    monkeypatch.setattr(dsgnar, "count_sketch_draws",
                        lambda key, rows, s, k, dtype: (jnp.ones((1, rows), dtype=dtype),
                                                        jnp.arange(rows)[None, :]))


def test_the_residual_blocks_cover_the_vector(tmp_path):
    """FlatObjective.blocks: one (label, n_rows) per condition, summing to the vector's length."""
    from stationary.flat_objective import FlatObjective
    cfg = _cfg(tmp_path, ("--pin-lam-robin",))
    model, state, exact_fields = build(cfg)
    obj = FlatObjective(state, make_batch(jax.random.PRNGKey(0), cfg), cfg, model, exact_fields)
    labels = [b[0] for b in obj.blocks]
    assert sum(b[1] for b in obj.blocks) == obj.residual(obj.flat0).shape[0]
    assert any(l.startswith("pde_") for l in labels) and any(l.startswith("inner_") for l in labels)
    assert len(labels) == len(set(labels))


def test_an_overshooting_step_is_rejected_and_the_radius_grows(monkeypatch):
    """Paper 3.4: with an exact model the ratio is 1, far above the 0.15 target -- rejected."""
    from stationary import dsgnar
    from stationary.problem import Config
    _identity_row_sketch(monkeypatch)
    A = jnp.diag(jnp.array([3.0, 2.0, 4.0, 1.5]))
    obj = _linear_objective(A, jnp.array([1.0, -2.0, 0.5, 3.0]))
    cfg = Config(dsgnar_steps=1, dsgnar_sketch=4, dsgnar_delta0=0.1, dsgnar_calibrate=False,
                 dsgnar_reject_overshoot=True, dsgnar_line_scan=False, seed=0, n_coll=8, n_bnd=4)
    _, hist, info = dsgnar.dsgnar_phase(obj, jnp.zeros(4), cfg, verbose=False)
    assert info["accepted"] == 0 and info["overshoot_rejections"] == 1, info
    assert hist[1]["radius"] > 0.1                       # expanded, not shrunk


def test_reweighting_moves_toward_the_lagging_condition(monkeypatch):
    """UpdateWeights: the condition with the SMALLER loss gains weight; history keeps the base loss."""
    from stationary import dsgnar
    from stationary.problem import Config
    _identity_row_sketch(monkeypatch)
    A = jnp.diag(jnp.array([3.0, 2.0, 4.0, 1.5]))
    b = jnp.array([10.0, -10.0, 0.01, 0.02])             # condition "a" big, "b" small
    obj = _linear_objective(A, b, blocks=[("a", 2), ("b", 2)])
    cfg = Config(dsgnar_steps=3, dsgnar_sketch=4, dsgnar_delta0=1e-3, dsgnar_calibrate=False,
                 dsgnar_reweight_alpha=0.05, seed=0, n_coll=8, n_bnd=4)
    _, hist, info = dsgnar.dsgnar_phase(obj, jnp.zeros(4), cfg, verbose=False)
    w = info["condition_weights"]
    assert w["b"] > w["a"], w
    assert abs(0.5 * (w["a"] + w["b"]) - 1.0) < 1e-12    # mean weight 1
    assert abs(hist[0]["loss"] - obj.loss(jnp.zeros(4))) < 1e-12
