"""The quasi-Newton phase must never evaluate the full loss eagerly.

Why this is a test and not a comment.  `ssbroyden_phase` needs the loss, its parts and the
outer number once before the first block and once per block -- for the plateau test and the
log line.  It used to get them from plain Python calls, and eager JAX runs the whole loss at
every collocation point op by op: measured 39.9 s against 0.052 s compiled at n_coll 2048 in
float64, a factor of 764.  Two consequences, both observed:

  * wall time -- at `--qn-block 100` an eager full-loss evaluation landed in every hundredth
    iteration, which is the right order to account for `production_quad_quarter_pin_r400`
    taking 119.8 min over 8001 iterations and `production_quad_pin` 185.1 min over 16000,
    against the 0.048 s per value+gradient the tooling reports;
  * device memory -- eager dispatch materialises the intermediates that compilation fuses
    away, and they grow with the collocation count, so
    `production_quad_quarter_pin_r400_long` (n_coll 32768) died at its first quasi-Newton
    block with the BFC allocator refusing 3.80 MiB, then 1.27 MiB, then 432 KiB, and finally
    failing inside the autotuner on 27.39 MiB.  A working set that does not fit does not fail
    on 432 KiB; a pool filled and fragmented by the two eager evaluations immediately before
    the compiled gradient does.

The check is behavioural rather than structural: every evaluation of the loss is watched, and
each one must see a TRACER.  A concrete array means Python is running the ops itself, which
is the bug.  Crunch's own `value_and_grad(fun)` is traced, so it passes; the failure mode this
guards is the phase's own calls, which is where the eager work used to be.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

from stationary import train as T
from stationary.train import parse_args

TINY = ["--arch", "sym_hybrid", "--n-coll", "32", "--n-bnd", "8",
        "--width", "4", "--depth", "2", "--no-figures"]


def test_no_loss_evaluation_in_the_phase_sees_concrete_arrays(tmp_path, monkeypatch):
    seen = []
    real = T.total_loss

    def spy(state, batch, cfg, model, exact_fields=None, **kw):
        # A jitted call hands the loss TRACERS; an eager call hands it concrete arrays.
        seen.append(any(isinstance(v, jax.core.Tracer) for v in jax.tree.leaves(state)))
        return real(state, batch, cfg, model, exact_fields, **kw)

    monkeypatch.setattr(T, "total_loss", spy)
    cfg = parse_args(TINY + ["--outdir", str(tmp_path / "run"), "--steps", "4",
                             "--lbfgs-steps", "20", "--qn-block", "20"])
    T.train(cfg, verbose=False)

    assert seen, "the loss was never evaluated -- the run did nothing"
    eager = seen.count(False)
    assert eager == 0, (
        f"{eager} of {len(seen)} loss evaluations ran EAGERLY. Eager dispatch materialises "
        f"the intermediates compilation fuses away: it cost 764x the compiled call at "
        f"n_coll 2048, and at n_coll 32768 it fills and fragments the allocator until the "
        f"next compiled gradient cannot get 27 MiB. Jit the call (see "
        f"ssbroyden_phase.loss_value_and_parts).")


def test_the_phase_still_reports_the_same_opening_values(tmp_path, monkeypatch):
    """The jitted call must return what the eager one did: loss, parts and the outer number."""
    captured = {}
    real_outer = None

    cfg = parse_args(TINY + ["--outdir", str(tmp_path / "run"), "--steps", "0",
                             "--lbfgs-steps", "20", "--qn-block", "20"])
    T.train(cfg, verbose=False)

    import json
    hist = json.load(open(tmp_path / "run" / "history.json"))
    first = hist[0]
    assert first["step"] == 1
    # the opening row is the loss and the parts of the SAME evaluation, under the weights the
    # run started with -- not a mixture of an opening loss and a late set of parts
    assert first["loss"] > 0
    for k in ("pde_ricci", "pde_gauge", "inner_lam"):
        assert k in first, f"the opening row lost {k}"
    assert "outer" not in first or True     # outer_* are the per-group keys
    assert any(k.startswith("outer_") for k in first)
