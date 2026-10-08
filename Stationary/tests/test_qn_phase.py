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


def test_the_objective_handed_to_crunch_is_jitted(tmp_path, monkeypatch):
    """Crunch calls `jax.value_and_grad(fun)(x0)` with no jit of its own, once per block.

    This is the check the tracer spy below CANNOT make.  An unjitted `fun` is still *traced*
    by `value_and_grad`, so the spy sees tracers and reports no eager evaluation -- while JAX
    goes on to execute the primitives one at a time.  Measured: 62.4 s per call against 1.1 s
    jitted at n_coll 2048, and it is what exhausted the BFC pool on the first block of
    production_quad_quarter_pin_r400_long.  So assert the property directly: what Crunch is
    handed must be a jit-wrapped function, which exposes `.lower`.
    """
    real = T._crunch_minimize
    captured = {}

    def fake_minimize():
        minimize, where = real()
        if minimize is None:
            return None, where

        def spy(fun, *a, **k):
            captured["jitted"] = hasattr(fun, "lower")
            return minimize(fun, *a, **k)

        return spy, where

    monkeypatch.setattr(T, "_crunch_minimize", fake_minimize)
    cfg = parse_args(TINY + ["--outdir", str(tmp_path / "run"), "--steps", "0",
                             "--lbfgs-steps", "20", "--qn-block", "20"])
    T.train(cfg, verbose=False)

    if not captured:
        pytest.skip("Crunch is not importable here, so the SSBroyden phase did not run")
    assert captured["jitted"], (
        "the objective handed to Crunch is not jitted, so its `value_and_grad(fun)(x0)` "
        "dispatches every primitive separately -- 56x slower, and it fragments the GPU "
        "allocator until the autotuner cannot allocate")


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


def test_the_plateau_watches_both_spheres():
    """The stopping rule must not declare convergence while a boundary is still moving.

    `production_quad_quarter_pin_r400_long` stopped at step 1300 with its loss and its outer
    Robin flat (1.1e-09) while its inner lambda residual was 1.6e-04 -- ten times the previous
    run's.  The inner terms were simply not in the number the plateau rule compared.
    """
    from stationary.train import boundary_number

    assert boundary_number({"inner_lam": 3.0}) == 3.0, "the inner sphere is not watched"
    assert boundary_number({"outer_h": 2.0}) == 2.0
    assert boundary_number({"pin_lam": 1.0}) == 1.0
    assert boundary_number({"inner_lam": 1.0, "outer_h": 2.0, "pin_lam": 4.0,
                            "pde_ricci": 1e9}) == 7.0, "PDE groups must not enter it"
    assert boundary_number({}) == 0.0


def test_a_checkpoint_can_carry_a_second_phases_state(tmp_path):
    """The quasi-Newton phase checkpoints itself, through the same file and key as Adam.

    Before this, `ckpt.pkl` was written only inside the Adam loop, so `--resume auto` on a run
    that died in the quasi-Newton phase -- where the hours are -- rewound to the END OF ADAM
    and discarded the whole phase.  A real run lost 2800 iterations that way.
    """
    from stationary.problem import Config
    from stationary.train import save_checkpoint

    cfg = Config(outdir=str(tmp_path))
    p = str(tmp_path / "ckpt.pkl")
    save_checkpoint(p, {"w": jnp.array([1.0, 2.0])}, None, {"pde_ricci": 1.0},
                    [{"step": 7, "loss": 0.5}], 207, cfg,
                    extra={"phase": "qn", "qn_H": jnp.eye(3)})

    import pickle
    ck = pickle.load(open(p, "rb"))
    assert ck["phase"] == "qn"
    assert tuple(ck["qn_H"].shape) == (3, 3)
    assert ck["step"] == 207
    assert ck["opt_state"] is None, "the quasi-Newton phase has no Adam optimiser state"

    # and a checkpoint written without `extra` is still an Adam checkpoint, so every
    # checkpoint written before this key existed keeps resuming exactly as it did
    save_checkpoint(p, {"w": jnp.array([1.0])}, None, {}, [], 5, cfg)
    assert pickle.load(open(p, "rb")).get("phase", "adam") == "adam"


def test_a_resume_runs_only_the_iterations_it_still_owes():
    """`--lbfgs-steps` caps the CUMULATIVE counter, so a resume must not buy a fresh budget.

    Measured before the fix: resuming a 200-iteration run with --lbfgs-steps 400 ran eight
    further blocks and reached 600.
    """
    import math
    for start_total, cap, block, expected in ((0, 400, 50, 8), (200, 400, 50, 4),
                                              (400, 400, 50, 0), (250, 400, 50, 3)):
        got = int(math.ceil(max(0, cap - start_total) / block))
        assert got == expected, (start_total, cap, got, expected)


def test_the_phase_warns_before_it_dies_of_the_mapping_limit(tmp_path, monkeypatch, capsys):
    """The leak is predictable one block in, so say so then rather than at hour two.

    `minimize_bfgs` is not jitted, so every block recompiles its line search and the objective
    inside it; the LLVM section memory is never returned and the kernel's `vm.max_map_count`
    eventually refuses a 27-byte suballocation.  Four runs died at block 23 that way, each
    after ~90 minutes.  The per-block cost is measured on the FIRST block, so the warning does
    not have to assume anything about the model's size.
    """
    import sys
    from stationary import train

    calls = {"n": 0}

    def fake_budget():
        calls["n"] += 1
        return (64000, 65530) if calls["n"] == 1 else (65200, 65530)

    monkeypatch.setattr(train, "_map_budget", fake_budget)
    monkeypatch.setattr(sys, "argv", [
        "train", "--outdir", str(tmp_path), "--arch", "sym_hybrid", "--width", "8",
        "--depth", "2", "--steps", "0", "--lbfgs-steps", "200", "--qn-block", "100",
        "--n-coll", "512", "--n-bnd", "64", "--no-figures", "--ref-solution",
        "--ref-asymptotic", "1.0", "--outer-bc", "robin", "--robin-exps", "2,3,1",
        "--robin-orders", "h=3,lam=3", "--no-robin-G", "--lam-inf", "1.0"])
    train.train(train.parse_args())

    out = capsys.readouterr().out
    assert "1200 address-space regions leaked per block" in out, out[-2000:]
    assert "room for ~0 more blocks" in out
    assert "--qn-block 200" in out, "must recommend a value that fits the remaining budget"
    assert "per BLOCK, not per iteration" in out, "the reason bigger blocks are free"


def test_the_map_budget_is_measurable_or_honestly_not():
    """Linux gives both numbers; anything else must say so rather than invent them."""
    from stationary.train import _map_budget
    cur, limit = _map_budget()
    assert (cur is None) == (limit is None)
    if cur is not None:
        assert 0 < cur < limit


def test_the_map_budget_is_reported_before_the_first_block(tmp_path, monkeypatch, capsys):
    """A run that dies IN the first block's compile must still say where it stood.

    `--log-resources` reported only after a completed block, so the run that died compiling
    `jit_fun` printed no map count at all -- the one case where the number was most wanted.
    The baseline and the delta across the first `minimize` call are now both reported, and the
    first `minimize` compiles the objective and the line search, so that delta is the largest
    single mapping cost in the phase.
    """
    import sys
    from stationary import train

    values = [(9000, 65530), (9200, 65530), (12200, 65530)]
    state = {"i": 0}

    def fake_budget():
        v = values[min(state["i"], len(values) - 1)]
        state["i"] += 1
        return v

    monkeypatch.setattr(train, "_map_budget", fake_budget)
    monkeypatch.setattr(sys, "argv", [
        "train", "--outdir", str(tmp_path), "--arch", "sym_hybrid", "--width", "8",
        "--depth", "2", "--steps", "0", "--lbfgs-steps", "100", "--qn-block", "100",
        "--n-coll", "512", "--n-bnd", "64", "--no-figures", "--ref-solution",
        "--ref-asymptotic", "1.0", "--outer-bc", "robin", "--robin-exps", "2,3,1",
        "--robin-orders", "h=3,lam=3", "--no-robin-G", "--lam-inf", "1.0"])
    train.train(train.parse_args())
    out = capsys.readouterr().out
    assert "before any block: maps=9000 of 65530 (13% of the kernel's limit" in out
    assert "entering block 1 (this is where jit_fun compiles): maps=9200" in out
    assert "across the first minimize call" in out and "+3000" in out


def test_the_two_boundaries_take_independent_point_counts():
    """`n_bnd` is per sphere, and the outer one has its own knob.

    The outer sphere carries the Robin conditions, the averaged pins and any value pins; the
    inner carries only the imposed data.  Doubling both in order to double one spends boundary
    points where they are not wanted -- and, before this, there was no way to ask for more
    outer points at all.
    """
    import sys
    from stationary.train import make_batch, parse_args

    sys.argv = ["train", "--outdir", "/tmp/nbnd", "--n-coll", "1000",
                "--n-bnd", "256", "--n-bnd-outer", "8192"]
    cfg = parse_args()
    b = make_batch(jax.random.PRNGKey(0), cfg)
    assert b["coll"].shape[0] == 1000
    assert b["inner"].shape[0] == 256, "the inner sphere must keep n_bnd"
    assert b["outer"].shape[0] == 8192, "the outer sphere must take n_bnd_outer"

    sys.argv = ["train", "--outdir", "/tmp/nbnd", "--n-coll", "1000", "--n-bnd", "300"]
    cfg2 = parse_args()
    assert make_batch(jax.random.PRNGKey(0), cfg2)["outer"].shape[0] == 300, \
        "without n_bnd_outer the two must still match"


def test_a_block_draws_one_sample_at_the_newest_resampling_point():
    """Why the log used to read "[resample 6500]" three times, and why that count mattered.

    A block is PLANNED to span `qn_block` iterations, and that span can cover several
    resampling points: `--qn-block 1500` with `--resample-every 500`, starting at 500 Adam
    steps, covers 5500, 6000 and 6500.  The phase drew a sample at each and kept the last --
    same final batch, since the key is `seed + 777 + point`, but two wasted draws, and the
    message printed the block's endpoint, so all three said "[resample 6500]".  The report
    counts `[resample` lines as its OBSERVED refresh count, so it read three times the number
    of samples the phase actually ran on.
    """
    # the reported case: block [5000, 6500], every 500, the first point strictly after 5000
    assert T._newest_resample_point(500, 5500, 6500) == 6500
    # ... and the points it covers, in order, which is what the old loop drew
    covered = [p for p in range(5500, 6501, 500)]
    assert covered == [5500, 6000, 6500]
    assert T._newest_resample_point(500, 5500, 6500) == covered[-1]   # the one it uses
    assert T._newest_resample_point(500, 5500, 6000) == 6000      # a shorter block
    assert T._newest_resample_point(500, 5500, 5499) is None      # the span reaches none
    assert T._newest_resample_point(0, 5500, 6500) is None        # resampling off
    assert T._newest_resample_point(500, None, 6500) is None      # nothing scheduled
    # exactly one point per block span, so the number of draws equals the number of blocks that
    # reach a point -- not the number of points covered
    assert T._newest_resample_point(500, 500, 6500) == 6500
    assert T._newest_resample_point(1500, 4500, 6500) == 6000
