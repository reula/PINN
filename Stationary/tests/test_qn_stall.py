"""A quasi-Newton phase that cannot move must say so and stop.

pq_c100_vac ran 120 blocks -- every one `status 3`, the loss bit-identical at 2.213840e-02, the
recorded step frozen at 500 Adam + 49 -- while RSS grew 268 MB per block to 32 GB.  Three
defects made that possible, and this file pins the two that live in the phase's loop:

* `if res.hess_inv is not None: H = res.hess_inv` accepted the estimate from a FAILED call, so
  every later block started from it, took zero iterations and failed again;
* `initial_scale` was engaged only for `b == 0`, i.e. never for the retry that needs it;
* nothing counted the failed-and-unchanged blocks, and the plateau rule could not: its gate is
  `total >= plateau_min_iters`, and `total` is the counter a zero-iteration block never moves.

The minimiser here always fails and always returns the input state, which is what Crunch does
when its line search fails.  The plan is ten blocks; a phase that cannot move must use three
(`plateau_patience`) and stop.
"""
from __future__ import annotations

import types

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from stationary import train as T
from stationary.train import parse_args

TINY = ["--arch", "sym_hybrid", "--n-coll", "32", "--n-bnd", "8",
        "--width", "4", "--depth", "2", "--no-figures"]
POISON = 99.0          # a hess_inv that must NOT reach the next block


def _stalled_run(tmp_path, monkeypatch, patience=3, lbfgs_steps=100, qn_block=10):
    calls = []

    def fake_minimize(fun, x0, args=(), method=None, options=None):
        calls.append({"H": np.asarray(options["initial_H"]),
                      "scale": options["initial_scale"]})
        v = fun(x0)
        loss = float(v[0]) if isinstance(v, (tuple, list)) else float(v)
        n = int(np.asarray(x0).size)
        return types.SimpleNamespace(x=x0, fun=loss, nit=0, status=3,
                                     hess_inv=jnp.full((n, n), POISON))

    monkeypatch.setattr(T, "_crunch_minimize", lambda *a, **k: (fake_minimize, "test-fake"))
    cfg = parse_args(TINY + ["--outdir", str(tmp_path / "run"), "--steps", "0",
                            "--lbfgs-steps", str(lbfgs_steps), "--qn-block", str(qn_block),
                            "--plateau-patience", str(patience)])
    T.train(cfg, verbose=False)
    return cfg, calls


def test_a_stalled_phase_stops_instead_of_grinding_through_every_block(tmp_path, monkeypatch):
    # plateau_min_iters is left at its default (100), and `total` never advances from 0, so the
    # PLATEAU rule cannot fire however many blocks run: the three calls can only be the stall
    # guard.  Without it this would be ceil(100/10) = 10 calls.
    _cfg, calls = _stalled_run(tmp_path, monkeypatch, patience=3)
    assert len(calls) == 3, f"{len(calls)} blocks ran instead of stopping after 3"


def test_a_failed_block_does_not_donate_its_hessian(tmp_path, monkeypatch):
    _cfg, calls = _stalled_run(tmp_path, monkeypatch)
    assert len(calls) >= 2
    assert not np.any(np.isclose(calls[1]["H"], POISON)), \
        "the failed call's hess_inv was accepted and became the next block's initial_H"
    assert np.allclose(calls[1]["H"], calls[0]["H"]), \
        "H changed despite the block having failed"


def test_the_retry_re_engages_initial_scale(tmp_path, monkeypatch):
    _cfg, calls = _stalled_run(tmp_path, monkeypatch)
    assert len(calls) >= 2
    assert calls[0]["scale"] is True, "block 1 always scales"
    assert calls[1]["scale"] is True, \
        "the block after a failure must scale too: that is the only rescaling of the first step"


def test_a_healthy_phase_is_not_stopped_by_the_guard(tmp_path, monkeypatch):
    """A block that IMPROVES the loss is not a failure, even when it reports status 3.

    pq_c200_vac converged exactly this way: its final blocks reported status 3 (a line search
    cannot improve a loss already at 1e-14) while the loss kept falling, and it was the plateau
    rule that stopped it.  The guard counts only failures that changed NOTHING.
    """
    calls = []
    state = {"f": 1.0}

    def fake_minimize(fun, x0, args=(), method=None, options=None):
        calls.append(1)
        state["f"] *= 0.5                      # a real improvement every block
        return types.SimpleNamespace(x=x0, fun=state["f"], nit=5, status=3, hess_inv=None)

    monkeypatch.setattr(T, "_crunch_minimize", lambda *a, **k: (fake_minimize, "test-fake"))
    cfg = parse_args(TINY + ["--outdir", str(tmp_path / "run"), "--steps", "0",
                            "--lbfgs-steps", "40", "--qn-block", "10",
                            "--plateau-patience", "3"])
    T.train(cfg, verbose=False)
    # four blocks planned; all four must run, because none of them stalled
    assert len(calls) == 4, f"the stall guard stopped a phase that was still improving ({len(calls)})"


def _plateau_run(tmp_path, monkeypatch, min_iters, patience=3, nblocks=10):
    """Blocks that improve by 0.1% each: never a NEW BEST (that needs 1%), never a stall.

    This is the state pq_c200_vac ended in -- a line search that cannot improve a loss already at
    the floor, so every block reports status 3 with a tiny gain -- and it is the case the plateau
    rule exists for, and the one the stall guard must NOT claim.
    """
    calls = []
    # A DECREASING reported loss with a field that hardly moves: that is what pq_c200_vac's last
    # blocks looked like (status 3, a line search that cannot do better, a tiny gain).  The loss
    # must actually fall -- a constant one IS a stall and the guard is right to stop it.
    state = {"f": None}

    def fake_minimize(fun, x0, args=(), method=None, options=None):
        calls.append(1)
        v = fun(x0)
        base = float(v[0]) if isinstance(v, (tuple, list)) else float(v)
        state["f"] = base if state["f"] is None else state["f"] * 0.999
        return types.SimpleNamespace(x=x0, fun=state["f"], nit=5, status=3, hess_inv=None)

    monkeypatch.setattr(T, "_crunch_minimize", lambda *a, **k: (fake_minimize, "test-fake"))
    cfg = parse_args(TINY + ["--outdir", str(tmp_path / f"run{min_iters}"), "--steps", "0",
                            "--lbfgs-steps", str(nblocks * 10), "--qn-block", "10",
                            "--plateau-patience", str(patience),
                            "--plateau-min-iters", str(min_iters)])
    T.train(cfg, verbose=False)
    return calls


def test_a_negative_plateau_min_iters_disables_the_stop(tmp_path, monkeypatch):
    """0 means "may plateau from the first block"; a negative value means "never plateau".

    Without the switch the only way to reach --lbfgs-steps was to pass a minimum ABOVE the cap,
    and passing exactly the cap stops one iteration short -- the defect NEXT.md recorded.
    """
    stopped_early = _plateau_run(tmp_path / "a", monkeypatch, min_iters=0)
    assert len(stopped_early) < 10, \
        f"the plateau rule should have stopped this after {3} blocks, not run {len(stopped_early)}"
    to_the_cap = _plateau_run(tmp_path / "b", monkeypatch, min_iters=-1)
    assert len(to_the_cap) == 10, \
        f"a negative --plateau-min-iters must run all 10 planned blocks, ran {len(to_the_cap)}"
