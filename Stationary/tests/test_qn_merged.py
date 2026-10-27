"""The SSBroyden phase on the merged Crunch of 2026-10-17 (`--qn-crunch merged`, the default).

The merged `Crunch/Optimizers/minimize_backtracking.py` (the collaborators' merge, the one SBH uses)
has a different interface from the legacy one: `minimize(fun_val, fun_val_grad, x0, options)`, a
Python loop around two jitted functions, no status code, and line searches that never fail (Armijo
stops at alphamin and takes that step).  What the phase must do with it:

* hand it the loss and the loss+gradient of the SAME objective;
* carry the block's inverse Hessian whenever the block lowered the loss -- the legacy rule
  (`status == 0`, i.e. converged) dropped it after every normal block, so the next one restarted
  from the unscaled identity;
* after a block that did not lower the loss, restart from the identity with initial_scale, and
  stop after `plateau_patience` such blocks (the stall guard);
* never retrace: batch and weights are arguments of the jitted functions, so neither the next block
  nor a resample compiles anything new.
"""
from __future__ import annotations

import re
import types

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from stationary import train as T
from stationary.train import parse_args

TINY = ["--arch", "sym_hybrid", "--n-coll", "32", "--n-bnd", "8",
        "--width", "4", "--depth", "2", "--no-figures", "--qn-crunch", "merged"]
MARK = 0.25            # the H a successful fake block returns


def _fake_run(tmp_path, monkeypatch, improve, nblocks=4, patience=3):
    calls = []
    state = {"f": None}

    def fake_minimize(fun_val, fun_val_grad, x0, options):
        v = float(fun_val(x0))
        vg, g = fun_val_grad(x0)
        calls.append({"H": np.asarray(options["initial_H"]), "scale": options["initial_scale"],
                      "v": v, "vg": float(vg), "g": np.asarray(g), "opts": dict(options)})
        state["f"] = (v if state["f"] is None else state["f"]) * (0.5 if improve else 1.0)
        n = int(np.asarray(x0).size)
        return types.SimpleNamespace(x=x0, fun=state["f"], nit=jnp.array(options["maxiter"]),
                                     hess_inv=MARK * jnp.eye(n),
                                     stats={"n_update": 1, "n_skip": 0, "n_first": 0})

    monkeypatch.setattr(T, "_crunch_minimize", lambda *a, **k: (fake_minimize, "test-fake"))
    cfg = parse_args(TINY + ["--outdir", str(tmp_path / "run"), "--steps", "0",
                            "--lbfgs-steps", str(10 * nblocks), "--qn-block", "10",
                            "--plateau-patience", str(patience), "--plateau-min-iters", "-1"])
    T.train(cfg, verbose=False)
    return calls


def test_the_two_callables_are_the_same_objective(tmp_path, monkeypatch):
    calls = _fake_run(tmp_path, monkeypatch, improve=True, nblocks=1)
    c = calls[0]
    assert np.isclose(c["v"], c["vg"], rtol=1e-12), "fun_val and fun_val_grad disagree"
    assert c["g"].shape == c["H"].shape[:1] and np.all(np.isfinite(c["g"]))
    o = c["opts"]
    assert o["update_method"] == "ssbroyden2" and o["scale_invariant"] is True
    assert o["line_search"] in ("armijo", "wolfe") and o["max_iter"] == 100


def test_a_block_that_lowered_the_loss_carries_its_hessian(tmp_path, monkeypatch):
    calls = _fake_run(tmp_path, monkeypatch, improve=True)
    assert len(calls) == 4
    assert calls[0]["scale"] is True and np.allclose(calls[0]["H"], np.eye(len(calls[0]["H"])))
    for c in calls[1:]:
        assert np.allclose(c["H"], MARK * np.eye(len(c["H"]))), \
            "a block that ran out of iterations (status 1) did not hand its H to the next"
        assert c["scale"] is False


def test_a_block_that_did_not_move_restarts_from_the_scaled_identity_and_stalls(tmp_path,
                                                                                 monkeypatch):
    calls = _fake_run(tmp_path, monkeypatch, improve=False, nblocks=10, patience=3)
    assert len(calls) == 3, f"{len(calls)} blocks ran; the stall guard should stop after 3"
    for c in calls[1:]:
        assert np.allclose(c["H"], np.eye(len(c["H"]))), "the no-progress block's H was carried"
        assert c["scale"] is True, "the retry must rescale the identity's first step"


def test_the_real_merged_crunch_runs_and_never_retraces(tmp_path, capsys):
    minimize, where = T._crunch_minimize(linesearch="armijo", impl="merged")
    if minimize is None:
        pytest.skip(f"no merged Crunch here: {where}")
    cfg = parse_args(TINY + ["--outdir", str(tmp_path / "run"), "--steps", "5",
                            "--lbfgs-steps", "40", "--qn-block", "10", "--resample-every", "10",
                            "--plateau-min-iters", "-1", "--log-resources"])
    T.train(cfg, verbose=True)
    out = capsys.readouterr().out
    traces = [int(t) for t in re.findall(r"resources: traces=(\d+)", out)]
    assert len(traces) == 4, out[-2000:]
    assert len(set(traces)) == 1, f"the objective was retraced across blocks/resamples: {traces}"
    assert "[resample" in out, "the run did not resample, so the test proves nothing"
    m = re.search(r"SSBroyden: loss (\S+) -> (\S+) in", out)
    assert m and float(m.group(2)) < float(m.group(1)), "the merged phase did not lower the loss"


def test_a_non_finite_block_is_rejected(tmp_path, monkeypatch):
    """qp_nocom_w1 (2026-10-26): one block came back NaN (a strong-Wolfe step into overflow) and
    the phase carried the NaN state to the end.  A non-finite block must leave the parameters
    where they were, restart from the scaled identity, and not end the run NaN."""
    calls = []

    def fake_minimize(fun_val, fun_val_grad, x0, options):
        calls.append({"x": np.asarray(x0), "H": np.asarray(options["initial_H"]),
                      "scale": options["initial_scale"]})
        n = int(np.asarray(x0).size)
        if len(calls) == 2:                                   # the second block blows up
            return types.SimpleNamespace(x=x0 * jnp.nan, fun=jnp.nan,
                                         nit=jnp.array(options["maxiter"]),
                                         hess_inv=jnp.eye(n), stats={"n_update": 0, "n_skip": 0, "n_first": 0})
        v = float(fun_val(x0))
        return types.SimpleNamespace(x=x0 * 0.999, fun=0.5 * v, nit=jnp.array(options["maxiter"]),
                                     hess_inv=MARK * jnp.eye(n),
                                     stats={"n_update": 1, "n_skip": 0, "n_first": 0})

    monkeypatch.setattr(T, "_crunch_minimize", lambda *a, **k: (fake_minimize, "test-fake"))
    cfg = parse_args(TINY + ["--outdir", str(tmp_path / "run"), "--steps", "0",
                            "--lbfgs-steps", "40", "--qn-block", "10", "--plateau-min-iters", "-1",
                            "--resample-every", "0"])
    T.train(cfg, verbose=False)
    assert len(calls) == 4
    assert np.allclose(calls[2]["x"], calls[1]["x"]), "the NaN block's result was kept"
    assert np.allclose(calls[2]["H"], np.eye(len(calls[2]["H"]))) and calls[2]["scale"] is True
    import json
    hist = json.load(open(tmp_path / "run" / "history.json"))
    assert all(np.isfinite(r["loss"]) for r in hist), "a NaN loss reached the history"
