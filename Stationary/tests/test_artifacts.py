"""What a run must leave on disk, whatever kills it.

The failure these pin down is not a wrong number but an absent one.  `runs/` has twice lost
work to it: `production_quad_quarter_pinrad` was refused a 4.00 GiB allocation at
quasi-Newton iteration 0 (JAX's default preallocation had taken 75% of a 12 GiB A30 slice),
and a run killed inside its first quasi-Newton block left a directory containing only
`params_adam.pkl` -- the Adam warm-up, none of the work -- because

  * `config.json` was written at the very END of train(), after the quasi-Newton phase and
    after the diagnostics block, so a run that died anywhere in between was not a "run
    directory" at all and `postprocess.sh` refused it outright (exit 1); and
  * the quasi-Newton phase had no checkpoint of its own, and with `--steps 0` there is no
    ckpt.pkl either, so nothing in it was ever on disk.

Both are fixed: `config.json` goes down before anything is computed, and each completed
quasi-Newton block writes `params.pkl`.  These tests hold that line, and hold the two
properties the writes need to be safe -- atomic, and readable back by `--init-from`.
"""
from __future__ import annotations

import json
import os
import pickle

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

from stationary import train as T
from stationary.train import parse_args, write_params

TINY = ["--arch", "sym_hybrid", "--n-coll", "32", "--n-bnd", "8",
        "--width", "4", "--depth", "2", "--no-figures"]


def test_config_json_exists_even_if_the_quasi_newton_phase_dies(tmp_path, monkeypatch):
    """The one the run directory is recognised by must not depend on finishing."""
    outdir = tmp_path / "run"

    def explode(*a, **k):
        raise RuntimeError("simulated device OOM in the quasi-Newton phase")

    monkeypatch.setattr(T, "ssbroyden_phase", explode)
    cfg = parse_args(TINY + ["--outdir", str(outdir), "--steps", "0", "--lbfgs-steps", "10"])
    with pytest.raises(RuntimeError):
        T.train(cfg, verbose=False)

    assert (outdir / "config.json").exists(), (
        "a crash in the quasi-Newton phase left no config.json, so postprocess.sh "
        "refuses the directory and NOTHING can be recovered from it")
    assert json.load(open(outdir / "config.json"))["arch"] == "sym_hybrid"


class SimulatedCrash(Exception):
    """Deliberately NOT a RuntimeError.

    `jax.errors.JaxRuntimeError` IS a subclass of RuntimeError, so `pytest.raises(RuntimeError)`
    around a call that compiles a graph silently absorbs a real device failure -- a broken
    autotune cache on the hub, say -- and the test then reports whatever the assertion after it
    trips over.  That is exactly what happened: an `Input/output error` writing the GPU
    autotune cache surfaced as `KeyError: 'params'`, pointing at this file instead of at the
    machine.  Catching a type nothing else uses, and recording what reached disk in a
    `finally`, keeps an infrastructure failure looking like one.
    """


def test_a_completed_block_writes_params_pkl(tmp_path, monkeypatch):
    """Each block must put the weights on disk, so a crash costs at most one block."""
    outdir = tmp_path / "run"
    real = T.ssbroyden_phase
    captured = {}

    def wrapped(state, batch, weights, loss_fn, cfg, verbose=True, gradnorms=None,
                history=None):
        cfg.lbfgs_steps = 1
        cfg.qn_block = 1
        try:
            real(state, batch, weights, loss_fn, cfg, verbose, gradnorms, history)
        finally:
            # In `finally` so that a failure inside the block is still reported as itself,
            # with the record of what did reach disk attached.
            captured["params"] = os.path.exists(outdir / "params.pkl")
            captured["history"] = os.path.exists(outdir / "history.json")
        # The block is done and has written; now die, as a device error would.
        raise SimulatedCrash("crash after one completed block")

    monkeypatch.setattr(T, "ssbroyden_phase", wrapped)
    cfg = parse_args(TINY + ["--outdir", str(outdir), "--steps", "0", "--lbfgs-steps", "1"])
    with pytest.raises(SimulatedCrash):
        T.train(cfg, verbose=False)
    assert captured["params"], "the completed block did not write params.pkl"
    assert captured["history"], (
        "the completed block did not write history.json, so the report of a crashed run "
        "could not say how far it got")


def test_write_params_is_atomic_and_survives_a_previous_file(tmp_path):
    """It is the only copy of the trained weights, so it must not be left half-written."""
    p = write_params(str(tmp_path), {"net": {"w": jnp.arange(4.0)}}, "params.pkl")
    first = open(p, "rb").read()
    write_params(str(tmp_path), {"net": {"w": jnp.arange(4.0) + 1.0}}, "params.pkl")
    assert not os.path.exists(p + ".tmp"), "the temp file was left behind"
    got = pickle.load(open(p, "rb"))
    assert list(got["net"]["w"]) == [1.0, 2.0, 3.0, 4.0]
    assert first != open(p, "rb").read()


def test_params_pkl_is_readable_by_init_from(tmp_path):
    """`--init-from` reads `saved['net'] if 'net' in saved else saved`; keep both working."""
    state = {"net": {"w": jnp.arange(3.0)}}
    write_params(str(tmp_path), state)
    saved = pickle.load(open(tmp_path / "params.pkl", "rb"))
    net = saved["net"] if "net" in saved else saved
    assert list(net["w"]) == [0.0, 1.0, 2.0]


def test_no_stray_temp_file_after_a_normal_run(tmp_path):
    """A successful run leaves no `*.tmp` for a later reader (or `ls`) to trip over."""
    outdir = tmp_path / "run"
    cfg = parse_args(TINY + ["--outdir", str(outdir), "--steps", "2", "--lbfgs-steps", "0"])
    T.train(cfg, verbose=False)
    assert not [f for f in os.listdir(outdir) if f.endswith(".tmp")]
    for name in ("config.json", "params.pkl", "params_adam.pkl", "history.json", "report.json"):
        assert (outdir / name).exists(), name
