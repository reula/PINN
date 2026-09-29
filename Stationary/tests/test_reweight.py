"""Gradient-norm adaptive reweighting, and the fact that it now runs in EVERY optimiser.

Why this file exists: the rule `w ~ 1/||d group/d theta||` used to live inside the Adam loop,
so it ran only as often as the Adam phase had iterations.  `runs/production_quad_quarter_pin_r400`
asked for `--reweight-every 1500`, ran 500 Adam steps and 8001 quasi-Newton iterations, and
logged no `[reweight]` line at all: its whole Adam phase was shorter than one reweighting
period, and the 8000 iterations that actually did the work were invisible to the rule.

These tests pin down the two halves of the fix:

  * the SCHEDULE -- `reweight_every` counts optimiser iterations on one counter that Adam and
    the quasi-Newton phase both advance, so the points are the multiples of the period, they
    are used once each, and the quasi-Newton phase picks up strictly after the Adam phase
    rather than replaying points Adam already consumed;
  * the UPDATE -- only the four interior equation groups move, the boundary weights are data
    and are left alone, a group whose gradient is at the round-off floor is skipped rather
    than grown without bound, and no weight escapes `reweight_band` times its configured value.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

from stationary.losses import GROUP_KEYS, default_weights
from stationary.problem import Config
from stationary.train import _first_reweight_after, _reweight_crosses, _reweighted

PDES = ("compat", "ricci", "gauge", "lam_eq")


def cfg_with(**over):
    cfg = Config(R0=0.005773502691896258, rho_in=0.01, inner_radius=0.01, rho_out=1.0)
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def grads(**by_name):
    """A per-group gradient-norm vector, in GROUP_KEYS order."""
    return jnp.asarray([float(by_name.get(k, 1.0)) for k in GROUP_KEYS])


# ------------------------------------------------------------------ the schedule
def test_trigger_points_are_the_multiples_of_the_period():
    """Adam's semantics must not move: it reweights before iteration `it` iff it % period == 0."""
    cfg = cfg_with(reweight_every=1500)
    fired = [it for it in range(1, 5001) if _reweight_crosses(cfg, it - 1, it)]
    assert fired == [1500, 3000, 4500]


def test_quasi_newton_picks_the_schedule_up_after_adam():
    """The first quasi-Newton point is strictly after the Adam phase: no point is used twice."""
    for steps, period in ((500, 1500), (0, 1500), (1500, 1500), (2000, 2000), (7, 3)):
        cfg = cfg_with(steps=steps, reweight_every=period)
        consumed_by_adam = {it for it in range(1, steps + 1) if _reweight_crosses(cfg, it - 1, it)}
        first_qn = _first_reweight_after(cfg, steps)
        assert first_qn is not None and first_qn > steps
        assert first_qn not in consumed_by_adam
        assert first_qn % period == 0


def test_the_two_phases_together_use_each_point_once():
    """Continuity: Adam then the quasi-Newton counter, with `qn_block` not dividing the period."""
    period, steps, block, n_blocks = 1500, 500, 100, 40
    cfg = cfg_with(steps=steps, reweight_every=period)

    used = [it for it in range(1, steps + 1) if _reweight_crosses(cfg, it - 1, it)]
    total, nxt = 0, _first_reweight_after(cfg, steps)
    while total < block * n_blocks:                       # mimic the block loop
        while nxt is not None and steps + total + block >= nxt:
            used.append(nxt)
            nxt += period
        total += block
    assert used == [1500, 3000, 4500]                     # the first is Adam's, the rest are QN's


def test_period_zero_disables_it_everywhere():
    cfg = cfg_with(reweight_every=0)
    assert _first_reweight_after(cfg, 0) is None
    assert _first_reweight_after(cfg, 1234) is None
    assert not any(_reweight_crosses(cfg, it - 1, it) for it in range(1, 200))


# -------------------------------------------------------------------- the update
def test_boundary_weights_are_data_and_never_move():
    cfg = cfg_with(reweight_every=10)
    w = default_weights(cfg)
    new = _reweighted(cfg, w, grads(compat=1e-3, ricci=1e0, gauge=1e-2, lam_eq=1e-1),
                      10, verbose=False)
    assert float(new["inner"]) == float(w["inner"])
    assert float(new["outer"]) == float(w["outer"])


def test_a_dead_group_is_left_alone_and_the_live_ones_are_balanced():
    """`compat` is identically satisfied in the metric-only schemes (gradient ~1e-36).

    The rule is w ~ 1/||grad||, so the group with the LARGEST gradient is driven DOWN and the
    smallest live one comes UP -- which is exactly why the boundary terms are excluded from it.
    """
    cfg = cfg_with(reweight_every=10)
    w = default_weights(cfg)
    new = _reweighted(cfg, w, grads(compat=1e-36, ricci=1e-3, gauge=1e-4, lam_eq=1e-5),
                      10, verbose=False)
    assert float(new["compat"]) == float(w["compat"])      # at the floor: untouched
    assert float(new["ricci"]) < float(w["ricci"])         # largest gradient: driven down
    assert float(new["lam_eq"]) > float(w["lam_eq"])       # smallest live: brought up


def test_the_cumulative_band_is_respected():
    """Repeated reweights against a lopsided gradient cannot walk a weight out of the band."""
    cfg = cfg_with(reweight_every=10, reweight_band=4.0)
    w = default_weights(cfg)
    g = grads(compat=1.0, ricci=1e6, gauge=1.0, lam_eq=1.0)
    for i in range(1, 51):
        w = _reweighted(cfg, w, g, 10 * i, verbose=False)
    for k in PDES:
        assert float(w[k]) <= 4.0 * float(default_weights(cfg)[k]) + 1e-12
        assert float(w[k]) >= float(default_weights(cfg)[k]) / 4.0 - 1e-12


def test_an_all_floor_gradient_leaves_the_weights_alone():
    """A converged state can underflow all four gradients to zero; that must not raise."""
    cfg = cfg_with(reweight_every=10)
    w = default_weights(cfg)
    new = _reweighted(cfg, w, grads(compat=0.0, ricci=0.0, gauge=0.0, lam_eq=0.0),
                      10, verbose=False)
    assert all(jnp.isfinite(v) and float(v) > 0 for v in new.values())
    assert {k: float(v) for k, v in new.items()} == {k: float(v) for k, v in w.items()}


# ------------------------------------------------- the regression the fix is for
def test_reweighting_fires_with_zero_adam_steps(tmp_path, capsys):
    """`--steps 0 --reweight-every N` must reweight in the quasi-Newton phase.

    This is the reported failure: with the rule inside the Adam loop a `--steps 0` run
    reweighted zero times however long it ran.  `--qn-method lbfgs` keeps the test
    independent of the Crunch checkout (present in this repo, absent on some machines),
    and both quasi-Newton phases reweight on the same schedule.
    """
    from stationary.train import parse_args, train

    cfg = parse_args([
        "--outdir", str(tmp_path / "rw"), "--arch", "sym_hybrid",
        "--steps", "0", "--lbfgs-steps", "12", "--qn-method", "lbfgs",
        "--reweight-every", "5", "--n-coll", "32", "--n-bnd", "8",
        "--width", "4", "--depth", "2", "--no-figures",
    ])
    train(cfg, verbose=True)
    out = capsys.readouterr().out
    fired = [ln for ln in out.splitlines() if ln.startswith("[reweight ")]
    assert fired, f"no reweighting in a --steps 0 run:\n{out[-2000:]}"
    assert "[reweight 5]" in out and "[reweight 10]" in out
