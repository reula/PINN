"""The report must say how many points were used, and how long one batch lives.

Neither was in it.  The sampling line carried `n_coll` and `n_bnd` but not the OUTER sphere's
count, and nothing anywhere stated the refresh cadence -- how many iterations share a batch --
which is half of what a PINN's error depends on and the first thing a reader of a report asks.
And run_hub.sh post-processes a CRASHED training on purpose, so the report needs to be able to
say that its numbers come from an incomplete run: pq_c100_vac3 died of a device OOM inside block
1 and its report read exactly like a finished one's.
"""
from __future__ import annotations

import json

from stationary.problem import Config
from stationary.report import sampling_lines, unfinished


def test_sampling_provenance_names_both_spheres_and_the_refresh_cadence(tmp_path):
    cfg = Config(n_coll=27768, n_bnd=4548, n_bnd_outer=4548, resample_every=500,
                 reweight_every=0)
    log = tmp_path / "run.log"
    log.write_text("[resample 6500] new collocation sample of 27768 points\n"
                   "[resample 7000] new collocation sample of 27768 points\n")
    text = "\n".join(sampling_lines(cfg, str(log)))
    assert "n_coll 27768" in text
    assert "4548 inner + 4548 outer" in text, text
    assert "one batch for every 500 iterations" in text, text
    assert "2 resample line(s)" in text, "the OBSERVED count, not just the cadence"
    assert "reweight_every = 0" in text and "off" in text


def test_no_refresh_is_stated_plainly():
    cfg = Config(n_coll=1024, n_bnd=64, resample_every=0, reweight_every=200)
    text = "\n".join(sampling_lines(cfg, None))
    assert "whole phase (no refresh)" in text, text
    assert "reweight_every = 200" in text and "off" not in text


def test_a_crashed_training_is_bannerred(tmp_path):
    (tmp_path / "train.exit").write_text("1\n")
    assert "DID NOT FINISH" in (unfinished(str(tmp_path)) or "")
    (tmp_path / "train.exit").write_text("0\n")
    assert unfinished(str(tmp_path)) is None, "a clean exit must not be bannerred"
    (tmp_path / "train.exit").unlink()
    (tmp_path / "run.status").write_text(
        json.dumps({"train_exit": 137, "at": "2026-10-05T23:10"}))
    banner = unfinished(str(tmp_path)) or ""
    assert "137" in banner and "2026-10-05T23:10" in banner, banner
