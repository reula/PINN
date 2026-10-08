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


def test_the_report_runs_to_the_end_and_past_the_multipoles(tmp_path):
    """A NameError in a LATER section used to truncate report.txt silently.

    `report.py` printed the section header "CHART-INDEPENDENT (geometric invariants)" and then
    died on a stale name (`power`, left behind when the amplitude-based multipole lines were
    replaced by the S_lm form), because the name was computed in an earlier block.  The report
    of every run after that change ended at that header: the multipole tables were there and
    the geometric invariants, the family fit and the plane comparison were simply absent, with
    nothing in the file to say so.  This test runs the whole pipeline -- a one-step training to
    produce a real run directory, then `python -m stationary.report` on it -- and requires the
    sections on BOTH sides of the multipoles to be present.
    """
    import os
    import subprocess
    import sys

    from stationary.train import parse_args, train

    out = tmp_path / "r"
    cfg = parse_args(["--outdir", str(out), "--arch", "sym_hybrid", "--steps", "1",
                      "--lbfgs-steps", "0", "--n-coll", "32", "--n-bnd", "8",
                      "--width", "4", "--depth", "2", "--no-figures", "--ckpt-every", "0"])
    train(cfg, verbose=False)

    proc = subprocess.run([sys.executable, "-m", "stationary.report", "--outdir", str(out)],
                          capture_output=True, text=True, cwd=os.getcwd())
    assert proc.returncode == 0, proc.stderr[-3000:]
    text = proc.stdout
    for needle in ("MULTIPOLES OF lambda", "SPURIOUS DIPOLE",
                   "CHART-INDEPENDENT (geometric invariants)",
                   "family read off the solution"):
        assert needle in text, (needle, text[-3000:])
    # ... and the section that died is not merely headed: it has content after the header
    tail = text.split("CHART-INDEPENDENT (geometric invariants)")[1]
    assert "family read off the solution" in tail
