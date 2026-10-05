# Where the work stands

Read `COMMANDS.md` for the environment contract and the units convention.  This file is the
handover: what is committed, what is in flight, and what to do first.

## First action: post-process, do not retrain

`runs/pq_c200_vac` is finished (loss 2.0905e-14, params.pkl written).  Everything we want from it
can be obtained by re-running the post-processing, which needs no training:

    git pull
    JAX_PLATFORMS=cpu POST_THREADS=16 JAX_CACHE=0 PY=$PWD/.venv/bin/python \
      ./run_hub.sh --post --outdir runs/pq_c200_vac
    grep -c 'finished in' logs/pq_c200_vac.post.log      # want 5

The step log is separate from the training log, and that count is the check that all five steps
ran -- before today, three defects in run_hub.sh made steps silently not run.

## The change in flight: radius-independent multipoles

`multipoles.py` and `report.py` report the amplitude at rho_out and a fitted power.  Both are
radius-dependent, so neither can be compared with the inner data, and the same symbol S_l means
different things in the two places:

* inner data are imposed as `lambda = lambda_0 + S_1 z/rho + S_2 (z^2-(x^2+y^2)/2)/rho^2`, i.e.
  powers `1/rho^l`;
* the far field decays as `1/rho^{l+1}`, which is what the report's "expected -(l+1)" means.

Report the radius-independent coefficient instead:

    S_l(rho) = amplitude_l(rho) * rho^(l+1)

as a table against rho, so its constancy is visible, with the inner data's converted value beside
it.  It should be constant for a correct tail and should equal the inner value.

Evidence that this matters: in pq_c200_vac, l=2 is 1.9872e-07 with fitted power -2.686 against -3
expected.  A tail anchored to the inner data cannot give an amplitude that large, and -2.69 is not
a rho^-3 tail -- so the reported quadrupole is not the physical one, and only the fitted power
shows it today.

## Verified, and NOT verified

Verified by running: `--eq-weights` (including per-field `inner_h`/`outer_h`), `--ricci-lam-source`,
`--n-bnd-outer`, the plateau decision on a fixed sample, the best-field restoration, the units of
the COMMANDS.md section.

NOT verified, and worth one look each:

* **the loss-history panel as rendered.**  1242118 gives it fixed y-limits taken from the loss's
  own range, because pq_c200_vac's groups reach 1e-21 and auto-scaling collapsed every curve into
  a vertical line at the left edge.  The code runs; the figure has not been looked at.
* **h_rr at angles.**  `profile.py` writes profiles_vs_rho.png with lambda and h_rr panels.  Both
  figures are written for every run I have tried, but never for a converged run, so I do not know
  whether the h_rr panel says what it should.

## Known defects not fixed

* the plateau message still prints a stray `)` -- `tol 0.0001 of the best);`
* `--plateau-min-iters N` delays the plateau to N but does not disable it; to run to the cap
  without any stop, N must EXCEED the cap (30001 for a 30000 run).

## What the experiment found

`pq_c200_vac` (`--ricci-lam-source 0`, metric boundary data off) relaxed to a near-vacuum metric --
max |dh| at rho_out 7.3e-06 against 2.0e-03 in the sourced run, Hawking mass flat to 0.13% against
a 20-40% drift, M_lapse and m_H agreeing -- and its dipole fell from 1.86e-04 to 2.26e-06.  So the
dipole is predominantly lambda responding to the metric's error through the Ricci source, not
something generated in lambda's own sector.  A residual dipole of 2.3e-06 survives with fitted
power +1.13 against -2, i.e. still a non-decaying kernel mode of the truncated conditions, and
11x above l=3.  That is the next question.
