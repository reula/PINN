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

**It has to run ON THE HUB NODE, in the JupyterLab terminal.**  From the login node
(`serafin`) the venv has no interpreter: `.venv/bin/python -> python3.13` is a DANGLING
symlink there (`readlink -f` resolves to nothing; the login node has only `/usr/bin/python3.6`),
so `run_hub.sh` refuses with `interpreter '.../.venv/bin/python' not found`.  The run's record
scp's back in 40 kB, though, so the figures can be reproduced locally:

    scp hub:Julia/PINN/Stationary/runs/pq_c200_vac/{{params.pkl,config.json,history.json,report.json}} runs/pq_c200_vac/
    MPLCONFIGDIR=$PWD/.mplcache JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu \
      bash postprocess.sh runs/pq_c200_vac .venv/bin/python profile,evaluate,report

Local costs (8 threads, this laptop): profile ~1 min, evaluate ~6 min, report ~30 min; `vtk`
is much slower than either because the invariant fields take a second derivative at every node.

The step log is separate from the training log, and that count is the check that all five steps
ran -- before today, three defects in run_hub.sh made steps silently not run.

## DONE: radius-independent multipoles, S_lm in front of Y_lm/rho^(l+1)

    lambda - lambda_inf = sum_lm S_lm Y_lm / rho_phys^(l+1),      S_lm constant

`multipoles.multipole_constants` returns S_lm per (l, m) at a list of radii,
`multipoles.inner_constants` the same number read off the IMPOSED inner data, and
`multipoles.multipole_table` formats them together with the exact reference through the same
estimator.  `report.py` prints it (a table against rho, physical units) and
`lambda_multipole_decay.png` plots S_lm against rho per (l, m), flat = correct power, with the
inner-data values as dashed lines.  The plots are in PHYSICAL rho (cfg.vtk_physical_inner /
rho_in = 200 for this run, so rho in [1, 200]); only the coordinate is converted, the fields are
as the run computed them, exactly like vtk.py.

What it says about pq_c200_vac (l = 2 is the one NEXT.md asked about):

| rho_phys | S_20 network | S_20 exact | inner data |
|---|---|---|---|
| 1 (inner) | -0.132111 | -0.132111 | -0.132111 |
| 2 | -0.237402 | 0 | -0.132111 |
| 10 | -0.371144 | 0 | -0.132111 |
| 200 | -1.589762 | 0 | -0.132111 |

So the inner data are satisfied EXACTLY at the inner sphere and the quadrupole constant then
grows outward (drift 91.7%, 12x the imposed value at rho_out): the l = 2 content decays like
rho^-2.15, not rho^-3.  The dipole is the same story from an imposed zero: S_10 grows from
-1.4e-09 to +9.0e-02 with fitted power +1.13.  Both are homogeneous admixtures of the kind this
file already suspected -- non-decaying modes of the truncated outer condition.

THE MONOPOLE IS THE CONTROL, and it holds: S_00 goes -2.363 -> -3.897 over the shell while the
exact solution gives -2.363 -> -4.082, i.e. the same 39% vs 42% drift (which is the finite
R0/rho of the expansion, R0/rho in [0.003, 0.58], not a wrong power), with the ratio falling
smoothly 1.0000 -> 0.9547.  The estimator reproduces the exact solution's own drift and returns
zero for every exact l >= 1, so the quadrupole's 12x is a property of the run, not of the
measurement.  Two independent quadratures agree to six digits throughout.

Verified: tests/test_multipole_constants.py (7 tests) pins the definition against a synthetic
pure tail and against reconstruction of the inner data.

Historical note -- the amplitudes the report used to print, and why they were not comparable:

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

* **the loss-history panel as rendered.**  LOOKED AT, and it was drawing NOTHING: the plotting
  loop and the legend were indented inside the `else:` branch of the history loader (the
  checkpoint fallback), so for every run that has a history.json the panel came out as a titled
  frame reading "37 rows from history.json" over an empty interior.  Fixed, and the panel is now
  `evaluate._draw_loss_history`, a function, with tests/test_loss_history_panel.py pinning both
  that it draws and that `_plots` calls it at the top level rather than in a fallback branch.
  Also found while checking the multipole figure: its decay fit sampled
  `np.geomspace(max(2.0, rho_in*1.5), rho_out, 14)`, which for this run (rho_out = 1) starts
  ABOVE the outer sphere -- so the figure's fitted power was an extrapolation.  Fixed to sample
  inside the shell.
* **h_rr at angles.**  Looked at for pq_c200_vac: profiles_vs_rho.png shows lambda departing
  angularly at the inner sphere and merging with the spherical reference outward, and h_rr flat
  against its asymptote 1.0 -- which is what it should be for a run whose metric boundary data
  are switched off (inner_h/outer_h weights 0), so the panel says what it should.

## Known defects not fixed

* the plateau message still prints a stray `)` -- `tol 0.0001 of the best);`
* `--plateau-min-iters N` delays the plateau to N but does not disable it; to run to the cap
  without any stop, N must EXCEED the cap (30001 for a 30000 run).

## Still open

* the `)` that the plateau message prints stray (`tol 0.0001 of the best);`)
* `--plateau-min-iters N` delays the plateau rather than disabling it
* `vtk.py` converts the COORDINATES to physical units but leaves the curvature FIELDS (which
  have units of length^-4) in chart units, while `plane.py` divides them by factor^4.  One of the
  two is wrong; the fields written by the geometry layer are the ones that would move.

## What the experiment found

`pq_c200_vac` (`--ricci-lam-source 0`, metric boundary data off) relaxed to a near-vacuum metric --
max |dh| at rho_out 7.3e-06 against 2.0e-03 in the sourced run, Hawking mass flat to 0.13% against
a 20-40% drift, M_lapse and m_H agreeing -- and its dipole fell from 1.86e-04 to 2.26e-06.  So the
dipole is predominantly lambda responding to the metric's error through the Ricci source, not
something generated in lambda's own sector.  A residual dipole of 2.3e-06 survives with fitted
power +1.13 against -2, i.e. still a non-decaying kernel mode of the truncated conditions, and
11x above l=3.  That is the next question.
