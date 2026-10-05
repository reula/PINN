# Where the work stands

Read `COMMANDS.md` for the environment contract and the units convention.  This file is the
handover: what is committed, what is in flight, and what to do first.

## First action: post-process, do not retrain

`runs/pq_c200_vac` is finished (loss 2.0905e-14, params.pkl written).  Everything we want from it
can be obtained by re-running the post-processing, which needs no training:

    git pull
    JAX_PLATFORMS=cpu POST_THREADS=16 JAX_CACHE=0 PY=$PWD/.venv/bin/python \
      ./run_hub.sh --post --outdir runs/pq_c200_vac \
                    --only evaluate,profile,report,vtk,plane
    grep -c 'finished in' logs/pq_c200_vac.post.log      # want 5

`--only` is not optional here: `run_hub.sh --post` defaults to
`ONLY="evaluate,profile,report"`, so without it the count is 3 and there is no plane/.""

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

* FIXED: the plateau message's stray `)` -- and the clause it came from, which claimed the rule
  used `plateau_tol`.  It has used a 1% improvement since 293f328; the message says so now.
* FIXED: `--plateau-min-iters` can DISABLE the stop with a NEGATIVE value, in both the
  quasi-Newton and the Adam rule.  Before, the only way to reach the cap was a value above it
  (30001 for a 30000 run), and passing exactly the cap stopped one iteration short -- a plateau
  and an edge artefact looking the same in the log.

## Still open

* the `)` that the plateau message prints stray (`tol 0.0001 of the best);`)
* `vtk.py` converts the COORDINATES to physical units but leaves the curvature FIELDS (which
  have units of length^-4) in chart units, while `plane.py` divides them by factor^4.  One of the
  two is wrong; the fields written by the geometry layer are the ones that would move.
* `report.py`'s geometry block still samples through the CARTESIAN route with an axis clamp,
  while the figures use the chart route with an orthonormal frame.  The two agree off axis; near
  it the report keeps the clamped values, so the report and the picture are not the same numbers.
* the twin is PREPARED but not relaunched, and its warm-up question is open (see the section
  above): the stalled directory `runs/pq_c100_vac` should not be reused -- a fresh `--outdir`.
* nothing on the hub has been post-processed with the new code: the `--only` command above is
  still to run, for pq_c200_vac's five steps.

## Prepared: the same problem at rho in [1, 100]

The twin of pq_c200_vac in the physical chart, one outer radius closer in.  Every flag below was
derived from `runs/pq_c200_vac/config.json` -- the 28 fields that run changed from `Config()`
defaults -- and then checked by building the Config and diffing it against that config with only
the geometry rescaled: **zero fields differ** except `lam0_auto` (see below).

    # ON THE HUB NODE, in the JupyterLab terminal
    cd ~/serafin/Julia/PINN/Stationary
    git pull
    nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader    # want 12288 MiB

    export JAX_COMPILATION_CACHE_DIR=/tmp/jaxcache-$USER
    JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false PY=$PWD/.venv/bin/python \
      ./run_hub.sh --outdir runs/pq_c100_vac \
        --arch axisym_hybrid --steps 500 --lbfgs-steps 30000 --qn-block 250 \
        --n-coll 32768 --n-bnd 2048 --n-bnd-outer 2048 --ckpt-every 500 \
        --R0 0.5773502691896258 --rho-in 1.0 --rho-out 100.0 --inner-radius 1.0 \
        --vtk-physical-inner 1.0 \
        --lam0 0.33333333333333337 --lam-inf 1.0 --lam-bc-S2 -0.08333333333333333 \
        --outer-bc robin --robin-orders h=3,lam=3 --no-robin-G \
        --eq-weights compat=0,ricci=1,gauge=1,lam_eq=1,inner_h=0,outer_h=0 \
        --ricci-lam-source 0 --pin-lam-robin --ref-solution --ref-asymptotic 1.0 \
        --reweight-every 0 --decay-feature --log-resources --vtk

IDENTICAL means: the same INNER boundary conditions -- lambda_0 = 1/3, S_1 = 0, S_2 = -1/12 on
the same physical inner sphere, the same R0-relative geometry, the same exact reference, the same
Robin orders and exponents -- and the ONLY physical change is the outer boundary, at physical
rho = 100 instead of 200.  Everything else in the flag list is pq_c200_vac's own configuration.

Equivalently, in pq_c200_vac's own chart: rho_out = 0.5 with every other number untouched.  That
is what was checked -- two Configs, the old chart with rho_out = 0.5 and the physical chart
[1, 100], agree in EVERY field to 1e-12 once the 200x relabelling of the four length fields
(R0, rho_in, rho_out, inner_radius) is undone.  The flags exist because the outputs are drawn in
the chart: the old run was chart [0.005, 1] with R0 = 0.002886751345948129, and that chart is
200x the units its figures are in, so R0 becomes 1/sqrt(3) = 0.5773502691896258 and the inner
sphere becomes 1.

`--lam0` is pinned to the old run's value because the auto value at these scales comes out
0.3333333253563414, i.e. 1/3 to 8e-9: the imposed inner data are the thing being compared
between the two runs, so they are made equal.  The pin only sticks because the CLI clears
`lam0_auto` when `--lam0` is given (train.py, the block at `if a.lam0 is not None`) -- a Config
built with `lam0` but `lam0_auto` left true has it recomputed and the pin is silently lost, which
is what a check that went through `Config(**cfg)` rather than the CLI does.  The only
consequence is `lam0_auto = False` in the new config, which records provenance rather than data;
drop the flag if the auto path is preferred.

Run it on the GPU: `--qn-block 250` is the GPU choice (COMMANDS.md section 5 -- CPU costs +2534
mapped regions per block, so ~28 blocks of this size would exceed the 65530 limit and die of it,
which is also the evidence that pq_c200_vac itself was a GPU run).

Post-process it exactly as the other one:

    JAX_PLATFORMS=cpu POST_THREADS=16 JAX_CACHE=0 PY=$PWD/.venv/bin/python \
      ./run_hub.sh --post --outdir runs/pq_c100_vac --only evaluate,profile,report,vtk,plane

With rho_in = 1 and vtk_physical_inner = 1 the factor is 1, so the figures and the S_lm table
come out in [1, 100] directly, needing no rescaling -- and S_20's inner-data value to compare
against is the SAME -0.1321 as before, since the imposed angular data are unchanged.

## pq_c100_vac stalled in its quasi-Newton phase: where it is NOT

The twin was launched and its quasi-Newton phase did nothing: 109+ blocks, every one `status 3`,
TWO distinct loss values in the whole log, the recorded step frozen at 549 (500 Adam + 49) so the
plateau gate could never fire, and RSS climbing +268 MB per block to 32 GB.  Bisected before
blaming the new geometry -- the outer Robin residual, split into its two terms, on four states:

| state | rho_phys | h term | lam term |
|---|---|---|---|
| stalled pq_c100_vac | 100 | 4.071e+02 | 9.7e-06 |
| its exact reference | 100 | 4.3e-21 | 6.0e-16 |
| converged pq_c200_vac, at its own rho_out | 200 | 9.7e-10 | 1.0e-17 |
| converged pq_c200_vac, at physical rho = 100 | 100 | 7.0e-10 | 8.7e-09 |

So the condition is FINE at rho_out = 100: the exact solution satisfies it to 6e-16 and a
converged network of the same problem satisfies it at that radius to 9e-9.  The whole 4.07e2 is
the metric term, and it is a property of the post-Adam STATE -- 27x worse than the twin's at the
same stage.  Nothing about the chart, the Robin orders or the outer radius is implicated.

The phase's own failure mode is what turned one bad state into 120 wasted blocks:

* block 1 runs 49 iterations, the line search fails (`status 3`) and Crunch returns the INPUT
  state, so the loss is bit-identical from then on;
* `if res.hess_inv is not None: H = res.hess_inv` then accepts the FAILED call's Hessian with no
  status guard, and `initial_scale` is engaged only for `b == 0`, so every later block takes ZERO
  iterations (the "49 iterations" printed on each line is the cumulative counter, which is why it
  never moves);
* the plateau test compares the loss, which cannot change, behind a `plateau_min_iters` gate that
  the frozen counter never passes -- so the run grinds out all ceil(30000/250) = 120 blocks;
* the first-block warning watches the MAPS budget (never at risk: 3556/65530) while what grows is
  RSS, +268 MB per block, the GPU figure in COMMANDS.md section 5.  A stalled phase therefore eats
  the node before the warning speaks.

FIXED (train.py, tests/test_qn_stall.py): `hess_inv` is refused when `status != 0`, `initial_scale`
is re-engaged after a failed block, a stall stops the phase after `plateau_patience` blocks that
failed AND changed nothing (ungated -- a stalled phase is not warm-up), the best field is restored
on that stop as on a plateau one, the summary prints `stop=cap|plateau|converged|stall` and a
block-based `stopped=` instead of the `total < lbfgs_steps` that lies when blocks return nit = 0,
and the first-block warning projects RSS against MemTotal as well as maps against the kernel's
limit.  Still open for this chart: WHY the same 500 Adam steps leave the metric 27x worse -- the
loss balance the Adam phase sees, not the condition it aims at.  A warm-up-only run measures it:
`--steps 5000 --lbfgs-steps 0` on the same flags (that path is exercised now, see the `b = -1`
guard) and compare `outer_h` with the 4.07e+02 left at 500 steps.

## What the experiment found

`pq_c200_vac` (`--ricci-lam-source 0`, metric boundary data off) relaxed to a near-vacuum metric --
max |dh| at rho_out 7.3e-06 against 2.0e-03 in the sourced run, Hawking mass flat to 0.13% against
a 20-40% drift, M_lapse and m_H agreeing -- and its dipole fell from 1.86e-04 to 2.26e-06.  So the
dipole is predominantly lambda responding to the metric's error through the Ricci source, not
something generated in lambda's own sector.  A residual dipole of 2.3e-06 survives with fitted
power +1.13 against -2, i.e. still a non-decaying kernel mode of the truncated conditions, and
11x above l=3.  That is the next question.
