# SBH handoff, 2026-11-04 (laptop session → Mac mini session)

Written for the Claude Code session on the Mac mini that continues the SBH single-puncture work. It assumes
nothing beyond this repo and the copied notes. Details and every number live in `SBH/README.md`, section 7
(the long `sing_*` part, in chronological order); this file is the map.

## 1. How the user works (also in the memory notes copied to the mini)

1. The user launches every GPU run on the CCAD JupyterHub (one A30, 24 GB) by pasting a complete command you give
   them. You cannot launch there. You monitor over ssh on serafin, which shares the filesystem: the tree is
   `~/Julia/PINN` on serafin, `~/serafin/Julia/PINN` on the hub. The hub's processes are invisible from serafin
   (`pgrep` there sees nothing): judge from log write times. The hub container restarts without warning,
   every day or two. Everything is resumable: relaunching the same command continues.
2. Commit and push before giving a command, since the hub pulls from GitHub. Stage only your own files: other
   sessions edit `Stationary/` and `Mini.md`.
3. Document every tool, run and finding in `SBH/README.md` in the same commit, with the measured numbers;
   plots go to `SBH/plots/<date>/`. The user likes plots more than tables.
4. Comparison runs use the baseline's settings (128000 interior / 8000 end / 12800 IC / 16000 boundary points,
   SSBroyden, windows of 0.25); only the tested idea differs.
5. Number the items in replies (1., 2., 2a). Name references explicitly ("the 1D reference", the
   finite-difference evolution `reference_1d/ref_single_m0.5_dr0.005_t20.npz`). Phone notifications:
   ntfy topic in the memory notes, results only.
6. Command hygiene on the hub: never chain `git pull && cd SBH && ...; ...` (if the pull fails, the `;` part runs
   from the wrong directory). On 2026-11-04 the hub lost its GitHub SSH key for a while; the repo is public, so
   `git remote set-url origin https://github.com/reula/PINN.git` on the hub checkout fixes pulls if it recurs.

## 2. The problem and the chain of results (README section 7)

Single Brill-Lindquist puncture, m = 0.5, vacuum CCZ4 (1+log, Gamma driver, eta = 0.1), r <= 24, windows of
0.25. Graded against the 1D reference (spherical, exact for this problem) and the Misner-Sharp mass on spheres
(exact: M everywhere).

1. Baseline `long_con10` (collaborators' `EEccz4_BH_puncture.py`, 3D MLP 40x6, Theta/Z penalty 10). Sphere files:
   `eval_data/long_con10_w8.npz` (t <= 2), `long_con10_w32.npz` (t <= 8).
2. `sing_pinn.py`: template (the 1D evolution near the puncture, cutoff W) + rho N, hard initial data
   (`--hard_ic`). The 3D MLP version (`sing_bl_hard`) has a fixed error bump at the template seam r = 0.2-0.7.
   Inner networks (`--r_in`), larger r_c, rho^4 did not remove it.
3. 1D twin `sing_1d.py` (same ansatz, radial network, the 3D residual on the axis): no seam error, loss 1e-7.
   So the 3D MLP's representation was the problem.
4. Lifted radial network in 3D (`--radial_net`: an MLP of (t, r) giving 10 radial functions, lifted to the 22
   fields): reaches the 1D level (window 0 loss 4e-8). Without the template (`--no_template`) it is as good.
   The user's conclusion: for binaries the symmetry must not be imposed. Design going forward: no templates,
   exact data as fixed offset, per-hole networks near the holes plus a third network for the rest (not built).
5. To t = 8 (`sing_radial_notpl`, 3D radial, no template; `1d_T8`, 1D): both saturate at 1e-4 to 1e-3. The 1D
   twin shows the windowed scheme itself has a ceiling, and the far-field mass drifts in every run, 1D
   included (+2.5e-2 at r = 10 by t = 8 in 1D).
6. The colleagues' 3 Nov update (`docs/BH_PINN_progress_2026-11-03.pdf`, J. Masso): 6x80 net, soft excision,
   r^2-weighted Hamiltonian/momentum penalty lambda = 30: mass within 1.4 % to t = 10, chi/alpha 4e-4 at r = 1,
   but the lapse peak at t ~ 2 is 20 % low. Our runs track the lapse at r = 0.03 to <= 1e-3 throughout (dense
   sigma = 0.05 cloud), not yet told to them.
7. **Division-free ADM constraint penalty** (the user's idea): `adm_constraints.conformal_constraints_one` gives
   chi H and chi M_i with every 1/chi cancelled by hand (tested in `tools/test_adm_constraints.py`, check 4).
   `sing_1d.py --w_adm L --adm_r_power 2`: L mean(r^2 [(chi H)^2 + |chi M|^2]). Not yet in `sing_pinn.py`.
8. 1D results to t = 8 (`plots/2026-11-04/sing_1d_adm.png`, `tools/plot_1d_runs_t.py`):

| run | mass at r = 0.3 / 1 / 3 / 10 (t = 8) | worst mass, all r >= 0.2, all t | alpha rms r = 1-3 / 3-10 | chi rms r = 0.2-1 |
|---|---|---|---|---|
| `1d_T8` (control) | +2.5e-3 / +2.7e-3 / +1.9e-3 / +2.5e-2 | 3.7e-2 | 3e-4 / 4e-4 | 5e-4 |
| `1d_T8_adm30` | +2.9e-4 / +3.3e-4 / +2.9e-4 / -1.1e-4 | 1.2e-3 | 8e-4 / 1e-3 | 5e-5 |
| `1d_T8_adm100` | -3.9e-4 / -3.6e-4 / -3.5e-4 / -7.2e-4 | 1.6e-3 | 1e-3 / 2e-3 | 3e-4 |

   lambda = 30 fixes the mass (30x on the worst value) at the cost of alpha 2.5x worse in r = 1-10. Rejected or
   unhelpful: slope (Sobolev) hand-off `--ic_d1/--ic_d2` (helps t <= 2, worse later), windows of 0.5 (worse),
   R = 48 (colleagues: a larger domain only spreads the deficit). **The user rejected hard hand-overs.**

## 3. Runs on the hub (as of 2026-11-05 11:30 UTC)

0. **The mini can drive the hub (2026-11-06).** `tools/hub_run.py status` / `run "CMD" [--wait S] [--close]` /
   `close "N ..."` (run with ../Stationary/.venv/bin/python from SBH/): opens a JupyterLab terminal in the user's
   hub server through the Jupyter API and types the command, printing the terminal's output. The hub API lives at
   https://research.labs.ccad.unc.edu.ar (not jupyterhub.ccad.unc.edu.ar, which only redirects); token in
   ~/.config/ccad/hub_token on the mini (user-made, scopes servers!user access:servers!user read:users!user).
   Unlike ssh to serafin, this sees the hub's own processes (ps, nvidia-smi). --start starts a stopped server
   (a new container); never stop or restart a running one. Launch only what the user asked for, and tell them.
1. `jobs/hub/sing1d_batch.sh` runs 1D runs NPAR at a time. One 1D run needs 3.1 GB (the dense SSBroyden matrix,
   8730^2 doubles) and saturates the A30: parallel runs gain nothing, so NPAR 1-2, priority order, and never
   two batches at once (each memory pool keeps what it grows to).
2. Batch of 2026-11-05 07:25 UTC: `1d_T8_adm10`, `1d_T8_adm30_ic`, `1d_T8_adm30_icd1` completed and graded (README).
   Done 2026-11-05 15:20 UTC: `1d_T8_adm30_ic_aw` (`--adm_r0 5 --adm_q 2`) and `1d_T8_adm30_ic_rw` (`--res_r0 5
   --res_q 2`), the outer-boosted weights, single changes against `1d_T8_adm30_ic`. Neither helps the outer fields:
   aw worse everywhere (alpha 3-10 2e-3, larger negative mass drift); rw stops the mass drift (+5e-4, flat) but
   the outer alpha is back at adm30's 1e-3. README. Nothing running on the hub now.
   Done 2026-11-05 21:33 UTC: `1d_T8_adm30_ic_log25` (`--log_frac 0.25`, a quarter of the points log-uniform in r
   down to 1e-3): fails in window 0 (loss stuck at 1.3e-3, errors 1e-2, mass +7e-2, carried to t = 8). Sampling
   r < 0.01 slows the cold start: window 0 was still falling fast when its 4000 iterations ran out (not a
   plateau stop). Batch of 2026-11-06: `1d_T8_adm30_ic_logonly05` (`--log_frac 1 --log_rmin 0.05`, all points
   log-uniform on [0.05, 24]), then `1d_T8_adm30_ic_log25_it0` (`--log_frac 0.25 --iters0 12000`). README.
   The hub server stopped ~09:50 UTC 2026-11-06 (logonly05 at 24 / 32). Relaunched by the mini (tools/hub_run.py)
   as one batch: logonly05 (resume), s2_adm30_ic, 1d_T8_adm30_ic_vol, s2_1d_T8, 1d_T8_adm30_ic_log25_it0 (README:
   the colleagues' recipe, --adm_vol and --stage2). Split the same day (the user's suggestion): the hub keeps
   logonly05 and s2_adm30_ic; `1d_T8_adm30_ic_vol`, `s2_1d_T8`, `1d_T8_adm30_ic_log25_it0` went to Mendieta (jobs
   1185909-11, `jobs/mendieta/sing1d.sbatch`, submitted from the mini; outputs in Mendieta's own
   ~/Julia/PINN/SBH/runs/, rsync them back). The hub batch skips them through runs/NAME/ELSEWHERE markers on
   serafin (`jobs/hub/sing.sh`). Mendieta needed the two git-ignored reference files and runs/1d_T8's windows.
   Status 14:20 UTC 2026-11-06: logonly05 finished (32 / 32, report.npz on serafin). `s2_adm30_ic` died at once
   out of GPU memory (eager SSBroyden update, 5-7 copies of the 2.9 GB inverse Hessian); fixed by jitting the
   update in Crunch (README, same losses, 6 x 60 stage 2 now 9.3 GB). It needs relaunching on the hub (the server
   was stopped at 14:20). Mendieta: `1d_T8_adm30_ic_vol` running (window 24 / 32), `s2_1d_T8` and log25_it0
   pending; pull the fix on Mendieta before s2_1d_T8 starts (the job runs the checkout as it is).
   Later: `s2_adm30_ic` relaunched and done (loss 2.5x down, mass alike, outer alpha 2x worse; README).
   `lo_adm30_ic_w0` (hub, `--loss_orders 3`, window 0 alone, 12000 iterations; the user's loss-vs-error
   experiment): errors ~ sqrt(loss) down to held loss 1e-6, then decoupled; plot
   plots/2026-11-06/loss_vs_error_w0.png, README.
   Mendieta runs done and graded (README): adm30_ic_vol (trade: mass at r = 10 5x better, outer alpha 4-5x
   worse), s2_1d_T8 (mass 7x better than the control, outer alpha 5-8x worse), log25_it0 (dropped). Nothing
   running. Pattern: whatever lowers the mass error or the loss raises outer alpha.
   Per-window second stage built (`--s2_window_iters`, README); `1d_T8_adm30_ic_s2w` (`--w_adm 30 --w_adm_ic 30
   --s2_window_iters 2000`) sent to Mendieta (hub left free for the Stationary recipe, which the hub stop cut at
   wr_prod_w1 iteration 7500; to be resumed from the Stationary session 7692b0d6).
   2026-11-07: `1d_T8_adm30_ic_s2w` FAILED (second stage raised the held loss, damage handed on, NaN from t = 3;
   cancelled; README has the diagnosis and the fixes to try).
   Seams measured (`--eval_seams`, tools/plot_seams.py, README): jumps 5-20x below the error; they add up to the
   error only at r >= 10 (both runs) and for chi in adm30_ic at r = 1-3; icd1 has 2-4x smaller jumps and no
   smaller errors. Warm-start offset ~100x the final jump (time rescaling per window); bias-shift fix
   `--warm_shift` built; `1d_T8_adm30_ic_ws` (adm30_ic + --warm_shift, Mendieta 1185943) done: alike adm30_ic in
   errors, mass and seam jumps (README). The warm-start offset does not matter; adm30_ic stays the base.
   Launched on Mendieta (README): `1d_T8_adm30_ic_s8` (second seed), `1d_T8_adm30_ic_s2wb` (revised per-window
   second stage: fresh N2, keep best held, 3 x 30), `1d_T8_adm30_ic_cold` (`--cold_start`). Graded 2026-11-08
   (README): seed scatter 3-7x in outer alpha at t = 8 (one-seed verdicts there are noise); s2wb alike (stage 2
   gains ~7% held loss); cold start worse (mass 3x). Next cheap fix: keep the best held iterate in stage 1 too.
   The 1D reference is made by reference_1d/sph_reference.py (any dt; command in jobs/mini/fit_ref.sh); the user's
   dt 0.0625 file is the same solution and was not needed (README).
   2026-11-08: all 1D tries in one table with lessons and next candidates: TRIES_1D.md (keep it current).
   Sobolev+dt fits done (README): d_t was the missing constraint (fit converges, PINN loss 10-1e4x below the r-only
   fit), still 1e3-1e7x the trained windows'. `--zero_z` built (Theta = Z = 0); `1d_T8_zz` (Mendieta 1185965,
   `--zero_z --w_constraint 0`) queued: grade vs 1d_T8 (fields) and adm30_ic (mass).
   Then three single changes vs adm30_ic on Mendieta (README): `1d_T8_adm30_ic_kb` (--keep_best), `_6x60` (--hidden 60),
   `_mid` (--n_mid 6000: 4x the points, log-uniform on r = 0.2-24, inner clouds untouched). The user: grade on the mass
   and t <= 4 (or two seeds) before judging outer alpha at t = 8; the outer boundary is ruled out as the error source
   (no incoming front; R = 48 was alike the control: README, TRIES_1D 4); item 6 (3D) waits for a better scheme.
   Results (README, TRIES_1D 40-43): kb mass 2x better and t <= 4 fields 2-4x better at no cost (adopt: the base
   should become adm30_ic + --keep_best); 6x60 mass 2.2x better, early alpha worse, loss floor unmoved; mid loss floor
   2-3x lower, errors alike; zz alike the control to t = 2.5 (no mass gain) then NaN from window 10 (1/r, sqrt(a+b)
   near the puncture). Nothing running.
   On the mini, done: `jobs/mini/fit_ref.sh` (supervised fit to the reference, value-only and Sobolev, 1D and 3D;
   README). A network fitted to the answer has the PINN's field error (1e-4, 1D) but residuals 10^3 - 10^6 x the
   trained runs'; adding r f', r^2 f'' does not lower it (no d_t term). Residual maps on (t, r):
   `plots/2026-11-05/tr_loss_1d.png`, the residual sits at r < 0.01 where the clouds have no points.
3. Stopped on purpose, not to be finished unless asked: `1d_T8_icd12`, `1d_T8_ic100`, `1d_T8_R48`, `1d_T8_far`
   (8-9 windows), `1d_T8_icd1` (27), `1d_T8_w05` (13 / 16).

13. 2026-11-08 ~11:30 UTC, Mendieta, three jobs (the user's decisions on the four results: "1. OK" = keep_best
   adopted; "2. continue with 6x60"; "4. there is a lot to try there" = zero_z with r / chi weights and |a + b|):
   `1d_T8_adm30_ic_kb60` (adm30_ic + `--keep_best --hidden 60`, the new base candidate), `1d_T8_zz_rin` (control +
   `--zero_z --w_constraint 0 --keep_best --res_rin 0.1`), `1d_T8_zz_chi` (same with `--res_chi 1` instead). Grade
   kb60 against adm30_ic, kb and 6x60; the zz runs against the control `1d_T8` and `1d_T8_zz` (which went NaN in
   window 10). Loss plot of every grading: `tools/plot_1d_loss.py` (the user's standing request). Job ids in the
   reply / `squeue`; logs `runs/slurm_NAME_JOBID.out` on Mendieta.

14. 2026-11-08 ~13:00 UTC, Mendieta, the template again (the user's direction: "go back to the runs with a
   spherically symmetric approximation near the origin, and only looking outside that area"): `1d_T8_tpl1`
   (adm30_ic + `--keep_best --r_c 1.0 --w_power 2 --template_file reference_1d/sing_bl_m0.5_dense_r3_t8.npz`,
   BASE `--hard_ic --N_window 32 --tfinal 8`, i.e. without --no_template) and `1d_T8_tpl1_rmin` (+ `--r_min 0.5`,
   no loss points inside r = 0.5). The table `sing_bl_m0.5_dense_r3_t8.npz` (r <= 3, t <= 8, dt 0.02) is
   git-ignored: made on the mini by `reference_1d/sph_reference.py --dr 0.005 --tfinal 8 --snap_dt 0.02 --r_save
   3.0`, copied to Mendieta's reference_1d/. Grade at r >= 1 (bands 1-3, 3-10, mass at r = 1, 3, 10) against
   adm30_ic_kb / kb60; inside r_c the fields are the template's. README "Back to the template".

15. 2026-11-09: items 13 and 14 done and graded (README "Results of the five runs of 8 Nov"). Best: `1d_T8_tpl1_rmin`
   (template r_c 1 + `--r_min 0.5`): fields 2-4x below ic, held floor 60x lower, mass 3.2e-3 flat. tpl1 (points everywhere)
   fails. kb60 not adopted. zz_rin / zz_chi survive but do not fix the mass. Nothing running. Next: second seed and r_min 0.25 / 1
   of tpl1_rmin; the user's call.

16. 2026-11-09: template + Theta = Z = 0 graded (README "Template + Theta = Z = 0"). tplzz60 has the smallest fields of the study but
   the far mass drifts to 5e-2 (no ADM penalty); `--res_r0 1` (tplzz_r1) fixes the mass at 2.8e-3 but costs the outer fields 5-20x.
   Next: the r^2 weight on the Theta and Ghat rows only (not built). Nothing of SBH running.

## 4. Next steps

1. Seed pair graded (2026-11-04): the mass gain of lambda = 30 holds with both seeds (worst 1.2e-3 / 1.4e-3 vs
   3.7e-2 / 4.4e-2); inside r = 3 the fields are within the controls' seed spread (2-3x) or better; in r = 3-10
   both lambda = 30 runs are ~3x worse than both controls from t ~ 5 (alpha 1e-3 / 3e-3 vs 4e-4 / 8e-4 at t = 8).
   Next `1d_T8_adm10` (gentler penalty: keeps the mass, gives back the outer fields?). Grading: scp
   `runs/<name>/report.npz` and `history.csv` from serafin into `SBH/eval_data/<name>/`, then
   `python tools/plot_1d_runs_t.py OUT.png "control=eval_data/1d_T8" "name=eval_data/<name>" ...`
   and band rms from `report.npz` (`err` (n_t, n_r, 10): radial functions a, b, c, d, chi, Khat, Theta, g, alpha, B;
   `mass` m/M - 1).
2. Proposed to the user, waiting for these results before approval: put `--w_adm` (division-free, r^2) into
   `sing_pinn.py` (the same `conformal_constraints_from_derivs` on the interior stacks `u_derivs` gives; add the
   `adm` part as in `sing_1d.py`) and run the 3D version closest to the binary design: plain 3D MLP (no
   `--radial_net`), `--template bl --hard_ic --no_template --w_adm <lambda>`, baseline settings, to t = 8 with
   `--stop_rel 0.02 --stop_patience 1000` (about 15-25 GPU hours with restarts). The 3D runs need the full A30
   (10-15 GB); one at a time.
3. Known weaknesses: the plateau early stop in `sing_pinn.py` keeps a window's last state, not its best
   (`sing_1d.py` has `--keep_best` since 2026-11-08, adopted). The held loss of `sing_1d.py` is one point's residual
   in the windows where the seed's held draw lands a point at r << 0.01 (seed 7: windows 9, 16, 27; seed 8: 11, 13;
   every run, README "Loss histories of the five runs"); `--keep_best` selects on that point there. Proposed, not
   built: a held loss restricted to r >= 0.01. Never read a "held spike" as a property of a run before checking
   the other runs of the same seed.

## 5. Data and environment

1. `SBH/eval_data/` (git-ignored, 25 MB): sphere-evaluation files of every 3D run, the baseline's, and the 1D
   reports. Copied to the mini and to serafin (`~/Julia/PINN/SBH/eval_data`).
2. `SBH/reference_1d/*.npz` (git-ignored): the 1D reference and the template tables, copied to the mini; also on
   serafin. Regenerate with `reference_1d/sph_reference.py` (commands in `jobs/hub/sing.sh`).
3. Run outputs stay on serafin `~/Julia/PINN/SBH/runs/<name>/` (window_*.pkl, sing_history.csv or history.csv,
   report files). 3D sphere evaluation: `JAX_ENABLE_X64=1 python sing_pinn.py --out DIR <the run's flags>
   --eval_spheres OUT.npz`; plots: `tools/plot_sing_mass.py`, `tools/plot_sphere_errors.py`,
   `tools/plot_error_by_radius.py`, `tools/plot_T8_compare.py`.
4. Python: `../Stationary/.venv/bin/python` with `JAX_ENABLE_X64=1` (CPU on the mini; fine for evaluation and
   1D smoke tests, ~2 min per 250 iterations at 500 points).
