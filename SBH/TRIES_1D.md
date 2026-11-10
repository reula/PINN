# All 1D tries so far (sing_1d.py), 1 - 8 Nov 2026

One line per run or experiment, in order. "vs" names the run each one is judged against (one change each; the
factor-2 rule: under 2x is "alike"). Field numbers are band rms of alpha at t = 8 unless said otherwise; "mass" is
the worst |m/M - 1| over r >= 0.2 and all t. Full numbers: README.md (section per run), SUMMARY_2026-11-06.md.
Settings common to every run from B on: `--hard_ic --no_template --N_window 32 --tfinal 8`, 2000 interior / 250
end / 400 IC / 2000 held-out points, 4000 iterations per window, SSBroyden, resampled every 250 iterations.

## A. Setting up the 1D twin (1 Nov; two windows to t = 0.5)

| # | run | change | result | verdict / lesson |
|---|---|---|---|---|
| 1 | window-0 twin of sing_bl_hard (r_c 0.2) | the 3D ansatz in spherical symmetry | held loss 2.2e-7 (3D: 7.6e-5), no seam bump, alpha 1e-5 | the 3D seam error was the 3D MLP's representation, not the ansatz |
| 2 | r_c 0.1 / 0.2 / 0.3 / 0.5 / 1.0 | template cutoff radius | monotonically better with r_c; 1.0 is 10-40x better than 0.2 in the seam band and the mass | r_c 1.0 |
| 3 | rho_power 4 | correction factor | mass better, Khat not | not adopted |
| 4 | w_power 2 | gentler template blend | Khat and mass 3-7x better | adopted, then moot (no template) |
| 5 | x_scale 1 | input scale | alike | irrelevant |
| 6 | soft IC | vs hard IC | alpha/Khat 2x worse at t = 0.5, mass alike | hard IC kept |
| 7 | `--no_template` | the exact initial data as a fixed offset, no template | same fields | the user's design decision: no templates (reversed 8 Nov: runs 47, 48) |

## B. To t = 8: the windowed scheme (3 - 4 Nov)

| # | run | change | result | verdict / lesson |
|---|---|---|---|---|
| 8 | `1d_T8` (control) | 32 windows of 0.25 | fields saturate at 1e-4 - 6e-4 from t ~ 2 although the loss is 1e-8 - 1e-6; far mass drifts to +2.5e-2 at r = 10 | the scheme's ceiling; the loss is no proxy for the error |
| 9 | `1d_T8_seed8` | seed | alpha 2-3x worse everywhere, mass 4.4e-2 vs 3.7e-2 | seed spread of 2-3x in the fields |
| 10 | `1d_T8_icd1` (27 windows) | first-derivative hand-off `--ic_d1 1` | Khat 2x better at t <= 2; alpha, chi in r = 3-10 3e-3 / 1e-3 later; mass drifts as the control | worse |
| 11 | `1d_T8_w05` (13 / 16 windows) | windows of 0.5, 8000 iterations | mass +3.5e-2 at r = 10 by t = 6 | worse |
| 12 | `1d_T8_R48` (10 windows, to t = 2.5) | outer boundary at R = 48 instead of 24 | alike the control at every t: fields within 1.5x, worst mass 3.7e-3 vs 7.9e-3 at t = 2.5 | **the outer boundary is not the error source**; stopped as pointless |
| 12b | `1d_T8_ic100`, `1d_T8_far`, `1d_T8_icd12` | IC weight 100; focus cloud sigma 5; second-derivative hand-off | stopped at 8-9 windows when the ADM batch took priority | no verdict |

## C. ADM constraint penalty (4 - 5 Nov)

| # | run | change | result (vs) | verdict / lesson |
|---|---|---|---|---|
| 13 | `1d_T8_adm30` | `--w_adm 30` (r^2-weighted H, M penalty) | mass 30x better (1.2e-3), chi 5-10x better inside r = 1; alpha 2.5x worse in r = 1-10 (vs `1d_T8`) | adopted; the first trade |
| 14 | `1d_T8_adm30_seed8` | seed | mass gain seed-independent (1.4e-3); outer alpha 3e-3 (vs 13) | outer alpha cost is real, 3x above both controls |
| 15 | `1d_T8_adm100` | lambda 100 | mass alike adm30 with a -4e-4 bias; alpha in 3-10 5x worse (vs 13) | worse |
| 16 | `1d_T8_adm10` | lambda 10 | mass 3x worse than adm30 and drifting; outer alpha as bad as adm30 (vs 13) | the outer-alpha cost does not scale with lambda; 30 kept |
| 17 | `1d_T8_adm30_ic` | + `--w_adm_ic 30` (penalty on the hand-off slice) | outer fields back to the control's level; mass 2.5e-3 (late drift) (vs 13) | **the base run since** |
| 18 | `1d_T8_adm30_icd1` | + `--ic_d1 1` instead of 17's slice penalty | best mass (9.9e-4) and inner chi (1e-5); alpha in 0.2-3 3x worse (vs 13) | alternative base for the mass |
| 19 | `1d_T8_adm30_ic_aw` | outer-boosted penalty weight | outer alpha 5x worse (vs 17) | worse |
| 20 | `1d_T8_adm30_ic_rw` | outer-boosted residual weight | no mass drift; outer alpha 2.5x worse (vs 17) | worse |

## D. Point sampling (5 - 6 Nov)

| # | run | change | result (vs 17) | verdict / lesson |
|---|---|---|---|---|
| 21 | `1d_T8_adm30_ic_log25` | 1/4 of the points log-uniform down to r = 1e-3 | window 0 stuck on a plateau; 4e-2 errors | much worse: points inside r = 0.01 hurt |
| 22 | `1d_T8_adm30_ic_log25_it0` | + 12000 iterations in window 0 | window 0 still at held 4.9e-6; 15x worse than 17 | dropped |
| 23 | `1d_T8_adm30_ic_logonly05` | all points log-uniform on [0.05, 24] | 2-8x better at t = 4, alike at t = 6, outer alpha 2.5-4x worse at t = 8 | alike (within seed scatter) |

## E. The colleagues' recipe in 1D (6 Nov)

| # | run | change | result | verdict / lesson |
|---|---|---|---|---|
| 24 | `1d_T8_adm30_ic_vol` | penalty on its own volume-uniform points, r >= 1 | mass at r = 10 5x better; outer alpha 4-5x worse (vs 17) | a trade |
| 25 | `s2_1d_T8` | global stage 2 (6 x 60 correction net, frozen windows) over the control | mass 7x better, outer alpha 5-8x worse (vs 8); still 2x above 17 in mass | penalising during the run beats correcting afterwards |
| 26 | `s2_adm30_ic` | global stage 2 over 17 | loss 2.5x down; mass alike, outer alpha 2-2.5x worse (vs 17) | alike; the loss went down, the error did not |

## F. Windows and hand-offs (6 - 8 Nov)

| # | run | change | result (vs 17) | verdict / lesson |
|---|---|---|---|---|
| 27 | `1d_T8_adm30_ic_s2w` | per-window second stage, 6 x 60, N2 warm-started, last iterate kept | held loss rose in most windows, NaN from t = 3 | failed: 19k parameters on 2000 points overfit the draw |
| 28 | `1d_T8_adm30_ic_s2wb` | the same with a fresh 3 x 30 N2 and the best held iterate kept | no blow-up; held loss down 7% (median); mass 2.0e-3, fields alike | alike; little to gain inside a window |
| 29 | `1d_T8_adm30_ic_ws` | `--warm_shift` (continuous warm start across the seam) | errors, mass and seam jumps alike | the final jump is set by training, not by the start |
| 30 | `1d_T8_adm30_ic_cold` | `--cold_start` (fresh random network every window) | mass 3x worse, fields 2-5x worse | worse: the warm start carries real information |
| 31 | `1d_T8_adm30_ic_s8` | seed | outer alpha 3-7x worse at t = 8, chi 3-10 3x; mass alike; alike at t = 4 | **seed scatter is as large as most verdicts** on outer alpha at t = 8 |

## G. Diagnostics (no verdict: what they showed)

| # | experiment | what was measured | finding |
|---|---|---|---|
| 32 | `lo_adm30_ic_w0` (loss vs error, window 0, 12000 iterations) | errors at every loss check | error ~ sqrt(loss) down to a held loss ~1e-6, then decoupled; alpha held at 2e-5 by r < 0.2; the held loss floors at 3e-8 - 3e-7 while training loss keeps falling |
| 33 | value fits to the reference (windows 0, 16, 31) | a network told the answer: its error and PINN loss | same 1e-4 field error as the PINN; PINN loss 1e3 - 1e8 x the trained runs' |
| 34 | Sobolev fits in r (values + r f' + r^2 f'') | the same with spatial derivatives | residual still 1e3 - 1e6 x the PINN's; d_t never constrained |
| 35 | the reference's own residual (degree-5 interpolant) | is the reference a loss minimiser? | yes for r >= 1 (1e5 - 1e11 below the PINN); inside r = 1 alike or above the PINN's |
| 36 | (t, r) residual maps of trained windows | where the loss sits | inside r = 0.01 (no points there); outer part ~1e-8, i.e. rms residual 1e-4 = the outer error |
| 37 | H diagnostic (colleagues' relation dm/dR = R^2 H / 4) | rebuilt vs measured mass | holds to 2-10%; adm30_ic leaves a one-sign <H> ~ -1e-6 outside r = 3 (the colleagues' bias, 100x weaker) |
| 38 | seams (`--eval_seams`, runs 17, 18, 29) | jump at each hand-off vs the error | jumps 5-20x below the error; they add up to the error only at r >= 10; inside r = 10 the error comes from the evolution within the windows |
| 39 | Sobolev + d_t fits (`--fit_dt 1`, windows 0, 16, 31) | the same with d_t f matched to the reference's time differences | converges (errors 1e-5 - 1e-4, at or below the PINN's); PINN loss 7.5e-5 / 7.5e-2 / 1.1, 10 - 1e4 x below the r-only fit, still 1e3 - 1e7 x the trained windows' |
| 40 | `1d_T8_zz` (`--zero_z --w_constraint 0`) | Theta = 0, Z^i = 0 (g from the conformal metric): the residual is the Einstein system, no zeta term, no ADM penalty | to t = 2.5 alike the control in fields and mass (worst 6.1e-3 vs 5.6e-3); NaN in window 10 (1/r, sqrt(a + b) near the puncture). (Its window-9 held spike is the seed's held draw, present in every run: README) | no mass gain from the constraints at the residual weight; blew up; followed by 45, 46 |
| 41 | `1d_T8_adm30_ic_kb` (`--keep_best`) | each window keeps its lowest-held-loss iterate | mass 1.7-2.2x better at every t (1.5e-3); t = 4 alpha 2-4x better; loss floor unchanged; last iterate was best in only 6 of 32 windows | **better, free: adopt into the base** |
| 42 | `1d_T8_adm30_ic_6x60` (`--hidden 60`) | main network 6 x 60 (19k parameters) instead of 6 x 40 | mass 2.2x better (1.1e-3, best of the w_adm 30 runs); t = 4 alpha 2.5x worse, t = 8 alike; loss floor unchanged; 2x the time | mixed: mass yes, fields no |
| 43 | `1d_T8_adm30_ic_mid` (`--n_mid 6000`) | 4x the points, the extra ones log-uniform on r = 0.2-24 (around and outside the horizon, r_h ~ 0.25-0.45) | held-loss floor 2-3x lower (8.6e-8); mass and fields alike at every t | alike: the loss moved, the error did not |
| 44 | `1d_T8_adm30_ic_kb60` (`--keep_best --hidden 60`) | 41 + 42 together: the new base candidate | mass 1.5e-3 as kb, but alpha t = 4-8 2-3x worse than kb/ic (6 x 60 hurts early alpha) | not adopted |
| 45 | `1d_T8_zz_rin` (`--zero_z --w_constraint 0 --keep_best --res_rin 0.1`) | 40 with sqrt|a + b|, keep-best, and the residual weighted by r^2 / (0.1^2 + r^2) (the user's r-weight: damps the puncture rows) | survives to t = 8 (held floor 2.4e-8); fields as the control or better; mass drifts to +3.0e-2 at r = 10 | no mass fix without the ADM penalty |
| 46 | `1d_T8_zz_chi` (`--zero_z --w_constraint 0 --keep_best --res_chi 1`) | 40 with sqrt|a + b|, keep-best, and the residual weighted by chi (the user's chi-weight) | survives to t = 8 (held floor 5.9e-9); smallest non-template fields (alpha 1-4e-4); mass -5e-3 at r = 10 | better than 40; mass not fixed |
| 47 | `1d_T8_tpl1` (`--r_c 1.0 --w_power 2 --template_file ..._r3_t8.npz`, no --no_template; + adm30_ic + keep_best) | back to the template (tries 1-7): the 1D solution inside r_c = 1, blended to 3; the network corrects outside (rho = r^2/(r^2 + 1)) | mass 1.9e-2, alpha 1e-2 at t = 6, held floor 8e-6, 14 windows above 1e-5 | fails: points near the puncture spoil it |
| 48 | `1d_T8_tpl1_rmin` (47 + `--r_min 0.5`) | no loss points inside r = 0.5: only the outer region is trained | alpha 1-2e-4 in every band at t = 8 (2-4x below ic), chi 5e-5 - 3e-4, mass flat 3.2e-3, held floor 2.6e-9, no spikes | **best run in the fields**; next: second seed, r_min 0.25 / 1 |
| 49 | `1d_T8_tplzz` (`--zero_z --w_constraint 0 --res_rin 0.1 --keep_best --r_c 1 --w_power 2 --r_min 0.5` + the template table, no ADM penalty) | template + Theta = Z = 0 + r-weight: the user's combination (9 Nov) | alpha 5e-4, 2e-4, 9e-5 and mass within 1e-3 to r = 10, but worst mass 5e-2 beyond r = 10 (growing from t = 4); held floor 3.4e-10 | fields best, far mass drifts |
| 50 | `1d_T8_tplzz60` (49 + `--hidden 60`) | the same with a 6 x 60 network | smallest fields of any run (alpha 1-2e-4 in every band), same far-mass drift (5e-2); held floor 3.8e-10 | fields best; 6 x 60 helps here |
| 51 | `1d_T8_tplzz_r1` (49 with `--res_r0 1 --res_q 2` instead of `--res_rin 0.1`) | residual weighted by (1 + r^2/r0^2), r0 = 1: 1 inside, 100 at r = 10, so the constraint rows count more outside and the inner region stays in (the user, 9 Nov) | worst mass 2.8e-3 flat (no ADM penalty); alpha 6e-4, 1e-3, 2e-3 (5-20x worse than 49); held floor 2.1e-9 | outer weight fixes the mass at the fields' cost |
| 52 | `1d_T8_tplzz60_con` (`1d_T8_tplzz60` + `--con_r0 1 --con_q 2`) | weight (1 + r^2/r0^2), r0 = 1, ONLY on the Theta row and the three Ghat rows of the Einstein residual (= Hamiltonian and momentum constraints with Theta = Z = 0); the other 14 rows weight 1; 6 x 60 (the user, 9 Nov, after 51 weighted every row) | worst mass 1.9e-3 / 2.5e-3 / 2.5e-3 / 4.5e-3 at t = 2 / 4 / 6 / 8 (50: 5e-2); fields best to t = 6 (alpha 4e-5 - 2e-4 inside r = 3); t = 8 outer alpha 8e-4 / 2e-3 in r = 1-10 as 51; held floor 6.5e-10 | the mass fixed by the two constraint rows alone; late outer alpha is the weight's cost: try r0 3 - 5 or q 1 |

## What it adds up to

1. The windowed scheme saturates at 1e-4 - 6e-4 in the fields from t ~ 2 whatever the loss (8, 32, 36): the outer
   loss of 1e-8 is an rms residual of 1e-4, the size of the outer error; the loss would have to fall 1e4 further.
2. The ADM penalty fixes the mass (30x) at a 2-3x cost in outer alpha; nothing since has improved both (13-26).
3. The seed scatter in outer alpha at t = 8 is 3-7x (9, 14, 31), so every one-seed verdict on outer alpha at t = 8
   (19, 20, 23, 24, 26, 28, 29) is inside the noise. Mass and the errors up to t = 4 are steady and can be trusted.
4. The hand-off is not the bottleneck inside r = 10 (10, 29, 30, 38): the jumps are 5-20x below the error, making
   them smaller (18) or continuous (29) does not lower the error, and removing the warm start (30) raises it.
5. The in-window error is where the gain must come from, but more iterations (32), a second network (28) and
   points nearer the puncture (21, 22) did not lower it. A network with the reference's values and derivatives (39)
   has the same 1e-4 error and a loss 1e3 - 1e7 x the PINN's: the loss-error relation is many-to-one and the PINN's
   loss is already on the solution branch; the error floor is not a failure to lower the residual. 41-43 repeat it:
   4x the points lowered the loss 2-3x and left the error alone; keep-best and 6 x 60 halved the mass error without
   touching the loss.

## How to follow (candidates, cheapest first; the user's decisions of 8 Nov in brackets)

1. Keep the best held-out iterate in the first stage, as 28 does in the second (adm30_ic window 9 ended on a
   held loss of 1.6; two windows of 28 on 0.6 and 3.5). One line, one run. [ok: run 41; adopted 8 Nov. Those held spikes are the seed's held draw (one point at r << 0.01), in every run of the seed: windows 9/16/27 for seed 7, 11/13 for seed 8; a held loss restricted to r >= 0.01 would make keep-best robust there: proposed]
2. Grade every change on the mass and on t <= 4, or run two seeds, before calling outer alpha at t = 8 (item 3). [agreed]
3. The in-window floor: the held loss floors at 3e-8 - 3e-7 because of points the draw misses (32). Untested
   single changes against 17: a wider network (6 x 60, run 42) and 4x the points, put around and outside the horizon,
   not into the puncture (run 43). [done: 42 mass 2.2x better, early alpha worse; 43 loss 2-3x lower, errors alike; the user: continue with 6 x 60 -> run 44 = 41 + 42]
4. ~~The outer boundary~~ Ruled out (8 Nov, the user's point): the error does not come in from r = 24. In adm30_ic
   the alpha error grows first at r = 3-6 (3e-4 by t = 3) and reaches 6-10 and 10-14 only at t >= 5.5 / 6.5, moving
   outward; the bands 14-22 stay flat at 2e-4 from t = 2.5; the mass deficit at r = 18-22 is -1e-3 at all those radii
   at once by t = 2, with no front. And R = 48 (12) was alike the control. The deficit is a coherent bias of the
   solution the loss selects (the colleagues' <H> bias), not a reflection.
5. 39 says the loss as formulated does not see the remaining 1e-4: changes to the formulation (40: Theta = Z = 0;
   the ADM rows at the residual weight) are the kind of thing to try, not more optimisation. [40 blew up at the puncture; the user (8 Nov): "a lot to try there": r-weight r^2/(r_0^2 + r^2) (run 45), chi-weight (run 46), |a + b|, keep-best; both running]
6. `--w_adm_ic 10`, and the penalty in 3D (`--w_adm` in sing_pinn.py): the step back to 3D once 1D settles. [waits for a
   better scheme]
7. The template again (the user, 8 Nov, after 41-46): the spherically symmetric solution inside r_c = 1 (the
   sweep's best, try 2), the network only outside, graded only outside. Runs 47 (template) and 48 (template + no
   points inside r = 0.5). In 1D this is the ceiling of the approach (the template is exact); in 3D it is the
   single-hole approximation near each puncture.
