# Verification of the paper's numbers (started 2026-10-23)

Every number in `paper.tex` is recomputed here from the data the runs left, or by rerunning, or
by recomputing from the exact solutions. Each entry gives the paper's value, the recomputed value
and how it was obtained. ✓ means the two agree to the quoted digits; ✎ means the paper is
corrected to the recomputed value; ✗ means the number is removed from the paper.

## Values for the paper's blue [pending] markers (filled as the runs finish)

From the hub chains (Paper/hub_qp.sh: Adam 5000 on Mendieta, then SSBroyden with the strong-Wolfe
line search in stages of 10000; 16384 points):

| marker (section) | value | source |
|---|---|---|
| Training: "no other optimiser is needed" | confirmed: Adam + one Wolfe SSBroyden stage reach 1.13e-10 (20 min), two stages 9.2e-11 | qp_com_w1/w2 |
| fresh/training ratio | 1.00 to 1.01 for the paper's runs | qp_com_w2 1.01, qp_nocom_w2 1.00 |
| V.B: the error without the condition, near the inner sphere | rms dlam 4.6e-6, dh 5.9e-5 (rho 1-3) | qp_nocom_w2 |
| V.B: d_z without the condition | +7.63e-5, the same on rho = 1, 2, 10, 100 | com_flux |
| V.B: ratio lambda error (rho 1-3) / d_z | 0.061 (the 0.062 law, now on a mode 300x smaller) | |
| V.B: the factor gained | 80x in lambda, 70x in h (overall rms 2.5e-6 -> 3.1e-8, 2.9e-5 -> 4.2e-7) | |
| V.B: below the perturbation, with the condition | rms dlam 3.1e-8 vs 9.6e-3 (3e5x), dh 4.2e-7 vs 1.1e-2 (3e4x) | qp_com_w2 |
| V.B: the loss | LOWER without the condition: 5.3e-11 against 9.2e-11 | the loss does not see the mode |
| Table (error by region), without | lam 4.63e-6 / 2.31e-6 / 3.53e-7 / 4.20e-8; h 5.86e-5 / 8.59e-6 / 8.28e-7 / 8.69e-8 | grade_rotated |
| Table (error by region), with | lam 6.44e-8 / 6.55e-9 / 1.55e-9 / 9.10e-10; h 8.63e-7 / 4.18e-8 / 4.71e-9 / 3.03e-9 | grade_rotated |
| Table (moments), with: net - ref | M0 +1.4e-8, M2 +9.1e-7, M4 -7.1e-6 | gh_moments_runs |
| Table (moments), without: net - ref | M0 -3.5e-8, M2 +5.8e-7, M4 -2.5e-6 | gh_moments_runs |
| Appendix wall times | Adam 6 min; SSBroyden 10000 iterations ~20 min (A30) | reports |

Note for the text of V.B: with this optimiser the translation came out ~300x smaller than in the
older runs (d_z 7.6e-5 against 2.6e-2), so the error without the condition is 4.6e-6, not 1e-3:
the mode costs almost nothing in the loss, and how far a run drifts along it depends on the path.
The statement to make is that the error is uncontrolled without the condition and that it is
removed with it, not a fixed size.

## Ablations: are the Robin ladder and the averaged pins needed? (hub_qp.sh, 2026-10-27)

qp_com's settings (rho^4 weights, centre-of-mass condition) from scratch, Adam 5000 + two Wolfe
stages, graded against the rotated fine reference. Both were introduced while the inner condition
still imposed h_rA = 0 (NOTES.md timeline); with h_rA free and the centre of mass fixed they are not needed.

| run | loss | fresh/train | rms dlam | rms dh | dlam rho 1-3 / 30-100 | dh rho 1-3 / 30-100 |
|---|---|---|---|---|---|---|
| qp_com_w2 (ladder + pins) | 9.2e-11 | 1.01 | 3.1e-8 | 4.2e-7 | 6.4e-8 / 9.1e-10 | 8.6e-7 / 3.0e-9 |
| ab_noladder_w2 (Robin at rho_out only) | 3.7e-11 | 1.00 | 6.9e-9 | 1.8e-7 | 1.4e-8 / 5.9e-10 | 3.7e-7 / 3.9e-9 |
| ab_nopins_w2 (no averaged pins) | 1.9e-11 | 1.04 | 1.0e-8 | 1.9e-7 | 2.0e-8 / 5.4e-9 | 3.9e-7 / 3.6e-9 |
| ab_none_w2 (neither) | 1.25e-11 | 0.99 | 9.1e-9 | 1.45e-7 | 1.25e-8 / 7.0e-9 | 3.0e-7 / 4.2e-9 |

ab_nopins_w2 had one block rejected by the non-finite guard (block 21) and recovered.
ab_none: d_z on rho = 1, 2, 10, 100 is -7e-10, -5e-8, +1e-9, +1e-9.
Conclusion: the formulation without ladder and pins is as accurate or better (3x in lambda, 3x in h
over qp_com); the far-field lambda error rises from 9e-10 to 7e-9, still 4e4 below the perturbation.

## Off-axis quadrupole (3-D hybrid, data about (1,0,1)/sqrt 2)

| run | settings | loss | rms dlam | rms dh | dh rho 1-3 | d_z rho = 2 |
|---|---|---|---|---|---|---|
| qp_rot_w3 | ladder + pins, COM 24 x 8 (aliased) | 5.4e-7 | 1.82e-5 | 7.7e-4 | — | — |
| qp_rot2_w3 | ladder + pins, COM 48 x 32 | 1.35e-6 | 8.9e-6 | 1.84e-3 | 3.8e-3 | 4.8e-6 |
| qp_rot3_w4 | no ladder, no pins, COM 48 x 32 (hub_qp3.sh) | 1.17e-6 | 9.05e-6 | 1.19e-3 | 2.5e-3 | 1.0e-5 |
| qp_rot4_n32768_w4 | as qp_rot3, width 40, 32768 / 2048 points (hub_qp4.sh) | 4.31e-9 | 3.39e-6 | 2.58e-4 | 5.3e-4 | 7.5e-7 |

qp_rot2 still decreasing slowly (~20% per 10000 Wolfe iterations, Ricci and gauge residuals
dominant); fresh/train 1.03. The rotated exact Weyl run (weyl_rot45_hyb, no ladder, no pins)
reached 5.1e-9 with the same 3-D network, hence qp_rot3 in the final formulation.
qp_rot3: stages 2.1e-6, 1.6e-6, 1.3e-6, 1.17e-6 (10 min each, about twice as fast as qp_rot2: no
ladder spheres); fresh/train 1.05. Same plateau as qp_rot2, so the ladder and pins are not what
limits the 3-D case; the Ricci and gauge residuals dominate, and the error sits in h near the
inner sphere (rho 1-3: 2.5e-3 against a perturbation of 2.3e-2). Likely capacity: 2347 parameters
for a strong-field (lambda ~ 1/3 at rho_in) tilted datum, against the weak-field rotated Weyl test.

Moments and tails (gh_moments_runs.py, tail_decay.py, on the Mac from the copied params):

| run | M0 net-ref | M2 net-ref | M4 net-ref | spurious M3 | max odd c_l (rho 2-100) |
|---|---|---|---|---|---|
| ab_none_w2 | -6.3e-8 | +7.3e-7 | -3.2e-6 | -2.4e-7 | 3.9e-9 |
| qp_rot3_w4 (about the data axis) | -5.9e-6 | +2.5e-5 | -1.2e-3 | +8.6e-5 | 1.3e-6 |
| qp_rot4_n32768_w4 | -5e-7 | +4.0e-6 | -3.9e-5 | -1.3e-6 | 4.1e-8 |

qp_rot3 tails c_l rho^(l+1): l=0 -> 0.58386; l=2 holds 0.1748 to rho ~30 then drifts (0.1437 at 100);
l=4 holds to rho ~5 only. Relative moment errors 1e-5 (M0), 1.4e-4 (M2), 5% (M4).
Stage times: ab_none Adam 161 s, w1 649 s (7.7e-11), w2 654 s (1.25e-11); qp_com w1/w2 ~1290 s each.

Paper updated 2026-10-27 to the final formulation (no extra outer spheres, no averaged pins):
II.D, loss (no w_pin), Sec. V.C new (tab:ablation), V.D off-axis filled from qp_rot3, tab:moments
two new rows, decay paragraph, appendix, conclusions. qp_rot4 (2026-10-28): Adam 523 s (0.60), stages 3.99e-8, 4.34e-9, 4.37e-9, 4.31e-9 (~1770 s each, flat
after w2); fresh/train 1.00; region errors lam 6.9e-6 / 1.1e-6 / 5.0e-8 / 1.6e-8, h 5.3e-4 / 2.3e-5 /
6.8e-7 / 3.0e-8; tails: l=2 constant to rho 100, l=4 to ~20. Filled into V.D, tab:moments, decay.

## Final-formulation pair for Table I, and the tangential inner term (hub_qp5.sh, 2026-10-28)

| run | loss | fresh | d_z (all spheres) | rms dlam | rms dh | dlam by region | dh by region |
|---|---|---|---|---|---|---|---|
| ab_none_w2 (with COM) | 1.25e-11 | 0.99 | <= 5e-8 | 9.12e-9 | 1.45e-7 | 1.25e-8 / 8.90e-9 / 7.47e-9 / 6.98e-9 | 2.98e-7 / 3.37e-8 / 4.94e-9 / 4.22e-9 |
| fn_nocom_w2 (without) | 1.64e-11 | 1.02 | -1.0633e-3 (constant) | 3.53e-5 | 3.99e-4 | 6.45e-5 / 3.22e-5 / 4.92e-6 / 5.85e-7 | 8.17e-4 / 1.20e-4 / 1.15e-5 / 1.21e-6 |
| fn_tan_w2 (with COM, --inner-tangential) | 9.38e-12 | 1.05 | <= 2.3e-8 | 5.86e-9 | 1.15e-7 | 6.94e-9 / 5.41e-9 / 5.54e-9 / 5.54e-9 | 2.36e-7 / 1.95e-8 / 5.23e-9 / 5.07e-9 |

Factors without/with: 3870 (lam), 2750 (h). lam error rho 1-3 / |d_z| = 0.0607. l=1 peak of dlam:
1.35e-4 at rho 1.9 (without), 3.9e-9 (with). Moments net - ref (gh_moments_runs.py): fn_nocom
M0 -2.5e-8, M2 +1.8e-7, M4 -4.1e-7; fn_tan -4.1e-8, +5.1e-7, -4.3e-6. Centre of mass from the
sphere of radius rho (scratch com_profile.py: axis_profile(gap_rho=rho), shift at order 8;
reproduces qp_nocom's 1.2e-7 / 5.8e-5 / 1.17e-4 / 1.29e-4): fn_nocom -7.7e-7 / -8.08e-4 /
-1.629e-3 / -1.794e-3 at rho 1 / 2 / 10 / 70, a = d_z/M0 = -1.82e-3; ab_none <= 6e-8.
Paper: V.B, tab:com, fig:com (fig_com.py now fn_nocom_w2 vs ab_none_w2), tab:moments, the moments
paragraph and the appendix now use this pair; tab:ablation keeps qp_com as the "all" row.
fn_tan: 1.8x lower lam error at rho 1-3, 1.3-1.5x overall; not yet in the paper.
rot_tan (qp_rot4_n32768 + --inner-tangential, 3-D width 40, finished 2026-10-28 11:49): Adam 522 s
(0.55), stages 9.93e-8, 6.20e-8, 5.13e-8, 2.95e-8 (~1775 s each, still falling, unlike qp_rot4 which
was flat at 4.3e-9 from stage 2); fresh/train 0.99; |d| <= 1.2e-6. rms dlam 6.12e-6, dh 4.05e-4;
by region lam 1.25e-5 / 1.87e-6 / 7.9e-8 / 1.9e-8, h 8.39e-4 / 3.51e-5 / 1.02e-6 / 5.7e-8. That is
1.6-1.8x WORSE than qp_rot4 after the same budget. The tangential term itself is not what is left
(bc_in 2.65e-11 of 2.95e-8): the PDE residuals are 7x higher (Ricci rms 8.4e-5 vs 3.3e-5), i.e. this
run converged more slowly. Continued two stages (hub_qp6.sh, finished 2026-10-28 13:14): w5 2.09e-8,
w6 1.95e-8 (levelling off, 4.5x qp_rot4's 4.3e-9); fresh/train 0.97; |d| <= 8.4e-7. rms dlam 4.66e-6,
dh 3.81e-4 (1.4x / 1.5x qp_rot4's 3.39e-6 / 2.58e-4); by region lam 9.53e-6 / 1.41e-6 / 6.5e-8 /
7.3e-9, h 7.88e-4 / 3.27e-5 / 9.6e-7 / 5.1e-8 (rho 1-3 h 1.5x qp_rot4's 5.3e-4); Ricci rms 6.7e-5
(2x). Verdict: modest gain in axisymmetry, none in 3-D even with 1.5x the budget; not part of the
paper's formulation (paper: end of the H^1 paragraph).

## The inner condition (for the sentence on h_rr, h_rA in II.D)

| claim | value | source |
|---|---|---|
| spectral reference with h_rA = 0 imposed | loss 1.8e-5, a floor: Newton stalled after 2 steps, non-decaying l = 2 metric part | PERTURBATION.md:287 (spectral solves before 0773b6a) |
| the same without it (l <= 6, 24 radial, rho^2 weights) | loss 3.0e-9, Newton converged in 3 steps | PERTURBATION.md:304 |
| h_rr pinned | over-determines the radial gauge (inner_h_rr stays free) | README.md §4 / problem.py |

## Exact solutions and analytic checks (Mac; `Paper/recheck_local.py`, `verify_weyl.py`, tests)

| claim | paper (draft) | recomputed | |
|---|---|---|---|
| exact Schwarzschild through the full loss | < 1e-16 | < 1e-16 (`test_loss_is_zero_on_exact_solution`) | ✓ |
| same, Robin-source chart with the averaged pins | — | 4.3e-10 from the tangential pin; zero without the pins (`test_sym_hybrid_manufactured_robin_is_exact`, updated) | new |
| exact Weyl rods, full loss, cylindrical source | 1.3e-28 | 1.26e-28 without the pins (verify_weyl, 17/17); 1.6e-9 with the h_rr pin | ✓ (pins off) |
| Weyl chart's harmonic residual at rho = 7.5 | 3.9e-3 | max 9.2e-3, rms 7.1e-3 over the sphere | ✎ |
| rotated exact Weyl: compat / Ricci / lam_eq / gauge | 1.9e-16 / 1.6e-15 / 7.1e-17 / 3.4e-16 | max over 256 points in [4, 40]: 8.9e-16 / 2.6e-14 / 5.9e-17 / 1.9e-15 | ✎ |
| rotated exact Weyl in harmonic gauge | 3.0e-2 | 2.6e-2 | ✎ |
| Robin decay ratios h / Gamma / lam at rho = 20 | 2.01 / 3.03 / 0.955 | 2.00 / 3.005 / 0.952 (local log-derivative) | ✎ |
| order-1 lambda pin residual on the exact solution at rho = 100 | -6.6e-5 | -6.590e-5 | ✓ |
| its cost at w_pin = 100 | 4.3e-7 | 4.34e-7 | ✓ |
| the same pin on P | 1.5e-11 | 1.48e-11 | ✓ |
| monopole of P for Schwarzschild in harmonic coordinates | M/rho + M^3/(3 rho^3) + ... | P = x/(1-x^2) = M/rho + M^3/rho^3 + M^5/rho^5 + ... (sympy; the M^3/3 is U's: U = -artanh(M/rho)) | ✎ (paper l.164, PERTURBATION.md) |
| parameter counts (axisym / + decay / 3-D hybrid / mlp) | 2265 / 2285 / 2347 / 2725 | identical | ✓ |
| R_ab R^ab of the exact rods (1, 0.1) at rho_in = 7.5 / 4 / 3.5 | 3.3e-6 / 5.8e-3 / 5.3e-2 | 3.28e-6 / 5.79e-3 / 5.28e-2 (`weyl.curvature_invariant`, max over the sphere) | ✓ |
| GH moments of exact assets (Schwarzschild, rods, Curzon) | — | `tests/test_gh_moments.py` | new |

## Weyl runs re-evaluated from their saved networks (`stationary.evaluate` on copies)

| run | quantity | paper | recomputed | |
|---|---|---|---|---|
| weyl_prod | final loss | 2.06e-13 | 2.059e-13 | ✓ |
| | max dh / max dlam | 6.7e-7 / 9.0e-8 | 6.70e-7 / 9.04e-8 | ✓ |
| | decay powers l = 0, 1, 2 | -0.926, -2.05, -2.85 | report.txt fit: -0.924, -2.97 (l = 1 is at the 1e-12 noise), -2.84 | ✗ (replaced by the GH moments) |
| weyl_unequal | loss / max dh | 2.4e-14 / 3.5e-7 | 2.37e-14 / 3.41e-7 | ✎ dh |
| | a_l0 coefficients, dipole 1.398415 | (from README) | not recomputed | ✗ (replaced by the GH moments) |
| weyl_close4 | loss / max dh | 7.1e-13 / 1.6e-5 | 7.13e-13 / 1.54e-5 | ✎ dh |
| weyl_close35 | loss / max dh / max dlam | 3.1e-12 / 7.7e-5 / 1.8e-6 | 3.06e-12 / 7.29e-5 / 1.76e-6 | ✎ dh |
| weyl_rot45 | loss / max dh / max dlam | 1.75e-4 / 7.1e-2 / 5.2e-3 | 1.753e-4 / 7.12e-2 / 5.19e-3 | ✓ |
| weyl_rot45_long | loss / max dh / max dlam | 2.28e-5 / 3.6e-1 / 6.4e-2 | 2.282e-5 / 3.57e-1 / 6.44e-2 | ✓ |
| weyl_rot45_hub | loss / max dh / max dlam | 6.93e-8 / 1.1e-3 / 1.0e-4 | 6.928e-8 / 1.13e-3 / 1.04e-4 | ✓ |
| chart-invariance loss | 1.720418e-2 on three shells | not recomputed | ✗ |
| weyl_rot45 -> _long | loss reduction "factor 13" (README) | 1.753e-4 / 2.282e-5 = 7.7 | ✎ (factor 8) |

## Schwarzschild tests

The original runs (`m2R2_asym1` first order, `m2R3_symhybrid`, `m2R4_realrobin`) were deleted from
`runs/` in commit 92cb622. They are recovered from 92cb622~1 into `runs/_restored/`.

| quantity | paper | restored report.json | |
|---|---|---|---|
| first order: final loss | 1.2e-5 | 6.31e-6 | ✎ (the 1.2e-5 has no source) |
| first order: max dh / dGamma / dlam | 3.3e-1 / 1.7e-1 / 1.3e-2 | 0.328 / 0.173 / 1.28e-2 | ✓ |
| metric only: loss, max dh / dGamma / dlam | 5.8e-8, 5.3e-5 / 2.3e-4 / 2.9e-5 | 5.75e-8, 5.30e-5 / 2.34e-4 / 2.92e-5 | ✓ |
| real Robin data: max dh / dGamma / dlam | 8.5e-5 / 1.9e-4 / 1.9e-4 | 8.50e-5 / 1.88e-4 / 1.93e-4 | ✓ |

These were float32 runs of older code. All three were rerun with the current code and without the
averaged pins (`Paper/commands_schwarzschild.sh`, jobs 1182114-1182116, 9-11 min each). The paper
quotes the reruns:

| rerun | loss | max dh | max dGamma | max dlam | M0 (exact 1) |
|---|---|---|---|---|---|
| schw_first_order | 1.50e-9 | 1.71e-2 | 8.61e-3 | 1.27e-3 | |
| schw_metric_manuf | 2.40e-11 | 6.51e-6 | 7.86e-6 | 1.01e-6 | 0.999991 |
| schw_metric_real | 2.42e-11 | 7.00e-6 | 5.97e-6 | 6.32e-7 | 0.999997 |

M2 of the Schwarzschild runs is 1.5e-4 +- 2.7e-4 (consistent with 0; with the inner sphere at 2M
the axis series converges slowly, so the window spread is large).

### Schwarzschild with Wolfe (2026-11-09; supersedes the reruns above in the paper)

The author asked to drop Armijo everywhere.  Phase K of `hub_recipe.sh` reran the three tests with
Wolfe AND the quadrupole recipe's Robin conditions (order 3 on h and P, none on Gamma; order 3 on
Gamma too for the first-order network).  The errors grew 20-90x at the same loss (hub and mini agree):

| K run (order 3) | loss | max dh | max dGamma | max dlam |
|---|---|---|---|---|
| sw_first_order | 3.2e-9 | 0.32 | 0.12 | 1.8e-2 |
| sw_metric_manuf | 4.1e-11 | 1.4e-4 | 6.1e-5 | 1.5e-4 |
| sw_metric_real | 2.5e-11 | 6.1e-4 | 3.5e-4 | 2.1e-4 |

One change at a time on Mendieta (metric only, manufactured source unless noted; runs/sx_*):

| run | order | Gamma cond. | line search | weights p | loss | max dh | max dGamma | max dlam | s |
|---|---|---|---|---|---|---|---|---|---|
| sx_wolfe_o2G | 2 | yes | Wolfe | (2,1,2) | 8.6e-12 | 7.6e-7 | 9.8e-6 | 9.0e-8 | 540 |
| **sx_wolfe_o2noG** | 2 | no | Wolfe | (2,1,2) | 6.9e-12 | 8.9e-7 | 4.1e-6 | 1.1e-6 | 511 |
| sx_wolfe_o3G | 3 | yes | Wolfe | (2,1,2) | 4.9e-11 | 2.7e-4 | 1.2e-4 | 3.0e-4 | 1840 |
| sx_armijo_o3noG | 3 | no | Armijo | (2,1,2) | 7.0e-11 | 1.7e-4 | 7.5e-5 | 2.2e-4 | 838 |
| sx_o3noG_rho4 | 3 | no | Wolfe | (4,3,4) | 1.3e-8 | 2.9e-6 | 6.3e-5 | 1.4e-6 | 862 |
| **sx_first_o2** (first order) | 2 (h, G, P) | yes | Wolfe | (2,1,2) | 6.9e-10 | 1.0e-2 | 5.2e-3 | 7.7e-4 | 429 |
| **sx_real_o2noG** (homogeneous) | 2 | no | Wolfe | (2,1,2) | 1.1e-11 | 9.5e-7 | 3.8e-6 | 2.1e-6 | 415 |

The order, not the line search, is the cause: Wolfe at order 2 is 8x better than the Armijo rerun
(6.5e-6); dropping Gamma's condition changes little; order 3 with p = (2,1,2) is stiff (README 8.8:
the order-n residual scales as gain^n, so the same weight is a different weight at every order), and the
rho^4 weights recover most of it.  The paper (Sec. schw, tab:firstorder) quotes the bold runs, states
order 2 for this test and the order-3 numbers.  M_ADM (gh_moments_runs.py): sx_real_o2noG 1.000020
+- 1.2e-5, sx_wolfe_o2noG 0.999990 +- 5.7e-6.  Commands: `Paper/commands_schwarzschild.sh`.

## Rotated Weyl with the metric-only 3-D network (`Paper/commands_weyl.sh`, job 1182117)

| | loss | max dh | max dlam | iterations / time | M0 / M2 / M3 / centre (exact 0.0314286 / 6.785e-6 / -3.33e-7 / 0.0374026) |
|---|---|---|---|---|---|
| weyl_rot45_hyb | 5.13e-9 | 1.64e-4 | 1.73e-5 | 7000 (plateau) / 16.6 min | 0.0314289 / 6.792e-6 / -3.26e-7 / 0.03740 |
| (old first-order weyl_rot45_hub) | 6.93e-8 | 1.13e-3 | 1.04e-4 | 12165 / 141 min | 0.031446 / 5.3e-6 / -2.5e-7 |

## The spectral reference (`Paper/commands_specref.sh`, job 1182235)

The rerun of the reconstructed command gives loss 2.793e-11 after Newton step 3, identical to the
original (job 1180550). ✓

## Quadrupole (Mendieta; `Paper/recheck_mendieta.sh`, report_metrics and grade_rotated rerun)

Recomputed with jobs 1182087-1182094 (`plots/recheck_report_metrics.json` and the grade logs). The
earlier tables reproduce. For example, m5_ssb_norsmp has lambda error 1.558e-3 at rho 1-3 (table
1.6e-3) and d_z = -0.02567 on all four spheres; m7_dsg_com has 6.67e-8 (6.7e-8); m8_ssb_com has
lambda 3.18e-8 and h 1.43e-7 (3.2e-8 and 1.4e-7). The rotated grades reproduce to the quoted digits:
rot_ssbm 1.56e-5 / 1.16e-3, rot_from_fit 6.49e-6 / 3.23e-4, ctl_hyb_ssbm2 1.17e-5 / 5.52e-4,
rot_coll256k_2g 1.43e-5 / 7.47e-4. These runs are no longer quoted in the paper, which uses the
from-scratch round P instead.

Wolfe line search on the rotated case (round 28): rot_ssbw (from rot_ssb) reaches loss 1.73e-6,
rms dlam 5.6e-6 and dh 8.5e-4, against Armijo rot_ssbm's 2.25e-6, 1.56e-5 and 1.16e-3.

## The paper's from-scratch runs (round P of `mendieta_submit.sh`)

`qp_com`, `qp_nocom` and `qp_rot`, each Adam + SSBroyden stages with a final check. Also the
rotated Weyl test with the metric-only network, `weyl_rot45_hyb` (`Paper/commands_weyl.sh`).
Pending.

## The mass as a function of S2 (hub_mass.sh, hub, finished 2026-10-31 00:26)

Fine spectral references (l = 0..10, 36 radial, 1008 unknowns, 2500 points, final formulation +
COM, --iters 10) at five amplitudes; networks (ab_none settings, only --lam-bc-S2 changed) at
-1/24 and -1/6.  Moments by scripts/gh_moments_runs.py (<run>/gh_moments.json), R0 = 0.5773503.

| S2 | run dir | ref M0 | M0 - R0 | /S2^2 | ref M2 | M0 spread | GN |r| | net - ref M0 |
|---|---|---|---|---|---|---|---|---|
| -1/24 | ms_s2h_w2 | 0.5788859 | 1.536e-3 | 0.885 | 0.08325 | 4e-7 | 6.3e-9 | -1.9e-9 |
| -1/12 | ab_none_w2 | 0.5838468 | 6.497e-3 | 0.936 | 0.17471 | 4e-6 | 4.0e-7 | -3.6e-8 |
| -1/8 | ms_s2t_cfg | 0.5930546 | 1.570e-2 | 1.005 | 0.27929 | 3e-5 | 5.4e-6 | (no network) |
| -1/6 | ms_s2d_w2 | 0.6079697 | 3.062e-2 | 1.102 | 0.40523 | 1.1e-4 | 3.7e-5 | -4.3e-7 |
| +1/12 | ms_s2p_cfg | 0.5829155 | 5.565e-3 | 0.801 | -0.15402 | 5e-7 | 2.2e-7 | (no network) |

Fits of M0 - R0 (scratch, numpy lstsq): a S2^2 alone, a = 1.06, max residual 1.8e-3 (the cubic and
quartic terms matter at 1/6); a S2^2 + b S2^3: 0.870, -1.33, residual 4.9e-4; + c S2^4: 0.830,
-0.769, 5.18, residual 3.7e-5 (within the -1/6 reference's spread); with a linear term added
(4 parameters, 5 points): linear coefficient -1.4e-3, i.e. 2% of the quadratic term at 1/12, not
significant.  Even part at +-1/12: 6.03e-3 = 0.868 S2^2.  Ratios dM(2 S2)/dM(S2): 4.23, 4.71.
Reference at -1/12 in the final formulation: M0 identical to the old (extra outer conditions)
reference to 7 digits; ab_none_w2 against it: rms dlam 7.1e-9, dh 1.45e-7 (vs 9.1e-9 / 1.5e-7
against the old one).  Networks: ms_s2h_w2 loss 3.19e-12 (fresh 0.98), dlam 2.66e-9, dh 1.04e-7
(perturbation 4.6e-3 / 5.2e-3); ms_s2d_w2 loss 2.53e-10 (fresh 1.00), dlam 2.18e-7, dh 1.86e-6
(perturbation 2.2e-2 / 2.4e-2).  The -1/6 reference's Gauss-Newton stalled at |r| 3.7e-5 (l <= 10
is short for a 50% datum), so part of the 2.2e-7 may be the reference's.
Paper: IV.A paragraph, tab:mass and fig:mass (Paper/fig_mass.py -> figs/mass_s2.png, reads the
run dirs' gh_moments.json); appendix.  PERTURBATION.md item 4 ("mass untouched") was
wrong and is corrected there.

## A second-stage network on the rotated Weyl test (hub_stage.sh, hub, 2026-10-30 21:42)

Base weyl_rot45_hyb (width 20, 7000 SSBroyden iterations, loss 5.1e-9).  All arms its settings
(8192 + 512 points, resampled every 500, SSBroyden up to 20000), errors against the exact rotated
Weyl solution, fresh = mean loss on 5 fresh samples / training loss (scripts/fresh_sample_loss.py,
Mac CPU).

| run | what | train loss | fresh | iters | max dh | max dlam | rms dh | rms dlam |
|---|---|---|---|---|---|---|---|---|
| base | width 20 | 5.13e-9 | 1.18 | 7000 | 1.64e-4 | 1.73e-5 | 2.31e-5 | 1.85e-6 |
| st_ctl | base continued | 4.09e-9 | 1.22 | 2000 | 1.48e-4 | 1.04e-5 | 2.18e-5 | 1.79e-6 |
| st_a | + corr. w20, eps 1e-4 | 2.81e-9 | 2.2e5 | 1750 | 1.95e-4 | 1.24e-5 | 2.59e-5 | 2.48e-6 |
| st_b | + corr. w20, fourier 4 | 1.53e-10 | 95 | 3750 | 8.75e-5 | 4.74e-6 | 1.07e-5 | 7.96e-7 |
| st_c | + corr. w40, fourier 4 | 1.19e-11 | 436 | 3168 | 5.80e-5 | 3.01e-6 | 9.74e-6 | 6.09e-7 |
| st_w40 | one width-40 net | 2.22e-9 | 1.58 | 8000 | 7.68e-5 | 5.02e-6 | 1.29e-5 | 9.94e-7 |

Verdict: the staged networks overfit (fresh/training 95-2e5; on fresh points st_c's loss 5.2e-9 is
the base's), although the errors fell 2.4-5.7x; report residuals on the report's own points are no
lower than the base's.  Not in the paper.  The honest result is st_w40: width 40 beats width 20 by
2-3.5x in error at fresh 1.6.  README §5 has the entry; a second round would need denser points,
more frequent redraws and no Fourier features on the correction.

## The hard end of the S2 scan: S2 = -0.25 and -0.30 (hub_mass.sh, hub_hardref.sh, hub_hard2.sh; Mendieta, 2026-10-31)

Datum lambda = 1/3 + S2 P2: minimum 1/12 (S2 = -0.25) and 1/30 (-0.30) at the poles.

1. The first references (tag fine, l <= 14, 48 radial, 4000 points, 1824 unknowns, undamped
   Gauss-Newton) DIVERGED: |r| 0.98 -> 5.2 (-0.25) and 2.9 -> 14.5 (-0.30) on step 1; the loop
   stopped on "residual grew" and kept that iterate.  Their masses (0.676, 0.734) and every
   "net - ref" printed for these two runs are meaningless.  cond(J) was 1.5e11 (4.6e9 at -1/6).
2. Fix (stationary/specref.py, 242e33e): backtracking step (halve until |r| <= 1.001 |r_old|,
   keep the iterate and stop if none helps) and --rcond.  Variants on ms_s2_025_w2 at full size
   (Mendieta, hub_hardref.sh), each its own tag:
   - `damp` (backtracking, rcond 1e-13): CONVERGED, |r| 9.8e-1 -> ... -> 4.58e-5 (loss 2.1e-9) in
     10 steps (steps 1, 0.25, 1, 0.5, 0.5, 1, 1, ...); M0 = 0.669610, M2 = 0.80673.  The scan's
     network ms_s2_025_w2 against it: M0 +4.6e-7, M2 +1.5e-5.
   - `rc9` (backtracking, rcond 1e-9): WORSE, M0 = 0.5373, M2 = 1.110 (nonsense).  A coarser
     cutoff removes real directions at this resolution; the near-singular directions were NOT the
     problem, the undamped step was.  Do not use --rcond above the default for these bases.
   - on a small basis (l <= 6, 10 radial, CPU) every step is taken in full at -1/12, -0.25 and
     -0.30: the damping is inactive where the old solver worked, so the earlier references stand.
3. Networks (ab_none settings, S2 changed): ms_s2_025: loss 3.2e-8 after w2 (fresh 1.5); three more
   stages at 32768 points / redraws every 250 (hub_hard2.sh, w3-w5): 2.5e-8, stall, fresh/training
   1.17.  ms_s2_030: loss 1.6e-6 after w2 (fresh 265, overfitting); after w3-w5 ~2e-6 on the last
   sample, fresh/training 38: not converged, the loss changes by factors of 4 between samples.
   The -0.30 network's mass is not used.
4. Next (hub_hard3.sh, Mendieta): DSGNAR on top of the stalls (5 cycles x 60, sketch 2000) and a
   width-40 network, at both amplitudes; the -0.30 reference with the corrected solver
   (hub_hardref.sh --run 30 damp).

## The hard end graded (2026-11-01; Mendieta: hub_hardref.sh, hub_bigref.sh, hub_hard2.sh, hub_hard3.sh)

Networks against the corrected spectral references; moments by scripts/gh_moments_runs.py --tag.

| S2 | run | loss (train) | fresh/train | rms dlam | rms dh | M0 net - spec | M2 net - spec |
|---|---|---|---|---|---|---|---|
| -0.25 | ms_s2_025_w5 (width 20) | 2.5e-8 | 1.17 | 3.9e-7 | 7.0e-6 | -6.4e-7 | +2.6e-5 |
| -0.25 | ms_s2_025_b40_w3 (width 40) | 2.3e-10 | 11 | 1.9e-7 | 3.4e-6 | +6.5e-7 | -6.2e-6 |
| -0.25 | ms_s2_025_dsg (DSGNAR on w5) | 2.1e-8 | 1.5 | - | - | - | - |
| -0.30 | ms_s2_030_w2 (width 20) | 1.6e-6 | 265 | 4.7e-5 | 8.1e-5 | -7.4e-5 | -4.8e-2 |
| -0.30 | ms_s2_030_b40_w3 (width 40) | 4.0e-8 | 170 | 1.2e-5 | 1.5e-5 | -1.1e-3 | +6.2e-2 |
| -0.30 | ms_s2_030_dsg (DSGNAR on w5) | 3.6e-7 | 1022 | - | - | - | - |

Spectral references: S2 = -0.25: l <= 14, 48 radial, backtracking, |r| 4.6e-5 (loss 2.1e-9), M0 0.669610,
M2 0.80673.  S2 = -0.30: l <= 14 (tag damp) stalled at |r| 1.15e-3, loss 1.3e-6, M0 0.765980; l <= 20
(tag big_l20) |r| 7.3e-5, loss 5.3e-9, M0 0.766484 +- 4.2e-3 (window spread), M2 1.4072; l <= 14 with
72 radial (big_r72): cond(J) 8e13 (7e11 for l <= 20), the damped steps never leave |r| 0.32, M0 0.521
(nonsense).  Memory: peak device 6.3 GB at 2544 unknowns (24 GB card), host 14.5 GB at 1824 unknowns.
(M0 - R0)/S2^2: 0.80 (+1/12), 0.885, 0.936, 1.005, 1.102 (-1/6), 1.476 (-1/4), 2.101 (-0.30); the
five-value fit gives 0.0841 at -1/4 (measured 0.0923) and 0.1374 at -0.30 (0.1891).
DSGNAR on the stalls (5 cycles x 60, sketch 2000): -0.25 every cycle ends on the trust radius, 2.0e-8
-> 1.7e-8 in cycle 1, no net gain; -0.30 accepts 55 of 60 steps per cycle (10-70x down on its sample)
but each new sample starts higher, fresh/training 1022: it fits its sample at 16384 points.
Width 40 at -0.30 overfits (fresh/training 170, fresh samples spread 4e-8 to 3e-5): the best fresh
loss of the -0.30 networks is not below ~7e-6.  Paper: IV.A paragraphs, tab:mass, fig:mass, appendix.

## Collocation points, metric-only 3-D network on the rotated quadrupole (for V.C of the paper)

From DSGNAR.md rounds 20-26 (older formulation: rho^4 weights with the Robin ladder, width 20, SSBroyden,
fine spectral reference rotated; not rerun in the final formulation):

| run | points | loss | rms dh | dh rho 1-3 | rms dlam |
|---|---|---|---|---|---|
| rot_ssbm3 | 16384 | 1.3e-6 | 1.17e-3 | 2.4e-3 | 1.7e-5 |
| rot_coll64k | 65536 | 2.6e-7 | 7.2e-4 | 1.5e-3 | 1.4e-5 |
| rot_coll128k | 131072 | 1.16e-7 | 9.9e-4 | 2.1e-3 | 1.0e-5 |
| rot_coll256k_2g | 262144 | 1.18e-7 | 7.5e-4 | 1.6e-3 | 1.4e-5 |

4x points: loss 5x, dh 1.6x, dlam 1.2x; 16x: loss 11x, dh 1.6x; the h error then wanders between 7e-4 and 1e-3.

## One recipe for every result (Paper/hub_recipe.sh, 2026-11-06)

Prompted by the evidence review in NOTES.md. The author asked that everything be done as close as
possible to a single recipe. The recipe is ab_none's (final formulation + centre of mass). Its flags,
built through train.parse_args, reproduce ab_none_w2's config.json field for field; the only
differences are three fields added since then (inner_tangential, stage_base, stage_eps), at defaults
that match the old behaviour.

* **Weyl in the recipe.** The axisymmetric Weyl runs (2026-09-25..28) imposed the full exact h on the
  inner sphere (`pack_sym(f.h - e.h)`, before 0773b6a), with 4096 points, order-2 Robin on h, Gamma
  and lambda, w = 10, ρ² weights and SSBroyden-Armijo from cold. They are redone in the recipe in a
  chart with ρ_in = 1, rods scaled by 1/ρ_in(physical): wr_prod, wr_unequal, wr_close4, wr_close35,
  and the rotated wr_rot45 in the 3-D recipe. The Robin source and the inner data still come from the
  exact solution. The full recipe loss on the exact unequal rods is 4e-27.
* **Centre of mass for exact data** (losses.com_target, tests/test_com_flux.py). With
  inner_bc = "reference" the condition drives d to the exact solution's own flux, not to 0. For the
  unequal rods, d_z = M z_cm = 0.0256 on every sphere in the ρ_in = 1 chart (z_cm = 0.17455 chart =
  1.3091 physical, the closed-form value). With d = 0 the term would have been 3.3e-3 on the exact
  solution, i.e. a pull towards a chart that the data do not describe.
* New tools: scripts/compare_refs.py (two spectral references of one problem against each other), and
  `grade_rotated.py --tag` (several references per run).
* The hard-end runs and their references (ms_s2_025_*, ms_s2_030_*) were copied from Mendieta to the
  hub's runs/ for phase E.

### Results (hub 2026-11-06..08, phases A-F; Mendieta 2026-11-07, phases G-H)

The hub batch ran twice into the hub's limits and resumed from its checkpoints each time: idle
culling after 1 h (HUB.md 19), then a server lifetime of about 24 h. Errors are against the exact
solution (Weyl) or against the final spectral reference, runs/ab_none_w2/specref_nonlinear_fine.npz
(S2 = -1/12; the hard end uses its own spectral references). "fresh" is the mean loss on 5 fresh
samples divided by the training loss.

**A, F: Weyl in the recipe.** The chart is a homothety of the physical one (lengths and masses both
scaled by 1/ρ_in), so dh and dλ are dimensionless and compare directly with the old runs. The old
runs had the full exact h imposed on the inner sphere; the new ones only λ and the induced metric.

| run | loss | fresh | max dh (old) | max dλ (old) | rms dh | rms dλ | M2 net - exact |
|---|---|---|---|---|---|---|---|
| wr_prod_w2 | 6.0e-15 (2.1e-13) | 0.96 | 1.6e-7 (6.7e-7) | 4.7e-9 (9.0e-8) | 5.7e-9 | 2.4e-10 | -8.8e-8 |
| wr_unequal_w2 | 1.1e-14 (2.4e-14) | 1.00 | 2.7e-7 (3.4e-7) | 4.3e-9 | 1.2e-8 | 3.9e-10 | 2.1e-9 |
| wr_close4_w2 | 1.7e-14 (7.1e-13) | 1.73 | 2.9e-7 (1.5e-5) | 7.3e-9 | 7.6e-9 | 4.4e-10 | 2.3e-6 |
| wr_close35_w2 | 7.2e-14 (3.1e-12) | 11.7 | 4.7e-7 (7.3e-5) | 7.7e-9 (1.8e-6) | 3.6e-8 | 2.0e-9 | 5.4e-7 |
| wr_rot45_w4 (3-D, 45 deg) | 1.3e-11 (6.9e-8, weyl_rot45_hub) | 1.18 | 3.3e-6 (1.1e-3) | 1.2e-7 (1.0e-4) | 5.3e-7 | 1.3e-8 | 5.2e-7 |

* wr_close35's fresh ratio comes from one sample of five (3.9e-12; the other four 7.3e-14..8.1e-14).
* The centre of mass sits at the exact flux on every inner sphere (d_z = 0.026, 0.090, 0.12 for
  unequal, close4, close35); only ρ = 100 (outside the ρ_out = 10 domain) differs.
* GH moments (FHP normalisation, about the fitted centre): M0 within 1.1e-7 of the exact rods for
  all; M2 as in the table, M3 within 4.6e-5 (close rods) -- below the extraction spread (7.7e-5 in M2
  for close35). The axisymmetric and the rotated close35 networks give the same moments to 1e-7.
* wr_rot45 stages: 1.6e-11, 1.5e-11, 1.4e-11, 1.3e-11 (about 1 h 45 min each on the A30).

**B, G, H: off-axis quadrupole, width against points** (rotated data about (1,0,1)/sqrt 2, 4 stages).

| | 16384 points | 32768 points |
|---|---|---|
| width 20 | qp_rot3_w4: loss 1.2e-6, rms dh 1.19e-3, rms dλ 9.1e-6 | rp_w20_n32768_w4: 5.4e-7, 8.2e-4, 1.9e-5 |
| width 40 | rp_w40_n16384_w4: 1.4e-8, 7.2e-4, 1.4e-5 | rp_w40_n32768_w4: 4.4e-9, 3.3e-4, 3.6e-6 |

* qp_rot3 and qp_rot4_n32768 (hub_qp3.sh, hub_qp4.sh) have the recipe's config field for field;
  regraded against the final reference (G) they give the same numbers as before (qp_rot4: 2.58e-4,
  3.39e-6). H retrains qp_rot4 in the recipe on Mendieta: loss 4.37e-9 against 4.31e-9, rms dh
  3.3e-4 against 2.6e-4: it reproduces.
* Doubling only the width or only the points gains at most a factor 1.6 in h; both together give 3.6 in
  h and 2.5 in λ. Max dh: 7.9e-3, 5.1e-3, 5.1e-3, 2.2e-3. All fresh ratios 0.98-1.06.
* Stage losses (width 40, 16384): 2.5e-8, 1.6e-8, 1.5e-8, 1.4e-8; (width 20, 32768): 4.4e-6, 6.4e-7,
  5.7e-7, 5.4e-7; (width 40, 32768): 3.6e-8, 4.5e-9, 4.5e-9, 4.4e-9. Flat after the second stage.

**C: the yardstick at S2 = -1/12.** The l <= 14 reference (48 radial) against the final one (l <= 10,
36 radial): rms dh 1.7e-8, max 3.3e-7; rms dλ 1.9e-9, max 2.4e-8 -- ten times below the network
errors. Regrades (final / l14): ab_none_w2 rms dh 1.45e-7 / 1.44e-7, rms dλ 7.1e-9 / 5.8e-9;
fn_tan_w2 1.15e-7 / 1.12e-7, 3.7e-9 / 3.0e-9; fn_nocom_w2 4.0e-4 / 4.0e-4, 3.5e-5 / 3.5e-5.

**D: seeds** (seed 0 = the original run; all fresh ratios 1.01-1.17).

| run | seed | loss | rms dh | rms dλ | d_z (all spheres) |
|---|---|---|---|---|---|
| ab_none (COM) | 0 / 1 / 2 | - / 2.3e-11 / 5.8e-11 | 1.45e-7 / 2.1e-7 / 3.1e-7 | 7.1e-9 / 1.7e-8 / 1.9e-8 | ~1e-8 |
| fn_tan (COM, H^1 inner data) | 0 / 1 / 2 | - / 7.5e-12 / 6.9e-12 | 1.15e-7 / 1.6e-7 / 9.2e-8 | 3.7e-9 / 2.9e-9 / 3.5e-9 | ~1e-8 |
| fn_nocom (no COM) | 0 / 1 / 2 | - / 6.4e-12 / 1.7e-11 | 4.0e-4 / 2.5e-3 / 6.4e-4 | 3.5e-5 / 2.2e-4 / 5.6e-5 | 0 / -6.6e-3 / +1.7e-3 |

Without the centre of mass the loss is as low but the error 1e3-1e4 times larger and seed-dependent:
each seed lands on a different translate along the axis. With it, three seeds agree within a factor 3.

**E: the hard end at 65536 points, redrawn every 250** (width 40, Adam + 3 stages; the later stages
stopped on stall).

| S2 | run | loss | fresh | rms dh | rms dλ | M0, M2 net - spec |
|---|---|---|---|---|---|---|
| -0.25 | ms_s2_025_b40_w3 (32768) | 2.3e-10 | 11 | 3.4e-6 | 1.9e-7 | +6.5e-7, -6.2e-6 |
| -0.25 | hs_025_n65k_w3 | 2.2e-10 | 0.95 | 3.3e-6 | 2.0e-7 | +6.1e-7, -1.9e-6 |
| -0.30 | ms_s2_030_b40_w3 (32768) | 4.0e-8 | 170 | 1.5e-5 | 1.2e-5 | -1.1e-3, +6.2e-2 |
| -0.30 | hs_030_n65k_w3 | 4.3e-9 | 39 | 1.6e-5 | 1.3e-5 | +8.7e-4, -5.1e-2 |

At -0.25 the overfitting is gone and the error unchanged: the width-40 result is now clean. At -0.30
more points reduce the overfitting (170 -> 39) but not the error, and M2 is still 5% off: the limit
of the method at this resolution, not a sampling artefact.

### Moments and the translation illustration against the final reference (2026-11-08)

tab:moments regraded (scripts/gh_moments_runs.py --ref runs/ab_none_w2/specref_nonlinear_fine.npz --tag
final; reference M_ADM 0.583847, M2 0.174715, M4 0.021964, window spread 4.2e-6 / 1.16e-4 / 7.4e-4):

| run | M_ADM net - spec | M2 | M4 |
|---|---|---|---|
| ab_none_w2 (with COM) | -3.6e-8 | +2.4e-7 | +2.4e-9 |
| fn_nocom_w2 (without) | +2.1e-9 | -3.1e-7 | +2.8e-6 |
| qp_rot3_w4 (3-D, width 20; phase G) | -5.9e-6 | +2.4e-5 | -1.2e-3 |
| qp_rot4_n32768_w4 (3-D, width 40; phase G) | -4.0e-7 | +3.5e-6 | -3.6e-5 |
| rp_w40_n32768_w4 (its rerun; phase H) | -8.4e-8 | -3.5e-8 | -3.0e-5 |

II.D illustration (review item 10), replaced by the paper's own pair: the l = 1 coefficient of
P = (1 - lambda^2)/(4 lambda) on the sphere rho = 100 (48-point Gauss-Legendre, meridian) is -1.104e-7 for
fn_nocom_w2 and +3.9e-11 for ab_none_w2; d_z / rho^2 = M_ADM a_z / rho^2 = -1.063e-7.  At rho = 10:
-9.60e-6 against -1.063e-5.  The old "|a| ~ 0.04 -> 2.5e-6" (COM_GAUGE.md, old formulation) is
consistent with the same law (0.58 x 0.0445 / 1e4 = 2.6e-6).

## Width 60 off-axis and the rotation condition (phases I, J, L, M of hub_recipe.sh; 2026-11-10/11)
Single-run scheme (Adam plateau rule, one SSBroyden run with plateau rule, best field restored), resample 500.
Off-axis errors against the rotated spectral reference specref_nonlinear_fine (grade_rotated_final.json);
Weyl errors against the exact solution.  Figures: Paper/fig_training.py; cache Paper/aux/training_errors.json.

| run | settings beyond the recipe | stop | loss | fresh ratio | rms dh | max dh | rms dlam | max dlam |
|---|---|---|---|---|---|---|---|---|
| I  rp_w60_n65k_one | width 60, 65536 pts | Adam ~1600, SSB cap 40000 | 1.6e-11 | 1.006 | 6.7e-4 | - | 8.5e-6 | - |
| J  rp_w60_n65k_tan_one | + inner data in H^1; + rotation condition from its resume at 27000 | SSB plateau 32250 | 7.9e-10 | 1.028 | 3.09e-6 | 4.3e-5 | 3.2e-8 | 3.0e-7 |
| L  rp_w60_n65k_rot | I's final network + rotation condition (--rot-pin 1,1.5,3,10,30,100) | SSB plateau 11000 | 1.62e-11 | 1.026 | 2.59e-7 | 3.3e-6 | 7.6e-9 | 1.0e-7 |
| wr_rot45_w4 | rotated Weyl, width 40, staged | - | 1.32e-11 | 1.18 | 5.3e-7 | 3.3e-6 | 1.35e-8 | 1.2e-7 |
| M  wr_rot45_rot | wr_rot45_w4 + rotation condition (--rot-pin 1,1.5,3,6,10, exact target) | SSB plateau 5250 | 1.36e-11 | 1.15 | 2.0e-7 | 3.1e-6 | 1.5e-8 | 1.4e-7 |

Mode diagnostics (scripts/error_modes.py): in I one constant k_R, |k_R| = 2.3e-3, explains 96-99% of dh on
rho 1.2-50, and |F_R| = 2.3e-3 on every sphere 1-100 (spectral reference < 2e-12); in L |k_R| < 1e-9 from 3000
iterations on.  Flux identities (scripts/green_identities.py, wr_rot45_w4): exact solution to quadrature; network
F_T = exact to 2e-9, F_R (0.4-1.1)e-6 with its change between spheres = the gauge-residual volume integral.
J and L each ran out of device memory once (13.6 GiB request) and were resumed from ckpt.pkl (J on Mendieta at
22000; L moved from the hub to Mendieta at 7000, job 1186619).
