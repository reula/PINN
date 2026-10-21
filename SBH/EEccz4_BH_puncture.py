"""
VACUUM CCZ4 PINN FOR BRILL-LINDQUIST / BOWEN-YORK BLACK HOLE BINARIES
======================================================================

This file merges two prior scripts:

- EEccz4_BH.py: the CCZ4 evolution equations (1+log slicing + Gamma-driver
  shift, moving-puncture gauge, no excision), the Adam+BFGS windowed
  training loop, and the Brill-Lindquist + Bowen-York analytic BH initial
  data (supporting --n_stars 1 or 2, --BL_masses/--BL_momenta/--BL_spins).
- bbh_headon_bssn_pinn_shift_driven_puncture.py: the shift-driven puncture
  tracker (dx_A/dt = -beta^x(x_A), RK4-integrated through a trained
  network -- the puncture is never a trainable variable) and the
  independent chi-minimum cross-check diagnostic, both generalized here to
  independent per-puncture tracking on the x-axis (no left/right symmetry
  assumed).

The scalar-field (Klein-Gordon) sector and boson_star/schwarzschild_ks
initial-data paths from EEccz4_BH.py have been removed entirely: this file
is vacuum-only, Brill-Lindquist/Bowen-York BH initial data only.
"""

import os
import csv
import math
import sys
import pickle
import time  # [GPUS]
import argparse
import collections

import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt

from tqdm import tqdm
from flax import linen as nn
import flax.core
from typing import Sequence
from functools import partial, lru_cache
from jax import flatten_util
from jax import random as randomjax

from Crunch.Models.layers import *
from Crunch.Models.polynomials import *
from Crunch.Auxiliary.metrics import *
from Crunch.Optimizers.minimize_backtracking import minimize

jax.config.update("jax_enable_x64", True)

# --------------------------------------------------------------------------------
# PATHS
# --------------------------------------------------------------------------------
file_path = os.getcwd()
project_root = os.path.dirname(file_path)
print(f"Project root: {project_root}")
if project_root not in sys.path:
    sys.path.append(project_root)

# --------------------------------------------------------------------------------
# ARGUMENTS
# --------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="3D Cartesian vacuum CCZ4 PINN for BH binaries")

# File name
parser.add_argument("--Equation", type=str, default="VacuumCCZ4_3D")

# IC enforcement mode
parser.add_argument("--ic_mode", type=str, default="soft", choices=["hard", "soft"],
                    help="IC enforcement: 'hard' uses the tau^2 wrapper; 'soft' adds penalty terms")

# Collocation point counts
parser.add_argument("--Nint",     type=int,   default=80000)
parser.add_argument("--NI",       type=int,   default=8000,  help="IC and tapa surface points")
parser.add_argument("--Nb",       type=int,   default=16000,  help="Boundary points")
parser.add_argument("--N_change", type=int,   default=250,   help="Resampling interval (epochs/BFGS restarts)")

# Loss weights
parser.add_argument("--lam_bc",       type=float, default=1.0)
parser.add_argument("--lamg0",        type=float, default=10.0,   help="Soft IC weight for metric")
parser.add_argument("--sampling_centers", type=str, default="track", choices=["track", "initial"],
                    help="Where the focused/core sampling clouds are centred. 'track' (default): the "
                         "current puncture centres (--bh_center_mode). 'initial': the punctures' t = 0 "
                         "positions in every window, while tracking (and the moving chi background) "
                         "still run -- a diagnostic for the drift (SBH/README.md): it removes the "
                         "loop tracked position -> off-centre clouds -> asymmetric shift.")
parser.add_argument("--symmetric_focus", action="store_true",
                    help="Draw each focused/core Gaussian cloud in point-reflected pairs (c + d, c - d), "
                         "so the cloud is exactly symmetric about its centre (up to --min_puncture_dist "
                         "rejections and projection at the outer sphere). Off by default (bitwise as before).")
parser.add_argument("--follow_puncture_sampling", action=argparse.BooleanOptionalAction, default=False,
                    help="Make the near-puncture focused collocation clouds follow the punctures' "
                         "motion INSIDE a window (default off = previous behavior). Without this the "
                         "focus centers are fixed at their window_start positions for the whole window, "
                         "while the punctures keep moving (~0.1-0.2 units per window on the orbital "
                         "runs, comparable to the ~0.3-wide puncture core), so late-in-window points "
                         "are progressively centered on where the puncture WAS. With it on, each "
                         "interior point drawn near puncture k (within 3x --int_focus_radius of its "
                         "window_start position) is rigidly shifted by that puncture's displacement "
                         "at the point's own time t, using the within-window trajectory that "
                         "--chi_bg_mode moving already refreshes at every resample, and the "
                         "window-end (tapa) surface is shifted to the puncture's window_end position. "
                         "Only takes effect under --bh_center_mode shift_driven with --chi_bg_mode "
                         "moving (the only configuration that has a within-window trajectory); the "
                         "first batch of each window, drawn before any trajectory exists, is unshifted. "
                         "Near-puncture rejection (--min_puncture_dist) is applied in the "
                         "window_start frame before the shift, which preserves each shifted point's "
                         "distance to its own puncture exactly.")
parser.add_argument("--orbit_focus_fraction", type=float, default=0.0,
                    help="Fraction of the interior AND initial/terminal-surface collocation points drawn "
                         "uniformly in a ball centered on the midpoint of the two punctures (the origin "
                         "for the symmetric setups) with radius --orbit_focus_mult times their window_start "
                         "separation, so the whole orbital region is covered directly instead of only "
                         "through the tails of the two per-puncture clouds (--int_focus_radius / "
                         "--ic_focus_radius). Default 0.0 = off (previous behavior). Taken out of the total: "
                         "with 0.15, 15%% of points are orbit-ball points and the other 85%% are split by "
                         "--int_focus_fraction/--ic_focus_fraction as before. Only for --n_stars 2. "
                         "Combined with --follow_puncture_sampling, orbit-ball points that lie within the "
                         "follow cutoff of a puncture are shifted along with it like any other nearby point.")
parser.add_argument("--orbit_focus_mult", type=float, default=1.5,
                    help="Radius of the --orbit_focus_fraction ball as a multiple of the punctures' "
                         "window_start separation (default 1.5), capped at --R_domain.")
parser.add_argument("--lam_ic_log",   type=float, default=0.0,
                    help="Weight of an extra RELATIVE (log-ratio) soft-IC term on chi and alpha: "
                         "lam_ic_log * mean[(ln chi - ln chi_prev)^2 + (ln alpha - ln alpha_prev)^2] "
                         "over the IC points, added inside the --lamg0 IC term (so it is scaled by "
                         "--lamg0 too). Logged in its own loss_ic_log column; loss_IC/loss_g0 keep "
                         "the plain absolute term only, so they stay comparable across runs. "
                         "Sign-blind: it sees |chi|, so it cannot detect chi flipping negative "
                         "(--w_pos guards that). Default 0.0 = off (previous "
                         "behavior). The plain IC term is an ABSOLUTE squared error, so near a puncture "
                         "-- where chi and alpha are small by construction -- even a total loss of the "
                         "puncture structure costs almost nothing; this term is scale-free, so erasing "
                         "a puncture costs the same as a comparable relative error anywhere. Start "
                         "small (0.01-0.1): a 1%% relative error contributes ~1e-4 here versus ~1e-6 for the "
                         "absolute term at the same accuracy, so 1.0 would make this term dominate the "
                         "Einstein loss. Only meaningful with --ic_mode soft "
                         "(hard IC satisfies the IC exactly, making this ~0).")
parser.add_argument("--lamgt0",       type=float, default=10.0,   help="Soft IC weight for dg/dt")
parser.add_argument("--lam_einstein", type=float, default=1.0)
parser.add_argument("--lam_gauge",    type=float, default=1.0)
parser.add_argument("--reinit_mode",  type=str,   default="warm",
                    choices=["warm", "full", "last_layer"],
                    help="warm: inherit previous window params (default); "
                         "full: re-initialize all params every window; "
                         "last_layer: warm-start but replace the output Dense layer with fresh weights")
parser.add_argument("--lam_tapa",     type=float, default=2.0,
                    help="Weight for the tapa (t_end surface) PDE residual relative to the interior (1=equal, 2=double weight)")
parser.add_argument("--gauge",        type=str,   default="ccz4",
                    choices=["ccz4", "static"],
                    help="Gauge evolution: 'ccz4' (default) uses the dynamical 1+log slicing + "
                         "Gamma-driver shift equations below (f_alpha, f_beta, eta); "
                         "'static' is a debugging fallback that freezes alpha=1, beta^i=0 for the "
                         "whole run (equivalent to f_alpha=f_beta=0, eta=0), useful for validating "
                         "the matter+CCZ4-gravity coupling in geodesic slicing before trusting the "
                         "dynamical gauge.")
parser.add_argument("--f_alpha", type=float, default=1.0,
                    help="1+log slicing coefficient: d_t alpha = beta^i d_i alpha - 2 alpha f_alpha Khat. "
                         "Paper's standard choice is f_alpha=1 (constant).")
parser.add_argument("--f_beta", type=float, default=1.0,
                    help="Gamma-driver shift coefficient: "
                         "d_t beta^i = beta^j d_j beta^i + (3/4) f_beta Ghat^i - eta*beta^i. "
                         "Paper's standard choice is f_beta=1 (constant).")
parser.add_argument("--eta", type=float, default=0.1,
                    help="Gamma-driver shift damping. Paper's guidance: eta ~ 2/M for the total "
                         "ADM mass M of the system -- set explicitly per run, no auto-detection. "
                         "Default 0.1 (bbh_headon_bssn_pinn_shift_driven_puncture.py's own default "
                         "is 0.0 -- pass --eta 0.0 explicitly to match it).")
parser.add_argument("--w_constraint", type=float, default=0.0,
                    help="Weight of a constraint penalty mean(Theta^2 + |Z^i|^2) on the interior points "
                         "(Z^i = chi (Ghat^i - Gt^i)/2). Both vanish for the true solution, and a constraint-"
                         "violating solution still satisfies the CCZ4 evolution equations, so the residual "
                         "loss alone sees that mode only through the initial data (SBH/README.md, mass "
                         "budget). 0 = off (default). Not with --dsgnar_steps (not in its residual vector). "
                         "Only for constraint-satisfying initial data (single puncture; NOT the Bowen-York "
                         "binary, whose Hamiltonian constraint is not solved).")
parser.add_argument("--w_pos", type=float, default=10.0,
                    help="Weight of the positivity penalty relu(-chi)^2 + relu(-alpha)^2 on the "
                         "interior collocation points, keeping chi and alpha from crossing zero "
                         "during training. Ported from "
                         "bbh_headon_bssn_pinn_shift_driven_puncture.py's --w-pos/point_terms.")
parser.add_argument("--kappa_z", type=float, default=0.0,
                    help="Z4 constraint-damping coefficient (drives Theta, Z_i -> 0). "
                         "Paper's guidance for neutron-star-scale scenarios: kappa_z <~ 1/M.")
parser.add_argument("--kappa_c", type=float, default=0.0,
                    help="Conformal-constraint-damping coefficient (drives det(gamma-tilde)->1, "
                         "trace(A-tilde)->0). Paper's guidance: kappa_c <~ 0.1/M.")
parser.add_argument("--kappa_2", type=float, default=0.0,
                    help="Secondary Z4 damping parameter appearing in the Khat/Theta equations as "
                         "kappa_z*(1-kappa_2) and kappa_z*(2+kappa_2). Standard CCZ4 default 0.")
parser.add_argument("--metric_bc",    type=str,   default="sommerfeld",
                    choices=["dirichlet", "sommerfeld"],
                    help="Boundary condition for the 20 CCZ4 metric-sector fields: 'dirichlet' fixes "
                         "each field to its t=0 initial-data value at the boundary; 'sommerfeld' "
                         "(default) applies the outgoing-wave condition on the perturbation from the "
                         "flat-space target (gamma-tilde_ij->delta_ij, chi,alpha->1, all others->0): "
                         "d_t X + c*(x/r)*d_r X + c*(X-X_flat)/r = 0.")
parser.add_argument("--loss_save_every", type=int, default=200)
parser.add_argument("--snapshot_save_every", type=int, default=None,
                    help="Cadence (in steps) for the 4 diagnostic PNG snapshots (fields, residuals, "
                         "gauge metric, native fields) -- pure visualization, never read back into "
                         "training, so raising this relative to --loss_save_every speeds up a run "
                         "with zero effect on the trained solution. Default None: defaults to "
                         "--loss_save_every (current/previous behavior: a snapshot every time a loss "
                         "line is logged). Was hardcoded to 2000 regardless of this help text's own "
                         "claim -- silently making snapshots 10x sparser than --loss_save_every's "
                         "own default of 200 -- until fixed.")
parser.add_argument("--snapshot_2d_every", type=int, default=1,
                    help="Write the per-window 2D (x,y) slice snapshots (diagnostics_2D_w*.png and "
                         "diagnostics_2D_zoom_w*.png) every this many windows (default 1 = every window; "
                         "the last window is always written; 0 disables them). Each render evaluates "
                         "the network and the Hamiltonian constraint (second derivatives) on two "
                         "~10^4-point grids, and is NOT tied to --snapshot_save_every, so with many "
                         "windows this is the way to thin it out (and under --ic_mode hard each render "
                         "also pays the ancestor sum, growing with the window index).")
parser.add_argument("--snapshot_2d_zoom_halfwidth", type=float, default=8.0,
                    help="Half-width (in the same M=1 coordinate units as --star_separation/"
                         "--R_domain) of a SECOND, zoomed-in per-window 2D (x,y) slice snapshot "
                         "(diagnostics_2D_zoom_w{window}.png), covering [-halfwidth,halfwidth] in "
                         "both x and y -- default 8.0 gives an 16M-wide view for the default "
                         "equal-mass (BL_masses 0.5,0.5, summing to M=1) setup. The full-domain "
                         "2D slice (diagnostics_2D_w{window}.png, spanning the whole "
                         "[-R_domain,R_domain] box) dilutes the near-binary dynamics across a much "
                         "larger plot area than the punctures actually occupy; this one reuses the "
                         "same point COUNT (see _SNAP_NXY_2D) over a much smaller physical area, "
                         "so it's also higher-resolution near the punctures, not just cropped.")
parser.add_argument("--log_term_grad_norms", action=argparse.BooleanOptionalAction, default=False,
                    help="Diagnostic (default off): also log ||d(term)/d(params)|| separately for "
                         "each loss term (Einstein, gauge, BC, IC, positivity) at the same "
                         "loss_save_every cadence, into the loss CSV/plot (columns gn_ein, gn_gauge, "
                         "gn_bc, gn_ic, gn_pos). Distinguishes 'this term's own gradient is tiny' "
                         "from 'its gradient is healthy but outweighed by the others' when a loss "
                         "term won't come down. Meaningfully more expensive than the default "
                         "combined-gradient logging (5 extra backward passes per logged step, each "
                         "through the same jacfwd(jacfwd(...))-based CCZ4 residual apply_model uses) "
                         "-- leave off for production runs, turn on only while debugging a specific "
                         "stuck-loss-term question.")

# PDE parameters
parser.add_argument("--c",        type=float, default=1.0, help="Wave speed for Sommerfeld BC")
parser.add_argument("--metric_det_eps",  type=float, default=1e-14)
parser.add_argument("--chi_floor", type=float, default=1e-5,
                    help="Floor for chi wherever the CCZ4 residual divides by it: chi_safe = "
                         "max(chi, chi_floor), the usual finite-difference choice. Caps every "
                         "1/chi factor at 1/chi_floor and keeps the divisor positive when an "
                         "under-trained network gives chi<=0. Must stay below the smallest "
                         "PHYSICAL chi at the collocation points (~2.9e-5 at t=0 for the default "
                         "masses and --min_puncture_dist 0.02; checked at startup). 0 restores "
                         "the pre-2026-10-10 behaviour, the sign-blind smooth floor "
                         "sqrt(chi^2+metric_det_eps^2).")

parser.add_argument("--constraint_ic", type=str, default=None,
                    help="Path to pinn_ic_state.pkl from EE_constraint_IC.py --fit_pinn. "
                         "When set, used as initial state for window 0 (soft IC mode only).")

# Binary boson-star / BH initial data
parser.add_argument("--n_stars",         type=int,   default=2, choices=[1, 2],
                    help="Number of boson stars/BHs in the initial data (1 or 2).")
parser.add_argument("--star_separation", type=float, default=4.0,
                    help="Distance between centres along the x-axis (n_stars=2 only). "
                         "bbh_headon_bssn_pinn_shift_driven_puncture.py's own reference run uses "
                         "2.0 (its D0=2*z0, z0=1.0); this file's own default of 4.0 is a wider "
                         "starting separation for a quasi-circular orbit, not a match to that "
                         "reference.")

# Initial data: always Brill-Lindquist + Bowen-York BH puncture data
# (conformally-flat, pre-collapsed lapse alpha=1/psi, zero shift), evolved
# with the moving-puncture gauge (1+log slicing + Gamma-driver) -- no
# excision, the puncture is evolved through directly. Vacuum (no matter
# field). Time-symmetric (K_ij=0) by default; --BL_momenta/--BL_spins
# generalize this to Bowen-York data (boosted/spinning punctures, K_ij!=0).
parser.add_argument("--BL_masses", type=str, default="0.5,0.5",
                    help="Comma-separated bare masses m_A for Brill-Lindquist initial data. "
                         "Number of entries must equal --n_stars. Default 0.5,0.5 (equal-mass "
                         "binary) matches bbh_headon_bssn_pinn_shift_driven_puncture.py.")
parser.add_argument("--BL_momenta", type=str, default="0.0,0.0",
                    help="Comma-separated x-momenta P_A (Bowen-York linear momentum) for each "
                         "puncture -- the RADIAL component when combined with --BL_momenta_y's "
                         "tangential component (e.g. for a quasi-circular setup). A single value "
                         "broadcasts to every BH; otherwise the count must equal --n_stars. 0.0 "
                         "recovers the original time-symmetric Brill-Lindquist data exactly (for "
                         "any --n_stars, provided --BL_momenta_y is also all-zero). Default 0.0,0.0: "
                         "no radial momentum -- this file's own default is a quasi-circular orbit "
                         "driven entirely by --BL_momenta_y's tangential component, not a head-on "
                         "collision. NOTE for --n_stars 1: this 2-value default does NOT broadcast "
                         "(broadcasting only applies to a SINGLE value) -- like --BL_masses, "
                         "--n_stars 1 requires an explicit single-value override here (and for "
                         "--BL_momenta_y below), e.g. --BL_momenta 0.0. For a head-on collision "
                         "instead, pass e.g. 0.3,-0.3 (inward "
                         "Bowen-York momentum of magnitude 0.3: puncture A at negative x gets +0.3, "
                         "puncture B at positive x gets -0.3), matching "
                         "bbh_headon_bssn_pinn_shift_driven_puncture.py's --P 0.3. NOTE: the "
                         "punctures THEMSELVES (--star_separation's placement) "
                         "still start exactly on the x-axis regardless of --BL_momenta_y -- a "
                         "nonzero tangential momentum is what curves the subsequent trajectory "
                         "into an orbit, the standard Bowen-York way to set up quasi-circular data, "
                         "not an initial off-axis position.")
parser.add_argument("--BL_momenta_y", type=str, default="-0.16,0.16",
                    help="Comma-separated y-momenta P_A (Bowen-York linear momentum's TANGENTIAL "
                         "component, for --n_stars 2 in the x-y plane) for each puncture -- same "
                         "broadcast rule as --BL_momenta (a single value broadcasts to every BH). "
                         "Default -0.16,0.16: a bound orbit out of the box (opposite-"
                         "signed tangential momenta so the combined linear momentum stays zero, "
                         "same convention as --BL_momenta's own broadcast rule) -- pass 0 (or 0,0) "
                         "for pure radial motion, i.e. the original head-on-collision behavior. "
                         "NOTE for --n_stars 1: this 2-value default does NOT broadcast -- pass an "
                         "explicit single value here too (e.g. --BL_momenta_y 0), see --BL_momenta's "
                         "own note. "
                         "For a DIFFERENT quasi-circular setup, give the two punctures OPPOSITE-"
                         "signed values here (e.g. 0.1,-0.1) so their combined linear momentum "
                         "stays zero. All three of --bh_center_mode shift_driven/chi_min/"
                         "lapse_min, and --chi_bg_mode moving, follow the resulting off-axis "
                         "(x,y) motion (chi_min/lapse_min via a genuine 2D x-y plane scan, not "
                         "just the x-axis -- see locate_chi_minima/find_bh_centers_lapse). The "
                         "one remaining x-axis-only exception is shift_driven's own chi-min "
                         "CROSS-CHECK column in puncture_track.txt (diagnostic only, not used "
                         "for sampling/training).")
parser.add_argument("--BL_spins", type=str, default="0,0,0",
                    help="Semicolon-separated x,y,z spin-vector triples S_A (Bowen-York spin) "
                         "for each puncture. A single triple broadcasts to every BH; otherwise "
                         "the number of semicolon-separated entries must equal --n_stars, each "
                         "with exactly 3 comma-separated components. Default 0,0,0: no spin "
                         "contribution, for any --n_stars. "
                         "Example: --BL_spins 0,0,0.1;0,0,-0.1.")
parser.add_argument("--bh_center_mode", type=str, default="shift_driven",
                    choices=["fixed", "lapse_min", "shift_driven", "chi_min"],
                    help="BH-center tracking, used both to steer the sampling-focus centers each "
                         "window and as the post-training trajectory diagnostic. All three of "
                         "'shift_driven'/'chi_min'/'lapse_min' write the same puncture-track "
                         "diagnostic outputs each window (puncture_track.txt, "
                         "puncture_track_fine.txt, diagnostics_puncture_track.png) -- 'shift_driven' "
                         "logs a within-window RK4 sub-grid there (a genuine continuous "
                         "trajectory), while 'chi_min'/'lapse_min' log a single re-located point "
                         "per window (they re-scan from scratch each window, so there is no "
                         "sub-grid to log). 'fixed' logs nothing (centers never move). "
                         "Mode semantics: "
                         "'shift_driven' (default): each puncture's x-position is propagated by "
                         "RK4-integrating dx_A/dt = -beta^x(x_A,y=eps,z=eps,t) through the "
                         "PREVIOUS window's trained network across [window_start,window_end] "
                         "(--track_steps substeps) -- ported from the shift-driven puncture "
                         "transport in bbh_headon_bssn_pinn_shift_driven_puncture.py, generalized "
                         "to independent per-puncture tracking (no left/right symmetry assumed), "
                         "so unequal --BL_masses/--BL_momenta are tracked correctly. "
                         "'lapse_min': from window 1 on, centers are relocated to the minima of "
                         "the lapse alpha=1/sqrt(-g^tt) of the PREVIOUS window's network, scanned "
                         "over the whole x-y plane (z=0) at t=window start -- follows an orbiting "
                         "puncture off the x-axis too (--BL_momenta_y), not just a moving-along-x "
                         "one. For n_stars=2 this finds ALL local minima of alpha across the whole "
                         "scanned plane, ranks them by depth, and keeps the deepest n_stars "
                         "subject to a minimum (Euclidean) separation -- NOT a fixed x<0/x>0 "
                         "half-axis split, so it stays correct if the two minima cross x=0 or "
                         "merge into one basin near merger; for n_stars=1 the global minimum is "
                         "used. "
                         "'chi_min': same idea as 'lapse_min' but tracks the minima of the "
                         "conformal factor chi instead of the lapse -- reuses locate_chi_minima's "
                         "multi-stage refinement/candidate-selection (the same routine already used "
                         "as a post-hoc cross-check under --bh_center_mode shift_driven), so centers "
                         "follow wherever chi is currently deepest on the PREVIOUS window's network, "
                         "rather than a physically-integrated (shift_driven) or lapse-based "
                         "(lapse_min) notion of puncture position. Incompatible with --chi_bg_mode "
                         "moving, same restriction as 'lapse_min'/'fixed' (see --chi_bg_mode). "
                         "'fixed': sampling centers stay at the initial analytic positions for "
                         "all windows.")
parser.add_argument("--track_steps", type=int, default=1000,
                    help="RK4 substeps used by --bh_center_mode shift_driven to integrate each "
                         "puncture's trajectory across one window. Independent of "
                         "--moving_bg_refresh_steps (its own separate, hardcoded-default flag, "
                         "not derived from this one) -- see there.")
parser.add_argument("--moving_bg_refresh_steps", type=int, default=20,
                    help="RK4 substeps used ONLY for the --chi_bg_mode moving background's "
                         "mid-training refreshes (_refresh_traj), as opposed to --track_steps which "
                         "is still used for the one-per-window FINAL/logged trajectory in "
                         "advance_shift_tracked_centers. _refresh_traj/_integrate_puncture_trajectory "
                         "is an unjitted, Python-looped RK4 integration (each substep is its own "
                         "tiny, undispatched network forward pass), and it reruns from scratch on "
                         "EVERY BFGS outer restart and Adam resample event (tens of times per "
                         "window) -- at --track_steps' default of 1000 that is ~1000*4*n_stars "
                         "unjitted forward passes PER REFRESH, tens of thousands total per window, "
                         "which is what makes --chi_bg_mode moving dramatically slower than "
                         "--chi_bg_mode physical (which never calls this mid-training at all). The "
                         "background only needs to track the APPROXIMATE current position between "
                         "refreshes, not a high-precision trajectory, so a much smaller step count "
                         "here is fine -- the final per-window trajectory that actually gets logged/ "
                         "reported still uses the full --track_steps precision.")
parser.add_argument("--chi_bg_mode", type=str, default="moving",
                    choices=["physical", "moving"],
                    help="'physical': the network's chi output IS the physical "
                         "conformal factor, exactly as in EEccz4_BH.py. "
                         "'moving' (default): ported from bbh_headon_bssn_pinn_shift_driven_puncture.py's "
                         "moving_bl_chi -- the network's chi output instead represents the SMOOTH "
                         "regular deviation from the analytic Brill-Lindquist background evaluated "
                         "at each puncture's live shift-driven-tracked position; the physical chi "
                         "used everywhere (residual, BC, diagnostics) is background+regular, "
                         "reconstructed by a wrapper around the network's apply function. The "
                         "background here is generalized to independent per-puncture tracking "
                         "(unequal --BL_masses/--BL_momenta, --n_stars 1 or 2) and works under "
                         "EITHER --ic_mode, unlike the source file (hard-only). Requires "
                         "--bh_center_mode shift_driven, since the tracked positions are shared "
                         "with that mechanism.")
parser.add_argument("--min_puncture_dist", type=float, default=0.02,
                    help="Minimum Cartesian distance interior/IC-surface collocation points must "
                         "keep from each puncture's CURRENT tracked center (the same 'centers' "
                         "already used to steer the focused-sampling cloud, i.e. --bh_center_mode's "
                         "shift-driven/lapse-min/fixed positions -- moves with the puncture in "
                         "shift_driven mode). 0.0 (default): disabled, unchanged behavior -- the "
                         "focused sampling cloud is free to place points arbitrarily close to a "
                         "puncture, same as bbh_headon_bssn_pinn_shift_driven_puncture.py, which "
                         "has no such mechanism at all. >0: any point closer than this is rejected "
                         "and redrawn from the same focused distribution, repeating until none "
                         "remain (capped retries) -- this file still evolves straight through the "
                         "puncture with no excision, so this only avoids SAMPLING collocation "
                         "points in the steepest part of the near-singular region, unlike a true "
                         "excision zone which would also exclude it from the physics itself.")
parser.add_argument("--mask_rel_threshold", type=float, default=50.0,
                    help="Relative-magnitude threshold used by _mask_divergent/_apply_plot_nan to "
                         "decide what counts as 'divergent enough to hide' in diagnostic plots: a "
                         "value is masked to NaN if it is non-finite or exceeds this many times the "
                         "array's own median-of-finite magnitude. Lower values hide more (mask more "
                         "aggressively, e.g. to also suppress mild near-puncture noise); higher "
                         "values hide less (only the most extreme outliers). Purely cosmetic -- "
                         "does not affect training, only what is NaN'd out of saved plots.")
parser.add_argument("--min_center_separation", type=float, default=0.2,
                    help="Minimum x-distance between two DETECTED puncture positions (lapse minima "
                         "for --bh_center_mode lapse_min's find_bh_centers_lapse, or chi minima for "
                         "locate_chi_minima's cross-check) before they are treated as distinct -- "
                         "closer than this, _pick_bh_centers_2d collapses them into one repeated "
                         "position, reporting the deepest minimum for every remaining slot instead "
                         "of a spurious split. Was previously a hardcoded 0.1 with no way to tune it "
                         "without editing source; lowering it delays that merge until the punctures' "
                         "detected minima are genuinely closer together (e.g. during late inspiral, "
                         "well before the true horizon merger this default was tuned for).")

# Random seed
parser.add_argument("--SEED", type=int, default=42)

# Sampling focus
parser.add_argument("--ic_focus_fraction",  type=float, default=0.6)
parser.add_argument("--ic_focus_radius",    type=float, default=2.0)
parser.add_argument("--int_focus_fraction", type=float, default=0.6)
parser.add_argument("--int_focus_radius",   type=float, default=2.0)
# Extra puncture-core points (2026-10-13). ADDED on top of --Nint / --NI (nothing is
# taken from the other clouds), drawn from an independent random stream, so with
# 0 (the default) every point set is bitwise identical to before, and with > 0 the
# base points are still the same as without them.
parser.add_argument("--core_points", type=int, default=0,
                    help="Extra interior collocation points per resample, drawn from a Gaussian of "
                         "std --core_radius around each puncture (split equally), with the same time "
                         "distribution, --min_puncture_dist rejection and --follow_puncture_sampling "
                         "shift as the other interior points. Added to --Nint. Default 0 = off.")
parser.add_argument("--core_ic_points", type=int, default=0,
                    help="Extra points on the IC surface AND on the tapa surface, from the same core "
                         "Gaussian. Added to --NI. Default 0 = off.")
parser.add_argument("--core_radius", type=float, default=0.5,
                    help="Std of the core Gaussian(s) used by --core_points / --core_ic_points.")

# Time distribution of interior collocation points
parser.add_argument("--time_dist", type=str, default="exponential",
                    choices=["uniform", "exponential"],
                    help="Time distribution for interior collocation points: "
                         "'uniform' (default): t ~ Uniform[t_start, t_end]; "
                         "'exponential': t ~ truncated-Exp from t_start with width sigma_time, "
                         "which grows linearly with resampling steps from sigma_time_0 * window_width "
                         "to ~20x window_width at f_uniform * total_resamplings (then uniform).")
parser.add_argument("--sigma_time_0", type=float, default=0.001,
                    help="Initial sigma_time as a fraction of window width (exponential mode only). "
                         "At resampling step 0: sigma_time = sigma_time_0 * (t_end - t_start).")
parser.add_argument("--f_uniform", type=float, default=0.7,
                    help="Fraction of total resampling steps after which the exponential time "
                         "distribution reaches the uniform limit (sigma = 20 * window_width). "
                         "sigma_time grows linearly from sigma_time_0 to 20x window_width "
                         "over this fraction (exponential mode only).")
parser.add_argument("--causal_weight", action=argparse.BooleanOptionalAction, default=True,
                    help="Diagnostic/experimental (default on; pass --no-causal_weight to disable): "
                         "reweight the Einstein+gauge "
                         "residual loss (loss_ccz4's ri/rf terms) by a temporal-causality factor "
                         "a la Wang/Sankaran/Perdikaris 'Respecting causality is all you need for "
                         "training physics-informed neural networks'. The window is split into "
                         "--causal_buckets sequential time buckets; each bucket's contribution is "
                         "weighted by exp(-causal_eps * cumulative mean residual from EARLIER "
                         "buckets, stop-gradiented), so a late bucket only gets to matter once "
                         "earlier buckets are actually solved. Complementary to --time_dist "
                         "exponential (which only biases where collocation points LAND in t; this "
                         "biases how much they COUNT in the loss once sampled) -- the two can be "
                         "combined. No extra forward/backward pass: reuses the already-computed "
                         "per-point residual, just changes its reduction.")
parser.add_argument("--causal_eps", type=float, default=1.0,
                    help="Causal weighting steepness (--causal_weight only). Larger = stricter "
                         "(later buckets contribute ~0 until earlier ones are small); 0 recovers "
                         "the unweighted loss exactly. Fixed for the whole run -- no annealing "
                         "schedule yet (c.f. --sigma_time_0/--f_uniform's schedule for the "
                         "sampling-side curriculum).")
parser.add_argument("--causal_buckets", type=int, default=8,
                    help="Number of sequential time buckets to discretize the window into for "
                         "--causal_weight.")

# Domain
parser.add_argument("--R_domain", type=float, default=24.0,
                    help="Radius of the spherical spatial domain. "
                         "bbh_headon_bssn_pinn_shift_driven_puncture.py's own reference run uses "
                         "--L 6.0 (box half-width, same linear extent though that file samples a "
                         "cube, not a sphere); this file's own default of 24.0 is a larger domain "
                         "for a quasi-circular orbit's wider excursion, not a match to that "
                         "reference.")
parser.add_argument("--tfinal", type=float, default=50.0,
                    help="Final evolution time. bbh_headon_bssn_pinn_shift_driven_puncture.py's "
                         "run script uses --t-final 2.0 for its own (shorter) reference run; "
                         "this file's own default of 50.0 is a longer production evolution, not "
                         "a match to that reference.")
parser.add_argument("--r_eps",  type=float, default=1e-8)

# Training
parser.add_argument("--N_ramp",       type=int,   default=0,
                    help="Steps over which lam_einstein ramps from 0 to target (0 = no ramp)")
parser.add_argument("--EPOCHS_ADAM", type=int,   default=0)
parser.add_argument("--EPOCHS_QN",   type=int,   default=12000)
parser.add_argument("--N_window",    type=int,   default=200,
                    help="Number of temporal training windows: window_edges = "
                         "linspace(0, tfinal, N_window+1). Default 200 (with --tfinal 50.0) gives "
                         "0.25-wide windows. bbh_headon_bssn_pinn_shift_driven_puncture.py's run "
                         "script uses --stage-dt for its own (shorter) reference run (8 windows "
                         "of 0.25 width at its --t-final 2.0); this file's own defaults are a "
                         "longer production evolution, not a match to that reference.")

parser.add_argument("--blowup_factor", type=float, default=0.0,
                    help="Loss blow-up guard for the BFGS restarts (0 = off, the default). If a restart "
                         "ends with a finite loss above blowup_factor x the best restart loss of the "
                         "window so far, its weights are discarded: training rolls back to the best "
                         "restart's weights, resets the Hessian to identity and resamples. After "
                         "--blowup_max rollbacks in one window the window ends with the best weights. "
                         "The NaN guard alone keeps the last FINITE weights, which can be garbage "
                         "(single_m05_jax09_core48k_qn24k_t10_2gpu, window 17: loss 2e-6 -> 1 -> 1e31). "
                         "Production: 100.")
parser.add_argument("--blowup_max", type=int, default=3,
                    help="Rollbacks per window allowed by --blowup_factor before the window ends.")
parser.add_argument("--patience_bfgs", type=int,   default=500,
                    help="Early-stopping patience for BFGS outer restarts (0 = disabled). "
                         "Stops if no relative improvement > min_delta for this many restarts.")
parser.add_argument("--min_delta",     type=float, default=1e-3,
                    help="Minimum relative loss improvement (best-new)/|best| to reset "
                         "the patience counter.")
# DSGNAR (doubly-sketched Gauss-Newton, arXiv:2607.02194), vendored from Stationary/ (2026-10-18).
# Flags mirror Stationary's --dsgnar-* ones; see SBH/README.md section 4 and Stationary/DSGNAR.md.
# Flags with default None are the ones --dsgnar_paper presets; None = "not given".
parser.add_argument("--dsgnar_steps", type=int, default=0,
                    help="DSGNAR iterations per window and cycle (0 = off, default). Runs on the window's "
                         "parameters before or after the QN restarts (--dsgnar_order).")
parser.add_argument("--dsgnar_order", type=str, default="before_qn", choices=["before_qn", "after_qn"],
                    help="before_qn (default): Adam -> DSGNAR -> QN polish. after_qn: Adam -> QN -> DSGNAR. "
                         "--EPOCHS_QN 0 gives DSGNAR alone.")
parser.add_argument("--dsgnar_cycles", type=int, default=1,
                    help="Restart cycles per window: each from the previous cycle's parameters with a FRESH "
                         "sample and a FRESH trust region (delta0, initial lambda) -- Stationary's scheme "
                         "against the trust region collapsing. Iterations per cycle: --dsgnar_cycle_steps.")
parser.add_argument("--dsgnar_cycle_steps", type=int, default=None,
                    help="Iterations per cycle (default: --dsgnar_steps).")
parser.add_argument("--dsgnar_resample_every", type=int, default=0,
                    help="Redraw the collocation sample every N DSGNAR iterations inside a cycle, keeping the "
                         "trust region (0 = fixed sample within a cycle, default).")
parser.add_argument("--dsgnar_sketch", type=int, default=0,
                    help="Sketch size s (0 = n_params/3, the paper's lower end). Cost per iteration ~ s JVPs.")
parser.add_argument("--dsgnar_hashes", type=int, default=None, help="CountSketch hashes K (default 4).")
parser.add_argument("--dsgnar_chunk", type=int, default=4,
                    help="JVP tangents pushed through the residual at once (memory ~ chunk x residual "
                         "workspace; 4 is safe at the default point counts).")
parser.add_argument("--dsgnar_delta0", type=float, default=1.0, help="Initial trust-region radius.")
parser.add_argument("--dsgnar_delta_min", type=float, default=None, help="Stop below this radius (default 1e-14).")
parser.add_argument("--dsgnar_omega", type=float, default=1e-8, help="LM regularisation floor.")
parser.add_argument("--dsgnar_stage1_ratio", type=float, default=None, help="Target ratio, stage 1 (default 0.15).")
parser.add_argument("--dsgnar_stage2_ratio", type=float, default=None, help="Target ratio, stage 2 (default 0.5).")
parser.add_argument("--dsgnar_backtrack", type=float, default=0.5, help="Line-scan factor.")
parser.add_argument("--dsgnar_backtrack_steps", type=int, default=4, help="Line-scan fractions 1, 1/2, ... (4).")
parser.add_argument("--dsgnar_line_scan", action=argparse.BooleanOptionalAction, default=None,
                    help="Scan fractions of the step (default on; the paper evaluates the step once).")
parser.add_argument("--dsgnar_calibrate", action=argparse.BooleanOptionalAction, default=None,
                    help="Median calibration of the sketched model's predicted decrease (default on).")
parser.add_argument("--dsgnar_shrink_on_accept", action="store_true", default=None,
                    help="Radius to the PCHIP radius on accept (the paper); default: grow-only.")
parser.add_argument("--dsgnar_radius_floor", type=float, default=None,
                    help="Radius floor as a fraction of its high-water mark (default 1e-3; paper 0).")
parser.add_argument("--dsgnar_reweight", type=float, default=None,
                    help="Per-condition reweighting exponent (paper: 0.05; default 0 = off). Conditions: "
                         "einstein/gauge x interior/tapa, bc, ic, (ic_log), pos.")
parser.add_argument("--dsgnar_fixed_sketch", action="store_true", help="One sketch for the whole cycle.")
parser.add_argument("--dsgnar_reject_overshoot", action=argparse.BooleanOptionalAction, default=None,
                    help="Reject a step whose ratio overshoots target + 0.1 and expand the radius (the paper's "
                         "TEXT, 3.4; on in --dsgnar_paper). --no-dsgnar_reject_overshoot: accept any step that "
                         "lowers the loss, as the paper's own CODE does (ldc/ldc.ipynb).")
parser.add_argument("--dsgnar_jacobian", type=str, default="cols", choices=["cols", "rows"],
                    help="How the sketched Jacobian C J Omega S is built. cols (default): s forward "
                         "tangents through the whole residual. rows: the paper code's way -- per-point "
                         "reverse-mode Jacobians vmapped over --dsgnar_row_batch points, folded into the "
                         "CountSketch buckets, SRCT applied once; 2.5x cheaper on CPU, same J~ to round-off.")
parser.add_argument("--dsgnar_residual_sketch", type=str, default="count", choices=["count", "none"],
                    help="count (default, the paper): CountSketch on the residual rows too. none (needs "
                         "--dsgnar_jacobian rows): exact residual, only the parameters sketched -- the "
                         "s x s Gram matrix (J Omega S)^T (J Omega S) and (J Omega S)^T r are accumulated over "
                         "every row. Near the residual floor the row sketch's noise swamps the true gradient "
                         "and makes ~half the steps uphill (tools/sketch_noise.py); this mode does not.")
parser.add_argument("--dsgnar_row_batch", type=int, default=64,
                    help="Points per macro-batch of --dsgnar_jacobian rows (memory ~ batch x 24 x n_params x 8 B).")
parser.add_argument("--dsgnar_max_seconds", type=float, default=0.0,
                    help="Wall-time budget of each DSGNAR cycle in seconds (0 = none): the phase stops after "
                         "the iteration that exceeds it -- for comparisons at equal GPU time.")
parser.add_argument("--dsgnar_paper", action="store_true",
                    help="The paper's settings (reweight 0.05, reject overshoot, no line scan, no "
                         "calibration, shrink on accept, no floor, K 4, targets 0.075/0.5, delta_min 1e-8); "
                         "explicit flags override.")
parser.add_argument("--dsgnar_log_every", type=int, default=1,
                    help="Write the loss history every N DSGNAR iterations (1 = every iteration).")
parser.add_argument("--qn_fixed_batch", action="store_true",
                    help="Do NOT redraw the collocation sample between QN restarts (Stationary's best "
                         "SSBroyden scheme: one fixed sample, inverse Hessian carried). The NaN, excursion "
                         "and blow-up guards still redraw it.")
parser.add_argument("--deriv_mode", type=str, default="autodiff", choices=["autodiff", "taylor"],
                    help="How r_ccz4_evolution gets the network's derivatives. 'autodiff' (default): "
                         "per-point jvp (t) + jacfwd (x,y,z) + jacfwd-of-jacfwd (second spatial "
                         "derivatives), bit-for-bit the behaviour of earlier runs. 'taylor': one batched "
                         "second-order forward pass through the tanh MLP (value, 4 first and the 6 "
                         "distinct second spatial derivatives), plus autodiff of the analytic moving chi "
                         "background; same mathematics, agrees with autodiff to round-off (~1e-16 "
                         "relative), fewer operations. Falls back to autodiff for --ic_mode hard.")
parser.add_argument("--line_search", type=str, default="wolfe", choices=["armijo", "wolfe"],
                    help="BFGS/SSBroyden line search. 'wolfe' (default): strong-Wolfe "
                         "bracketing/zoom search with cubic/quadratic interpolation (gradients "
                         "only at sufficient-decrease points, steps may exceed 1, curvature "
                         "condition y.s>0 guaranteed on success), as used in the SSBroyden PINN "
                         "paper (arXiv:2405.04230). 'armijo': backtracking by halving from step 1 "
                         "until sufficient decrease (one loss evaluation per trial; the "
                         "historical default, used by every run before 2026-10).")
parser.add_argument("--wolfe_c2", type=float, default=0.9,
                    help="Curvature constant c2 of the strong-Wolfe search (--line_search wolfe).")
parser.add_argument("--es_min_loss", type=float, default=0.0,
                    help="Early stopping (and the Hessian resets it triggers) only fires when the best "
                         "loss of the window is also below this absolute value (0 = no condition, "
                         "default). Guards against stopping a window that is flat at a high loss. "
                         "Note: patience counts BFGS restarts (EPOCHS_QN/N_change = 48 per window with "
                         "the defaults), so the default --patience_bfgs 500 never fires; to use early "
                         "stopping set e.g. --patience_bfgs 10 --min_delta 1e-2 --es_min_loss <floor>.")
parser.add_argument("--h0_scaling", type=int, default=1, choices=[0, 1],
                    help="1 (default): when a BFGS restart starts from the identity inverse Hessian "
                         "(first restart of a window, or after a reset), rescale it to (y.s/y.y) I "
                         "after the first line search (Nocedal & Wright eq. 6.20) so the first steps "
                         "are well scaled. 0: raw identity (behaviour of runs before 2026-10-16).")
parser.add_argument("--qn_scale_invariant", type=int, default=1, choices=[0, 1],
                    help="1 (default since 2026-10-19, the Crunch authors' recommendation): Crunch's "
                         "scale_invariant option -- relative curvature guards (y.s <= 1e-14 |s||y|) and the "
                         "H update computed on the rescaled pair, so the optimizer does not depend on the "
                         "loss scale. Without it every update is skipped once y.s < 1e-14 (loss ~1e-12) "
                         "and the QN 'floor' can be that threshold, not the network. Same cost. 0 = the "
                         "behaviour of every run before 2026-10-19 (bitwise; use it to compare with them).")
parser.add_argument("--qn_gtol", type=float, default=1e-9,
                    help="QN stops a restart when |grad| <= this ABSOLUTE value (not made relative by "
                         "--qn_scale_invariant). Lower it, or set 0, when the loss is very small: e.g. a "
                         "network whose last layer starts at zero (gradient ~1e-11) stops at iteration 0.")
parser.add_argument("--qn_loss_norm", type=int, default=0, choices=[0, 1],
                    help="1: inside the QN optimizer, divide the loss by its value at the start of the "
                         "window's QN phase (fixed for the window, the span over which H is carried), so "
                         "the identity H of the first step is well scaled whatever the loss level; the "
                         "logs and guards still see the true loss, and --qn_gtol becomes relative. With "
                         "--qn_scale_invariant 1 this makes the QN trajectory independent of the loss "
                         "scale (tools/test_scale_invariance.py). 0 (default) = off.")
parser.add_argument("--excursion_ratio", type=float, default=100.0,
                    help="Excursion guard: if the loss at the end of a BFGS restart exceeds this "
                         "factor times the loss at the end of the previous accepted restart, the "
                         "restart is discarded (weights restored, inverse Hessian reset, batch "
                         "redrawn at the same sampling stage) and retried. Catches large finite "
                         "loss jumps (e.g. chi crossing zero under --chi_floor) that the NaN guard "
                         "cannot see. 0 = disabled.")
parser.add_argument("--excursion_abs", type=float, default=1e-2,
                    help="Excursion guard for the FIRST restart of windows after the first: discard "
                         "and retry if its loss exceeds max(this value, excursion_ratio x previous "
                         "window's final loss). 0 = disabled.")
parser.add_argument("--excursion_retries", type=int, default=3,
                    help="Maximum number of excursion-guard retries per window (0 = guard off).")
parser.add_argument("--qn_update", type=str, default="ssbroyden2", choices=["ssbroyden2", "score", "bfgs"],
                    help="Quasi-Newton inverse-Hessian update. 'ssbroyden2' (default): self-scaled "
                         "Broyden (arXiv:2405.04230). 'score': SCORE (Si, An, Ma & Yan, "
                         "arXiv:2608.04206): SSBroyden whose secant pair is shifted to (s, y + mu_k s) "
                         "with mu_k = clip(score_eta*lambda_k, score_mu_min, score_mu_max), lambda_k = "
                         "sqrt(g.H.g), and whose strong-Wolfe search starts from 1/(1+lambda_k). "
                         "Needs --line_search wolfe. 'bfgs': plain BFGS.")
parser.add_argument("--score_eta", type=float, default=0.1,
                    help="SCORE: shift per unit decrement, mu_k = clip(eta*lambda_k, mu_min, mu_max).")
parser.add_argument("--score_mu_min", type=float, default=1e-8,
                    help="SCORE: lower clip of the curvature shift mu_k (units loss/parameter^2).")
parser.add_argument("--score_mu_max", type=float, default=1e-4,
                    help="SCORE: upper clip of mu_k. The paper does not publish eta/mu_min/mu_max; "
                         "on a scaled-down copy of its Burgers benchmark eta=1, mu_max=1e-2 stalled "
                         "~30x above SSBroyden and these defaults tied it (no gain found).")

# Architecture
parser.add_argument("--N_LAYERS",  type=int, default=6)
parser.add_argument("--HIDDEN",    type=int, default=40)
parser.add_argument("--FEATURES",  type=int, default=22,
                    help="Must be 22 (vacuum CCZ4, no matter sector): gamma-tilde_ij(6), "
                         "A-tilde_ij(6), chi, Khat, Theta, Gamma-hat^i(3), alpha, beta^i(3)")

# Learning rate
parser.add_argument("--LR",         type=float, default=1e-3)
parser.add_argument("--decay_rate", type=float, default=0.9)
parser.add_argument("--decay_step", type=int,   default=5000)

# Resume from a previous checkpoint
parser.add_argument("--resume_checkpoint", type=str, default=None,
                    help="Path to pinn_checkpoint.pkl from a previous run -- this is a small "
                         "manifest listing each completed window's own pinn_checkpoint_w####.pkl "
                         "file (which must still sit alongside it in the same directory), not the "
                         "full per-window data itself. Completed windows are loaded and skipped; training "
                         "resumes from the first window not yet in the checkpoint. "
                         "Architecture/window flags (N_LAYERS, HIDDEN, FEATURES, "
                         "tfinal, N_window, ic_mode) must match the original run "
                         "exactly -- N_window is included because window_edges is "
                         "recomputed as linspace(0, tfinal, N_window+1), so changing "
                         "N_window shifts EVERY window boundary, not just adds one, "
                         "desyncing it from the loaded windows' recorded t_start/t_end.")

parser.add_argument("--outdir", type=str, default=".",
                    help="Directory for every output (CSV, PNG, TXT, checkpoints). Created if "
                         "absent; relative paths are relative to the current directory. Default "
                         "'.', i.e. the current directory, as before. --resume_checkpoint is a path "
                         "of its own (usually OUTDIR/pinn_checkpoint.pkl).")

parser.add_argument("--resume_windows", type=int, default=0,
                    help="With --resume_checkpoint: load only the first K completed windows (and their "
                         "puncture-track entries) and continue from window K+1, i.e. fork a run at a "
                         "window boundary. 0 (default) = all windows in the checkpoint. Note: the RNG "
                         "key is not checkpointed, so any resume draws different samples than the "
                         "uninterrupted run would; fork every arm of a comparison the same way.")
parser.add_argument("--stop_after_window", type=int, default=0,
                    help="Stop after this many windows are complete (1-based, as in the log's "
                         "'Training window k/N'); 0 = run to N_window. The checkpoint is written as usual.")
parser.add_argument("--fast_hessian", action="store_true",  # [GPUS]
                    help="[GPUS] Compile Crunch's SSBroyden inverse-Hessian update with jax.jit (one "  # [GPUS]
                         "fused kernel instead of ~35 separate GPU operations and three materialised "  # [GPUS]
                         "n x n outer products) and keep the inverse Hessian replicated on every GPU. "  # [GPUS]
                         "Same mathematics; results agree with the default to round-off.")  # [GPUS]
parser.add_argument("--loss_plot_every", type=int, default=1,  # [GPUS]
                    help="[GPUS] Redraw loss_window*.png only every N loss-history saves (and at the "  # [GPUS]
                         "end of each window). 1 (default) = every save, as before; the CSV is always "  # [GPUS]
                         "written at every save.")  # [GPUS]
parser.add_argument("--profile_restarts", action="store_true",  # [GPUS]
                    help="[GPUS] Time every BFGS restart by component (loss evaluations, optimizer "  # [GPUS]
                         "overhead, Hessian post-processing, trajectory refresh, logging evaluation, "  # [GPUS]
                         "resampling) and print a per-window summary. Adds a device sync after each "  # [GPUS]
                         "component, so use it for profiling runs only.")  # [GPUS]
parser.add_argument("--n_gpus", type=int, default=1,  # [GPUS]
                    help="[GPUS] Data parallelism over collocation points on this many GPUs of one "  # [GPUS]
                         "node (automatic sharding: every per-point training array is split along "  # [GPUS]
                         "the point axis, parameters and the BFGS inverse Hessian are replicated; "  # [GPUS]
                         "the jitted loss is unchanged). 1 (default) = the original single-device "  # [GPUS]
                         "code path. Point counts must be divisible by n_gpus. See PINN_LESSONS §14.")  # [GPUS]

args, unknown = parser.parse_known_args()

if unknown:
    print(f"WARNING: unrecognized command-line argument(s), IGNORED: {unknown} "
          f"-- if this includes a misspelled/renamed flag, the corresponding "
          f"default value is being used instead of your intended override.")

for arg, value in vars(args).items():
    print(f"{arg}: {value}")

# Every output path below is os.path.join(file_path, ...): --outdir only
# changes this one directory (default: the current directory, as before).
file_path = os.path.abspath(args.outdir)
os.makedirs(file_path, exist_ok=True)
print(f"Output directory: {file_path}")


def _require_positive(flag_name, value, reason):
    """Shared validation for the CLI arguments used directly as divisors/
    array sizes/moduli below -- a non-positive value would otherwise
    produce a silent NaN, an opaque numpy/JAX error deep inside training,
    or (--decay_step, --loss_save_every) a hang/silent corruption, instead
    of a clear, immediate, argument-specific error message."""
    if value <= 0:
        raise ValueError(f"{flag_name} must be > 0 (got {value}); {reason}")


# --------------------------------------------------------------------------------
# HYPERPARAMETERS
# --------------------------------------------------------------------------------
Nint     = args.Nint
N0       = args.NI
Nb       = args.Nb
SEED     = args.SEED
N_ramp         = args.N_ramp
EPOCHS_ADAM    = args.EPOCHS_ADAM
EPOCHS_QN      = args.EPOCHS_QN
patience_bfgs  = args.patience_bfgs
blowup_factor  = float(args.blowup_factor)
blowup_max     = int(args.blowup_max)
min_delta      = args.min_delta
line_search    = args.line_search
deriv_mode   = args.deriv_mode
qn_fixed_batch = bool(args.qn_fixed_batch)
w_constraint = float(args.w_constraint)
if w_constraint > 0 and int(args.dsgnar_steps) > 0:
    raise SystemExit("--w_constraint is not part of DSGNAR's residual vector yet; use it with QN only.")
_DSGNAR_DEFAULTS = dict(dsgnar_hashes=4, dsgnar_delta_min=1e-14, dsgnar_stage1_ratio=0.15,
                        dsgnar_stage2_ratio=0.5, dsgnar_line_scan=True, dsgnar_calibrate=True,
                        dsgnar_shrink_on_accept=False, dsgnar_radius_floor=1e-3, dsgnar_reweight=0.0,
                        dsgnar_reject_overshoot=False)
_DSGNAR_PAPER = dict(dsgnar_hashes=4, dsgnar_delta_min=1e-8, dsgnar_stage1_ratio=0.075,
                     dsgnar_stage2_ratio=0.5, dsgnar_line_scan=False, dsgnar_calibrate=False,
                     dsgnar_shrink_on_accept=True, dsgnar_radius_floor=0.0, dsgnar_reweight=0.05,
                     dsgnar_reject_overshoot=True)
_dsg = dict(_DSGNAR_PAPER if args.dsgnar_paper else _DSGNAR_DEFAULTS)
_dsg.update({k: getattr(args, k) for k in _dsg if getattr(args, k) is not None})
dsgnar_steps = int(args.dsgnar_steps)
if dsgnar_steps > 0:
    print(f"DSGNAR: {dsgnar_steps} iterations x {args.dsgnar_cycles} cycle(s) per window "
          f"({args.dsgnar_order}), sketch {args.dsgnar_sketch or 'n/3'}, "
          + ", ".join(f"{k[7:]}={v}" for k, v in _dsg.items())
          + f", jacobian={args.dsgnar_jacobian}, residual_sketch={args.dsgnar_residual_sketch}" + (f", max {args.dsgnar_max_seconds:g} s/cycle" if args.dsgnar_max_seconds else "")
          + (" [paper preset]" if args.dsgnar_paper else ""))


def _dsgnar_cfg(**over):
    """The duck-typed config dsgnar.dsgnar_phase reads (Stationary's field names)."""
    import types
    cfg = types.SimpleNamespace(
        dsgnar_steps=int(args.dsgnar_cycle_steps or dsgnar_steps),
        dsgnar_sketch=int(args.dsgnar_sketch), dsgnar_hashes=int(_dsg["dsgnar_hashes"]),
        dsgnar_chunk=int(args.dsgnar_chunk), dsgnar_row_chunk=0,
        dsgnar_delta0=float(args.dsgnar_delta0), dsgnar_delta_min=float(_dsg["dsgnar_delta_min"]),
        dsgnar_omega=float(args.dsgnar_omega),
        dsgnar_stage1_ratio=float(_dsg["dsgnar_stage1_ratio"]),
        dsgnar_stage2_ratio=float(_dsg["dsgnar_stage2_ratio"]),
        dsgnar_backtrack=float(args.dsgnar_backtrack), dsgnar_backtrack_steps=int(args.dsgnar_backtrack_steps),
        dsgnar_line_scan=bool(_dsg["dsgnar_line_scan"]), dsgnar_calibrate=bool(_dsg["dsgnar_calibrate"]),
        dsgnar_grow_on_accept=not bool(_dsg["dsgnar_shrink_on_accept"]),
        dsgnar_radius_floor=float(_dsg["dsgnar_radius_floor"]),
        dsgnar_reweight_alpha=float(_dsg["dsgnar_reweight"]),
        dsgnar_fixed_sketch=bool(args.dsgnar_fixed_sketch),
        dsgnar_reject_overshoot=bool(_dsg["dsgnar_reject_overshoot"]),
        seed=0, resample_every=int(args.dsgnar_resample_every), log_every=int(args.dsgnar_log_every),
        dsgnar_jacobian=str(args.dsgnar_jacobian), dsgnar_max_seconds=float(args.dsgnar_max_seconds),
        dsgnar_residual_sketch=str(args.dsgnar_residual_sketch))
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg
wolfe_c2       = args.wolfe_c2
# An older Crunch silently ignores unknown minimize() options, so a request
# for the Wolfe search must fail loudly rather than quietly run Armijo.
from Crunch.Optimizers import minimize_backtracking as _minimize_mod
if line_search not in getattr(_minimize_mod, "LINE_SEARCHES", ("armijo",)):
    raise SystemExit(f"--line_search {line_search!r} is not supported by the imported Crunch "
                     f"({_minimize_mod.__file__}); update Crunch/Optimizers/minimize_backtracking.py.")
es_min_loss       = float(args.es_min_loss)
h0_scaling        = args.h0_scaling
qn_scale_invariant = bool(args.qn_scale_invariant)
qn_gtol           = float(args.qn_gtol)
qn_loss_norm      = bool(args.qn_loss_norm)
if qn_scale_invariant and not getattr(_minimize_mod, "SCALE_INVARIANT", False):
    raise SystemExit(f"--qn_scale_invariant 1 needs the merged Crunch of 2026-10-17 "
                     f"({_minimize_mod.__file__} lacks it); pass --qn_scale_invariant 0 or update Crunch/.")
excursion_ratio   = args.excursion_ratio
excursion_abs     = args.excursion_abs
excursion_retries = args.excursion_retries
qn_update    = args.qn_update
score_eta    = args.score_eta
score_mu_min = args.score_mu_min
score_mu_max = args.score_mu_max
if qn_update not in getattr(_minimize_mod, "UPDATE_METHODS", ("bfgs", "ssbroyden", "ssbroyden2")):
    raise SystemExit(f"--qn_update {qn_update!r} is not supported by the imported Crunch "
                     f"({_minimize_mod.__file__}); update Crunch/Optimizers/minimize_backtracking.py.")
if qn_update == "score":
    if line_search != "wolfe":
        raise SystemExit("--qn_update score requires --line_search wolfe")
    if not (score_eta >= 0.0 and 0.0 <= score_mu_min <= score_mu_max):
        raise SystemExit(f"--qn_update score needs score_eta >= 0 and 0 <= score_mu_min <= score_mu_max "
                         f"(got {score_eta}, {score_mu_min}, {score_mu_max})")
if patience_bfgs > 0 and EPOCHS_QN > 0 and patience_bfgs >= max(1, EPOCHS_QN // args.N_change):
    print(f"NOTE: --patience_bfgs {patience_bfgs} is >= the number of BFGS restarts per window "
          f"({max(1, EPOCHS_QN // args.N_change)} = EPOCHS_QN/N_change): early stopping and the "
          f"stall Hessian resets can never fire.")
print(f"Quasi-Newton: update={qn_update}  line_search={line_search}  h0_scaling={h0_scaling}  "
      f"scale_invariant={int(qn_scale_invariant)}  gtol={qn_gtol:g}  loss_norm={int(qn_loss_norm)}  "
      f"early stop: patience={patience_bfgs} min_delta={min_delta:g} es_min_loss={es_min_loss:g}  "
      f"excursion guard: ratio={excursion_ratio:g} abs={excursion_abs:g} retries={excursion_retries}"
      + (f"  SCORE eta={score_eta} mu_min={score_mu_min:g} mu_max={score_mu_max:g}" if qn_update == "score" else ""))
N_LAYERS    = args.N_LAYERS
HIDDEN      = args.HIDDEN
FEATURES    = args.FEATURES
N_window    = args.N_window
Nchange     = args.N_change
loss_save_every = args.loss_save_every
snapshot_save_every = args.snapshot_save_every if args.snapshot_save_every is not None else loss_save_every
snapshot_2d_zoom_halfwidth = args.snapshot_2d_zoom_halfwidth
snapshot_2d_every = args.snapshot_2d_every
log_term_grad_norms = args.log_term_grad_norms
ic_mode = args.ic_mode
if deriv_mode == "taylor" and ic_mode == "hard":
    print("NOTE: --deriv_mode taylor does not cover --ic_mode hard; the residual uses autodiff.")
resume_checkpoint = args.resume_checkpoint

if FEATURES != 22:
    raise ValueError("FEATURES must be 22 for the vacuum CCZ4 formulation: gamma-tilde_ij(6), "
                     "A-tilde_ij(6), chi, Khat, Theta, Gamma-hat^i(3), alpha, beta^i(3)")

lr0        = args.LR
decay_rate = args.decay_rate
decay_step = args.decay_step

c          = args.c
metric_det_eps = args.metric_det_eps
chi_floor      = args.chi_floor


def _smooth_floor(x, eps):
    """Smooth division-safety floor: sqrt(x**2 + eps**2).

    Returns |x| whenever |x| >> eps (so a merely-mildly-wrong value, e.g.
    chi=-1e-3, gives a tame ~1e3 after dividing by it, not a manufactured
    ~1e14 distortion) and only approaches eps itself when x is ACTUALLY
    near zero. Equal to x to float64 rounding in the healthy (x>0, not near the
    floor) regime (the eps**2 term is absorbed) -- and SIGN-BLIND: for x<0 it
    returns |x|, so a divisor that should be negative comes out positive (see
    the chi_safe known issue in summary_review.md). Unlike jnp.maximum(x, eps), which clamps ANY
    non-positive x all the way down to eps regardless of how far from
    zero it actually was, manufacturing a huge spurious distortion out of
    an otherwise mildly-wrong value (CONFIRMED not hypothetical: this is
    what caused a real production run's loss_einstein to spike to ~1e36
    then NaN one BFGS step into a window -- see r_ccz4_evolution's
    chi_safe use of this helper for the full incident writeup).
    """
    return jnp.sqrt(x**2 + eps**2)  # NOTE: sign-blind -- |x| for negative x


def _chi_safe(chi):
    """Divisor-safe chi (--chi_floor). Default: max(chi, chi_floor), the
    finite-difference convention -- positive, and every 1/chi is bounded by
    1/chi_floor. It equals chi exactly wherever chi >= chi_floor, which
    holds for all physical values at the collocation points (startup check
    below). Gradient through the divisor is zero below the floor; --w_pos
    and the unfloored chi terms still push chi back up. chi_floor=0 gives
    the old sign-blind smooth floor (_smooth_floor, |chi| for chi<0)."""
    if chi_floor > 0.0:
        return jnp.maximum(chi, chi_floor)
    return _smooth_floor(chi, metric_det_eps)
lam_bc       = args.lam_bc
w_pos        = args.w_pos
lamg0        = args.lamg0
lam_ic_log   = args.lam_ic_log
follow_puncture_sampling = args.follow_puncture_sampling
orbit_focus_fraction = args.orbit_focus_fraction
orbit_focus_mult = args.orbit_focus_mult
lamgt0       = args.lamgt0
lam_einstein = args.lam_einstein
lam_gauge    = args.lam_gauge
lam_tapa     = args.lam_tapa
reinit_mode  = args.reinit_mode
metric_bc_mode = args.metric_bc
gauge_mode   = args.gauge
f_alpha      = args.f_alpha
f_beta       = args.f_beta
eta_shift    = args.eta
kappa_z      = args.kappa_z
kappa_c      = args.kappa_c
kappa_2      = args.kappa_2
if gauge_mode == "static":
    f_alpha = 0.0
    f_beta  = 0.0
    eta_shift = 0.0

R_domain     = args.R_domain
x_min, x_max = -R_domain, R_domain
y_min, y_max = -R_domain, R_domain
z_min, z_max = -R_domain, R_domain
tfinal       = args.tfinal
r_eps        = args.r_eps

# Vacuum Brill-Lindquist/Bowen-York BH puncture data only (no boson_star,
# no schwarzschild_ks/excision -- BH punctures are evolved through directly
# with the moving-puncture gauge, no excision).
bh_center_mode    = args.bh_center_mode
track_steps       = args.track_steps
moving_bg_refresh_steps = args.moving_bg_refresh_steps
chi_bg_mode       = args.chi_bg_mode
if chi_bg_mode == "moving" and bh_center_mode != "shift_driven":
    raise ValueError("--chi_bg_mode moving requires --bh_center_mode shift_driven "
                     "(the moving background is driven by the same tracked positions).")
min_puncture_dist = args.min_puncture_dist
mask_rel_threshold = args.mask_rel_threshold
min_center_separation = args.min_center_separation
# The shift-driven tracker (_beta_xy_near_puncture) reads beta^x/beta^y at
# this distance from the tracked center to RK4-integrate the puncture's motion --
# see the SHIFT-DRIVEN PUNCTURE TRACKER section below. If min_puncture_dist
# excludes a shell around the puncture that is wider than this, every
# interior/IC-surface collocation point used to TRAIN the network stays
# outside that shell while the tracker keeps reading beta^x from just
# inside it: the value driving the puncture's velocity is then a pure
# extrapolation into a region with zero training signal, and generically
# undershoots the true (peaked) shift there -- the network has no reason to
# reproduce a sharp feature it was never shown. This silently starves the
# reported motion regardless of --ic_mode or --input_encoding.
_PUNCTURE_TRACK_EPS = 0.03
if bh_center_mode == "shift_driven" and min_puncture_dist >= _PUNCTURE_TRACK_EPS:
    print(f"WARNING: --min_puncture_dist={min_puncture_dist} excludes training points out to "
          f"a wider radius than the {_PUNCTURE_TRACK_EPS} offset the shift-driven tracker reads "
          f"beta^x from (see comment above). The tracked puncture velocity is likely an "
          f"untrained extrapolation and will tend to undershoot the true speed. Consider "
          f"--min_puncture_dist well below {_PUNCTURE_TRACK_EPS} (e.g. 0.01), or 0.0 if that "
          f"remains numerically stable.")
BL_masses         = [float(m) for m in args.BL_masses.split(",")]
BL_momenta        = [float(p) for p in args.BL_momenta.split(",")]
BL_momenta_y      = [float(p) for p in args.BL_momenta_y.split(",")]
BL_spins          = [tuple(float(s) for s in trip.split(",")) for trip in args.BL_spins.split(";")]

n_stars         = args.n_stars
star_separation = args.star_separation
x_star_A = -star_separation / 2.0 if n_stars == 2 else 0.0
x_star_B =  star_separation / 2.0

# --chi_floor must stay below the smallest PHYSICAL chi the residual sees.
# Interior/IC points closer than --min_puncture_dist to a puncture are
# rejected, so at t=0 that minimum is chi_BL = psi^-4 at that distance.
if chi_floor > 0.0 and min_puncture_dist > 0.0:
    _xs = [x_star_A] + ([x_star_B] if n_stars == 2 else [])
    _chi_min_phys = min(
        (1.0 + sum(m / (2.0 * max(abs(xa + min_puncture_dist - xb), 1e-12))
                   for xb, m in zip(_xs, BL_masses))) ** -4
        for xa in _xs)
    print(f"chi_floor={chi_floor:.1e}; smallest t=0 Brill-Lindquist chi at the collocation "
          f"points (distance --min_puncture_dist={min_puncture_dist} from a puncture) = "
          f"{_chi_min_phys:.2e}")
    if chi_floor > 0.5 * _chi_min_phys:
        print(f"WARNING: --chi_floor={chi_floor:.1e} is not well below the physical chi "
              f"{_chi_min_phys:.2e} at the closest collocation points -- the floor would "
              f"alter the equations there. Lower --chi_floor or raise --min_puncture_dist.")


def _bh_center_list(*per_star_values):
    """[(x_star_A, v1_A, v2_A, ...)] plus, for n_stars==2, (x_star_B, v1_B, ...).
    Each positional arg is a list/tuple indexed by star (length n_stars) --
    reused by the plot-zone block, _bl_psi_jax/np, and _bowen_york_K_lower_jax
    so those three at least share one enumeration. NOT actually a single
    source of truth for the puncture layout in general: --n_stars is
    currently restricted to {1, 2} by argparse (choices=[1, 2]), and this
    function itself hardcodes the n_stars==2 case rather than looping over
    n_stars generically. At least 6 OTHER call sites (find_bh_centers_lapse,
    locate_chi_minima's half-axis split, the puncture-track plotting/color
    palette code, etc.) also hand-branch on n_stars==2 independently.
    Extending to n_stars>2 would need ALL of these updated, not just here."""
    centers = [(x_star_A,) + tuple(v[0] for v in per_star_values)]
    if n_stars == 2:
        centers.append((x_star_B,) + tuple(v[1] for v in per_star_values))
    return centers


if len(BL_masses) != n_stars:
    raise ValueError(f"--BL_masses has {len(BL_masses)} value(s) but --n_stars={n_stars}; "
                     f"they must match.")


def _broadcast_or_require(flag_name, values, n_stars, noun):
    """If len(values)==1, broadcast to n_stars (a single value applies to
    every BH); otherwise require an exact length match. Shared validation
    for --BL_momenta/--BL_spins, which both have a universally-sensible
    single-value default (no boost, no spin) that should broadcast rather
    than require repeating, keeping the documented "default recovers plain
    Brill-Lindquist" claim true for any --n_stars -- unlike --BL_masses,
    which has no such universal default and always requires an exact
    match."""
    if len(values) == 1 and n_stars > 1:
        return values * n_stars
    if len(values) != n_stars:
        raise ValueError(f"{flag_name} has {len(values)} {noun}(s) but --n_stars={n_stars}; "
                         f"provide either 1 {noun} (broadcast to every BH) or exactly {n_stars}.")
    return values


BL_momenta   = _broadcast_or_require("--BL_momenta", BL_momenta, n_stars, "value")
BL_momenta_y = _broadcast_or_require("--BL_momenta_y", BL_momenta_y, n_stars, "value")
BL_spins   = _broadcast_or_require("--BL_spins", BL_spins, n_stars, "semicolon-separated triple")
if any(len(s) != 3 for s in BL_spins):
    raise ValueError(f"--BL_spins triples must each have exactly 3 comma-separated x,y,z "
                     f"components; got {BL_spins}.")

if any(p != 0.0 for p in BL_momenta_y):
    # --BL_momenta_y gives the initial data a tangential kick (see its own
    # help). All three of --bh_center_mode shift_driven/chi_min/lapse_min
    # now follow the resulting off-axis (x,y) motion (shift_driven via its
    # RK4 tracker, chi_min/lapse_min via a genuine 2D x-y plane scan --
    # see locate_chi_minima/find_bh_centers_lapse), and --chi_bg_mode
    # moving's background follows it too. The one remaining x-axis-only
    # exception is shift_driven's own chi-min CROSS-CHECK column in
    # puncture_track.txt (a diagnostic only, not used for sampling/
    # training) -- its search now covers the full plane, but only the
    # x-component is compared/reported there.
    print(f"NOTE: --BL_momenta_y={BL_momenta_y} is nonzero -- --bh_center_mode={bh_center_mode} "
          f"tracks the resulting off-axis (x,y) motion.")

# Pure plot-annotation zones (small masked/circled region on diagnostic
# plots so the puncture's divergent field values don't blow out the color
# scale) -- NOT training-time exclusion, there is no excision in this file.
# _horizon_x_locs itself is populated by _refresh_display_mask_zones below
# (called unconditionally at the start of every train_window, including
# window 0, before any plot that reads it) -- an initial computation here
# would never actually be read, and previously used a DIFFERENT radius
# convention (per-star mass/2 vs. _refresh_display_mask_zones's shared
# max-of-masses/2) than the value it was invisibly replaced by, so it
# couldn't even be read as a preview of what that function computes.
_bh_plot_zones = []
_horizon_x_locs = []
for _cx, _mA in _bh_center_list(BL_masses):
    _bh_plot_zones.append((_cx, 0.0, 0.0, 0.25 * _mA / 2.0))

_bh_display_centers: dict = {}
for (cx, cy, cz, r_pl) in _bh_plot_zones:
    _bh_display_centers[(cx, cy, cz)] = max(_bh_display_centers.get((cx, cy, cz), 0.0), r_pl)
_bh_display_mask_zones = [(cx, cy, cz, r) for (cx, cy, cz), r in _bh_display_centers.items()]


def _display_mask_zones_for(star_centers):
    """(zones, r_ah_shared) for the given puncture positions: zones is the
    list of (cx, cy, cz, r_mask) near-puncture display-mask circles and
    r_ah_shared the shared analytic-horizon radius (see
    _refresh_display_mask_zones for why one shared radius, not one per
    BL_masses entry). Pure function of star_centers, so a snapshot can build
    the zones for ITS OWN time instead of relying on the module-global ones
    _refresh_display_mask_zones sets once at window_start."""
    r_shared = max((0.25 * mass / 2.0 for mass in BL_masses), default=0.0)
    zones = [(cx, cy, cz, r_shared) for (cx, cy, cz) in star_centers]
    r_ah_shared = max((mass / 2.0 for mass in BL_masses), default=0.0)
    return zones, r_ah_shared


def _refresh_display_mask_zones(star_centers):
    """Rebuild _bh_display_mask_zones (the near-puncture NaN-masking used
    by every snapshot plotter's _apply_plot_nan call) AND _horizon_x_locs
    (the dashed analytic-horizon-radius vertical lines on 1D-cut plots)
    from THIS window's current star_centers, instead of leaving them fixed
    at the static t=0 analytic positions computed above once at module
    load. Without this, once a puncture has actually moved -- the whole
    point of --bh_center_mode lapse_min/shift_driven/chi_min -- the mask
    sits at the wrong, stale location and genuine near-puncture divergence
    goes unmasked in every diagnostic plot for that window; _horizon_x_locs
    used to have exactly this bug (fixed at t=0 forever, even though this
    function already existed and was already called every window to fix
    the mask zones) -- the dashed horizon-radius markers no longer
    bracketed the puncture they were meant to indicate on any plot taken
    after the puncture had moved, while the mask circles right next to
    them correctly followed it.

    Uses ONE shared radius (the largest of BL_masses' individual mass-
    derived radii) for every zone, rather than pairing star_centers[k] with
    BL_masses[k] positionally: star_centers' order is NOT guaranteed to
    still match BL_masses' original A/B assignment -- find_bh_centers_lapse
    explicitly sorts its returned centers left-to-right, so if the two
    lapse minima ever cross x-order (possible with unequal --BL_masses),
    a positional zip would size a puncture's mask circle (or horizon-radius
    marker) with the WRONG mass. A shared max-radius sidesteps the identity
    question entirely at the cost of (harmlessly) over-sizing the smaller
    BH's marker a bit in an unequal-mass run -- fine for purely cosmetic
    plot annotations."""
    global _bh_display_mask_zones, _horizon_x_locs
    _bh_display_mask_zones, _r_ah_shared = _display_mask_zones_for(star_centers)
    _horizon_x_locs = [x for (cx, cy, cz) in star_centers for x in (cx - _r_ah_shared, cx + _r_ah_shared)]


def _snap_cut_line(star_centers, n):
    """The 1D line the per-window training snapshots (fields_w*.png/
    residuals_w*.png/gauge_metric_w*.png/native_fields_w*.png) sample along,
    parametrized by ARC LENGTH s along the line through the punctures
    (returns s, x, y, mid, u with (x,y) = mid + s*u).

    n_stars==2: the straight line through both puncture centers, origin of s
    at their midpoint, unit direction u pointing from puncture 0 to puncture
    1 (so puncture 0 is always at s<0 and the plot orientation is continuous
    as the pair orbits, given shift_driven keeps the puncture identities).
    s spans exactly the chord of the R_domain sphere along that line, so
    every sampled point is inside the domain the network was trained on.
    n_stars==1: the horizontal line through that star's y, s = x.

    Reduces EXACTLY to the original y=0, x in [-R,R] cut whenever both
    punctures sit symmetrically on the x-axis (the head-on case, s = x).
    This replaced a y(x) parametrization: as an orbit turns the pair toward
    the y-axis (x1-x0 -> 0) the slope diverges, np.clip pinned nearly every
    sample to y=+-R_domain, and the 1D plots sampled the domain edge instead
    of the punctures (seen on a real run at t~17, where both punctures
    collapsed into one spike at x~0 and alpha read 0.95 while the tracked
    alpha at the punctures was 0.28)."""
    if len(star_centers) == 1:
        mid = np.array([0.0, float(star_centers[0][1])])
        u   = np.array([1.0, 0.0])
    else:
        p0 = np.asarray(star_centers[0][:2], dtype=float)
        p1 = np.asarray(star_centers[1][:2], dtype=float)
        mid = 0.5 * (p0 + p1)
        d   = p1 - p0
        L   = float(np.hypot(d[0], d[1]))
        u   = d / L if L > 1e-12 else np.array([1.0, 0.0])
    mu   = float(mid @ u)
    half = float(np.sqrt(max(mu**2 - float(mid @ mid) + R_domain**2, 0.0)))
    s    = np.linspace(-mu - half, -mu + half, n)
    xy   = mid[None, :] + s[:, None] * u[None, :]
    return s, xy[:, 0], xy[:, 1], mid, u


class _SnapCut:
    """One snapshot column's cut line (see _snap_cut_line) plus everything a
    snapshot plot needs about it: s (plot x-axis), x/y (numpy, for masking and
    analytic references), x_in/y_in/z_in ((n,1) jnp network inputs), the
    near-puncture display-mask zones and analytic-horizon markers for THIS
    time's puncture positions, and the t=0 analytic CCZ4 initial data at
    exactly these points (computed lazily, once per cut -- see ic())."""
    def __init__(self, star_centers, n):
        self.s, self.x, self.y, self.mid, self.u = _snap_cut_line(star_centers, n)
        self.x_in = jnp.asarray(self.x).reshape(-1, 1)
        self.y_in = jnp.asarray(self.y).reshape(-1, 1)
        self.z_in = jnp.zeros((n, 1), dtype=jnp.float64)
        self.zones, r_ah = _display_mask_zones_for(star_centers)
        s_star = [float((np.asarray(c[:2]) - self.mid) @ self.u) for c in star_centers]
        self.horizon_s = [v for sk in s_star for v in (sk - r_ah, sk + r_ah)]
        self._ic = None

    def ic(self):
        """(n, 22) analytic t=0 CCZ4 data at this cut's own (x, y) points --
        the like-for-like reference for the network values plotted along the
        same points (an earlier shared y=0 constant was compared against
        values sampled off the x-axis once an orbit moved the cut)."""
        if self._ic is None:
            self._ic = np.asarray(compute_ccz4_ic(self.x, self.y, np.zeros_like(self.x)))
        return self._ic


ic_focus_fraction  = args.ic_focus_fraction
ic_focus_radius    = args.ic_focus_radius
int_focus_fraction = args.int_focus_fraction
int_focus_radius   = args.int_focus_radius
core_points        = args.core_points
core_ic_points     = args.core_ic_points
core_radius        = args.core_radius
if core_points < 0 or core_ic_points < 0:
    raise ValueError("--core_points and --core_ic_points must be >= 0.")
if (core_points > 0 or core_ic_points > 0) and not core_radius > 0.0:
    raise ValueError(f"--core_radius must be > 0 (got {core_radius}).")

time_dist    = args.time_dist
sigma_time_0 = args.sigma_time_0
f_uniform    = args.f_uniform

causal_weight  = args.causal_weight
causal_eps     = args.causal_eps
causal_buckets = args.causal_buckets

# (flag_name, value, reason, active) -- one declarative table for every
# positive-only CLI argument, same pattern as _CHECKPOINT_FIELDS above
# (whose "required" column this table's "active" column mirrors), instead
# of a one-off _require_positive call scattered at each argument's own
# assignment site. active=True is always checked; active=causal_weight
# means "only meaningful (and only checked) when --causal_weight is on",
# since --causal_buckets is unused when causal weighting is disabled.
_POSITIVE_FIELDS = [
    ("--Nint", Nint, "it's a collocation-point count used directly as an array "
     "size in build_train_data -- a non-positive value raises an opaque "
     "'negative dimensions are not allowed' at first-window time instead of a "
     "clear argument-validation error.", True),
    ("--NI", N0, "it's a collocation-point count used directly as an array "
     "size in build_train_data -- a non-positive value raises an opaque "
     "'negative dimensions are not allowed' at first-window time instead of a "
     "clear argument-validation error.", True),
    ("--Nb", Nb, "it's a collocation-point count used directly as an array "
     "size in build_train_data -- a non-positive value raises an opaque "
     "'negative dimensions are not allowed' at first-window time instead of a "
     "clear argument-validation error.", True),
    ("--N_window", N_window, "window_edges = linspace(0, tfinal, N_window+1) and "
     "the main training loop both need at least one window -- 0 silently runs zero "
     "training iterations (appearing to 'complete' instantly with no trained "
     "windows, only surfacing later as a confusing error in post-training "
     "diagnostics), and a negative value raises an opaque numpy error instead of a "
     "clear one.", True),
    ("--N_change", Nchange, "it's used as a divisor (EPOCHS_ADAM // Nchange) and "
     "a modulus ((e+1) % Nchange), so 0 crashes and a negative value has no "
     "sensible meaning.", True),
    ("--loss_save_every", loss_save_every, "the BFGS logging loop advances via "
     "`next_loss_step += loss_save_every`, so a non-positive value never advances "
     "(0) or advances backwards (negative), hanging in an infinite "
     "append_loss_history loop.", True),
    ("--snapshot_save_every", snapshot_save_every, "`step % snapshot_save_every` "
     "would raise ZeroDivisionError (0) or behave nonsensically (negative).", True),
    ("--decay_step", decay_step, "optax.exponential_decay computes decay_rate ** "
     "(step / decay_step) internally, so 0 divides by zero from the very first Adam "
     "step -- this fails SILENTLY (inf/nan learning-rate multipliers NaN-poison "
     "every trained parameter with no crash/error), unlike most of this file's "
     "other validation gaps.", True),
    ("--metric_det_eps", metric_det_eps, "it's a jnp.maximum(...) safety floor "
     "guarding against division-by-zero/sqrt-of-negative near a puncture -- a "
     "non-positive value fails to floor anything, silently reopening exactly the "
     "NaN paths this parameter exists to guard.", True),
    ("--tfinal", tfinal, "window_edges = linspace(0, tfinal, N_window+1) needs "
     "positive width, and _causal_weights divides by window_end - window_start -- "
     "tfinal <= 0 makes every window zero-width, producing a 0/0 NaN feeding an "
     "int32 cast rather than a clear error.", True),
    ("--moving_bg_refresh_steps", moving_bg_refresh_steps,
     "_integrate_puncture_trajectory silently clamps n_steps to a minimum of 4 "
     "(`max(4, int(n_steps))`), so a non-positive value is not an error but a silent "
     "override to 4 RK4 substeps with no indication the requested value was ignored.", True),
    ("--track_steps", track_steps,
     "same _integrate_puncture_trajectory silent max(4, int(n_steps)) clamp as "
     "--moving_bg_refresh_steps above (via advance_shift_tracked_centers) -- a "
     "non-positive value silently produces a coarse 4-substep trajectory instead "
     "of an error, and under --chi_bg_mode moving that coarse trajectory is what "
     "gets persisted into the checkpoint and reconstructed as physical chi "
     "everywhere downstream.", bh_center_mode == "shift_driven"),
    ("--causal_buckets", causal_buckets, "it's used as an array size in "
     "_causal_weights (jnp.zeros((n_buckets,), ...)) -- a non-positive value "
     "raises 'negative dimensions are not allowed' on the very first training "
     "step.", causal_weight),
    ("--snapshot_2d_zoom_halfwidth", snapshot_2d_zoom_halfwidth, "it's used as "
     "the +/- extent of np.linspace(-h, h, _SNAP_NXY_2D) for the zoomed 2D "
     "snapshot grid -- 0 collapses the whole grid onto a single point (x=y=0, "
     "landing exactly on a puncture) and a negative value reverses the linspace "
     "endpoints, silently plotting a mirrored/flipped domain instead of erroring.",
     True),
]
for _flag_name, _val, _reason, _active in _POSITIVE_FIELDS:
    if _active:
        _require_positive(_flag_name, _val, _reason)

if follow_puncture_sampling and not (bh_center_mode == "shift_driven" and chi_bg_mode == "moving"):
    print("WARNING: --follow_puncture_sampling only takes effect with --bh_center_mode shift_driven "
          "and --chi_bg_mode moving (no within-window trajectory exists otherwise); ignored here.")
if snapshot_2d_every < 0:
    raise ValueError(f"--snapshot_2d_every must be >= 0 (got {snapshot_2d_every}); 0 disables the 2D snapshots.")
if orbit_focus_fraction > 0 and n_stars != 2:
    print("WARNING: --orbit_focus_fraction only takes effect with --n_stars 2 (it needs the "
          "midpoint of two punctures); ignored here.")
if not (0.0 <= orbit_focus_fraction < 1.0):
    raise ValueError(f"--orbit_focus_fraction must be in [0, 1) (got {orbit_focus_fraction}).")
if orbit_focus_mult <= 0:
    raise ValueError(f"--orbit_focus_mult must be > 0 (got {orbit_focus_mult}).")
if lam_ic_log < 0:
    raise ValueError(f"--lam_ic_log must be >= 0 (got {lam_ic_log}); it weights a squared log-ratio term.")
if lam_ic_log > 0 and ic_mode == "hard":
    print("WARNING: --lam_ic_log > 0 has no effect with --ic_mode hard (the IC is satisfied exactly).")
if ic_mode == "soft" and lamg0 == 0:
    print("WARNING: ic_mode=soft but lamg0=0. The metric IC will not be enforced.")

key = jax.random.PRNGKey(SEED)
key, subkey = jax.random.split(key, 2)
np.random.seed(SEED)

x_center = y_center = z_center = 0.0
x_half   = y_half   = z_half   = R_domain

print(f"\nIC enforcement: {ic_mode}  |  reinit_mode: {reinit_mode}  |  chi_bg_mode: {chi_bg_mode}")
print(f"Loss weights: lam_einstein={lam_einstein}  lam_gauge={lam_gauge}"
      f"  lam_bc={lam_bc}  lam_tapa={lam_tapa}  lamg0={lamg0}  lam_ic_log={lam_ic_log}  w_pos={w_pos}")

# --------------------------------------------------------------------------------
# INITIAL DATA: vacuum Brill-Lindquist / Bowen-York BH punctures
# --------------------------------------------------------------------------------
_bl_momenta_nonzero = any(p != 0.0 for p in BL_momenta) or any(p != 0.0 for p in BL_momenta_y)
_bl_spins_nonzero   = any(any(c != 0.0 for c in s) for s in BL_spins)
_bl_has_boost = _bl_momenta_nonzero or _bl_spins_nonzero
_bl_extra = ""
if _bl_momenta_nonzero:
    _bl_extra += f", momenta(x)={BL_momenta}"
    if any(p != 0.0 for p in BL_momenta_y):
        _bl_extra += f", momenta(y)={BL_momenta_y}"
if _bl_spins_nonzero:
    _bl_extra += f", spins={BL_spins}"
print(f"{'Bowen-York' if _bl_has_boost else 'Brill-Lindquist'} initial data: "
      f"{n_stars} BH(s), masses={BL_masses}{_bl_extra}, "
      f"pre-collapsed lapse alpha=1/psi, zero shift beta^i=0. Vacuum (no matter field)."
      + (" (momentum constraint solved exactly via the flat Bowen-York trick; "
         "Hamiltonian constraint approximate, residual trained down like any other "
         "constraint)." if _bl_has_boost else "."))

constraint_ic_state = None
if args.constraint_ic is not None:
    if ic_mode != "soft":
        raise ValueError(f"--constraint_ic is only supported with --ic_mode soft "
                         f"(got --ic_mode {ic_mode}). In hard mode, window 0's IC is "
                         f"pinned algebraically via compute_ccz4_ic, not loaded from "
                         f"a prior constraint-fit state.")
    with open(args.constraint_ic, "rb") as _cf:
        constraint_ic_state = pickle.load(_cf)
    print(f"Constraint IC state loaded from: {args.constraint_ic}  "
          f"(window_idx={constraint_ic_state.get('window_idx')}, "
          f"ic_mode={constraint_ic_state.get('ic_mode')})")

METRIC_COMPONENTS    = ((0,0),(0,1),(0,2),(0,3),(1,1),(1,2),(1,3),(2,2),(2,3),(3,3))
METRIC_COMPONENT_NAMES = ("tt","tx","ty","tz","xx","xy","xz","yy","yz","zz")

_FLAT_METRIC_VEC = jnp.array([-1., 0., 0., 0., 1., 0., 0., 1., 0., 1.], dtype=jnp.float64)

# --------------------------------------------------------------------------------
# CCZ4 FIELD LAYOUT (vacuum: no matter sector)
#
# Network output (22 components):
#   [0:6]   gamma-tilde_ij   (packed via CCZ4_METRIC3_COMPONENTS, symmetric 3x3)
#   [6:12]  A-tilde_ij       (same packing)
#   [12]    chi
#   [13]    Khat
#   [14]    Theta
#   [15:18] Gamma-hat^i  (x,y,z)
#   [18]    alpha
#   [19:22] beta^i       (x,y,z)
# --------------------------------------------------------------------------------
CCZ4_METRIC3_COMPONENTS = ((0,0),(0,1),(0,2),(1,1),(1,2),(2,2))
CCZ4_METRIC3_NAMES      = ("xx","xy","xz","yy","yz","zz")

# Packed index of each diagonal component -- DERIVED, not hardcoded, so
# MLP.__call__'s flat-space bias (Z.at[:, _GT_xx_IDX].add(1.0) etc.) can
# never silently drift out of sync with CCZ4_METRIC3_COMPONENTS's actual
# ordering if it's ever reordered.
_GT_XX_IDX = CCZ4_METRIC3_COMPONENTS.index((0, 0))
_GT_YY_IDX = CCZ4_METRIC3_COMPONENTS.index((1, 1))
_GT_ZZ_IDX = CCZ4_METRIC3_COMPONENTS.index((2, 2))

GT_SLICE    = slice(0, 6)
AT_SLICE    = slice(6, 12)
CHI_IDX     = 12
KHAT_IDX    = 13
THETA_IDX   = 14
GHAT_SLICE  = slice(15, 18)
ALPHA_IDX   = 18
BETA_SLICE  = slice(19, 22)
CCZ4_SLICE  = slice(0, 22)   # all 22 metric-sector fields (the whole output)

# The 11 outputs (gt(6), chi, alpha, beta(3)) that r_ccz4_evolution's
# residual_one needs a SECOND spatial derivative for, gathered into one
# fixed-order index array so a sub-function selecting just these 11 can be
# built once (see residual_one's full_dd) instead of double-jacfwd-ing all
# 22. _DD_* below are this 11-wide sub-vector's OWN local slice/index
# positions (not the original 22-wide ones).
_DD_OUTPUT_IDXS = jnp.array(
    list(range(GT_SLICE.start, GT_SLICE.stop)) + [CHI_IDX, ALPHA_IDX] + list(range(BETA_SLICE.start, BETA_SLICE.stop)))
_DD_GT_SLICE    = slice(0, 6)
_DD_CHI_IDX     = 6
_DD_ALPHA_IDX   = 7
_DD_BETA_SLICE  = slice(8, 11)

def pack_sym3(m):
    """(...,3,3) symmetric -> (...,6), CCZ4_METRIC3_COMPONENTS order."""
    return jnp.stack([m[..., i, j] for i, j in CCZ4_METRIC3_COMPONENTS], axis=-1)

def unpack_sym3(v):
    """(...,6) -> (...,3,3) symmetric, CCZ4_METRIC3_COMPONENTS order."""
    g = jnp.zeros(v.shape[:-1] + (3, 3), dtype=v.dtype)
    for idx, (i, j) in enumerate(CCZ4_METRIC3_COMPONENTS):
        g = g.at[..., i, j].set(v[..., idx])
        g = g.at[..., j, i].set(v[..., idx])
    return g


def _unpack_sym3_leading(v):
    """(6, *deriv_shape) -> (3, 3, *deriv_shape): unpack_sym3, but for a
    symmetric-metric field's 6 packed components sliced off AXIS 0 (JAX's
    jvp/jacfwd convention: the sliced function's own output axis comes
    first, any differentiation axes come after -- e.g. jacfwd(f)(xyz) for
    f: R^3 -> R^22 has shape (22,3), component axis first) of a SHARED
    22-component value/Jacobian array, rather than the last axis unpack_sym3
    itself expects. Used by r_ccz4_evolution's residual_one to slice
    gt/At (and gt's second derivative) out of one shared full-output
    value/jvp/jacfwd computation instead of recomputing the whole network
    forward pass once per field (see residual_one). Output axis order
    (metric indices first, then derivative axes) matches this file's
    existing gt_d[i,j,k]=d_k gt_ij / gt_dd[i,j,m,n] convention exactly,
    verified by direct comparison against the original per-field closures
    on random points before this was adopted."""
    moved    = jnp.moveaxis(v, 0, -1)                   # (*deriv_shape, 6)
    unpacked = unpack_sym3(moved)                       # (*deriv_shape, 3, 3)
    return jnp.moveaxis(unpacked, (-2, -1), (0, 1))     # (3, 3, *deriv_shape)

# Flat-space (Minkowski) target for the 22 CCZ4 metric-sector fields, in
# output order [0:22]: gamma-tilde_ij->delta_ij, A-tilde_ij->0, chi->1,
# Khat->0, Theta->0, Gamma-hat^i->0, alpha->1, beta^i->0.
_FLAT_GT_VEC   = jnp.array([1., 0., 0., 1., 0., 1.], dtype=jnp.float64)
_FLAT_CCZ4_VEC = jnp.concatenate([
    _FLAT_GT_VEC,                          # gamma-tilde_ij
    jnp.zeros(6, dtype=jnp.float64),       # A-tilde_ij
    jnp.array([1.0], dtype=jnp.float64),   # chi
    jnp.array([0.0], dtype=jnp.float64),   # Khat
    jnp.array([0.0], dtype=jnp.float64),   # Theta
    jnp.zeros(3, dtype=jnp.float64),       # Gamma-hat^i
    jnp.array([1.0], dtype=jnp.float64),   # alpha
    jnp.zeros(3, dtype=jnp.float64),       # beta^i
])

# --------------------------------------------------------------------------------
# BRILL-LINDQUIST ANALYTICAL INITIAL DATA
# --------------------------------------------------------------------------------
def _bl_psi_jax(x, y, z):
    psi = jnp.ones(x.shape[0], dtype=jnp.float64)
    for cx, mA in _bh_center_list(BL_masses):
        rA  = jnp.maximum(jnp.sqrt((x - cx)**2 + y**2 + z**2), 1.0e-10)
        psi = psi + mA / (2.0 * rA)
    return psi


def _bl_psi_np(x_np, y_np, z_np):
    psi = np.ones(len(x_np), dtype=np.float64)
    for cx, mA in _bh_center_list(BL_masses):
        rA  = np.maximum(np.sqrt((x_np - cx)**2 + y_np**2 + z_np**2), 1.0e-10)
        psi = psi + mA / (2.0 * rA)
    return psi


def _bowen_york_K_lower_jax(x, y, z, psi):
    """Physical extrinsic curvature K_ij (lower, Cartesian) from the
    Bowen-York solution, superposed over punctures A (Brandt-Bruegmann
    1997). Each puncture contributes a flat-space, exactly divergence-free,
    exactly trace-free Abar_ij_A (linear-momentum + spin terms -- momentum
    is a general (Px,Py,0) in-plane vector, --BL_momenta/--BL_momenta_y;
    the term_lin formula below never assumed it was purely radial, only
    Py's absence made it look that way; spin a general 3-vector); physical
    K_ij = psi^{-2} * sum_A Abar_ij_A. `psi` (the SAME
    total conformal factor used for the metric) is passed in by the caller
    -- for brill_lindquist, alpha=1/psi already, so _adm_ic_batch reuses
    its own alpha=1/psi instead of this function re-deriving psi from
    scratch via a second puncture loop. This is the standard trick that
    makes K^ij=psi^{-10}*Abar^ij solve the momentum constraint exactly for
    ANY psi, independent of the (approximate, un-solved-for-u) Hamiltonian
    constraint.

    The unit vector n_A=(x-cx,y,z)/r_A is built from a MAX-floored radius
    (jnp.maximum(r2, r_eps**2), not the additive r2+r_eps**2 used elsewhere
    in this file for puncture-term regularization): away from the
    puncture (r_A > r_eps, i.e. everywhere except a physically-irrelevant
    r_eps-radius ball around the true singularity -- r_eps defaults to
    1e-8) max-flooring leaves r_A EXACTLY unregularized, so n_A is an
    EXACT unit vector and Abar_ij_A is EXACTLY trace-free/divergence-free
    there; the additive form used elsewhere corrupts n_A's unit-norm
    everywhere by O(r_eps^2/r^2), which would make this "exact" claim only
    approximate and silently worse for larger --r_eps. Verified standalone
    (/tmp/ccz4_by_verify/bowen_york_check.py, updated to the max-floor
    convention): Abar trace and divergence at float64 machine precision
    (~1e-17--1e-20) at every test point (all at O(1) distances, far outside
    the 1e-8 floor radius), for momentum-only/spin-only/combined, single-
    and two-puncture cases. Identically zero when all BL_momenta/BL_spins
    are zero (exact no-op, recovers time-symmetric Brill-Lindquist data)."""
    N = x.shape[0]
    Abar = jnp.zeros((N, 3, 3), dtype=jnp.float64)
    eye3 = jnp.eye(3, dtype=jnp.float64)
    for cx, Px, Py, S in _bh_center_list(BL_momenta, BL_momenta_y, BL_spins):
        dx = x - cx
        r2 = dx**2 + y**2 + z**2
        rA = jnp.sqrt(jnp.maximum(r2, r_eps**2))
        n  = jnp.stack([dx / rA, y / rA, z / rA], axis=-1)          # (N,3)
        P  = jnp.array([Px, Py, 0.0], dtype=jnp.float64)
        Pn = jnp.einsum("ni,i->n", n, P)
        term_lin = (3.0 / (2.0 * rA**2))[:, None, None] * (
            jnp.einsum("i,nj->nij", P, n) + jnp.einsum("j,ni->nij", P, n)
            - (eye3[None, :, :] - jnp.einsum("ni,nj->nij", n, n)) * Pn[:, None, None]
        )
        Sv = jnp.array(S, dtype=jnp.float64)
        V  = jnp.cross(jnp.broadcast_to(Sv, (N, 3)), n)             # (S x n), (N,3)
        term_spin = (3.0 / rA**3)[:, None, None] * (
            jnp.einsum("ni,nj->nij", V, n) + jnp.einsum("nj,ni->nij", V, n)
        )
        Abar = Abar + term_lin + term_spin
    return Abar / psi[:, None, None]**2


def _bl_contra_jax(x, y, z):
    psi      = _bl_psi_jax(x, y, z)
    psi4_inv = 1.0 / psi**4
    N        = x.shape[0]
    zeros    = jnp.zeros(N, dtype=jnp.float64)
    return jnp.stack([
        -psi**2,
         zeros,
         zeros,
         zeros,
         psi4_inv,
         zeros,
         zeros,
         psi4_inv,
         zeros,
         psi4_inv,
    ], axis=1)


def _bl_contra_np(x_np, y_np, z_np):
    psi      = _bl_psi_np(x_np, y_np, z_np)
    psi4_inv = 1.0 / psi**4
    N        = len(x_np)
    zeros    = np.zeros(N, dtype=np.float64)
    return np.column_stack([
        -psi**2,
         zeros, zeros, zeros,
         psi4_inv, zeros, zeros,
         psi4_inv, zeros,
         psi4_inv,
    ])


def _ic_values_jax(x, y, z):
    g_tot  = _bl_contra_jax(x, y, z)
    gt_tot = jnp.zeros((x.shape[0], 10), dtype=jnp.float64)
    return g_tot, gt_tot


# --------------------------------------------------------------------------------
# HARD-IC WRAPPER
# Output is 22-dimensional (vacuum: the 22 CCZ4 metric-sector fields).
#
# Every one of the 22 CCZ4 metric-sector fields uses the SAME tau^1 (linear)
# construction: field_ic + tau*NN. All of them are first-order-in-time by
# construction: ONLY the VALUE at t0 is a free initial condition; the time
# derivative is whatever the evolution RHS says it should be, and is
# discovered by training (the residual at collocation points near t0 pulls
# tau*NN(0,x)/Delta_t, the value's raw t-derivative at tau=0, toward that
# RHS) rather than being pinned by construction. This still guarantees the
# field VALUE matches field_ic exactly at tau=0 (field_ic + 0*NN = field_ic).
# --------------------------------------------------------------------------------
# Single source of truth for "does jax.clear_caches() still need to run every
# window" (see its two call sites in train_window and the main loop below).
# ic_mode=="hard" is the ONLY reason it's still needed: make_hard_ic_apply_fn's
# ancestor-sum graph genuinely has a different SHAPE every window (one more
# stacked ancestor each time -- see its docstring), so a distinct compile per
# window is unavoidable there regardless of any caching, unlike train_apply_fn/
# unflatten_func/update_model/optimizer, which are all module-level or cached
# across windows now (see _wrap_with_moving_chi_bg/_get_stable_unflatten and
# the optimizer/update_model definition), so JAX's own compilation cache
# naturally reuses their compiled executables window-to-window with nothing
# left to bound. Centralized here as ONE flag (rather than each call site
# independently re-deriving "if ic_mode == 'hard':") so a future change to
# WHY clearing is needed has exactly one place to update instead of two
# copies that could silently drift out of sync.
_NEEDS_PERWINDOW_CACHE_CLEAR = (ic_mode == "hard")


class _IdentityLRUCache:
    """Shared bounded LRU memoization cache -- used by ALL THREE of this
    file's build-expensive-closure-once-per-distinct-input caches
    (_hard_ic_apply_fn_cache, _moving_chi_bg_wrap_cache,
    _unflatten_func_cache, see each). Previously three independently
    hand-rolled implementations of this exact pattern -- two as a bounded
    collections.OrderedDict with matching move-to-end/popitem logic, one
    as a plain UNBOUNDED dict -- with nothing tying their eviction
    discipline together, so a fix to the bound/eviction logic could
    easily be applied to one or two copies and forgotten in the third.

    get_or_build's `build` callable is invoked ONLY on a genuine cache
    miss -- each cache's own call site explains why a miss is always safe
    to recover from (it just costs one extra rebuild, never a correctness
    problem, since every one of these caches memoizes a pure function of
    its key). A hit moves the entry to "most recently used" before
    returning it, so eviction (once `maxsize` is exceeded) always drops
    the least-recently-used entry, exactly as the original hand-rolled
    versions did."""
    def __init__(self, maxsize):
        self._entries = collections.OrderedDict()
        self._maxsize = maxsize

    def get_or_build(self, key, build):
        cached = self._entries.get(key)
        if cached is not None:
            self._entries.move_to_end(key)
            return cached
        value = self._entries[key] = build()
        if len(self._entries) > self._maxsize:
            self._entries.popitem(last=False)  # evict least-recently-used
        return value

    def __len__(self):
        return len(self._entries)


# Tied to N_window (not an arbitrary constant), shared by all three caches
# below: id(ic_history)/raw_apply_fn/params-shape are each stable per window
# for the life of a run (see the respective docstrings), so there are never
# more than N_window+1 distinct keys for any of them to hold -- sizing the
# cap any smaller would evict and rebuild entries still in active use (e.g.
# the end-of-run diagnostics loop, which revisits every window's own
# eval_fn), and any larger is dead headroom. A cache miss just triggers a
# normal rebuild (correct, only slower for that one call), so capping this
# and evicting the least-recently-used entry is entirely safe -- without a
# bound, _hard_ic_apply_fn_cache in particular (each entry closing over a
# stacked copy of that window's ancestor params, on top of what
# window_states/the checkpoint already retain for the same data) would grow
# without limit over a long or many-window run.
_PERWINDOW_CACHE_MAXSIZE = N_window + 1

# NOTE _hard_ic_apply_fn_cache bounds entry COUNT, not memory: each entry
# for window k holds k stacked ancestor-parameter-pytree copies (see
# ic_history), so the LAST window's single entry is the dominant memory
# cost on a long run and is NOT shrunk by this cap -- only redundant
# *rebuilding* across many resamples/BFGS-restarts within one window is
# avoided by caching at all, which is this cache's actual purpose (see
# make_hard_ic_apply_fn's docstring: build_train_data/find_bh_centers_lapse
# hit it every resample).
_hard_ic_apply_fn_cache = _IdentityLRUCache(_PERWINDOW_CACHE_MAXSIZE)


def make_hard_ic_apply_fn(base_apply_fn, win_start, win_end, ic_history=None):
    """FLAT, NON-RECURSIVE hard-IC composition (was: a recursive closure
    chain wrapping every prior window's apply_fn+params back to window 0 --
    see the O(N^2)-compile discussion this replaced).

    KEY FACT (proved from the tau-wrapper's own algebra, not an
    approximation): every ancestor window is only ever queried at exactly
    its own win_end (= the next window's win_start, by contiguity), so its
    tau_norm = (win_end - win_start)/Delta_t = 1.0 EXACTLY, for every
    ancestor, at every recursion depth, regardless of window widths. By
    induction this holds all the way back to window 0. So the whole k-deep
    recursive chain telescopes into a flat, order-independent sum:

        IC_k(x,y,z) = compute_ccz4_ic(x,y,z)                                 [window 0 analytic base]
                    + sum_{j=0}^{k-1} base_apply_fn(params_j, tfinal, x,y,z, 0, tfinal)[:, CCZ4_SLICE]

    Every term in that sum uses IDENTICAL arguments (t_virtual=tfinal,
    window_start=0, window_end=tfinal) -- only params_j differs -- so it's
    exactly the shape jax.vmap wants: stack the k ancestor param pytrees
    once and vmap+sum a single evaluation function over them, instead of
    composing k nested Python closures.

    Why this actually fixes the compile-time blowup (not just the redundant
    rebuilding the earlier memoization step addressed): jax.vmap traces its
    body ONCE: the traced graph size is independent of k. The old recursive
    design forced whatever JIT call traced through this closure (e.g.
    apply_model/bfgs_step for window k) to unroll k sequentially-nested
    function calls into its graph -- trace cost genuinely grew with k. Here
    it's one vmap node regardless of k. JAX still compiles a distinct
    specialization per window (since the stacked array's leading dimension
    k differs -- ordinary shape-specialization, unavoidable), but each of
    those N compilations is now O(1) work instead of O(k), so total compile
    cost across a full run is O(N), not O(N^2).

    ic_history: a flat list of ancestor windows' param pytrees, [params_0,
    params_1, ..., params_{k-1}] (oldest first; order doesn't matter
    numerically since the sum is order-independent, kept for readability),
    or None/[] for window 0. This replaces the old nested ic_frozen dict
    ({"params","t_start","t_end","t_boundary","ic_frozen":<parent>}) --
    the per-ancestor timing bookkeeping it carried is exactly what the fact
    above proves unnecessary (every ancestor's timing algebra reduces to
    the same universal constants). See extend_ic_history() for how a
    window's own state extends this list for the next window, and
    _migrate_ic_frozen_to_history() for loading old-format checkpoints.

    Caching: still memoized (same rationale/call-site pattern as before --
    build_train_data's IC-fetch and find_bh_centers_lapse call this on
    every resample, not just once per window), now keyed off id(ic_history)
    instead of id(ic_frozen); ic_history lists are still built once per
    window and kept alive via window_states/checkpoint, so identity is
    stable within a run. Bounded LRU sized to N_window+1 (see
    _PERWINDOW_CACHE_MAXSIZE/_IdentityLRUCache) -- a miss just rebuilds, so this avoids
    redundant rebuilding across the many resamples/BFGS-restarts within
    one window; it caps entry COUNT, not memory (each entry's own size
    still grows with window index), so it is not a memory cap on its own.

    NOTE: the key deliberately does NOT include id(base_apply_fn). Every
    call site passes model.apply, and instance.method in Python creates a
    NEW bound-method object on every attribute access -- id(model.apply)
    differs on every single call, which silently made the cache miss
    EVERY time (verified: a closure capturing base_apply_fn keeps the
    first bound-method object alive, so a second `model.apply` access
    cannot even coincidentally recycle its address, unlike the misleading
    `id(f.apply) == id(f.apply)` you get when nothing holds the first one
    alive). There is only ever one model/apply_fn in a run, so this
    component of the key served no discriminating purpose anyway -- safer
    to drop it than to rely on every call site remembering to pass some
    separately-stored stable reference instead of model.apply directly.
    """
    _key = (id(ic_history), float(win_start), float(win_end))

    def _build():
        Delta_t = win_end - win_start
        _ws0 = jnp.array(0.0,    dtype=jnp.float64)
        _we0 = jnp.array(tfinal, dtype=jnp.float64)

        if ic_history:
            _stacked_params = jax.tree_util.tree_map(
                lambda *leaves: jnp.stack([jnp.asarray(v) for v in leaves]), *ic_history)

            def _ancestor_eval(p, x, y, z):
                # t_virtual = tfinal exactly for every ancestor (see docstring), but
                # must be broadcast to x's batch shape -- the network concatenates
                # t with x,y,z, so a bare 0-d scalar (unlike _ws0/_we0, which are
                # never batch-concatenated) doesn't work here.
                t_virtual = jnp.full_like(x, tfinal)
                return base_apply_fn(p, t_virtual, x, y, z, _ws0, _we0)[:, CCZ4_SLICE]

            _vmapped_ancestors = jax.vmap(_ancestor_eval, in_axes=(0, None, None, None))

        def hard_apply(params, t, x, y, z, window_start, window_end):
            tau_norm  = (t - win_start) / Delta_t
            t_virtual = tau_norm * tfinal
            nn_out    = base_apply_fn(params, t_virtual, x, y, z, _ws0, _we0)
            # compute_ccz4_ic (window 0's analytic base) appears EXACTLY ONCE in
            # the telescoped sum for EVERY window k>=0 -- it is NOT replaced by
            # the ancestor sum, it's added to it (see docstring derivation: the
            # recursion always bottoms out at window 0's base case, whose
            # contribution survives unconditionally). An earlier version of this
            # code made the two mutually exclusive (if/else), which is wrong for
            # every k>=1 -- caught via a real run showing window 1's IC-value
            # loss stuck at ~0.18 instead of ~0 (windows 0, 2, 3 were fine
            # because k=0 has no ancestor sum to omit the base from, and by k=2
            # the mistake looks superficially "smaller" relative to a bigger
            # sum, but is still there). The isolated equivalence test used a
            # dummy analytic base of exactly zero, which accidentally canceled
            # this exact bug -- lesson: stubbing a term as 0 can hide a missing-
            # term bug involving that same term.
            ccz4_ic = compute_ccz4_ic(x[:, 0], y[:, 0], z[:, 0])
            if chi_bg_mode == "moving":
                # Convert compute_ccz4_ic's PHYSICAL chi (the true analytic BL
                # value at the fixed initial positions) into the REGULAR
                # remainder the tau-wrapper should seed: at t=0 exactly, the
                # true data has zero deviation from its own background, so this
                # is an exact identity subtraction (see _wrap_with_moving_chi_bg's
                # module docstring). No equivalent adjustment is needed for the
                # ancestor-sum terms below: by induction, every ancestor's raw
                # (unwrapped) output is already regular-only, so no extra
                # arithmetic is needed there.
                _x0_positions = [(c[0], 0.0) for c in _bh_center_list(BL_masses)]
                chi_bg0 = _translated_psi(x, y, z, _x0_positions) ** -4
                ccz4_ic = ccz4_ic.at[:, CHI_IDX].add(-chi_bg0[:, 0])
            if ic_history:
                contributions = _vmapped_ancestors(_stacked_params, x, y, z)  # (k, N, 22)
                ccz4_ic = ccz4_ic + jnp.sum(contributions, axis=0)
            return ccz4_ic + tau_norm * nn_out[:, CCZ4_SLICE]

        return hard_apply

    return _hard_ic_apply_fn_cache.get_or_build(_key, _build)


def extend_ic_history(prev_state):
    """The flat-list equivalent of the old build_window_ic_frozen: append
    prev_state's own final trained params onto its inherited ic_history,
    giving the ancestor list a NEW window should be trained against. Called
    once per window transition (train_window), not per resample -- the
    resulting list is then stored on the new window's state and reused by
    every call site that needs it (identity-stable, cache-friendly)."""
    prev_params_np = jax.tree_util.tree_map(np.asarray, prev_state["params"])
    return (prev_state.get("ic_history") or []) + [prev_params_np]


def _migrate_ic_frozen_to_history(ic_frozen):
    """Old checkpoints (pre flat-vmap-sum rewrite) stored the nested
    linked-list ic_frozen dict instead of a flat ic_history list. Walk it
    from innermost (oldest window) outward and return the equivalent flat
    list, oldest-first, so resume_checkpoint keeps working on old files."""
    chain = []
    node = ic_frozen
    while node is not None:
        chain.append(node["params"])
        node = node.get("ic_frozen")
    chain.reverse()
    return chain


# --------------------------------------------------------------------------------
# DOMAIN & WINDOW SETUP
# --------------------------------------------------------------------------------
window_edges = np.linspace(0.0, tfinal, N_window + 1)
print(f"Temporal windows: {window_edges}")

# --------------------------------------------------------------------------------
# DATA GENERATION
# --------------------------------------------------------------------------------
def _project_to_sphere(xyz):
    r = jnp.linalg.norm(xyz, axis=1, keepdims=True)
    return jnp.where(r > R_domain, xyz * (R_domain / r), xyz)


def sample_sphere(key, n):
    key_dir, key_r, new_key = randomjax.split(key, 3)
    xyz = jax.random.normal(key_dir, (n, 3), dtype=jnp.float64)
    xyz = xyz / jnp.linalg.norm(xyz, axis=1, keepdims=True)
    u   = jax.random.uniform(key_r, (n, 1), dtype=jnp.float64)
    r   = R_domain * u ** (1.0 / 3.0)
    xyz = r * xyz
    return xyz[:, 0:1], xyz[:, 1:2], xyz[:, 2:3], new_key


def sample_sphere_focused(key, n, fraction, radius):
    key_g, key_u, key_sh, new_key = randomjax.split(key, 4)
    N_focused = int(n * fraction)
    N_uniform = n - N_focused
    xyz_f = radius * jax.random.normal(key_g, (N_focused, 3), dtype=jnp.float64)
    xyz_f = _project_to_sphere(xyz_f)
    x_u, y_u, z_u, _ = sample_sphere(key_u, N_uniform)
    xyz = jnp.vstack([xyz_f, jnp.hstack([x_u, y_u, z_u])])
    perm = jax.random.permutation(key_sh, n)
    xyz = xyz[perm]
    return xyz[:, 0:1], xyz[:, 1:2], xyz[:, 2:3], new_key


def sample_sphere_focused_multistar(key, n, fraction, radius, centers):
    n_centers = len(centers)
    N_focused  = int(n * fraction)
    N_uniform  = n - N_focused
    n_per      = N_focused // n_centers

    subkeys  = randomjax.split(key, n_centers + 3)
    new_key  = subkeys[0]

    chunks = []
    for k, (cx, cy, cz) in enumerate(centers):
        n_k  = N_focused - n_per * (n_centers - 1) if k == n_centers - 1 else n_per
        if args.symmetric_focus:
            # point-reflected pairs: the cloud is symmetric about its centre by construction
            d = radius * jax.random.normal(subkeys[k + 1], ((n_k + 1) // 2, 3), dtype=jnp.float64)
            d = jnp.vstack([d, -d])[:n_k]
        else:
            d = radius * jax.random.normal(subkeys[k + 1], (n_k, 3), dtype=jnp.float64)
        pts  = jnp.array([cx, cy, cz], dtype=jnp.float64) + d
        pts  = _project_to_sphere(pts)
        chunks.append(pts)

    x_u, y_u, z_u, _ = sample_sphere(subkeys[n_centers + 1], N_uniform)
    xyz  = jnp.vstack(chunks + [jnp.hstack([x_u, y_u, z_u])])
    perm = jax.random.permutation(subkeys[n_centers + 2], n)
    xyz  = xyz[perm]
    return xyz[:, 0:1], xyz[:, 1:2], xyz[:, 2:3], new_key


# --------------------------------------------------------------------------------
# MINIMUM-DISTANCE REJECTION  (--min_puncture_dist)
#
# xisco (bbh_headon_bssn_pinn_shift_driven_puncture.py) has NO mechanism at
# all to keep collocation points away from the puncture -- its focused
# sampling cloud is free to land arbitrarily close. This is an OPT-IN
# (default 0.0 = disabled) rejection-resampling mechanism, adapted from the
# pattern in EEdedonder.py's _fill_with_rejection/_excision_invalid_mask,
# but simplified from a hard-excision zone (which also masks those points
# out of the PDE residual) to a plain minimum-SAMPLING-distance filter:
# this file still evolves straight through the puncture with no excision
# anywhere else, so this only avoids drawing training points in the
# steepest part of the near-singular region, not excluding it from the
# physics. Measured against whatever `centers` the caller is already using
# to steer the focused-sampling cloud (the live --bh_center_mode positions),
# so it automatically follows a moving puncture under shift_driven mode.
# --------------------------------------------------------------------------------
def _too_close_mask(xyz_np, centers, min_dist):
    bad = np.zeros(len(xyz_np), dtype=bool)
    for (cx, cy, cz) in centers:
        r = np.sqrt((xyz_np[:, 0]-cx)**2 + (xyz_np[:, 1]-cy)**2 + (xyz_np[:, 2]-cz)**2)
        bad |= (r < min_dist)
    return bad


def _reject_near_puncture(x, y, z, key, centers, min_dist, resample_fn, max_tries=20):
    """Redraw (via resample_fn(key,n)->(x,y,z,_)) any of (x,y,z) closer than
    min_dist to any of `centers`, repeating until none remain or max_tries
    is hit (capped so a pathological min_dist relative to the sampling
    cloud's spread can't hang training)."""
    xyz_np = np.column_stack([np.asarray(x[:, 0]), np.asarray(y[:, 0]), np.asarray(z[:, 0])])
    bad = _too_close_mask(xyz_np, centers, min_dist)
    tries = 0
    while bad.any() and tries < max_tries:
        n_bad = int(bad.sum())
        key, subkey = randomjax.split(key)
        x_r, y_r, z_r, _ = resample_fn(subkey, n_bad)
        new_pts  = np.column_stack([np.asarray(x_r[:, 0]), np.asarray(y_r[:, 0]), np.asarray(z_r[:, 0])])
        new_bad  = _too_close_mask(new_pts, centers, min_dist)
        bad_idx  = np.where(bad)[0]
        good_idx = np.where(~new_bad)[0]
        n_fix    = min(len(good_idx), n_bad)
        xyz_np[bad_idx[:n_fix]] = new_pts[good_idx[:n_fix]]
        bad = _too_close_mask(xyz_np, centers, min_dist)
        tries += 1
    xyz = jnp.asarray(xyz_np)
    return xyz[:, 0:1], xyz[:, 1:2], xyz[:, 2:3], key


_FOLLOW_CUTOFF_MULT = 3.0


def _interp_trajectory(traj_tt, traj_xx, k, t):
    """(x, y) of puncture k at time(s) t along a within-window RK4 trajectory
    (traj_tt (T,), traj_xx (n_stars,2,T)). The one place this interpolation is
    written -- shared by _follow_puncture_shift, _puncture_strength_rows and
    train_window's _star_centers_at. t may be a scalar or an array."""
    tt = np.asarray(traj_tt)
    xx = np.asarray(traj_xx)
    return np.interp(t, tt, xx[k, 0]), np.interp(t, tt, xx[k, 1])


def _follow_puncture_shift(xyz, t, centers, traj_tt, traj_xx, focus_radius):
    """Rigidly shift each point that sits near puncture k (nearest window_start
    center, within _FOLLOW_CUTOFF_MULT * focus_radius -- the radius of the
    focus cloud those points were drawn from: --int_focus_radius for interior
    points, --ic_focus_radius for the window-end surface) by that puncture's
    displacement from its window_start position to its position at the
    point's own time t, read off the within-window RK4 trajectory
    (traj_tt (T,), traj_xx (n_stars,2,T)). Points farther than the cutoff (the
    uniform far-field cloud) are left alone. A rigid shift preserves each
    point's distance to its own puncture, so --min_puncture_dist rejection done
    beforehand in the window_start frame stays valid. The trajectory starts at
    the same position the focus centers were taken from (both come from
    _shift_track_xy), so the shift is zero at t=window_start."""
    c0  = np.asarray([[c[0], c[1], c[2]] for c in centers], dtype=np.float64)   # (K,3)
    pts = np.asarray(xyz, dtype=np.float64)
    tn  = np.asarray(t, dtype=np.float64)[:, 0]
    d      = np.linalg.norm(pts[:, None, :] - c0[None, :, :], axis=2)            # (N,K)
    owner  = d.argmin(axis=1)
    near   = d[np.arange(len(pts)), owner] < _FOLLOW_CUTOFF_MULT * focus_radius
    shift  = np.zeros_like(pts)
    for k in range(len(c0)):
        m = near & (owner == k)
        if m.any():
            xk, yk = _interp_trajectory(traj_tt, traj_xx, k, tn[m])
            shift[m, 0] = xk - c0[k, 0]
            shift[m, 1] = yk - c0[k, 1]
    return _project_to_sphere(jnp.asarray(pts + shift))


def _sample_orbit_ball(key, n, centers):
    """n points uniform in the ball of radius min(orbit_focus_mult * separation,
    R_domain) centered on the midpoint of the two punctures (see
    --orbit_focus_fraction)."""
    c    = np.asarray([[cc[0], cc[1], cc[2]] for cc in centers], dtype=np.float64)
    mid  = c.mean(axis=0)
    R_o  = min(orbit_focus_mult * float(np.linalg.norm(c[1] - c[0])), R_domain)
    key_dir, key_r, new_key = randomjax.split(key, 3)
    d   = jax.random.normal(key_dir, (n, 3), dtype=jnp.float64)
    d   = d / jnp.linalg.norm(d, axis=1, keepdims=True)
    u   = jax.random.uniform(key_r, (n, 1), dtype=jnp.float64)
    xyz = _project_to_sphere(jnp.asarray(mid) + R_o * u ** (1.0 / 3.0) * d)
    return xyz[:, 0:1], xyz[:, 1:2], xyz[:, 2:3], new_key


def _with_orbit_ball(base_fn, centers):
    """Wrap a resample_fn(key, n) -> (x, y, z, key) so a fraction
    --orbit_focus_fraction of every draw comes from _sample_orbit_ball instead.
    Returned unchanged when the option is off or there aren't exactly two
    punctures. The wrapper is what near-puncture rejection redraws from too,
    so replacement points follow the same mixture."""
    if orbit_focus_fraction <= 0.0 or len(centers) != 2:
        return base_fn
    def mixed(key, n):
        n_orb = int(n * orbit_focus_fraction)
        k1, k2, k3, new_key = randomjax.split(key, 4)
        xb, yb, zb, _ = base_fn(k1, n - n_orb)
        xo, yo, zo, _ = _sample_orbit_ball(k2, n_orb, centers)
        xyz = jnp.vstack([jnp.hstack([xb, yb, zb]), jnp.hstack([xo, yo, zo])])
        xyz = xyz[jax.random.permutation(k3, n)]
        return xyz[:, 0:1], xyz[:, 1:2], xyz[:, 2:3], new_key
    return mixed


_CORE_STREAM = 0x636F7265   # "core": fold_in tag of the independent --core_points stream


def _sample_times(key_t, n, t_start, t_end, sigma_time):
    if sigma_time is not None:
        # Truncated-exponential sampling: p(s) ∝ exp(-s/sigma_time) on [0, tau].
        tau  = t_end - t_start
        lam  = 1.0 / sigma_time
        u    = jax.random.uniform(key_t, (n, 1), dtype=jnp.float64)
        return t_start + (-jnp.log1p(-u * (1.0 - jnp.exp(-lam * tau))) / lam)
    return t_start + (t_end - t_start) * jax.random.uniform(key_t, (n, 1), dtype=jnp.float64)


def _core_cloud(key, n, centers):
    """n points from Gaussians of std core_radius around `centers` (split
    equally), with the --min_puncture_dist rejection. Returns x, y, z (n,1)."""
    key_xyz, key_rej = randomjax.split(key)
    resample_fn = lambda k, m: sample_sphere_focused_multistar(k, m, 1.0, core_radius, centers)
    x, y, z, _ = resample_fn(key_xyz, n)
    if min_puncture_dist > 0.0:
        x, y, z, _ = _reject_near_puncture(x, y, z, key_rej, centers, min_puncture_dist, resample_fn)
    return x, y, z


def generate_interior(n, key, t_start, t_end, centers=None, sigma_time=None, follow=None):
    key_core = jax.random.fold_in(key, _CORE_STREAM)   # independent stream; does not touch key
    key_t, key_xyz, new_key = randomjax.split(key, 3)
    t = _sample_times(key_t, n, t_start, t_end, sigma_time)
    if centers is not None:
        resample_fn = _with_orbit_ball(
            lambda k, m: sample_sphere_focused_multistar(k, m, int_focus_fraction, int_focus_radius, centers), centers)
    else:
        resample_fn = lambda k, m: sample_sphere_focused(k, m, int_focus_fraction, int_focus_radius)
    x, y, z, _ = resample_fn(key_xyz, n)
    if min_puncture_dist > 0.0 and centers is not None:
        x, y, z, new_key = _reject_near_puncture(x, y, z, new_key, centers, min_puncture_dist, resample_fn)
    if follow is not None and centers is not None:
        xyz = _follow_puncture_shift(jnp.hstack((x, y, z)), t, centers, *follow, int_focus_radius)
        x, y, z = xyz[:, 0:1], xyz[:, 1:2], xyz[:, 2:3]
    X = jnp.hstack((t, x, y, z))
    if core_points > 0 and centers is not None:
        kc_t, kc_xyz = randomjax.split(key_core)
        tc = _sample_times(kc_t, core_points, t_start, t_end, sigma_time)
        xc, yc, zc = _core_cloud(kc_xyz, core_points, centers)
        if follow is not None:
            xyz = _follow_puncture_shift(jnp.hstack((xc, yc, zc)), tc, centers, *follow, core_radius)
            xc, yc, zc = xyz[:, 0:1], xyz[:, 1:2], xyz[:, 2:3]
        X = jnp.vstack((X, jnp.hstack((tc, xc, yc, zc))))
    return X, new_key


def generate_ic_surface(n, key, t_val, centers=None):
    key_core = jax.random.fold_in(key, _CORE_STREAM)   # independent stream; does not touch key
    key_xyz, new_key = randomjax.split(key)
    if centers is not None:
        resample_fn = _with_orbit_ball(
            lambda k, m: sample_sphere_focused_multistar(k, m, ic_focus_fraction, ic_focus_radius, centers), centers)
    else:
        resample_fn = lambda k, m: sample_sphere_focused(k, m, ic_focus_fraction, ic_focus_radius)
    x, y, z, _ = resample_fn(key_xyz, n)
    if min_puncture_dist > 0.0 and centers is not None:
        x, y, z, new_key = _reject_near_puncture(x, y, z, new_key, centers, min_puncture_dist, resample_fn)
    if core_ic_points > 0 and centers is not None:
        xc, yc, zc = _core_cloud(key_core, core_ic_points, centers)
        x, y, z = jnp.vstack((x, xc)), jnp.vstack((y, yc)), jnp.vstack((z, zc))
    t = jnp.full((x.shape[0], 1), t_val, dtype=jnp.float64)
    return jnp.hstack((t, x, y, z)), new_key


def generate_boundary(n, key, t_start, t_end):
    key_t, key_dir, new_key = randomjax.split(key, 3)
    t   = t_start + (t_end - t_start) * jax.random.uniform(key_t, (n, 1), dtype=jnp.float64)
    xyz = jax.random.normal(key_dir, (n, 3), dtype=jnp.float64)
    xyz = R_domain * xyz / jnp.linalg.norm(xyz, axis=1, keepdims=True)
    return jnp.hstack((t, xyz[:, 0:1], xyz[:, 1:2], xyz[:, 2:3])), new_key


def _adm_ic_batch(xf, yf, zf):
    """Convert the analytic Brill-Lindquist/Bowen-York IC (g^{ab}_0, dot
    g^{ab}_0 from _ic_values_jax) into ADM quantities at t=0: alpha, beta^i,
    chi, gamma-tilde_ij, A-tilde_ij, K (trace), all as JAX-differentiable
    functions of (x,y,z).

    K_ij is obtained from the ADM relation d_t gamma_ij = -2 alpha K_ij
    (Lie-advection-by-shift terms dropped). This relation always gives K=0
    for the time-symmetric part (gt_flat is identically zero); the analytic
    Bowen-York extrinsic curvature (nonzero only when --BL_momenta/
    --BL_spins are set) is added on top, exactly zero otherwise (see
    _bowen_york_K_lower_jax).
    """
    g_flat, gt_flat = _ic_values_jax(xf, yf, zf)
    g_contra  = unpack_metric_components(g_flat)
    gt_contra = unpack_metric_components(gt_flat)
    g_cov  = jnp.linalg.inv(g_contra)
    dg_cov = -jnp.einsum("nac,nbd,ncd->nab", g_cov, g_cov, gt_contra)   # d/dt g_cov

    alpha = 1.0 / jnp.sqrt(jnp.maximum(-g_contra[:, 0, 0], metric_det_eps))
    beta  = alpha[:, None]**2 * g_contra[:, 0, 1:]
    gamma = g_cov[:, 1:, 1:]
    dgamma_dt = dg_cov[:, 1:, 1:]
    K = -dgamma_dt / (2.0 * alpha[:, None, None])
    # alpha=1/psi exactly for brill_lindquist (_bl_contra_jax sets
    # g^tt=-psi**2, and metric_det_eps's clamp is never active since
    # psi>=1), so reuse it instead of re-deriving psi from scratch.
    K = K + _bowen_york_K_lower_jax(xf, yf, zf, 1.0 / alpha)

    chi = jnp.linalg.det(gamma) ** (-1.0 / 3.0)
    gt  = chi[:, None, None] * gamma
    gamma_inv = jnp.linalg.inv(gamma)
    Ktr = jnp.einsum("nij,nij->n", gamma_inv, K)
    At  = chi[:, None, None] * (K - gamma * Ktr[:, None, None] / 3.0)

    return alpha.reshape(-1, 1), beta, chi.reshape(-1, 1), gt, At, Ktr.reshape(-1, 1)


def compute_ccz4_ic(xv_1d, yv_1d, zv_1d):
    """Full vacuum CCZ4 initial data from the analytic Brill-Lindquist/
    Bowen-York IC.

    Theta_0 = 0 and Z^i_0 = 0 (standard Z4 IC convention: the physical/
    constraint-satisfying initial data has no constraint violation "stored"
    in the Z4 auxiliary sector), hence Gamma-hat^i_0 = Gamma-tilde^i_0, the
    ALGEBRAIC contracted Christoffel of gamma-tilde_ij, obtained by
    differentiating the ADM conversion above via jax.jvp along each spatial
    direction (same directional-derivative pattern used elsewhere in this
    file, e.g. for boundary/gauge source terms) -- verified standalone
    against an independent single-point nested-jacfwd ground truth to
    float64 precision on a non-conformally-flat test metric.

    Returns ccz4_ic(N,22) in CCZ4_SLICE field order
    [gt(6), At(6), chi, Khat, Theta, Ghat(3), alpha, beta(3)].
    """
    xv = jnp.asarray(xv_1d, dtype=jnp.float64)
    yv = jnp.asarray(yv_1d, dtype=jnp.float64)
    zv = jnp.asarray(zv_1d, dtype=jnp.float64)
    N  = xv.shape[0]

    alpha, beta, chi, gt, At, Ktr = _adm_ic_batch(xv, yv, zv)

    def _gt_of_xyz(xf, yf, zf):
        return _adm_ic_batch(xf, yf, zf)[3]

    _, dgtx = jax.jvp(lambda xf: _gt_of_xyz(xf, yv, zv), (xv,), (jnp.ones_like(xv),))
    _, dgty = jax.jvp(lambda yf: _gt_of_xyz(xv, yf, zv), (yv,), (jnp.ones_like(yv),))
    _, dgtz = jax.jvp(lambda zf: _gt_of_xyz(xv, yv, zf), (zv,), (jnp.ones_like(zv),))
    gt_d = jnp.stack([dgtx, dgty, dgtz], axis=-1)   # (N,3,3,3): [n,i,j,k]=d_k gt_ij

    _, _, Ghat0, _ = jax.vmap(_christoffel3)(gt, gt_d)

    Theta0 = jnp.zeros((N, 1), dtype=jnp.float64)
    Khat0  = Ktr   # Khat = K - 2*Theta = K at t=0 (Theta=0)

    ccz4_ic = jnp.concatenate([
        pack_sym3(gt), pack_sym3(At), chi, Khat0, Theta0, Ghat0, alpha, beta,
    ], axis=1)
    return ccz4_ic


def unpack_metric_components(g_components):
    g = jnp.zeros((g_components.shape[0], 4, 4), dtype=g_components.dtype)
    for idx, (i, j) in enumerate(METRIC_COMPONENTS):
        vals = g_components[:, idx]
        g = g.at[:, i, j].set(vals)
        g = g.at[:, j, i].set(vals)
    return g


# --------------------------------------------------------------------------------
# INITIAL CONDITION EVALUATION
# --------------------------------------------------------------------------------
def evaluate_metric_initial_data_xyz(x_vals, y_vals, z_vals):
    x_np = np.asarray(x_vals); y_np = np.asarray(y_vals); z_np = np.asarray(z_vals)
    g0  = jnp.asarray(_bl_contra_np(x_np, y_np, z_np), dtype=jnp.float64)
    gt0 = jnp.zeros_like(g0)
    return g0, gt0


# --------------------------------------------------------------------------------
# WEIGHTED MEAN  (no excision in this file -- points near a BH puncture are
# evolved through directly with the moving-puncture gauge, not masked or
# excluded; _wmean/pw plumbing is kept generic since apply_model/
# compute_bfgs_loss are already written against it, but the weight is always
# 1 everywhere).
# --------------------------------------------------------------------------------
def _wmean(res_sq, w):
    """Weighted mean of squared residuals: sum(w*res_sq) / (sum(w)*n_components).
    res_sq: (N, C) squared residuals; w: (N, 1) fixed point weights.
    Reduces exactly to jnp.mean(res_sq) when w == 1 everywhere (always true
    here, since there is no excision -- kept for interface compatibility).
    """
    return jnp.sum(w * res_sq) / (jnp.sum(w) * res_sq.shape[1] + 1e-30)


_IC_LOG_EPS = 1e-6


def _ic_fit(ug, ic_ccz4, w_ic):
    """Soft-IC data-fit terms (before the --lamg0 factor), returned SEPARATELY
    as (fit_abs, fit_log): fit_abs is the weighted MSE on the 22 CCZ4 fields,
    fit_log is --lam_ic_log times a weighted MSE of the log-ratio of chi and
    alpha to their targets (0 when off; see --lam_ic_log for why). Callers add
    them; apply_model logs them separately so loss_IC/loss_g0 stay comparable
    across runs with different --lam_ic_log. Log arguments go through
    _smooth_floor (eps=_IC_LOG_EPS), which is symmetric in its argument -- the
    log term sees |chi|, so it cannot detect chi flipping sign (positivity_
    penalty/--w_pos guards that) -- and gives a large-but-finite penalty
    instead of NaN for a transiently non-positive network value. One helper
    for apply_model, compute_bfgs_loss and _gn_ic so the three can't drift."""
    fit_abs = _wmean((ug[:, CCZ4_SLICE] - ic_ccz4)**2, w_ic)
    fit_log = jnp.zeros((), dtype=fit_abs.dtype)
    if lam_ic_log > 0.0:
        idx = jnp.array([CHI_IDX, ALPHA_IDX])
        d = (jnp.log(_smooth_floor(ug[:, idx], _IC_LOG_EPS))
             - jnp.log(_smooth_floor(ic_ccz4[:, idx], _IC_LOG_EPS)))
        fit_log = lam_ic_log * _wmean(d**2, w_ic)
    return fit_abs, fit_log


def _lapse_weight(x_vals, y_vals, z_vals):
    """Always 1: no excision in this file (moving-puncture gauge evolves
    through the puncture directly). Kept as a function (rather than inlining
    jnp.ones at each call site) so build_train_data doesn't need to change.
    Reads x_vals.shape[0] (static metadata, zero data movement) rather than
    len(np.asarray(x_vals)), which would copy the whole array device->host
    just to measure its length."""
    n_pts = x_vals.shape[0]
    return jnp.ones((n_pts, 1), dtype=jnp.float64)


# --------------------------------------------------------------------------------
# MODEL (vacuum: no matter sector)
# Output (22): [gamma-tilde_ij(6), A-tilde_ij(6), chi, Khat, Theta,
#               Gamma-hat^i(3), alpha, beta^i(3)]
# --------------------------------------------------------------------------------
class MLP(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, t, x, y, z, window_start, window_end):
        init     = nn.initializers.glorot_normal()
        window_width = jnp.maximum(window_end - window_start, jnp.finfo(jnp.float64).eps)
        t_scaled = 2.0 * (t - window_start) / window_width - 1.0
        x_scaled = (x - x_center) / x_half
        y_scaled = (y - y_center) / y_half
        z_scaled = (z - z_center) / z_half
        Z = jnp.concatenate([t_scaled, x_scaled, y_scaled, z_scaled], axis=-1)
        for fs in self.features[:-1]:
            Z = nn.Dense(fs, kernel_init=init)(Z)
            Z = jnp.tanh(Z)
        small_init = nn.initializers.normal(stddev=1e-3)
        Z = nn.Dense(self.features[-1], kernel_init=small_init,
                     bias_init=nn.initializers.zeros)(Z)
        # Shift CCZ4 outputs to the flat-space background (see _FLAT_CCZ4_VEC):
        # gamma-tilde_ij -> delta_ij, A-tilde_ij -> 0, chi -> 1, Khat -> 0,
        # Theta -> 0, Gamma-hat^i -> 0, alpha -> 1, beta^i -> 0.
        Z = Z.at[:, _GT_XX_IDX].add(1.0)    # gamma-tilde_xx ~ 1
        Z = Z.at[:, _GT_YY_IDX].add(1.0)    # gamma-tilde_yy ~ 1
        Z = Z.at[:, _GT_ZZ_IDX].add(1.0)    # gamma-tilde_zz ~ 1
        # Under --chi_bg_mode moving, CHI_IDX is the REGULAR deviation from
        # the moving analytic background (see _wrap_with_moving_chi_bg), so
        # its baseline target is 0, not 1 -- the background itself already
        # carries the "1 + puncture terms" structure. Adding +1.0 here too
        # would double-count that baseline.
        if chi_bg_mode == "physical":
            Z = Z.at[:, CHI_IDX].add(1.0)     # chi ~ 1
        Z = Z.at[:, ALPHA_IDX].add(1.0)   # alpha ~ 1
        return Z


class PINN(nn.Module):
    features: Sequence[int]

    def setup(self):
        self.MLP = MLP(features=self.features)

    @nn.compact
    def __call__(self, t, x, y, z, window_start, window_end):
        return self.MLP(t, x, y, z, window_start, window_end)


# --------------------------------------------------------------------------------
# OUTPUT / EVAL HELPERS
# --------------------------------------------------------------------------------
def full_output(apply_fn, params, t, x, y, z, window_start, window_end):
    return apply_fn(params, t, x, y, z, window_start, window_end)


def ccz4_components_output(apply_fn, params, t, x, y, z, window_start, window_end):
    return full_output(apply_fn, params, t, x, y, z, window_start, window_end)[:, CCZ4_SLICE]


def adm_to_g4(chi, gt, alpha, beta):
    """Reconstruct the 4D metric from CCZ4 ADM fields (standard ADM block formulas):
        gamma_ij = gt_ij / chi,  gamma^ij = inverse(gamma_ij)
        beta_i   = gamma_ij beta^j
        g_ab = [[-alpha^2+beta_k beta^k, beta_j], [beta_i, gamma_ij]]
        g^ab = [[-1/alpha^2, beta^j/alpha^2], [beta^i/alpha^2, gamma^ij - beta^i beta^j/alpha^2]]
        sqrt(-g) = alpha * sqrt(det gamma_ij)
    chi,alpha:(N,1)  gt:(N,3,3)  beta:(N,3). Returns g_cov(N,4,4), g_contra(N,4,4), smg(N,1).
    Verified: reduces exactly to Minkowski at chi=1,gt=I,alpha=1,beta=0; g_cov@g_contra=I
    and det(g_cov)=-smg^2 hold to float64 precision for general inputs.
    """
    chi_safe  = _chi_safe(chi)
    gamma_ij  = gt / chi_safe[..., None]
    gamma_inv = jnp.linalg.inv(gamma_ij)
    beta_low  = jnp.einsum("nij,nj->ni", gamma_ij, beta)
    beta_sq   = jnp.einsum("ni,ni->n", beta_low, beta)[:, None]
    alpha2    = alpha ** 2

    N = chi.shape[0]
    g_cov = jnp.zeros((N, 4, 4), dtype=chi.dtype)
    g_cov = g_cov.at[:, 0, 0].set((-alpha2 + beta_sq)[:, 0])
    g_cov = g_cov.at[:, 0, 1:].set(beta_low)
    g_cov = g_cov.at[:, 1:, 0].set(beta_low)
    g_cov = g_cov.at[:, 1:, 1:].set(gamma_ij)

    g_contra = jnp.zeros((N, 4, 4), dtype=chi.dtype)
    g_contra = g_contra.at[:, 0, 0].set((-1.0 / alpha2)[:, 0])
    g_contra = g_contra.at[:, 0, 1:].set(beta / alpha2)
    g_contra = g_contra.at[:, 1:, 0].set(beta / alpha2)
    g_contra = g_contra.at[:, 1:, 1:].set(
        gamma_inv - jnp.einsum("ni,nj->nij", beta, beta) / alpha2[..., None])

    det_gamma    = jnp.linalg.det(gamma_ij)[:, None]
    sqrt_minus_g = alpha * jnp.sqrt(jnp.maximum(det_gamma, metric_det_eps))
    return g_cov, g_contra, sqrt_minus_g



# --------------------------------------------------------------------------------
# EINSTEIN METRIC EVOLUTION RESIDUAL
# R_{ab} - 8π T̄_{ab} = 0  (standard Einstein equations coupled to scalar field)
# --------------------------------------------------------------------------------
def point_output_from_q(apply_fn, params, q, window_start, window_end):
    t_q = q[0:1].reshape(1, 1)
    x_q = q[1:2].reshape(1, 1)
    y_q = q[2:3].reshape(1, 1)
    z_q = q[3:4].reshape(1, 1)
    return full_output(apply_fn, params, t_q, x_q, y_q, z_q, window_start, window_end)[0]


def _christoffel3(g, dg):
    """g:(3,3) symmetric.  dg[a,b,c]=d_c g_ab.
    Gamma1[a,b,c]=Gamma_abc (first kind, symmetric in b,c).
    Gamma2[k,i,j]=Gamma^k_ij.  Gamma_upper[k]=g^ij Gamma^k_ij.
    Verified (see /tmp dev scripts referenced in the CCZ4 plan) against a
    ground-truth Ricci tensor computed by literally re-differentiating
    Gamma2 -- floating-point-exact agreement."""
    Gamma1 = 0.5 * (jnp.transpose(dg, (0, 2, 1)) + dg - jnp.transpose(dg, (2, 0, 1)))
    g_inv = jnp.linalg.inv(g)
    Gamma2 = jnp.einsum("kl,lij->kij", g_inv, Gamma1)
    Gamma_upper = jnp.einsum("ij,kij->k", g_inv, Gamma2)
    return Gamma1, Gamma2, Gamma_upper, g_inv


# Number of points along each 1D snapshot cut (see _snap_cut_line/_SnapCut).
_SNAP_NX_SHARED = 300

# Same sharing rationale as the module-level snapshot constants above, for the per-window
# 2D (x,y) slice snapshot(s) (see save_2d_slice_snapshot in train_window):
# a fixed x-y grid and its analytic t=0 reference, computed once here
# rather than rebuilt every window. Resolution deliberately lower than the
# end-of-run "2D SLICES" block's nxy=160 (this one runs once per window --
# up to --N_window times over a run -- vs. that block's fixed 3 times
# total, so a coarser grid keeps the added per-run cost comparable).
_SNAP_NXY_2D = 100


class _Snap2DGrid:
    """One (x,y) evaluation grid + its analytic t=0 reference for
    save_2d_slice_snapshot -- bundled so a SECOND, differently-sized grid
    (see _snap_grid_2d_zoom below, --snapshot_2d_zoom_halfwidth) can reuse
    the exact same construction/masking logic as the full-domain one
    (_snap_grid_2d_full) instead of a second copy-pasted block of module
    constants. mask_radius is always the true simulation domain's own
    R_domain, regardless of this grid's own (possibly much smaller) extent
    -- a zoomed-in box well inside R_domain never actually masks anything,
    but this stays correct even if --snapshot_2d_zoom_halfwidth were ever
    set close to or past R_domain."""
    def __init__(self, x_lo, x_hi, y_lo, y_hi, nxy, mask_radius):
        self.nxy = nxy
        x_grid, y_grid = np.linspace(x_lo, x_hi, nxy), np.linspace(y_lo, y_hi, nxy)
        self.Xg, self.Yg = np.meshgrid(x_grid, y_grid, indexing="xy")
        self.outside_domain = (self.Xg**2 + self.Yg**2) > mask_radius**2
        self.xr = self.Xg.ravel()
        self.yr = self.Yg.ravel()
        self.zr = np.zeros_like(self.xr)
        self.ic_shared = np.asarray(compute_ccz4_ic(self.xr, self.yr, self.zr))


if snapshot_2d_every > 0:
    _snap_grid_2d_full = _Snap2DGrid(x_min, x_max, y_min, y_max, _SNAP_NXY_2D, R_domain)
    _snap_grid_2d_zoom = _Snap2DGrid(-snapshot_2d_zoom_halfwidth, snapshot_2d_zoom_halfwidth,
                                      -snapshot_2d_zoom_halfwidth, snapshot_2d_zoom_halfwidth,
                                      _SNAP_NXY_2D, R_domain)
else:
    # each grid costs a compute_ccz4_ic call on ~10^4 points at import
    _snap_grid_2d_full = _snap_grid_2d_zoom = None
_Q_LABELS_2D = [r"$\alpha$", r"$\beta^x$", r"$g_{xx}$", r"$H$"]


def _tf3(X, g, g_inv):
    """Trace-free part of symmetric (3,3) X w.r.t. metric g."""
    tr = jnp.einsum("ij,ij->", g_inv, X)
    return X - (1.0 / 3.0) * g * tr


def _lie_lower2(X, dX, beta, dbeta):
    """(Lie_beta X)_ij for symmetric lower-2-tensor X.
    dX[i,j,k]=d_k X_ij (symmetric i,j).  dbeta[i,j]=d_j beta^i."""
    adv    = jnp.einsum("k,ijk->ij", beta, dX)
    term_i = jnp.einsum("kj,ki->ij", X, dbeta)   # X_kj d_i beta^k
    term_j = jnp.einsum("ik,kj->ij", X, dbeta)   # X_ik d_j beta^k
    return adv + term_i + term_j


def _lie_scalar(dX, beta):
    return jnp.einsum("k,k->", beta, dX)


# --------------------------------------------------------------------------------
# --deriv_mode taylor: value, first and second derivatives of the network in ONE
# batched forward pass. For a tanh layer h = tanh(a), a = W z + b, with
# s1 = 1 - h^2 and s2 = -2 h s1:
#     h_k  = s1 a_k,                 a_k  = W z_k
#     h_kl = s1 a_kl + s2 a_k a_l,   a_kl = W z_kl
# (k, l derivative directions; z_kl = 0 at the input, which is affine in
# t,x,y,z). First derivatives in t,x,y,z (4 directions), second derivatives
# only in the 6 distinct spatial pairs, so per point the cost is about
# 1 + 4 + 6 network-sized passes instead of the autodiff stack's
# jvp + jacfwd + jacfwd-of-jacfwd (9 second-derivative directions, primal
# and first derivatives recomputed inside each). Exact arithmetic, not finite
# differences: agrees with autodiff to round-off (tools/taylor_proto.py).
# --------------------------------------------------------------------------------
_TAYLOR_PAIRS = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
_TAYLOR_SYM   = jnp.array([[0, 1, 2], [1, 3, 4], [2, 4, 5]])   # (k,l) -> pair index
# apply_fn -> "plain" (model.apply) | "moving" (_wrap_with_moving_chi_bg(model.apply));
# filled where those functions are created. Anything else uses autodiff.
_TAYLOR_KIND = {}


def _taylor_capable(apply_fn):
    ok = apply_fn in _TAYLOR_KIND
    if not ok:   # runs at trace time, so once per compilation, never per step
        print(f"NOTE: --deriv_mode taylor: {getattr(apply_fn, '__name__', apply_fn)!s} is not a "
              f"plain/moving-background MLP apply_fn; this residual uses autodiff.")
    return ok


def _taylor_mlp(net_params, t, x, y, z, window_start, window_end):
    """MLP.__call__ with derivatives. (N,1) inputs -> val (N,22), d_t (N,22),
    d_xyz (N,22,3) [field, k], d_xyz d_xyz of the _DD_OUTPUT_IDXS fields (N,11,3,3)."""
    mp = net_params["params"]["MLP"]
    layers = [mp[f"Dense_{i}"] for i in range(len(mp))]
    window_width = jnp.maximum(window_end - window_start, jnp.finfo(jnp.float64).eps)
    Z = jnp.concatenate([2.0 * (t - window_start) / window_width - 1.0,
                         (x - x_center) / x_half,
                         (y - y_center) / y_half,
                         (z - z_center) / z_half], axis=-1)
    scale = jnp.stack([2.0 / window_width, 1.0 / x_half, 1.0 / y_half, 1.0 / z_half])
    W, b = layers[0]["kernel"], layers[0]["bias"]
    a  = Z @ W + b                                    # (N,H)
    a1 = scale[:, None] * W                           # (4,H), the same at every point
    h  = jnp.tanh(a); s1 = 1.0 - h * h; s2 = -2.0 * h * s1
    h1 = s1[:, None, :] * a1[None]                    # (N,4,H)
    h2 = jnp.stack([s2 * (a1[1 + k] * a1[1 + l]) for k, l in _TAYLOR_PAIRS], axis=1)   # (N,6,H)
    for L in layers[1:-1]:
        W, b = L["kernel"], L["bias"]
        a, a1, a2 = h @ W + b, h1 @ W, h2 @ W
        h  = jnp.tanh(a); s1 = 1.0 - h * h; s2 = -2.0 * h * s1
        sp = a1[:, 1:, :]
        h2 = s1[:, None, :] * a2 + s2[:, None, :] * jnp.stack(
            [sp[:, k] * sp[:, l] for k, l in _TAYLOR_PAIRS], axis=1)
        h1 = s1[:, None, :] * a1
    W, b = layers[-1]["kernel"], layers[-1]["bias"]
    val = h @ W + b
    val = val.at[:, _GT_XX_IDX].add(1.0)          # same flat-space shifts as MLP.__call__
    val = val.at[:, _GT_YY_IDX].add(1.0)
    val = val.at[:, _GT_ZZ_IDX].add(1.0)
    if chi_bg_mode == "physical":
        val = val.at[:, CHI_IDX].add(1.0)
    val = val.at[:, ALPHA_IDX].add(1.0)
    d1 = h1 @ W                                       # (N,4,22)
    d2 = h2 @ W[:, _DD_OUTPUT_IDXS]                   # (N,6,11)
    return (val, d1[:, 0, :], jnp.moveaxis(d1[:, 1:, :], 1, 2),
            jnp.moveaxis(d2[:, _TAYLOR_SYM, :], 3, 1))


def _taylor_derivs(apply_fn, params, t, x, y, z, window_start, window_end):
    """Batched (val, d_t, d_xyz, d_xyz d_xyz) as stacked by derivs_one in
    r_ccz4_evolution, for an apply_fn registered in _TAYLOR_KIND."""
    if _TAYLOR_KIND[apply_fn] == "plain":
        return _taylor_mlp(params, t, x, y, z, window_start, window_end)
    val, dt, d, dd = _taylor_mlp(params["net"], t, x, y, z, window_start, window_end)
    traj_tt = jax.lax.stop_gradient(params["traj_tt"])
    traj_xx = jax.lax.stop_gradient(params["traj_xx"])

    def bg_q(q):   # the analytic chi background at one point (cheap: no network)
        return _chi_bg_from_traj(q[0:1].reshape(1, 1), q[1:2].reshape(1, 1), q[2:3].reshape(1, 1),
                                 q[3:4].reshape(1, 1), traj_tt, traj_xx)[0, 0]
    Q   = jnp.concatenate([t, x, y, z], axis=1)
    bv  = jax.vmap(bg_q)(Q)
    bg1 = jax.vmap(jax.grad(bg_q))(Q)                                # (N,4)
    bH  = jax.vmap(jax.jacfwd(jax.jacfwd(bg_q)))(Q)[:, 1:, 1:]       # (N,3,3)
    return (val.at[:, CHI_IDX].add(bv), dt.at[:, CHI_IDX].add(bg1[:, 0]),
            d.at[:, CHI_IDX, :].add(bg1[:, 1:]), dd.at[:, _DD_CHI_IDX].add(bH))


@partial(jax.jit, static_argnums=(0,))
def r_ccz4_evolution(apply_fn, params, t, x, y, z, window_start, window_end):
    """
    Vacuum CCZ4 evolution residual: for every one of the 22 metric-sector
    fields (gamma-tilde_ij, A-tilde_ij, chi, Khat, Theta, Gamma-hat^i, alpha,
    beta^i), the residual is d_t(field) - RHS(fields, spatial derivatives),
    matching the equations in equations.tex Sec. "CCZ4 evolution equations",
    with the matter stress-energy source terms (tau, S_i, S_ij) set
    identically to zero (vacuum). No second TIME derivatives anywhere
    (first-order-in-time system) -- spatial derivatives use jax.jacfwd
    restricted to x,y,z only.

    Verified standalone against independent ground truth before being
    transcribed here:
      - Rhat3_ij + Rchi_ij (the conformal Ricci split) matches a Ricci tensor
        computed by directly re-differentiating the physical-metric
        Christoffel symbols, to float64 precision, when Ghat = Gt (Z = 0)
        (this caught and fixed a missing (ij)-symmetrization on the
        Ghat^k*Gamma_ijk term). For Z != 0 it equals R_ij + D_iZ_j + D_jZ_i
        by construction (see the Z comment in residual_one).
      - 2026-10-10: all 22 rows match an independent implementation of
        Alic et al. (2012) / arXiv:1705.01071 to ~1e-16 on random fields
        with Z, Theta, beta != 0, and vanish for static isotropic
        Schwarzschild (all 22) and Kerr-Schild Kerr (the 18 gravity rows).
      - The conformal Hessian D_iD_jalpha matches a Ricci-free ground-truth
        Hessian (d_id_jalpha - Gamma^k_ij(physical)d_kalpha), to float64
        precision (this caught and fixed a sign error in the chi-correction
        term), AND its trace matches the paper's own given Laplacian formula.
      - Flat space (chi=1,gt=delta,At=0,Khat=Theta=0,Ghat=0,alpha=1,beta=0,
        all derivatives zero) gives an EXACT zero residual for all 8 fields
        (trivial vacuum-static-solution check).

    Returns (N,22) unweighted residual: [0:18]=gravity (gt(6),At(6),chi,
    Khat,Theta,Ghat(3)), [18:22]=gauge (alpha,beta(3)).
    """
    q_batch = jnp.concatenate([t, x, y, z], axis=1)

    def derivs_one(q):
        # Autodiff derivative stack (--deriv_mode autodiff, the default):
        # value, d_t, d_xyz of all 22 outputs and d_xyz d_xyz of the 11 that
        # need it, returned as (22,), (22,), (22,3), (11,3,3).
        # Single per-point forward pass (all 22 outputs), shared by every
        # field below instead of each field independently re-running the
        # full network via its own point_output_from_q call -- see
        # full_of_q's uses in the "values"/"time derivatives"/"spatial
        # derivatives" blocks.
        def full_of_q(ql):
            return point_output_from_q(apply_fn, params, ql, window_start, window_end)

        xyz = q[1:4]
        def q_of_xyz(p):
            return jnp.concatenate([q[0:1], p])

        # --- values: ONE forward pass shared by all 8 fields (was 8) --------
        full_val = full_of_q(q)

        # --- time derivatives (LHS of every residual): ONE jvp (was 8) -----
        # jvp commutes exactly with slicing a linear (index/select) function
        # of its output, so slicing full_dt per field below is bit-for-bit
        # identical to jax.jvp(field_of_q, (q,), (...))[1] for that field --
        # verified directly against the original per-field closures on
        # random points before this was adopted.
        full_dt   = jax.jvp(full_of_q, (q,), (jnp.array([1.,0.,0.,0.]),))[1]

        # --- spatial derivatives (x,y,z only): ONE jacfwd / jacfwd-of-jacfwd
        # each (was 8 + 4), same exact-commutation argument as above -------
        full_d  = jax.jacfwd(lambda p: full_of_q(q_of_xyz(p)))(xyz)              # (22,3)

        # Only gt, chi, alpha, beta (11 of the 22 outputs) need a second
        # spatial derivative. Unlike full_val/full_d above (where slicing
        # AFTER the shared call is free, since every field needs that same
        # derivative order anyway), a jacfwd-of-jacfwd's cost includes the
        # network's own per-layer tangent-of-tangent bookkeeping scaling
        # with output width at the final projection layer -- so selecting
        # the 11 needed outputs BEFORE the double-jacfwd (instead of
        # computing all 22 second derivatives and slicing 4 ways after)
        # avoids paying that bookkeeping for the 11 unused Khat/Theta/Ghat/
        # At columns. Selecting a fixed subset of a function's outputs
        # commutes exactly with differentiation (same index-then-linear
        # argument as _unpack_sym3_leading's callers above), so this is
        # bit-for-bit equivalent to the old "differentiate all 22, slice
        # after" version -- just cheaper.
        def _dd_of_q(ql):
            return full_of_q(ql)[_DD_OUTPUT_IDXS]  # (11,): gt(6),chi,alpha,beta(3)
        full_dd  = jax.jacfwd(jax.jacfwd(lambda p: _dd_of_q(q_of_xyz(p))))(xyz)  # (11,3,3)
        return full_val, full_dt, full_d, full_dd

    def residual_one(full_val, full_dt, full_d, full_dd):
        gt    = unpack_sym3(full_val[GT_SLICE])
        At    = unpack_sym3(full_val[AT_SLICE])
        chi   = full_val[CHI_IDX]
        Khat  = full_val[KHAT_IDX]
        Theta = full_val[THETA_IDX]
        Ghat  = full_val[GHAT_SLICE]
        alpha = full_val[ALPHA_IDX]
        beta  = full_val[BETA_SLICE]
        gt_dt     = unpack_sym3(full_dt[GT_SLICE])
        At_dt     = unpack_sym3(full_dt[AT_SLICE])
        chi_dt    = full_dt[CHI_IDX]
        Khat_dt   = full_dt[KHAT_IDX]
        Theta_dt  = full_dt[THETA_IDX]
        Ghat_dt   = full_dt[GHAT_SLICE]
        alpha_dt  = full_dt[ALPHA_IDX]
        beta_dt   = full_dt[BETA_SLICE]
        gt_d    = _unpack_sym3_leading(full_d[GT_SLICE])    # [i,j,k]=d_k gt_ij
        At_d    = _unpack_sym3_leading(full_d[AT_SLICE])
        chi_d   = full_d[CHI_IDX]
        Khat_d  = full_d[KHAT_IDX]
        Theta_d = full_d[THETA_IDX]
        Ghat_d  = full_d[GHAT_SLICE]        # [i,j]=d_j Ghat^i
        alpha_d = full_d[ALPHA_IDX]
        beta_d  = full_d[BETA_SLICE]        # [i,j]=d_j beta^i
        gt_dd    = _unpack_sym3_leading(full_dd[_DD_GT_SLICE])
        chi_dd   = full_dd[_DD_CHI_IDX]
        alpha_dd = full_dd[_DD_ALPHA_IDX]
        beta_dd  = full_dd[_DD_BETA_SLICE]  # [i,j,k]=d_k d_j beta^i


        Gamma1, Gamma2, Gt_upper, gt_inv = _christoffel3(gt, gt_d)
        # chi appears as a divisor throughout this residual (
        # Rchi, gt7b, gt8b, DaDa_full below) -- an undertrained/transient
        # network can drive chi through zero or negative at a point, which
        # would otherwise produce an Inf/NaN residual at that single point
        # and poison the whole batch's loss/gradient via the mean/sum
        # reduction. _chi_safe floors it at --chi_floor (default 1e-5, the
        # finite-difference convention), so every 1/chi factor is bounded by
        # 1/chi_floor. (A floor at 1e-14, as jnp.maximum(chi, metric_det_eps)
        # once was, is no protection: it was the cause of a real
        # loss_einstein spike to ~1e36 then NaN.) Only VALUES divided by chi
        # are floored -- chi_d/chi_dd are left untouched.
        chi_safe   = _chi_safe(chi)
        gamma_inv  = chi_safe * gt_inv
        Z          = chi * (Ghat - Gt_upper) / 2.0

        # No separate D_iZ^i / D_(iZ_j) terms are needed: building the Ricci
        # tensor with the EVOLVED Ghat^k (in Rhat3 and Rchi's last bracket)
        # plus Rchi's explicit Z term (rs4) gives exactly
        #     Rphys_ij = R_ij + D_iZ_j + D_jZ_i,   R3 = R + 2 D_iZ^i
        # (Bezares, Palenzuela & Bona 2017, arXiv:1705.01071, below their
        # Eq. (syseq2)). Until 2026-10-10 this function ALSO added DiZj_sym
        # to the At bracket and 2*DiZi to the Theta equation, so the Z terms
        # were counted twice (3 D_(iZ_j) in At, R + 4 D_iZ^i in Theta).
        # Verified: all 22 rows now match an independent Alic et al. (2012)
        # implementation (physical Ricci/D_iZ_j by autodiff of gt/chi) to
        # ~1e-16 on random fields with Z != 0.

        # --- matter: none (vacuum) -- stress-energy source terms are zero.
        tau  = 0.0
        Si_4 = jnp.zeros(3, dtype=gt.dtype)
        Sij  = jnp.zeros((3, 3), dtype=gt.dtype)

        kappa_2_l = kappa_2

        # ================= assemble RHS (see equations.tex) =================
        d_k_beta_k = jnp.trace(beta_d)

        rhs_chi = _lie_scalar(chi_d, beta) + (2.0/3.0)*chi*(alpha*(Khat+2.0*Theta) - d_k_beta_k)

        lap_alpha = chi*jnp.einsum("ij,ij->", gt_inv, alpha_dd) - chi*jnp.einsum("k,k->", Gt_upper, alpha_d) \
                    - 0.5*jnp.einsum("ij,i,j->", gt_inv, alpha_d, chi_d)
        # AtAt = At_ij At^ij, At^ij raised with the CONFORMAL metric (no chi
        # factor) -- this is what makes (1/3)*(Khat+2*Theta)**2 + AtAt equal
        # the standard K_ij*K^ij identity (verified standalone: K_ij*K^ij ==
        # (1/3)*K**2 + AtAt to float64 precision only with this raise; the
        # chi**2-scaled physical raise used here previously was off by chi**2).
        At_upper_conf = jnp.einsum("ik,jl,kl->ij", gt_inv, gt_inv, At)
        AtAt = jnp.einsum("ij,ij->", At, At_upper_conf)
        S_trace = jnp.einsum("ij,ij->", gamma_inv, Sij)
        rhs_Khat = _lie_scalar(Khat_d, beta) - lap_alpha \
            + alpha*((1.0/3.0)*(Khat+2.0*Theta)**2 + AtAt + 4.0*jnp.pi*(tau+S_trace)
                     + kappa_z*(1.0-kappa_2_l)*Theta) \
            + 2.0*jnp.einsum("i,i->", Z, alpha_d)

        At_trace = jnp.einsum("ij,ij->", gt_inv, At)
        lndet_gt = jnp.log(jnp.maximum(jnp.linalg.det(gt), 1e-14))
        rhs_gt = _lie_lower2(gt, gt_d, beta, beta_d) - (2.0/3.0)*gt*d_k_beta_k \
            - 2.0*alpha*(At - (1.0/3.0)*gt*At_trace) - (alpha/3.0)*kappa_c*gt*lndet_gt

        # Rhat3_ij (conformal-metric Ricci, standard BSSN form)
        rterm1 = -0.5*jnp.einsum("mn,ijmn->ij", gt_inv, gt_dd)
        rt2a = jnp.einsum("ki,kj->ij", gt, Ghat_d); rt2b = jnp.einsum("kj,ki->ij", gt, Ghat_d)
        rterm2 = 0.5*(rt2a+rt2b)
        rterm3_raw = jnp.einsum("k,ijk->ij", Ghat, Gamma1)
        rterm3 = 0.5*(rterm3_raw + rterm3_raw.T)     # explicit (ij) symmetrization
        rterm4 = jnp.einsum("mn,kmi,jkn->ij", gt_inv, Gamma2, Gamma1) \
               + jnp.einsum("mn,kmj,ikn->ij", gt_inv, Gamma2, Gamma1) \
               + jnp.einsum("mn,kmi,knj->ij", gt_inv, Gamma2, Gamma1)
        Rhat3 = rterm1 + rterm2 + rterm3 + rterm4

        rs1 = 0.5*chi_dd
        rs2 = -0.5*jnp.einsum("kij,k->ij", Gamma2, chi_d)
        rs3 = -(1.0/(4.0*chi_safe))*jnp.einsum("i,j->ij", chi_d, chi_d)
        rs4a = jnp.einsum("k,ki,j->ij", Z, gt, chi_d); rs4b = jnp.einsum("k,kj,i->ij", Z, gt, chi_d)
        rs4 = (1.0/chi_safe)*(rs4a+rs4b)
        rbracket = jnp.einsum("km,km->", gt_inv, chi_dd) - (3.0/(2.0*chi_safe))*jnp.einsum("km,k,m->", gt_inv, chi_d, chi_d) \
                   - jnp.einsum("k,k->", Ghat, chi_d)
        rs5 = 0.5*gt*rbracket
        chiRchi = rs1+rs2+rs3+rs4+rs5
        Rchi = chiRchi/chi_safe
        Rphys_ij = Rhat3 + Rchi          # = R_ij + D_iZ_j + D_jZ_i (see above)
        R3 = jnp.einsum("ij,ij->", gamma_inv, Rphys_ij)   # = R + 2 D_iZ^i

        rhs_Theta = _lie_scalar(Theta_d, beta) \
            + 0.5*alpha*(R3 + (2.0/3.0)*Khat**2 + (2.0/3.0)*Theta*(Khat-2.0*Theta) - AtAt) \
            - jnp.einsum("i,i->", Z, alpha_d) \
            - alpha*(8.0*jnp.pi*tau + kappa_z*(2.0+kappa_2_l)*Theta)

        gt1 = jnp.einsum("j,ij->i", beta, Ghat_d)
        gt2 = -jnp.einsum("j,ij->i", Ghat, beta_d)
        gt3 = (2.0/3.0)*Ghat*d_k_beta_k
        gt4 = jnp.einsum("jk,ijk->i", gt_inv, beta_dd)
        gt5 = (1.0/3.0)*jnp.einsum("ij,kjk->i", gt_inv, beta_dd)
        At_up_full = jnp.einsum("ik,jl,kl->ij", gt_inv, gt_inv, At)
        gt6 = -2.0*jnp.einsum("ij,j->i", At_up_full, alpha_d)
        gt7a = jnp.einsum("ijk,jk->i", Gamma2, At_up_full)
        gt7b = -1.5/chi_safe*jnp.einsum("ij,j->i", At_up_full, chi_d)
        gt7c = -(2.0/3.0)*jnp.einsum("ij,j->i", gt_inv, Khat_d)
        gt7d = -8.0*jnp.pi*jnp.einsum("ij,j->i", gt_inv, Si_4)
        gt7 = 2.0*alpha*(gt7a+gt7b+gt7c+gt7d)
        # alpha (the lapse) is EXPECTED to collapse toward zero right at the
        # puncture under 1+log slicing -- the correct physical moving-puncture
        # gauge behavior -- and can even dip slightly negative in an
        # under-trained network (seen: alpha_min ~ -0.02 on a real run). The
        # equation has 2*alpha*(-gt^ij (Theta/alpha) d_j alpha), i.e. alpha
        # times a 1/alpha, which is exactly -2*Theta*gt^ij d_j alpha: cancel it
        # ANALYTICALLY (gt8c) rather than dividing. This needs no floor at all
        # and is correct for either sign of alpha -- the previous
        # Theta/_smooth_floor(alpha) divisor was |alpha|-like, so for alpha<0
        # the term came out with the WRONG SIGN.
        gt8a = -jnp.einsum("ij,j->i", gt_inv, (1.0/3.0)*Theta_d)
        gt8b = -(1.0/chi_safe)*Z*(kappa_z+(2.0/3.0)*(Khat+2.0*Theta))
        gt8c = -jnp.einsum("ij,j->i", gt_inv, Theta*alpha_d)
        gt8 = 2.0*(alpha*(gt8a+gt8b) + gt8c)
        rhs_Ghat = gt1+gt2+gt3+gt4+gt5+gt6+gt7+gt8

        # D_iD_j alpha: standard conformal Hessian identity (verified against
        # a ground-truth physical-Christoffel Hessian to float64 precision,
        # AND its trace matches lap_alpha above exactly).
        DaDa_full = alpha_dd - jnp.einsum("kij,k->ij", Gamma2, alpha_d) \
            + (1.0/(2.0*chi_safe))*(jnp.einsum("i,j->ij", chi_d, alpha_d) + jnp.einsum("j,i->ij", chi_d, alpha_d)
                                - gt*jnp.einsum("kl,k,l->", gt_inv, chi_d, alpha_d))
        bracket_At = alpha*(Rphys_ij - 8.0*jnp.pi*Sij) - DaDa_full   # Rphys includes 2 D_(iZ_j)
        bracket_At_TF = _tf3(bracket_At, gt, gt_inv)
        # At_mixed[i,j] = At^i_j = gt^ik At_kj (j still lower). AtAt_mixed must
        # contract At's own lower k against At_mixed's UPPER k (one up, one
        # down) to form At_ik At^k_j = At_ik gt^kl At_lj -- operand order
        # matters here: einsum("ik,kj->ij", At, At_mixed) puts At's lower k
        # (its own second slot) against At_mixed's upper k (its own first
        # slot). The previous order (At_mixed, At) instead summed two LOWER
        # k's directly with no metric between them -- verified wrong against
        # an exact Kerr (Kerr-Schild) vacuum solution (rhs_At ~5e-3, should
        # be exactly 0 since d_t of every field is 0 in these coordinates);
        # this order matches that check to float64 precision.
        At_mixed = jnp.einsum("ik,kj->ij", gt_inv, At)          # At^i_j (mixed, one index raised)
        AtAt_mixed = jnp.einsum("ik,kj->ij", At, At_mixed)      # At_ik At^k_j
        rhs_At = _lie_lower2(At, At_d, beta, beta_d) - (2.0/3.0)*At*d_k_beta_k \
            - (alpha/3.0)*kappa_c*gt*At_trace + chi*bracket_At_TF \
            + alpha*(Khat*At - 2.0*AtAt_mixed)

        rhs_alpha = jnp.einsum("i,i->", beta, alpha_d) - 2.0*alpha*f_alpha*Khat
        rhs_beta  = jnp.einsum("j,ij->i", beta, beta_d) + 0.75*f_beta*Ghat - eta_shift*beta

        res_gt    = gt_dt    - rhs_gt
        res_At    = At_dt    - rhs_At
        res_chi   = chi_dt   - rhs_chi
        res_Khat  = Khat_dt  - rhs_Khat
        res_Theta = Theta_dt - rhs_Theta
        res_Ghat  = Ghat_dt  - rhs_Ghat
        res_alpha = alpha_dt - rhs_alpha
        res_beta  = beta_dt  - rhs_beta

        return jnp.concatenate([
            pack_sym3(res_gt.reshape(1,3,3))[0],
            pack_sym3(res_At.reshape(1,3,3))[0],
            res_chi.reshape(1), res_Khat.reshape(1), res_Theta.reshape(1),
            res_Ghat, res_alpha.reshape(1), res_beta,
        ])

    if deriv_mode == "taylor" and _taylor_capable(apply_fn):
        derivs = _taylor_derivs(apply_fn, params, t, x, y, z, window_start, window_end)
    else:
        derivs = jax.vmap(derivs_one)(q_batch)
    return jax.vmap(residual_one)(*derivs)


# --------------------------------------------------------------------------------
# BOUNDARY CONDITIONS
# --------------------------------------------------------------------------------
@partial(jax.jit, static_argnums=(0,))
def metric_dirichlet_residual(apply_fn, params, t, x, y, z, window_start, window_end, gb, lam_bc):
    return jnp.sqrt(lam_bc) * (ccz4_components_output(apply_fn, params, t, x, y, z, window_start, window_end) - gb)


@partial(jax.jit, static_argnums=(0,))
def metric_sommerfeld_residual(apply_fn, params, t, x, y, z, window_start, window_end, c, lam_bc, r_eps):
    """Sommerfeld outgoing-wave BC for the 22 CCZ4 metric-sector fields toward
    their flat-space targets (_FLAT_CCZ4_VEC).

    g/gt/gx/gy/gz come from ONE jax.linearize (one full nonlinear forward
    pass through the network, batched over all boundary points) followed by
    4 cheap applications of the resulting linear tangent map, instead of 4
    separate jax.jvp calls that each redundantly re-ran the full nonlinear
    forward pass just to get one direction's tangent (3 of the 4 discarded
    their own primal). This is exactly what jax.jvp(f, primals, (tangent,))
    already computes internally (linearize once, then apply the tangent) --
    sharing that linearization across all 4 directions instead of rebuilding
    it 4 times changes nothing about the arithmetic, only how many times the
    shared nonlinear forward pass is executed."""
    def f(tv, xv, yv, zv):
        return ccz4_components_output(apply_fn, params, tv, xv, yv, zv, window_start, window_end)
    g, f_jvp = jax.linearize(f, t, x, y, z)
    zt, zx, zy, zz = jnp.zeros_like(t), jnp.zeros_like(x), jnp.zeros_like(y), jnp.zeros_like(z)
    gt = f_jvp(jnp.ones_like(t), zx, zy, zz)
    gx = f_jvp(zt, jnp.ones_like(x), zy, zz)
    gy = f_jvp(zt, zx, jnp.ones_like(y), zz)
    gz = f_jvp(zt, zx, zy, jnp.ones_like(z))
    r       = jnp.maximum(jnp.sqrt(x**2 + y**2 + z**2), r_eps)
    dn      = x*gx + y*gy + z*gz
    delta_g = g - _FLAT_CCZ4_VEC
    return jnp.sqrt(lam_bc) * (gt + c*dn/r + c*delta_g/r)


# --------------------------------------------------------------------------------
# GAUGE: no longer a separate penalty term (contrast with EEdedonder_BS.py's
# --gauge static/harmonic_lapse/harmonic/ghg/... menu). The 1+log slicing and
# Gamma-driver shift equations are two more rows of r_ccz4_evolution's output
# (rhs_alpha, rhs_beta), on exactly the same footing as the other 20 CCZ4
# fields -- see r_ccz4_evolution above and Stage 7 of the CCZ4 plan.
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
# POSITIVITY PENALTY
#
# Ported from bbh_headon_bssn_pinn_shift_driven_puncture.py's point_terms:
# pos = relu(-chi)^2 + relu(-alpha)^2, a per-point penalty (not a PDE
# residual) on the raw field VALUES, keeping chi and alpha from crossing
# zero during training. Weighted by --w_pos, applied on the interior
# collocation points only (same as bbh_headon).
# --------------------------------------------------------------------------------
@partial(jax.jit, static_argnums=(0,))
def positivity_penalty(apply_fn, params, t, x, y, z, window_start, window_end):
    out   = full_output(apply_fn, params, t, x, y, z, window_start, window_end)
    chi   = out[:, CHI_IDX]
    alpha = out[:, ALPHA_IDX]
    return jax.nn.relu(-chi)**2 + jax.nn.relu(-alpha)**2


# --------------------------------------------------------------------------------
# CONSTRAINT DIAGNOSTICS
#
# Unlike EEdedonder_BS.py (which had to project the covariant Ricci residual
# onto n^a n^b to get a Hamiltonian-constraint-like scalar post hoc), CCZ4
# carries its constraint violations as EXPLICIT dynamical fields: Theta is
# (twice) the Hamiltonian constraint violation, and Z^i = chi(Ghat^i-Gt^i)/2
# is the momentum constraint violation, directly -- no extra projection.
# --------------------------------------------------------------------------------
@partial(jax.jit, static_argnums=(0,))
def hamiltonian_constraint(apply_fn, params, t, x, y, z, window_start, window_end):
    out = full_output(apply_fn, params, t, x, y, z, window_start, window_end)
    return out[:, THETA_IDX]


@partial(jax.jit, static_argnums=(0,))
def gauge_source_vector(apply_fn, params, t, x, y, z, window_start, window_end):
    """Momentum-constraint violation Z^i = chi*(Gamma-hat^i - Gamma-tilde^i)/2,
    per point (vmapped so each point keeps its own t, not just point 0's)."""
    def z_single(tp, xp, yp, zp):
        p3 = jnp.stack([xp, yp, zp])

        def gt_of_xyz(p3_):
            # Kept separate from the value call below so jax.jacfwd only
            # traces through GT_SLICE, not the whole 22-field output.
            out = full_output(apply_fn, params, tp.reshape(1,1), p3_[0:1].reshape(1,1),
                               p3_[1:2].reshape(1,1), p3_[2:3].reshape(1,1),
                               window_start, window_end)
            return unpack_sym3(out[0, GT_SLICE])

        # Single un-differentiated full_output call covers BOTH gt (the
        # value, not the jacfwd derivative) and chi_p/Ghat_p -- these were
        # previously two separate calls at the identical point, the second
        # purely to re-read fields already present in the first's output.
        out_val = full_output(apply_fn, params, tp.reshape(1,1), xp.reshape(1,1), yp.reshape(1,1),
                               zp.reshape(1,1), window_start, window_end)
        gt = unpack_sym3(out_val[0, GT_SLICE])
        chi_p, Ghat_p = out_val[0, CHI_IDX], out_val[0, GHAT_SLICE]
        gt_d = jax.jacfwd(gt_of_xyz)(p3)
        _, _, Gt_upper, _ = _christoffel3(gt, gt_d)
        return chi_p * (Ghat_p - Gt_upper) / 2.0

    return jax.vmap(z_single)(t[:, 0], x[:, 0], y[:, 0], z[:, 0])


# --------------------------------------------------------------------------------
# CAUSAL (TEMPORAL) LOSS WEIGHTING  (--causal_weight)
#
# Wang/Sankaran/Perdikaris-style: split the window into causal_buckets
# sequential buckets in t; weight each point's residual contribution by
# exp(-causal_eps * [stop-gradiented cumulative mean residual from buckets
# STRICTLY EARLIER than its own]). A late bucket only gets nonzero weight
# once earlier buckets are already small -- unlike --time_dist exponential
# (which only biases where collocation points are DRAWN in t), this biases
# how much an already-drawn point COUNTS in the loss. Takes the (N,C)
# per-point residual that loss_ccz4 already computed (r_ccz4_evolution's
# output) -- no extra forward/backward pass, just a different reduction.
# --------------------------------------------------------------------------------
def _causal_weights(t_batch, residual_nc, window_start, window_end, n_buckets, eps):
    """Returns (per_point_weight (N,1), bucket_mean (n_buckets,)) -- the
    latter is also reused by _causal_weighted_ic to gate the tapa term
    (see there): it's the per-bucket mean-squared-residual computed from
    THIS batch (the interior points, which actually spread across all
    n_buckets, unlike tapa's single-time batch)."""
    per_point = jax.lax.stop_gradient(jnp.sum(residual_nc**2, axis=1))
    width  = window_end - window_start
    bucket = jnp.clip(
        ((t_batch[:, 0] - window_start) / width * n_buckets).astype(jnp.int32),
        0, n_buckets - 1)
    bucket_sum   = jnp.zeros((n_buckets,), dtype=per_point.dtype).at[bucket].add(per_point)
    bucket_count = jnp.zeros((n_buckets,), dtype=per_point.dtype).at[bucket].add(1.0)
    bucket_mean  = bucket_sum / jnp.maximum(bucket_count, 1.0)
    cum_prior    = jnp.concatenate([jnp.zeros((1,), dtype=bucket_mean.dtype), jnp.cumsum(bucket_mean)[:-1]])
    bucket_w     = jnp.exp(-eps * cum_prior)
    per_point_w  = jax.lax.stop_gradient(bucket_w[bucket]).reshape(-1, 1)
    return per_point_w, jax.lax.stop_gradient(bucket_mean)


def _causal_weighted_ic(ri, w_int, w_tapa, t, window_start, window_end):
    """Shared causal-weighting step for the Einstein/gauge interior+tapa
    weights -- used by apply_model, compute_bfgs_loss, AND the _gn_ein/
    _gn_gauge per-term gradient-norm diagnostics, so a future change here
    can't silently desync the diagnostics from what's actually optimized.
    This was previously duplicated independently in the first two, and
    OMITTED ENTIRELY from the third: with --causal_weight on (the current
    default), gn_ein/gn_gauge were silently reporting the UNWEIGHTED
    gradient norm, not the one actually being minimized.

    TAPA GATING (was: calling _causal_weights independently on the tapa
    batch itself, which is a no-op -- generate_ic_surface puts every tapa
    point at the SAME time, window_end, so they all land in one bucket
    whose "prior" buckets are empty within that isolated computation,
    giving weight=exp(-eps*0)=1 always regardless of --causal_eps or
    training progress). tapa sits at window_end, i.e. temporally AFTER
    every interior bucket, so instead of gating it by buckets "strictly
    earlier than its own" (undefined/degenerate for a single-time batch),
    it's gated by ALL n_buckets of the INTERIOR's own bucket-mean residual
    -- tapa only reaches full weight once the entire window's interior
    residual is small, not just whichever buckets happen to precede it."""
    if not causal_weight:
        return w_int, w_tapa
    per_point_w, bucket_mean = _causal_weights(t, ri[:, :22], window_start, window_end, causal_buckets, causal_eps)
    w_int_c  = w_int  * per_point_w
    w_tapa_c = w_tapa * jnp.exp(-causal_eps * jnp.sum(bucket_mean))
    return w_int_c, w_tapa_c


# --------------------------------------------------------------------------------
# LOSS FUNCTION (Adam)
# train_data tuple: (t,x,y,z, tf,xf,yf,zf, tb,xb,yb,zb,gb,
#                    ti,xi,yi,zi, ic_ccz4, w_int,w_tapa,w_ic, ws,we)
#
# r_ccz4_evolution returns (N,22): [0:18]=gt(6)+At(6)+chi+Khat+Theta+Ghat(3)
# (gravity sector, weighted by lam_einstein), [18:22]=alpha+beta(3) (gauge
# sector, weighted by lam_gauge -- these are genuine evolution equations,
# see r_ccz4_evolution, not a separate penalty).
# lamgt0 is accepted for CLI compatibility but unused: CCZ4 fields are
# first-order-in-time, so only their VALUE is prescribed at t0, not a
# time-derivative IC.
# --------------------------------------------------------------------------------
@partial(jax.jit, static_argnums=(0,))
def apply_model(apply_fn, params,
                lam_bc, lamg0, w_pos, traj_tt, traj_xx,
                lam_einstein, lam_gauge, r_eps,
                *train_data):
    (t, x, y, z, tf, xf, yf, zf, tb, xb, yb, zb, gb,
     ti, xi, yi, zi, ic_ccz4,
     w_int, w_tapa, w_ic,
     window_start, window_end) = train_data

    def loss_ccz4(p):
        ri = r_ccz4_evolution(apply_fn, p, t,  x,  y,  z,  window_start, window_end)
        rf = r_ccz4_evolution(apply_fn, p, tf, xf, yf, zf, window_start, window_end)
        w_int_c, w_tapa_c = _causal_weighted_ic(ri, w_int, w_tapa, t, window_start, window_end)
        ein_l   = lam_einstein * (_wmean(ri[:, :18]**2, w_int_c)  + lam_tapa * _wmean(rf[:, :18]**2, w_tapa_c))
        gauge_l = lam_gauge    * (_wmean(ri[:, 18:22]**2, w_int_c) + lam_tapa * _wmean(rf[:, 18:22]**2, w_tapa_c))
        return ein_l, gauge_l

    def loss_bc(p):
        if metric_bc_mode == "dirichlet":
            met_res = metric_dirichlet_residual(apply_fn, p, tb, xb, yb, zb, window_start, window_end, gb, lam_bc)
        else:
            met_res = metric_sommerfeld_residual(apply_fn, p, tb, xb, yb, zb, window_start, window_end, c, lam_bc, r_eps)
        return jnp.mean(met_res**2)

    def loss_ics(p):
        def full_of_t(tv):
            return full_output(apply_fn, p, tv, xi, yi, zi, window_start, window_end)
        ug = full_of_t(ti)
        return _ic_fit(ug, ic_ccz4, w_ic)          # (fit_abs, fit_log)

    def loss_pos(p):
        return jnp.mean(positivity_penalty(apply_fn, p, t, x, y, z, window_start, window_end))

    def total_loss(net_params):
        # traj_tt/traj_xx (--chi_bg_mode moving) are CLOSED OVER here, not
        # part of net_params, so jax.value_and_grad below differentiates
        # net_params only -- see _wrap_with_moving_chi_bg's module comment.
        # apply_fn is a FIXED identity for the whole window regardless of
        # how many times traj_tt/traj_xx are refreshed (they're ordinary
        # dynamic array arguments of constant shape, not baked into a
        # rebuilt closure), so refreshing them never forces a recompile --
        # unlike an earlier version of this feature, which rebuilt apply_fn
        # itself every refresh and leaked a new compiled executable into
        # jax's (never-evicted) jit cache every time, causing unbounded
        # memory growth over a long BFGS run.
        p = net_params if traj_tt is None else _bundle_params(net_params, traj_tt, traj_xx)
        g_ic_l, ic_log_l = loss_ics(p)
        ein_l, gauge_l = loss_ccz4(p)
        bc_l           = loss_bc(p)
        pos_l          = loss_pos(p)
        con_l          = (constraint_penalty(apply_fn, p, t, x, y, z, window_start, window_end)
                          if w_constraint > 0 else jnp.zeros(()))
        tot = ein_l + gauge_l + bc_l + lamg0*(g_ic_l + ic_log_l) + w_pos*pos_l + w_constraint*con_l
        return tot, (ein_l, bc_l, gauge_l, g_ic_l, ic_log_l, pos_l, con_l)

    (tot, (ein_l, bc_l, gauge_l, g_ic_l, ic_log_l, pos_l, con_l)), grads = \
        jax.value_and_grad(total_loss, has_aux=True)(params)

    all_loss = {
        "Loss":          tot,
        "loss_Einstein": ein_l,
        "loss_BC":       bc_l,
        "loss_gauge":    gauge_l,
        "loss_IC":       lamg0*g_ic_l,
        "loss_g0":       g_ic_l,
        "loss_ic_log":   lamg0*ic_log_l,
        "loss_gt0":      jnp.array(0.0, dtype=jnp.float64),
        "loss_pos":      pos_l,
        "loss_constraint": w_constraint * con_l,
    }
    return all_loss, grads


# --------------------------------------------------------------------------------
# PER-TERM GRADIENT NORMS (diagnostic only, --log_term_grad_norms)
#
# apply_model's combined ||grads|| conflates all loss terms, so it can't tell
# you WHY the Einstein term is stuck: is its own gradient tiny (it has
# nothing left to push on, e.g. a contaminated frozen hard-IC baseline near
# the puncture -- see the --min_puncture_dist discussion), or is it a
# healthy gradient that's simply outweighed by the other terms' combined
# pull on a shared parameter vector? This computes ||d(term)/d(params)||
# independently for each term, weighted exactly as it enters total_loss.
#
# COMPILE-COST WARNING (learned the hard way): r_ccz4_evolution already
# nests jax.jacfwd(jax.jacfwd(...)) internally for second spatial
# derivatives (see its definition), so apply_model's combined
# jax.value_and_grad is already a 3rd layer of autodiff on top of that --
# proven to compile/run (if slowly on CPU). An earlier version of this
# diagnostic wrapped FIVE independent jax.grad calls of that same 3-layer
# nest inside ONE jax.jit function; bundling that much nested-autodiff work
# into a single XLA compilation unit caused compile time to blow up
# catastrophically (observed: 80+ minutes stuck on the very first call, no
# progress) rather than just being ~5x slower. Each term below is therefore
# its OWN separately-jitted function -- same complexity class as a SUBSET
# of what apply_model already compiles fine, so no single compilation unit
# is any bigger than what's already proven tractable. Still meaningfully
# expensive (each term redoes its own forward pass through r_ccz4_evolution/
# full_output/positivity_penalty), so this is OPT-IN via --log_term_grad_norms
# (default off) and only ever called at the loss_save_every cadence, never
# every step.
# --------------------------------------------------------------------------------
def _bundle_p(p, traj_tt, traj_xx):
    return p if traj_tt is None else _bundle_params(p, traj_tt, traj_xx)


@partial(jax.jit, static_argnums=(0,))
def _gn_ein(apply_fn, params, traj_tt, traj_xx, lam_einstein,
            t, x, y, z, tf, xf, yf, zf, w_int, w_tapa, window_start, window_end):
    def f(p):
        pp = _bundle_p(p, traj_tt, traj_xx)
        ri = r_ccz4_evolution(apply_fn, pp, t,  x,  y,  z,  window_start, window_end)
        rf = r_ccz4_evolution(apply_fn, pp, tf, xf, yf, zf, window_start, window_end)
        w_int_c, w_tapa_c = _causal_weighted_ic(ri, w_int, w_tapa, t, window_start, window_end)
        return lam_einstein * (_wmean(ri[:, :18]**2, w_int_c) + lam_tapa * _wmean(rf[:, :18]**2, w_tapa_c))
    flat, _ = flatten_util.ravel_pytree(jax.grad(f)(params))
    return jnp.linalg.norm(flat)


@partial(jax.jit, static_argnums=(0,))
def _gn_gauge(apply_fn, params, traj_tt, traj_xx, lam_gauge,
              t, x, y, z, tf, xf, yf, zf, w_int, w_tapa, window_start, window_end):
    def f(p):
        pp = _bundle_p(p, traj_tt, traj_xx)
        ri = r_ccz4_evolution(apply_fn, pp, t,  x,  y,  z,  window_start, window_end)
        rf = r_ccz4_evolution(apply_fn, pp, tf, xf, yf, zf, window_start, window_end)
        w_int_c, w_tapa_c = _causal_weighted_ic(ri, w_int, w_tapa, t, window_start, window_end)
        return lam_gauge * (_wmean(ri[:, 18:22]**2, w_int_c) + lam_tapa * _wmean(rf[:, 18:22]**2, w_tapa_c))
    flat, _ = flatten_util.ravel_pytree(jax.grad(f)(params))
    return jnp.linalg.norm(flat)


@partial(jax.jit, static_argnums=(0,))
def _gn_bc(apply_fn, params, traj_tt, traj_xx, lam_bc, r_eps,
           tb, xb, yb, zb, gb, window_start, window_end):
    def f(p):
        pp = _bundle_p(p, traj_tt, traj_xx)
        if metric_bc_mode == "dirichlet":
            met_res = metric_dirichlet_residual(apply_fn, pp, tb, xb, yb, zb, window_start, window_end, gb, lam_bc)
        else:
            met_res = metric_sommerfeld_residual(apply_fn, pp, tb, xb, yb, zb, window_start, window_end, c, lam_bc, r_eps)
        return jnp.mean(met_res**2)
    flat, _ = flatten_util.ravel_pytree(jax.grad(f)(params))
    return jnp.linalg.norm(flat)


@partial(jax.jit, static_argnums=(0,))
def _gn_ic(apply_fn, params, traj_tt, traj_xx, lamg0,
           ti, xi, yi, zi, ic_ccz4, w_ic, window_start, window_end):
    def f(p):
        pp = _bundle_p(p, traj_tt, traj_xx)
        ug = full_output(apply_fn, pp, ti, xi, yi, zi, window_start, window_end)
        fa, fl = _ic_fit(ug, ic_ccz4, w_ic)
        return lamg0 * (fa + fl)
    flat, _ = flatten_util.ravel_pytree(jax.grad(f)(params))
    return jnp.linalg.norm(flat)


@partial(jax.jit, static_argnums=(0,))
def _gn_pos(apply_fn, params, traj_tt, traj_xx, w_pos,
            t, x, y, z, window_start, window_end):
    def f(p):
        pp = _bundle_p(p, traj_tt, traj_xx)
        return w_pos * jnp.mean(positivity_penalty(apply_fn, pp, t, x, y, z, window_start, window_end))
    flat, _ = flatten_util.ravel_pytree(jax.grad(f)(params))
    return jnp.linalg.norm(flat)


def per_term_grad_norms(apply_fn, params,
                         lam_bc, lamg0, w_pos, traj_tt, traj_xx,
                         lam_einstein, lam_gauge, r_eps,
                         *train_data):
    """Plain (non-jitted) dispatcher: no-ops to {} unless --log_term_grad_norms
    is set, so a default run never pays for or even traces this."""
    if not log_term_grad_norms:
        return {}
    (t, x, y, z, tf, xf, yf, zf, tb, xb, yb, zb, gb,
     ti, xi, yi, zi, ic_ccz4,
     w_int, w_tapa, w_ic,
     window_start, window_end) = train_data
    return {
        "gn_ein":   float(_gn_ein(apply_fn, params, traj_tt, traj_xx, lam_einstein,
                                   t, x, y, z, tf, xf, yf, zf, w_int, w_tapa, window_start, window_end)),
        "gn_gauge": float(_gn_gauge(apply_fn, params, traj_tt, traj_xx, lam_gauge,
                                     t, x, y, z, tf, xf, yf, zf, w_int, w_tapa, window_start, window_end)),
        "gn_bc":    float(_gn_bc(apply_fn, params, traj_tt, traj_xx, lam_bc, r_eps,
                                  tb, xb, yb, zb, gb, window_start, window_end)),
        "gn_ic":    float(_gn_ic(apply_fn, params, traj_tt, traj_xx, lamg0,
                                  ti, xi, yi, zi, ic_ccz4, w_ic, window_start, window_end)),
        "gn_pos":   float(_gn_pos(apply_fn, params, traj_tt, traj_xx, w_pos,
                                   t, x, y, z, window_start, window_end)),
    }


# --------------------------------------------------------------------------------
# LOSS FUNCTION (BFGS scalar)
# --------------------------------------------------------------------------------
def constraint_penalty(apply_fn, params, t, x, y, z, window_start, window_end):
    """mean(Theta^2 + |Z^i|^2) over the given points (--w_constraint)."""
    theta = full_output(apply_fn, params, t, x, y, z, window_start, window_end)[:, THETA_IDX]
    zv = gauge_source_vector(apply_fn, params, t, x, y, z, window_start, window_end)
    return jnp.mean(theta ** 2 + jnp.sum(zv ** 2, axis=1))


@partial(jax.jit, static_argnums=(0,))
def compute_bfgs_loss(apply_fn, net_params,
                      lam_bc, lamg0, w_pos, traj_tt, traj_xx,
                      lam_einstein, lam_gauge, r_eps,
                      *train_data):
    # See apply_model's total_loss for why traj_tt/traj_xx are bundled here
    # rather than being part of the BFGS-optimized weight vector: bfgs_step
    # keeps `weights`/`unflatten_func` scoped to net_params only, so BFGS's
    # quasi-Newton Hessian approximation never sees (and so can never move)
    # the trajectory, regardless of gradient values.
    params = net_params if traj_tt is None else _bundle_params(net_params, traj_tt, traj_xx)
    (t, x, y, z, tf, xf, yf, zf, tb, xb, yb, zb, gb,
     ti, xi, yi, zi, ic_ccz4,
     w_int, w_tapa, w_ic,
     window_start, window_end) = train_data

    rei = r_ccz4_evolution(apply_fn, params, t,  x,  y,  z,  window_start, window_end)
    ref = r_ccz4_evolution(apply_fn, params, tf, xf, yf, zf, window_start, window_end)
    w_int_c, w_tapa_c = _causal_weighted_ic(rei, w_int, w_tapa, t, window_start, window_end)
    ein_l   = lam_einstein * (_wmean(rei[:, :18]**2, w_int_c)   + lam_tapa * _wmean(ref[:, :18]**2, w_tapa_c))
    gauge_l = lam_gauge    * (_wmean(rei[:, 18:22]**2, w_int_c) + lam_tapa * _wmean(ref[:, 18:22]**2, w_tapa_c))

    if metric_bc_mode == "dirichlet":
        met_res = metric_dirichlet_residual(apply_fn, params, tb, xb, yb, zb, window_start, window_end, gb, lam_bc)
    else:
        met_res = metric_sommerfeld_residual(apply_fn, params, tb, xb, yb, zb, window_start, window_end, c, lam_bc, r_eps)
    bc_l = jnp.mean(met_res**2)

    def full_of_t(tv):
        return full_output(apply_fn, params, tv, xi, yi, zi, window_start, window_end)
    ug = full_of_t(ti)

    pos_l = jnp.mean(positivity_penalty(apply_fn, params, t, x, y, z, window_start, window_end))

    ic_abs, ic_log = _ic_fit(ug, ic_ccz4, w_ic)
    con_l = (w_constraint * constraint_penalty(apply_fn, params, t, x, y, z, window_start, window_end)
             if w_constraint > 0 else 0.0)
    return (ein_l + gauge_l + bc_l
            + lamg0  * (ic_abs + ic_log)
            + w_pos  * pos_l
            + con_l)


def bfgs_residual_vector(apply_fn, net_params,
                         lam_bc, lamg0, w_pos, traj_tt, traj_xx,
                         lam_einstein, lam_gauge, r_eps,
                         *train_data, _blocks=None, _scales=None):
    """compute_bfgs_loss in least-squares form, for --dsgnar_steps (DSGNAR is a
    Gauss-Newton method and needs the residual VECTOR): rows r with

        sum(r**2) == compute_bfgs_loss(same arguments)

    term by term. Each _wmean(res**2, w) = sum(w res^2) / (sum(w) C) becomes the
    rows sqrt(w / (sum(w) C)) * res; the plain means (boundary, positivity) the
    rows res / sqrt(count). The scale factors are stop-gradiented: the causal
    weights already are (see _causal_weights), so the Jacobian is that of the
    weighted residual with the weights held fixed, exactly as in the QN gradient.
    `_blocks`, if a list, receives (label, n_rows) per condition in row order
    (static shapes; DSGNAR's per-condition reweighting, --dsgnar_reweight). `_scales`, if
    a list, receives each row's (stop-gradiented) scale factor, in row order (used by the
    rows-first sketched Jacobian, --dsgnar_jacobian rows)."""
    params = net_params if traj_tt is None else _bundle_params(net_params, traj_tt, traj_xx)
    (t, x, y, z, tf, xf, yf, zf, tb, xb, yb, zb, gb,
     ti, xi, yi, zi, ic_ccz4,
     w_int, w_tapa, w_ic,
     window_start, window_end) = train_data
    sg = jax.lax.stop_gradient
    rows = []

    def add(label, scale, res):
        r = (sg(scale) * res).reshape(-1)
        rows.append(r)
        if _blocks is not None:
            _blocks.append((label, int(r.shape[0])))
        if _scales is not None:
            _scales.append(jnp.broadcast_to(sg(scale), res.shape).reshape(-1))

    def wrows(weight, w, res):      # rows of weight * _wmean(res**2, w)
        return jnp.sqrt(weight * w / (jnp.sum(w) * res.shape[1] + 1e-30)), res

    rei = r_ccz4_evolution(apply_fn, params, t,  x,  y,  z,  window_start, window_end)
    ref = r_ccz4_evolution(apply_fn, params, tf, xf, yf, zf, window_start, window_end)
    w_int_c, w_tapa_c = _causal_weighted_ic(rei, w_int, w_tapa, t, window_start, window_end)
    add("einstein_int",  *wrows(lam_einstein,            w_int_c,  rei[:, :18]))
    add("einstein_tapa", *wrows(lam_einstein * lam_tapa, w_tapa_c, ref[:, :18]))
    add("gauge_int",     *wrows(lam_gauge,               w_int_c,  rei[:, 18:22]))
    add("gauge_tapa",    *wrows(lam_gauge * lam_tapa,    w_tapa_c, ref[:, 18:22]))

    if metric_bc_mode == "dirichlet":
        met_res = metric_dirichlet_residual(apply_fn, params, tb, xb, yb, zb, window_start, window_end, gb, lam_bc)
    else:
        met_res = metric_sommerfeld_residual(apply_fn, params, tb, xb, yb, zb, window_start, window_end, c, lam_bc, r_eps)
    add("bc", 1.0 / jnp.sqrt(float(met_res.size)), met_res)

    ug = full_output(apply_fn, params, ti, xi, yi, zi, window_start, window_end)
    add("ic", *wrows(lamg0, w_ic, ug[:, CCZ4_SLICE] - ic_ccz4))
    if lam_ic_log > 0.0:
        idx = jnp.array([CHI_IDX, ALPHA_IDX])
        d = (jnp.log(_smooth_floor(ug[:, idx], _IC_LOG_EPS))
             - jnp.log(_smooth_floor(ic_ccz4[:, idx], _IC_LOG_EPS)))
        add("ic_log", *wrows(lamg0 * lam_ic_log, w_ic, d))

    out = full_output(apply_fn, params, t, x, y, z, window_start, window_end)
    pos = jnp.stack([jax.nn.relu(-out[:, CHI_IDX]), jax.nn.relu(-out[:, ALPHA_IDX])], axis=1)
    add("pos", jnp.sqrt(w_pos / float(t.shape[0])), pos)
    return jnp.concatenate(rows)


class _DsgnarObjective:
    """R^n -> R^M view of compute_bfgs_loss for dsgnar.dsgnar_phase (Stationary's
    FlatObjective). The residual is scaled by sqrt(2M) so that
    0.5 * mean(residual**2) == compute_bfgs_loss: DSGNAR's model decrease is in
    those units (Stationary/DSGNAR.md section 2.2). `args` are compute_bfgs_loss's
    arguments after net_params; they are closed over, so a new sample is a new
    objective (the phase rebuilds its jitted maps on resample).

    sketch_rows (--dsgnar_jacobian rows, the paper's construction): C J for the
    CountSketch C given by (signs, buckets), built ROW-wise. Every residual row
    depends on one collocation point (its scale factor is stop-gradiented), so a
    point's rows come from one reverse-mode Jacobian of that point's residuals;
    jacrev is vmapped over macro-batches of points and each batch's rows are
    folded into the s hash buckets at once (CountSketch is linear, so
    C J Omega S = (C J) Omega S: the parameter sketch is applied once, to the
    s x n result). Equal to the column construction to round-off
    (tools/test_dsgnar_bh.py)."""

    def __init__(self, apply_fn, unflatten, args, flat_like, row_batch=64):
        self.n = int(flat_like.size)
        self.dtype = flat_like.dtype
        blocks = []
        jax.eval_shape(lambda f: bfgs_residual_vector(apply_fn, unflatten(f), *args, _blocks=blocks),
                       flat_like)
        self.blocks = blocks
        M = sum(b[1] for b in blocks)
        k = float(np.sqrt(2.0 * M))

        @jax.jit
        def _residual(f):
            return k * bfgs_residual_vector(apply_fn, unflatten(f), *args)

        self._residual = _residual
        self._loss = jax.jit(lambda f: 0.5 * jnp.mean(_residual(f) ** 2))

        # ---------------- rows-first sketched Jacobian ----------------
        (lam_bc_, lamg0_, w_pos_, traj_tt_, traj_xx_, lam_e_, lam_g_, r_eps_) = args[:8]
        (t, x, y, z, tf, xf, yf, zf, tb, xb, yb, zb, gb,
         ti, xi, yi, zi, ic_ccz4,
         w_int, w_tapa, w_ic, ws, we) = args[8:]
        off, o = {}, 0
        for lab, nr in blocks:
            off[lab] = o
            o += nr
        has_log = "ic_log" in off

        def P(f):
            p = unflatten(f)
            return p if traj_tt_ is None else _bundle_params(p, traj_tt_, traj_xx_)

        def one(q):
            return [q[i:i + 1].reshape(1, 1) for i in range(4)]

        def f_int(f, q, _):      # 22 CCZ4 rows + 2 positivity rows of one interior point
            T, X, Y, Z = one(q)
            r = r_ccz4_evolution(apply_fn, P(f), T, X, Y, Z, ws, we)[0]
            o_ = full_output(apply_fn, P(f), T, X, Y, Z, ws, we)[0]
            return jnp.concatenate([r, jnp.stack([jax.nn.relu(-o_[CHI_IDX]), jax.nn.relu(-o_[ALPHA_IDX])])])

        def f_tapa(f, q, _):
            T, X, Y, Z = one(q)
            return r_ccz4_evolution(apply_fn, P(f), T, X, Y, Z, ws, we)[0]

        def f_bc(f, q, g):
            T, X, Y, Z = one(q)
            if metric_bc_mode == "dirichlet":
                return metric_dirichlet_residual(apply_fn, P(f), T, X, Y, Z, ws, we, g.reshape(1, -1), lam_bc_)[0]
            return metric_sommerfeld_residual(apply_fn, P(f), T, X, Y, Z, ws, we, c, lam_bc_, r_eps_)[0]

        def f_ic(f, q, g):
            T, X, Y, Z = one(q)
            ug = full_output(apply_fn, P(f), T, X, Y, Z, ws, we)[0]
            out = [ug[CCZ4_SLICE] - g]
            if has_log:
                idx = jnp.array([CHI_IDX, ALPHA_IDX])
                out.append(jnp.log(_smooth_floor(ug[idx], _IC_LOG_EPS))
                           - jnp.log(_smooth_floor(g[idx], _IC_LOG_EPS)))
            return jnp.concatenate(out)

        def rows_of(n_pts, parts):
            """(n_pts, K) global row indices for a point group: parts = [(block, width), ...]."""
            cols = [off[b] + np.arange(n_pts)[:, None] * wdt + np.arange(wdt)[None, :] for b, wdt in parts]
            return np.concatenate(cols, axis=1)

        hcat = lambda *a: jnp.concatenate(a, axis=1)
        Ni, Nf, Nb_, N0 = int(t.shape[0]), int(tf.shape[0]), int(tb.shape[0]), int(ti.shape[0])
        groups = [
            (f_int,  hcat(t, x, y, z),     jnp.zeros((Ni, 1)),
             rows_of(Ni, [("einstein_int", 18), ("gauge_int", 4), ("pos", 2)])),
            (f_tapa, hcat(tf, xf, yf, zf), jnp.zeros((Nf, 1)),
             rows_of(Nf, [("einstein_tapa", 18), ("gauge_tapa", 4)])),
            (f_bc,   hcat(tb, xb, yb, zb), gb,
             rows_of(Nb_, [("bc", 22)])),
            (f_ic,   hcat(ti, xi, yi, zi), ic_ccz4,
             rows_of(N0, [("ic", 22)] + ([("ic_log", 2)] if has_log else []))),
        ]
        # the ROW order of each point function above must match these index columns
        for fn_, Q, G, idx in groups:
            K = jax.eval_shape(lambda f, q, g: fn_(f, q, g), flat_like, Q[0], G[0]).shape[0]
            assert K == idx.shape[1], (fn_.__name__, K, idx.shape)
        B = int(row_batch)

        def _scales(f):
            sc = []
            bfgs_residual_vector(apply_fn, unflatten(f), *args, _scales=sc)
            return jnp.concatenate(sc)

        def make(s_):
            @jax.jit
            def sketch_rows(f, signs, buckets, w):
                coef_all = k * _scales(f) * w                      # (M,): frozen scale x reweighting
                acc = jnp.zeros((s_, f.shape[0]), dtype=f.dtype)
                for fn_, Q, G, idx in groups:
                    n_pts = Q.shape[0]
                    pad = (-n_pts) % B
                    idx_p = np.concatenate([idx, np.zeros((pad, idx.shape[1]), dtype=idx.dtype)]) if pad else idx
                    valid = np.concatenate([np.ones(n_pts), np.zeros(pad)])
                    Qp = jnp.concatenate([Q, jnp.repeat(Q[:1], pad, 0)]) if pad else Q
                    Gp = jnp.concatenate([G, jnp.repeat(G[:1], pad, 0)]) if pad else G
                    nb = (n_pts + pad) // B
                    data = (Qp.reshape(nb, B, -1), Gp.reshape(nb, B, -1),
                            jnp.asarray(idx_p.reshape(nb, B, -1)), jnp.asarray(valid.reshape(nb, B)))
                    jac = jax.vmap(jax.jacrev(fn_), in_axes=(None, 0, 0))

                    def body(a, d):
                        q, g, ix, v = d
                        J = jac(f, q, g)                                   # (B, K, n)
                        cf = coef_all[ix] * v[:, None]                     # (B, K)
                        Jr = J.reshape(-1, J.shape[-1])
                        for h in range(signs.shape[0]):
                            sgn = (signs[h][ix] * cf).reshape(-1)
                            a = a + jax.ops.segment_sum(Jr * sgn[:, None], buckets[h][ix].reshape(-1),
                                                        num_segments=s_)
                        return a, None

                    acc, _ = jax.lax.scan(body, acc, data)
                return acc
            return sketch_rows

        def make_gram(s_):
            """--dsgnar_residual_sketch none: no CountSketch on the rows. Accumulates, over every
            row, A = (J Omega S) (rows x s, the SRCT applied to each row of J by a DCT) into
            G = A^T A (s x s) and b = A^T r (s): the parameter-sketched Gauss-Newton system with
            the EXACT residual, whose step is never uphill for a small enough radius."""
            import dsgnar as _dm

            @jax.jit
            def gram(f, srct_signs, srct_perm, w):
                coef_all = k * _scales(f) * w
                r_all = _residual(f) * w
                G = jnp.zeros((s_, s_), dtype=f.dtype)
                b = jnp.zeros((s_,), dtype=f.dtype)
                for fn_, Q, G_in, idx in groups:
                    n_pts = Q.shape[0]
                    pad = (-n_pts) % B
                    idx_p = np.concatenate([idx, np.zeros((pad, idx.shape[1]), dtype=idx.dtype)]) if pad else idx
                    valid = np.concatenate([np.ones(n_pts), np.zeros(pad)])
                    Qp = jnp.concatenate([Q, jnp.repeat(Q[:1], pad, 0)]) if pad else Q
                    Gp = jnp.concatenate([G_in, jnp.repeat(G_in[:1], pad, 0)]) if pad else G_in
                    nb = (n_pts + pad) // B
                    data = (Qp.reshape(nb, B, -1), Gp.reshape(nb, B, -1),
                            jnp.asarray(idx_p.reshape(nb, B, -1)), jnp.asarray(valid.reshape(nb, B)))
                    jac = jax.vmap(jax.jacrev(fn_), in_axes=(None, 0, 0))

                    def body(c, d):
                        Gc, bc = c
                        q, g, ix, v = d
                        J = jac(f, q, g)                                   # (B, K, n)
                        cf = (coef_all[ix] * v[:, None]).reshape(-1)       # (B*K,)
                        Jr = J.reshape(-1, J.shape[-1]) * cf[:, None]
                        A = _dm.apply_srct(Jr, (srct_signs, srct_perm))    # (B*K, s)
                        rr = r_all[ix].reshape(-1) * v.repeat(ix.shape[1])
                        return (Gc + A.T @ A, bc + A.T @ rr), None

                    (G, b), _ = jax.lax.scan(body, (G, b), data)
                return G, b
            return gram

        self._sketch_rows_cache = {}
        self._make_sketch_rows = make
        self._gram_cache = {}
        self._make_gram = make_gram

    def sketch_rows(self, flat, signs, buckets, s, w):
        """C J (the (s, n) bucketed row sketch); apply the SRCT to get C J Omega S."""
        if s not in self._sketch_rows_cache:
            self._sketch_rows_cache[s] = self._make_sketch_rows(s)
        return self._sketch_rows_cache[s](jnp.asarray(flat), signs, buckets, w)

    def gram_rows(self, flat, srct, s, w):
        """(G, b) = ((J Omega S)^T (J Omega S), (J Omega S)^T r) over every row (no row sketch)."""
        if s not in self._gram_cache:
            self._gram_cache[s] = self._make_gram(s)
        return self._gram_cache[s](jnp.asarray(flat), srct[0], srct[1], w)

    def residual(self, flat):
        return self._residual(jnp.asarray(flat))

    def loss(self, flat):
        return float(self._loss(jnp.asarray(flat)))


@partial(jax.jit, static_argnums=(1, 2))
def bfgs_loss_and_grad(weights, apply_fn, unflatten_func, *args):
    net_params = unflatten_func(weights)
    return compute_bfgs_loss(apply_fn, net_params, *args)


@partial(jax.jit, static_argnums=(1, 2))
def bfgs_val_and_grad(weights, apply_fn, unflatten_func, *args):
    return jax.value_and_grad(
        lambda w: compute_bfgs_loss(apply_fn, unflatten_func(w), *args)
    )(weights)


# --------------------------------------------------------------------------------
# BFGS STEP
# --------------------------------------------------------------------------------
def make_positive_definite(A, epsilon=1e-6):
    A_sym = np.asarray((A + A.T) / 2)
    n     = A_sym.shape[0]
    shift = 0.0
    for _ in range(54):
        try:
            np.linalg.cholesky(A_sym + shift * np.eye(n))
            return jnp.asarray(A_sym + shift * np.eye(n))
        except np.linalg.LinAlgError:
            shift = epsilon if shift == 0.0 else shift * 10.0
    return jnp.eye(n, dtype=jnp.float64)


_QN_LAST_STATS = {}


def bfgs_step(initial_weights, H0, key, train_data,
              lam_bc, lamg0, w_pos, traj_tt, traj_xx, lam_einstein, lam_gauge, r_eps,
              apply_fn, unflatten_func, static_options, loss_scale=1.0):
    """loss_scale (--qn_loss_norm): the optimizer minimizes loss/loss_scale; the
    returned loss is the true one."""
    key, _ = jax.random.split(key)
    all_args = (apply_fn, unflatten_func,
                lam_bc, lamg0, w_pos, traj_tt, traj_xx, lam_einstein, lam_gauge, r_eps,
                *train_data)
    fun_val      = lambda w: bfgs_loss_and_grad(w, *all_args)
    fun_val_grad = lambda w: bfgs_val_and_grad(w, *all_args)
    if loss_scale != 1.0:
        _inv = 1.0 / float(loss_scale)
        _fv, _fvg = fun_val, fun_val_grad
        fun_val      = lambda w: _fv(w) * _inv
        fun_val_grad = lambda w: tuple(v * _inv for v in _fvg(w))
    if profile_restarts:  # [GPUS]
        fun_val, fun_val_grad = _prof_wrap(fun_val, "f_val"), _prof_wrap(fun_val_grad, "f_valgrad")  # [GPUS]
    opts = dict(static_options)
    if fast_hessian and n_gpus > 1:  # [GPUS] keep H and the weights on every GPU: no implicit copies
        H0 = jax.device_put(H0, _SH_REPL)  # [GPUS]
        initial_weights = jax.device_put(initial_weights, _SH_REPL)  # [GPUS]
    opts["initial_H"] = H0

    result = minimize(
        fun_val=fun_val,
        fun_val_grad=fun_val_grad,
        x0=initial_weights,
        options=opts,
    )
    # Per-call optimizer diagnostics for the window summary (absent in an
    # older Crunch).
    _QN_LAST_STATS.clear()
    _QN_LAST_STATS.update(getattr(result, "stats", None) or {})

    _tpost = time.perf_counter()  # [GPUS]
    new_H0 = (result.hess_inv + result.hess_inv.T) / 2
    # jnp.linalg.cholesky returns NaN on failure rather than raising (unlike
    # np.linalg.cholesky/scipy), so the previous `except LinAlgError` branch
    # here could never fire -- every actual non-positive-definite Hessian
    # silently fell back to a bare jnp.eye reset, and make_positive_definite
    # (which tries small increasing diagonal shifts to preserve as much of
    # the original curvature as possible, instead of discarding it entirely)
    # was unreachable dead code. bfgs_step runs eagerly (not under jax.jit),
    # so a concrete Python `if` on the already-computed NaN check is fine
    # here -- jax.lax.cond couldn't have called make_positive_definite
    # anyway, since it uses host-side np.linalg.cholesky in a Python retry
    # loop, not JAX-traceable.
    Lc = jnp.linalg.cholesky(new_H0)
    if bool(jnp.any(jnp.isnan(Lc))):
        final_H0 = make_positive_definite(new_H0)
    else:
        final_H0 = new_H0

    _prof_add("hess_post", _tpost, final_H0)  # [GPUS]
    x_out = result.x  # [GPUS]
    if fast_hessian and n_gpus > 1:  # [GPUS]
        # Hand the weights back uncommitted (as without --fast_hessian), so the  # [GPUS]
        # diagnostics/IC targets that use them keep their single-device compiled  # [GPUS]
        # versions; replicated weights made XLA recompile all of them for the  # [GPUS]
        # 4-GPU mesh at the first window end (13 min on MN5, job 47307354).  # [GPUS]
        x_out = jnp.asarray(np.asarray(result.x))  # [GPUS]
    return x_out, final_H0, key, result.fun * loss_scale, result.nit


# --------------------------------------------------------------------------------
# MODEL INITIALIZATION
# --------------------------------------------------------------------------------
feat_sizes = tuple([HIDDEN] * N_LAYERS + [FEATURES])
print(feat_sizes)

model    = PINN(features=feat_sizes)
_TAYLOR_KIND[model.apply] = "plain"
apply_fn = jax.jit(model.apply)
_TAYLOR_KIND[apply_fn] = "plain"

dummy_params = model.init(subkey,
    jnp.ones((Nint, 1), dtype=jnp.float64),
    jnp.ones((Nint, 1), dtype=jnp.float64),
    jnp.ones((Nint, 1), dtype=jnp.float64),
    jnp.ones((Nint, 1), dtype=jnp.float64),
    jnp.array(0.0,    dtype=jnp.float64),
    jnp.array(tfinal, dtype=jnp.float64))
dummy_params = jax.tree_util.tree_map(lambda x: x.astype(jnp.float64), dummy_params)
print(f"Total parameters: {sum(x.size for x in jax.tree_util.tree_leaves(dummy_params))}")

# Built ONCE for the whole run (lr0/decay_step/decay_rate are
# fixed CLI args, identical for every window) instead of inside train_window,
# where a fresh optax.adam(...) object and a freshly-@jax.jit-decorated
# update_model were created every window -- a NEW Python function object
# every time, which jax.jit treats as an entirely new function to compile
# regardless of any static-argument caching (this is a different flavor of
# the same "identity churn defeats JIT reuse" issue _wrap_with_moving_chi_bg
# had, see there, except here it's the jitted function ITSELF being
# recreated, not one of its static arguments). optimizer (a stateless bundle
# of pure functions) is genuinely identical every window; only opt_state
# (the mutable Adam moment buffers, actual DATA not a static/closure
# argument) is window-specific, and that still gets freshly initialized
# inside train_window below exactly as before.
_scheduler = optax.exponential_decay(lr0, decay_step, decay_rate, staircase=False)
optimizer  = optax.adam(_scheduler)


@jax.jit
def update_model(p, opt_s, grads):
    updates, opt_s = optimizer.update(grads, opt_s)
    return optax.apply_updates(p, updates), opt_s


# --------------------------------------------------------------------------------
# WINDOW HELPERS
# --------------------------------------------------------------------------------
def find_window_index(t_val):
    t_scalar = float(t_val)
    if t_scalar <= window_edges[0]:
        return 0
    if t_scalar >= window_edges[-1]:
        return N_window - 1
    return int(np.clip(np.searchsorted(window_edges, t_scalar, side="right") - 1,
                       0, N_window - 1))


# --------------------------------------------------------------------------------
# ADAPTIVE BH-CENTER FINDER (lapse/chi minima in the x-y plane)
# --------------------------------------------------------------------------------
def _find_local_minima_2d(Z):
    """(i_idx, j_idx) interior grid indices where Z[i,j] is <= all 8
    neighbors AND strictly < at least one of them (the strict part rules
    out flat-plateau false positives) -- needed once punctures can move
    off the x-axis (--BL_momenta_y). Pure numpy (no scipy), matching this
    file's existing preference (see _match_to_reference's docstring)."""
    if Z.shape[0] < 3 or Z.shape[1] < 3:
        i0, j0 = np.unravel_index(int(np.argmin(Z)), Z.shape)
        return np.array([i0]), np.array([j0])
    le     = np.ones(Z.shape, dtype=bool)
    lt_any = np.zeros(Z.shape, dtype=bool)
    core = Z[1:-1, 1:-1]
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            nb = Z[1 + di:Z.shape[0] - 1 + di, 1 + dj:Z.shape[1] - 1 + dj]
            le[1:-1, 1:-1]     &= (core <= nb)
            lt_any[1:-1, 1:-1] |= (core < nb)
    minima = np.zeros(Z.shape, dtype=bool)
    minima[1:-1, 1:-1] = le[1:-1, 1:-1] & lt_any[1:-1, 1:-1]
    return np.nonzero(minima)


def _pick_bh_centers_2d(x_lin, y_lin, Z, n_stars, min_center_separation):
    """Shared candidate-selection algorithm for both find_bh_centers_lapse
    (on alpha) and locate_chi_minima (on chi): find ALL local minima of
    Z(x,y) across the whole scanned grid (see _find_local_minima_2d), rank
    by depth, and greedily keep up to n_stars of them subject to
    min_center_separation (a genuine Euclidean distance, via math.hypot)
    so two adjacent grid points on the same physical dip aren't double-
    counted. If fewer than n_stars well-separated minima remain (e.g.
    post-merger, a single common horizon), the deepest one is repeated for
    every remaining slot instead of forcing a spurious distinct split.
    Returns a list of (x,y) tuples, sorted by x -- NOT a fixed "A=negative,
    B=positive" identity, which is fine since these are only used to steer
    sampling focus / cross-check the trajectory diagnostic, not to persist
    per-star identity (see _refresh_display_mask_zones for the one place
    identity DOES matter, handled separately there)."""
    if n_stars != 2:
        i0, j0 = np.unravel_index(int(np.argmin(Z)), Z.shape)
        return [(float(x_lin[i0]), float(y_lin[j0]))]
    i_idx, j_idx = _find_local_minima_2d(Z)
    if i_idx.size == 0:
        i_idx, j_idx = np.unravel_index(int(np.argmin(Z)), Z.shape)
        i_idx, j_idx = np.array([i_idx]), np.array([j_idx])
    vals  = Z[i_idx, j_idx]
    order = np.argsort(vals)   # deepest minimum first
    picked = []
    for k in order:
        xv, yv = float(x_lin[i_idx[k]]), float(y_lin[j_idx[k]])
        if all(math.hypot(xv - px, yv - py) >= min_center_separation for px, py in picked):
            picked.append((xv, yv))
        if len(picked) == n_stars:
            break
    if len(picked) < n_stars:
        fallback = picked[0] if picked else (float(x_lin[i_idx[order[0]]]), float(y_lin[j_idx[order[0]]]))
        picked += [fallback] * (n_stars - len(picked))
    picked.sort(key=lambda xy: xy[0])
    return picked


def _match_to_reference(candidates, reference):
    """Reorder `candidates` (e.g. locate_chi_minima's output, which is
    always LEFT-TO-RIGHT SORTED with no persistent per-star identity -- see
    _pick_bh_centers_2d) to align with `reference`'s per-star identity (e.g.
    the shift-driven tracker's traj_xx, which is never resorted -- each
    puncture is its own independent RK4 integration). Without this, a
    diagnostic plot/table comparing the two under the same index/color
    silently swaps which curve is "puncture 0" vs "puncture 1" the moment
    the two punctures' x-order crosses (possible with unequal --BL_masses),
    corrupting the cross-check into showing spurious divergence or false
    agreement. n_stars is restricted to {1,2} by argparse, so brute-force
    over the (at most 2) permutations is exact and avoids a scipy
    dependency for what linear_sum_assignment would otherwise solve. The
    swapped-order branch below only ever returns 2 elements -- asserted,
    not generalized, since a real n>2 assignment needs an actual matching
    algorithm (e.g. linear_sum_assignment), not more hardcoded branches;
    this just fails loudly instead of silently dropping candidates[2:] if
    --n_stars's argparse restriction is ever loosened without updating
    this function too. Thin wrapper around _match_permutation (scalar
    distance) -- see there for the (x,y)-tuple-aware version, used e.g. by
    the shift_driven/lapse_min chi-min cross-check, which needs to match on
    the full 2D position, not just x."""
    perm = _match_permutation(candidates, reference)
    return [candidates[i] for i in perm]


def _match_permutation(candidates, reference, dist=None):
    """Indices into `candidates` (e.g. [0,1] or [1,0] for n_stars=2) that
    best match `reference`'s per-star identity -- the same brute-force
    2-permutation search _match_to_reference wraps for the common
    'just reorder this one list' case, factored out so OTHER associated
    per-star data indexed by the SAME per-star slot can be reordered
    consistently too, not just the list being matched itself.

    dist(a,b) -> distance: defaults to abs(a-b) for plain scalars; pass
    e.g. `lambda a, b: math.hypot(a[0]-b[0], a[1]-b[1])` for (x,y) tuples
    (see the chi-min cross-check, which has real (x,y) positions available
    post --BL_momenta_y but previously only compared x, which can pick the
    wrong identity when two punctures' x-positions are close/crossed but
    y-positions are not)."""
    if dist is None:
        dist = lambda a, b: abs(a - b)
    if len(candidates) <= 1:
        return list(range(len(candidates)))
    assert len(candidates) == 2 and len(reference) == 2, (
        f"_match_permutation only handles exactly 2 candidates (got "
        f"{len(candidates)}); n_stars>2 needs a real assignment algorithm here.")
    identity_cost = dist(candidates[0], reference[0]) + dist(candidates[1], reference[1])
    swapped_cost  = dist(candidates[1], reference[0]) + dist(candidates[0], reference[1])
    return [0, 1] if identity_cost <= swapped_cost else [1, 0]


def _xy_dist(a, b):
    """Euclidean distance between two (x,y) tuples -- the dist= to pass to
    _match_permutation wherever real (x,y) positions are available (post
    --BL_momenta_y) instead of just x, so identity-matching can't be
    fooled by two punctures whose x-positions are close/crossed but whose
    y-positions clearly aren't."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def find_bh_centers_lapse(prev_state, t_val, n_scan=2000, min_center_separation=min_center_separation,
                           search_centers=None, search_halfwidth=None):
    """Locate BH centers as minima of the lapse alpha = 1/sqrt(-g^tt) in
    the x-y plane (z=0), evaluated from the previous window's network at
    t=t_val -- lets sampling follow a moving/recoiling/ORBITING puncture
    (--bh_center_mode lapse_min). No excision in this file, so no scan
    points are excluded: the puncture's lapse minimum is found directly,
    wherever it has moved to, including off the x-axis (e.g. under
    --BL_momenta_y).

    search_centers/search_halfwidth (both None by default): if given
    (search_centers a list of n_stars (x,y) hints -- e.g. each puncture's
    own last-known position, search_halfwidth its half-width, e.g. from
    _search_halfwidth's |beta|*dt*safety estimate), scan a SMALL box
    around EACH hint independently instead of the whole domain -- much
    cheaper (the same point budget covers a far smaller area, so per-axis
    resolution is correspondingly finer) and avoids the two-punctures-vs-
    one-scan candidate-selection question entirely, since each box is
    already centered on a specific, known puncture. Falls back to the
    original whole-plane scan below if ANY box's result lands within half
    a cell of that box's edge (the estimate was too small -- the puncture
    moved further than expected, or the hint is stale) -- this keeps the
    box-search an opt-in, self-checking OPTIMIZATION on top of the
    original assumption-free behavior, not a new failure mode: whenever it
    can't confidently confirm the true minimum is inside the box, it
    just falls through to the exhaustive search instead.

    Generalized from an x-axis-only line scan to a genuine 2D (x-y plane)
    grid scan: n_scan is now a TOTAL point-count budget, split evenly
    between the two axes (see _n_axis_from_budget) so the computational
    cost stays comparable to the original 1D scan rather than literally
    squaring it -- this does mean coarser per-axis resolution than the old
    1D scan's (e.g. 45 points/axis at the n_scan=2000 default, vs. 2000
    points along a single line) for the whole-plane fallback path, a
    necessary trade-off for covering the whole plane instead of one fixed
    line (the box-search path above doesn't have this trade-off: the same
    point count over a much smaller area is finer, not coarser).

    For n_stars=2 (whole-plane path only): rather than blindly forcing
    exactly one minimum out of a FIXED x<0/x>0 split (which silently
    mislocates or duplicates centers once the two lapse minima cross x=0,
    or merge into a single basin near merger -- a real failure mode for
    this file's own default head-on collision setup, where BL_momenta
    point the punctures at each other), this finds ALL local minima of
    alpha across the WHOLE scanned plane, ranks them by depth, and
    greedily keeps up to n_stars of them subject to min_center_separation
    (a genuine Euclidean distance now) so two adjacent grid points on the
    same physical dip aren't double-counted. If fewer than n_stars
    well-separated minima remain (e.g. post-merger, a single common
    horizon), the deepest one is reported for every remaining slot instead
    of forcing a spurious split -- correctly reflecting that there is
    genuinely only one puncture left to sample around. Centers are
    returned sorted by x (no more fixed "A=negative, B=positive"
    identity), which is fine since these are only used to steer sampling
    focus, not to persist per-star identity elsewhere. For n_stars=1 the
    global minimum is used.
    """
    prev_p = prev_state["params"]
    _ps = jnp.array(prev_state["t_start"], dtype=jnp.float64)
    _pe = jnp.array(prev_state["t_end"],   dtype=jnp.float64)
    if prev_state.get("ic_mode", "hard") == "hard":
        prev_fn = make_hard_ic_apply_fn(
            model.apply, float(_ps), float(_pe),
            ic_history=prev_state.get("ic_history"))
    else:
        prev_fn = apply_fn

    if not _state_is_finite(prev_state, prev_fn, prev_p, _ps, _pe, t_val):
        _warn_network_diverged_once()
        return _diverged_fallback_centers(search_centers, include_z=True)

    def _alpha_scan(xlo, xhi, ylo, yhi, nx, ny):
        x_lin, y_lin, x_in, y_in, z_in, t_in = _scan_grid_2d(xlo, xhi, ylo, yhi, 0.0, t_val, nx, ny)
        out     = np.asarray(full_output(prev_fn, prev_p, t_in, x_in, y_in, z_in, _ps, _pe))
        chi_s   = jnp.asarray(out[:, CHI_IDX:CHI_IDX + 1])
        gt_s    = unpack_sym3(jnp.asarray(out[:, GT_SLICE]))
        alpha_s = jnp.asarray(out[:, ALPHA_IDX:ALPHA_IDX + 1])
        beta_s  = jnp.asarray(out[:, BETA_SLICE])
        _, g_contra_s, _ = adm_to_g4(chi_s, gt_s, alpha_s, beta_s)
        g_tt = np.asarray(g_contra_s[:, 0, 0])
        alpha = (1.0 / np.sqrt(np.maximum(-g_tt, metric_det_eps))).reshape(nx, ny)
        spacing = max((xhi - xlo) / max(nx - 1, 1), (yhi - ylo) / max(ny - 1, 1))
        return x_lin, y_lin, alpha, spacing

    n_axis = _n_axis_from_budget(n_scan)

    if search_centers is not None:
        results, ok = [], True
        for cx, cy in search_centers:
            xlo, xhi = max(x_min, cx - search_halfwidth), min(x_max, cx + search_halfwidth)
            ylo, yhi = max(y_min, cy - search_halfwidth), min(y_max, cy + search_halfwidth)
            x_lin, y_lin, alpha_box, spacing = _alpha_scan(xlo, xhi, ylo, yhi, n_axis, n_axis)
            i0, j0 = np.unravel_index(int(np.argmin(alpha_box)), alpha_box.shape)
            xv, yv = float(x_lin[i0]), float(y_lin[j0])
            at_edge = (
                (abs(xv - xlo) < 0.5 * spacing and xlo > x_min) or
                (abs(xv - xhi) < 0.5 * spacing and xhi < x_max) or
                (abs(yv - ylo) < 0.5 * spacing and ylo > y_min) or
                (abs(yv - yhi) < 0.5 * spacing and yhi < y_max)
            )
            if at_edge:
                ok = False
                break
            results.append((xv, yv))
        # Reject two independently-refined boxes that converged onto (near)
        # the SAME point -- near merger, or if the two hints/boxes overlap,
        # one box's argmin can drift onto the OTHER puncture's (possibly
        # deeper) minimum entirely INSIDE its own box, never triggering the
        # edge-hit check above. Falling back to the whole-plane scan here
        # restores min_center_separation's usual guarantee for that case.
        if ok and n_stars == 2 and math.hypot(results[0][0] - results[1][0],
                                               results[0][1] - results[1][1]) < min_center_separation:
            ok = False
        if ok:
            results.sort(key=lambda xy: xy[0])
            return [(x, y, 0.0) for x, y in results]
        # else: fall through to the whole-plane scan below -- the box
        # estimate was too small for at least one puncture this time, or
        # the two boxes collapsed onto the same minimum.

    x_lin, y_lin, alpha, _ = _alpha_scan(x_min, x_max, y_min, y_max, n_axis, n_axis)
    picked = _pick_bh_centers_2d(x_lin, y_lin, alpha, n_stars, min_center_separation)
    return [(x, y, 0.0) for x, y in picked]


# --------------------------------------------------------------------------------
# SHIFT-DRIVEN PUNCTURE TRACKER
#
# Ported from bbh_headon_bssn_pinn_shift_driven_puncture.py's
# beta_z_near_puncture / build_shift_driven_Z / locate_chi_minimum (single
# symmetric puncture pair on the z-axis, integrated via a learned Z(t) baked
# into the solution ansatz), generalized here to independent per-puncture
# tracking in the x-y PLANE (z=0): each puncture's (x,y) coordinates are
# propagated on their own by RK4-integrating d(x_A,y_A)/dt =
# -(beta^x,beta^y)(x_A, y_A, z=eps, t) through a TRAINED window's network --
# no left/right symmetry assumed, so unequal --BL_masses/--BL_momenta are
# tracked correctly, and it generalizes to --n_stars 1. The puncture is
# never a trainable variable, only a post-hoc diagnostic/sampling-focus
# quantity derived from the learned shift, exactly as in the source file.
# The y-component only ever moves away from 0 if --BL_momenta_y gives the
# puncture a tangential kick (see --BL_momenta_y's help) -- the original
# x-axis-only head-on-collision behavior is the y=0-for-all-time special
# case of this same tracker, not a separate code path.
#
# Unlike bbh_headon (which needs a rough PRE-training estimate to bake a
# moving analytic background into its hard-IC ansatz), this file's hard-IC
# wrapper just carries the previous window's learned field forward
# continuously, so there is no such ansatz to feed -- the tracker only
# needs to run ONCE PER WINDOW, AFTER that window's own network has been
# trained (well-conditioned: evaluated on the exact time range it was
# trained on, unlike an extrapolated previous-window network). The advanced
# position then serves as next window's sampling-focus center (in place of
# find_bh_centers_lapse) and is logged as the trajectory diagnostic.
# --------------------------------------------------------------------------------
_shift_track_xy = [[c[0], 0.0] for c in _bh_center_list(BL_masses)]   # [[x_A,y_A], ...], y=0 at t=0
_INITIAL_CENTERS = [(xy[0], xy[1], 0.0) for xy in _shift_track_xy]     # --sampling_centers initial


def _wrap_moving_chi_bg_if_needed(fn, p, traj_tt, traj_xx):
    """Shared 'if --chi_bg_mode moving and a trajectory is available: wrap
    fn with _wrap_with_moving_chi_bg and bundle params via _bundle_params'
    step -- previously copy-pasted identically in _network_apply_for_state
    and _get_eval_fn. Both callers' own docstrings already warn this exact
    step is easy to get wrong (see locate_chi_minima's prior real bug,
    where omitting it silently returned the raw regular-remainder chi
    instead of physical chi), so a future fix/change to this condition now
    only needs updating this one place. Returns (fn, p) unchanged if
    traj_tt is None (chi_bg_mode != 'moving', or no trajectory available
    yet for this state)."""
    if chi_bg_mode == "moving" and traj_tt is not None:
        fn = _wrap_with_moving_chi_bg(fn)
        p  = _bundle_params(p, traj_tt, traj_xx)
    return fn, p


def _network_apply_for_state(state, traj_tt=None, traj_xx=None):
    """(apply_fn_like, params, t_start, t_end) for a saved window state,
    honoring ic_mode='hard' the same way find_bh_centers_lapse does.

    Under --chi_bg_mode moving, state["params"] is the raw net params and
    the raw apply_fn's CHI_IDX output is only the REGULAR REMAINDER, not
    physical chi (see _wrap_with_moving_chi_bg) -- callers that need
    physical chi (e.g. locate_chi_minima) MUST pass this window's own
    (traj_tt, traj_xx) so it can be wrapped back into the physical field
    here. Omitting them under chi_bg_mode=='moving' silently returns the
    raw remainder instead of physical chi -- this was exactly the bug
    behind diagnostics_puncture_track.png's chi-min cross-check appearing
    to diverge while the shift-tracked position (which only ever reads
    beta, unaffected by chi_bg_mode) correctly showed convergence, and
    while fields*.png/native*.png (rendered from INSIDE train_window, which
    has direct closure access to the properly-wrapped train_apply_fn) were
    fine."""
    p  = state["params"]
    ws = jnp.array(state["t_start"], dtype=jnp.float64)
    we = jnp.array(state["t_end"],   dtype=jnp.float64)
    if state.get("ic_mode", "hard") == "hard":
        fn = make_hard_ic_apply_fn(model.apply, float(ws), float(we),
                                    ic_history=state.get("ic_history"))
    else:
        fn = apply_fn
    fn, p = _wrap_moving_chi_bg_if_needed(fn, p, traj_tt, traj_xx)
    return fn, p, ws, we


@partial(jax.jit, static_argnums=(0,))
def _full_output_near_puncture_jit(fn, p, ws, we, t_val, x_p, y_p, eps):
    """Full 22-component full_output at (x_p[k], y_p[k], eps, t_val) for
    every puncture k, ALL sharing the same t_val (always true here: every
    caller evaluates every puncture at the same t), batched into ONE call.
    x_p/y_p track each puncture's CURRENT (possibly off-axis, orbiting --
    see --BL_momenta_y) position; only z is offset by eps to avoid
    sampling exactly on the puncture itself (the old version fixed y=z=eps,
    a near-axis offset that only made sense when the puncture was assumed
    to never leave y=0 -- now that x_p/y_p already track the true position,
    a single z-offset serves the same purpose). Shared query-grid
    construction for _beta_xy_near_puncture (extracts beta^x/beta^y, used
    by the RK4 puncture-tracking loop) AND advance_shift_tracked_centers's
    end-of-window beta_x/beta_y/alpha/chi diagnostic -- the latter
    previously hand-rebuilt this exact same (fixed near-axis offset, single
    shared t, batched over punctures) grid independently."""
    n    = x_p.shape[0]
    t_in = jnp.full((n, 1), t_val, dtype=jnp.float64)
    x_in = x_p.reshape(-1, 1)
    y_in = y_p.reshape(-1, 1)
    z_in = jnp.full((n, 1), eps, dtype=jnp.float64)
    return full_output(fn, p, t_in, x_in, y_in, z_in, ws, we)


def _beta_xy_near_puncture(fn, p, ws, we, t_val, xy_p, eps=_PUNCTURE_TRACK_EPS):
    """(beta^x, beta^y) at (xy_p[k,0], xy_p[k,1], eps, t_val) for every
    puncture k -- the x-y-plane generalization of the old fixed-x-axis
    _beta_x_near_puncture (which assumed y=z=eps always): xy_p now carries
    each puncture's CURRENT tracked (x,y) position, nonzero y once a
    --BL_momenta_y-driven orbit curves it off the x-axis. Both components
    are direct network outputs (BETA_SLICE.start/.start+1), no metric
    reconstruction needed (see _bh_adm_from_out). xy_p: (n,2) array, one
    (x,y) row per puncture; returns an (n,2) jnp array, [:,0]=beta^x,
    [:,1]=beta^y.

    JIT-compiled (fn static, matching the apply_fn-as-static-arg pattern
    used throughout this file) AND batched across all punctures in ONE call
    (rather than one call per puncture) because _integrate_puncture_
    trajectory's RK4 loop calls this thousands of times per window
    (track_steps * 4 RK4 stages -- 4000 calls at the defaults, independent
    of n_stars now that every stage batches all punctures together) with
    the SAME fn object and the SAME shape every time. Left unjitted, EVERY
    call eagerly retraces+dispatches full_output's whole graph from scratch
    -- for window 0's hard-IC mode this includes compute_ccz4_ic's ~150+
    chained JAX ops (2 matrix inversions, a vmap, 3 jax.jvp calls), and
    paying that tracing overhead per call (not just once) turned a single
    window's puncture-tracking step into a multi-minute stall (confirmed
    via direct timing: the identical computation wrapped in jax.jit and
    called repeatedly runs in milliseconds after the first call; unjitted,
    each call independently cost ~100+ ms of pure Python/tracing overhead
    before any array math). Only fn is static -- t_val/xy_p vary every RK4
    stage but keep the same shape/dtype, so they stay traced (non-static)
    inputs and every call after the first reuses one compiled executable.

    Returns a jnp array (not numpy): _integrate_puncture_trajectory's RK4
    loop keeps its running position as a jnp array across all 4 stages
    (see there) so this never needs to leave the device between stages --
    same IEEE754 values either way, just without the host<->device round
    trip on every one of its thousands of calls per window."""
    xy_p = jnp.asarray(xy_p, dtype=jnp.float64)
    out = _full_output_near_puncture_jit(fn, p, ws, we, t_val, xy_p[:, 0], xy_p[:, 1], eps)
    return out[:, BETA_SLICE.start:BETA_SLICE.start + 2]


@partial(jax.jit, static_argnums=(0,))
def _rk4_integrate_jit(fn, p, ws, we, xy0, t0s, dts):
    """The entire RK4 sub-stepping loop as ONE compiled program (jax.lax.scan
    over the n_steps axis of t0s/dts), instead of _integrate_puncture_
    trajectory's previous plain Python for-loop, which dispatched 4 separate
    jitted _beta_x_near_puncture calls PER SUBSTEP from the host -- track_
    steps=1000 (the default, used once per window by advance_shift_tracked_
    centers for ALL --bh_center_mode shift_driven runs) meant 4000 individual
    dispatches; --chi_bg_mode moving's _refresh_traj calls this again at
    EVERY Adam resample and BFGS outer restart (moving_bg_refresh_steps*4
    more dispatches each time), so that overhead was paid over and over
    through a single window's training, not just once. Confirmed via direct
    A/B timing (identical run, only --chi_bg_mode differed): 'moving' cost
    18-25% more BFGS-phase wall time than 'physical' at these defaults, and
    the gap widened window-to-window as --ic_mode hard's growing ic_history
    made every one of those individually-dispatched forward passes more
    expensive.

    jax.lax.scan traces the loop body ONCE and lowers it to a single XLA
    While/Scan op, so the whole trajectory becomes one dispatch regardless
    of n_steps -- same RK4 arithmetic, same operation order, so results are
    bit-identical to the old Python-loop version (verified against it
    directly; not merely close to floating-point noise).

    xy0: (n_stars,2) initial (x,y) positions -- generalized from the
    original (n_stars,) x-only state to also carry y once a --BL_momenta_y
    orbit curves the trajectory off the x-axis; the RK4 arithmetic itself
    is unchanged, just applied to a 2-vector state per star instead of a
    scalar. t0s/dts: (n_steps,) each, the substep start-times and widths
    (dts computed via np.diff on the SAME np.linspace grid the old loop
    used, so per-substep dt matches exactly, including any last-cell
    floating-point rounding). Returns xx_rest, shape (n_steps, n_stars, 2):
    the position AFTER each substep (not including xy0 itself -- the
    caller prepends that)."""
    def _rk4_step(xy, t0_dt):
        t0, dt = t0_dt
        k1 = -_beta_xy_near_puncture(fn, p, ws, we, t0,          xy)
        k2 = -_beta_xy_near_puncture(fn, p, ws, we, t0 + 0.5*dt, xy + 0.5*dt*k1)
        k3 = -_beta_xy_near_puncture(fn, p, ws, we, t0 + 0.5*dt, xy + 0.5*dt*k2)
        k4 = -_beta_xy_near_puncture(fn, p, ws, we, t0 + dt,     xy + dt*k3)
        xy_new = xy + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return xy_new, xy_new
    _, xx_rest = jax.lax.scan(_rk4_step, xy0, (t0s, dts))
    return xx_rest


def _integrate_puncture_trajectory(fn, p, ws, we, t_left, t_right, xy0_list, n_steps):
    """RK4-integrate d(x,y)/dt = -(beta^x,beta^y)(x,y,t) from t_left to
    t_right in n_steps substeps, independently for EACH puncture in
    xy0_list (each puncture's beta depends only on its OWN position, so
    per-puncture integration is exact regardless of whether the network
    evaluations are batched together), through the given network. Every
    RK4 stage batches ALL punctures into one _beta_xy_near_puncture call
    instead of dispatching one call per puncture (n_stars separate calls
    each doing identical Python/tracing work), halving (at n_stars=2) or
    better the call count for the same numerical result. The full
    n_steps-substep loop itself is one more level of that same sharing,
    fused into a single dispatch by _rk4_integrate_jit's jax.lax.scan (see
    there) rather than n_steps separate Python-loop dispatches. Returns
    (traj_tt, traj_xx): traj_tt shape (n_steps+1,), traj_xx shape
    (n_stars, 2, n_steps+1) -- axis 1 is (x,y) -- the FULL trajectory grid
    (not just the final position), so a continuous, JAX-differentiable
    (jnp.interp-based) background can be built from it and used DURING
    training (see --chi_bg_mode moving), not just as a post-hoc
    diagnostic."""
    n_steps = max(4, int(n_steps))
    tt  = np.linspace(t_left, t_right, n_steps + 1)
    t0s = jnp.asarray(tt[:-1], dtype=jnp.float64)
    dts = jnp.asarray(np.diff(tt), dtype=jnp.float64)
    xy0 = jnp.asarray(xy0_list, dtype=jnp.float64)                     # (n_stars, 2)
    xx_rest = _rk4_integrate_jit(fn, p, ws, we, xy0, t0s, dts)         # (n_steps, n_stars, 2)
    xx = jnp.concatenate([xy0[None, :, :], xx_rest], axis=0)           # (n_steps+1, n_stars, 2)
    xx = jnp.transpose(xx, (1, 2, 0))                                  # (n_stars, 2, n_steps+1)
    return jnp.asarray(tt), xx


def find_bh_centers_shift_driven():
    """This window's sampling-focus centers: the persistent shift-driven
    track as it stands at the start of the window (last advanced at the end
    of the previous window's training by advance_shift_tracked_centers, or
    still at the analytic initial positions before window 0)."""
    return [(x, y, 0.0) for x, y in _shift_track_xy]


def advance_shift_tracked_centers(trained_state, window_start, window_end, n_steps=None):
    """Advance the persistent _shift_track_xy cache across
    [window_start, window_end] using trained_state's (this window's own,
    just-trained) network, one independent 2D (x-y plane) RK4 integration
    per puncture. Returns (centers, per_puncture_diagnostics, (traj_tt,
    traj_xx)): per_puncture_diagnostics is a list of dicts {beta_x, beta_y,
    alpha, chi} evaluated at each puncture's final position and window_end
    (mirrors bbh_headon's build_shift_driven_Z return of beta_z(p),
    alpha_p, chi_p, generalized to a 2-component beta); (traj_tt, traj_xx)
    is the FULL RK4 sub-grid across the window (not just the endpoint), as
    numpy arrays (traj_xx shape (n_stars,2,n_steps+1), axis 1 = (x,y)), so
    callers can see the trajectory WITHIN the window, not only its value at
    each window boundary."""
    global _shift_track_xy
    if n_steps is None:
        n_steps = track_steps
    fn, p, ws, we = _network_apply_for_state(trained_state)

    if not _state_is_finite(trained_state, fn, p, ws, we, window_start):
        _warn_network_diverged_once()
        # Freeze the tracker at its last known-good position instead of
        # RK4-integrating a NaN beta through _integrate_puncture_trajectory
        # and silently setting _shift_track_xy to NaN forever -- unlike
        # find_bh_centers_lapse/locate_chi_minima (which already have this
        # guard), this function had none, even though --bh_center_mode
        # shift_driven is the default and _shift_track_xy feeds every
        # downstream consumer (sampling focus, the moving chi background,
        # the puncture-track diagnostic) for the rest of the run once it
        # goes NaN. Reports honest NaN diagnostics (not a fabricated
        # beta/alpha/chi) and a flat (no-motion) trajectory of the same
        # shape a real integration would have returned, so callers don't
        # need special-casing for a short/malformed array.
        frozen_xy = [tuple(xy) for xy in _shift_track_xy]
        diags = [{"beta_x": float("nan"), "beta_y": float("nan"),
                  "alpha": float("nan"), "chi": float("nan")} for _ in frozen_xy]
        n_steps_eff = max(4, int(n_steps))
        traj_tt = np.linspace(window_start, window_end, n_steps_eff + 1)
        traj_xx = np.stack([np.stack([np.full(n_steps_eff + 1, x), np.full(n_steps_eff + 1, y)])
                             for x, y in frozen_xy])
        return [(x, y, 0.0) for x, y in frozen_xy], diags, (traj_tt, traj_xx)

    traj_tt, traj_xx = _integrate_puncture_trajectory(
        fn, p, ws, we, window_start, window_end, _shift_track_xy, n_steps)
    new_xy = [(float(traj_xx[k, 0, -1]), float(traj_xx[k, 1, -1])) for k in range(len(_shift_track_xy))]
    # beta/alpha are direct network outputs, unaffected by chi_bg_mode, but
    # CHI_IDX under --chi_bg_mode moving is only the regular remainder --
    # re-fetch a wrapped (fn, p) using the trajectory just computed so the
    # "chi" diagnostic below reports physical chi, not the raw remainder.
    fn_chi, p_chi, _, _ = _network_apply_for_state(trained_state, traj_tt, traj_xx)
    # One batched call across all punctures instead of a separate dispatch
    # per puncture (n_stars is small, but this is the same avoidable-
    # per-item-forward-pass pattern flagged elsewhere in this file, e.g.
    # _integrate_puncture_trajectory) -- shares _full_output_near_puncture_
    # jit's grid construction with _beta_xy_near_puncture instead of
    # independently rebuilding the same (fixed near-axis eps offset,
    # single shared t, batched-over-punctures) query grid by hand.
    n_pts = len(new_xy)
    x_p = jnp.asarray([xy[0] for xy in new_xy], dtype=jnp.float64)
    y_p = jnp.asarray([xy[1] for xy in new_xy], dtype=jnp.float64)
    out = _full_output_near_puncture_jit(fn_chi, p_chi, ws, we, window_end, x_p, y_p, _PUNCTURE_TRACK_EPS)
    diags = [
        {
            "beta_x": float(out[i, BETA_SLICE.start]),
            "beta_y": float(out[i, BETA_SLICE.start + 1]),
            "alpha":  float(out[i, ALPHA_IDX]),
            "chi":    float(out[i, CHI_IDX]),
        }
        for i in range(n_pts)
    ]
    _shift_track_xy = [list(xy) for xy in new_xy]
    return [(x, y, 0.0) for x, y in new_xy], diags, (np.asarray(traj_tt), np.asarray(traj_xx))


# Per-puncture last-known (x,y,beta_x,beta_y,t) from the most recent
# find_bh_centers_lapse/locate_chi_minima call as a --bh_center_mode
# chi_min/lapse_min SAMPLING-focus driver (set by _locate_centers_and_diags)
# -- used to seed the next call's local box search (see
# find_bh_centers_lapse/locate_chi_minima's search_centers/search_halfwidth
# params and _search_halfwidth). None per-star until the first successful
# locate; falls back to the original whole-plane scan until then. NOT
# persisted across --resume_checkpoint (a deliberate simplification: the
# first post-resume window just pays for one whole-plane scan, then this
# fills back in normally).
_scan_track_state = [None] * n_stars


def _locate_centers_and_diags(mode, trained_state, t_val):
    """(centers, per_puncture_diagnostics) for --bh_center_mode chi_min/
    lapse_min, mirroring advance_shift_tracked_centers's return shape
    (centers as (x,y,0) tuples; diags as a list of {beta_x, beta_y, alpha,
    chi} dicts at each center) so both modes can feed the same
    puncture-track history/plot/txt pipeline shift_driven already uses.

    Unlike shift_driven, there is no persistent trajectory to advance --
    chi_min/lapse_min re-locate the centers from scratch every window (a
    fresh chi/lapse scan over trained_state, evaluated at t_val -- now a
    genuine 2D x-y plane scan, see locate_chi_minima/find_bh_centers_lapse),
    which is exactly what train_window already does at window_start to pick
    the NEXT window's sampling focus; this just repeats that same scan at
    window_end against THIS window's own freshly-trained state, for the
    diagnostic record, same as shift_driven logs its post-training
    position rather than only its pre-training one.

    chi_bg_mode=='moving' is unreachable here: argparse enforces
    --chi_bg_mode moving requires --bh_center_mode shift_driven (see the
    ValueError near chi_bg_mode's parsing), so trained_state's CHI_IDX is
    always physical chi already and _network_apply_for_state never needs
    a traj_tt/traj_xx wrap for these two modes.

    Seeds the scan with _scan_track_state's last-known (x,y,beta) per
    puncture (see there and _search_halfwidth) when available -- a local
    box around each puncture's own last position instead of a whole-plane
    scan -- and updates _scan_track_state with this call's own result
    afterward, so the NEXT call (either this same diagnostic call next
    window, or train_window's sampling-focus call for the window in
    between) gets to use it too."""
    global _scan_track_state
    if all(h is not None for h in _scan_track_state):
        _search_centers = [(h["x"], h["y"]) for h in _scan_track_state]
        _dt = t_val - _scan_track_state[0]["t"]
        _halfwidth = _search_halfwidth([(h["beta_x"], h["beta_y"]) for h in _scan_track_state], _dt)
    else:
        _search_centers, _halfwidth = None, None

    if mode == "chi_min":
        xys = locate_chi_minima(trained_state, t_val,
                                 search_centers=_search_centers, search_halfwidth=_halfwidth)   # [(x,y), ...]
    elif mode == "lapse_min":
        xys = [(c[0], c[1]) for c in find_bh_centers_lapse(
            trained_state, t_val, search_centers=_search_centers, search_halfwidth=_halfwidth)]
    else:
        raise ValueError(f"_locate_centers_and_diags: unsupported mode {mode!r}")

    fn, p, ws, we = _network_apply_for_state(trained_state)
    # xys is already safe (locate_chi_minima/find_bh_centers_lapse above
    # each guard their own scan internally), but this diags query is a
    # SEPARATE network evaluation of that same trained_state -- guard it
    # too rather than letting a diverged network silently produce NaN
    # diags with no warning. _state_is_finite is memoized on trained_state
    # itself, so this reuses the result those calls already computed
    # rather than re-probing the network.
    if not _state_is_finite(trained_state, fn, p, ws, we, t_val):
        _warn_network_diverged_once()
        diags = [{"beta_x": float("nan"), "beta_y": float("nan"),
                  "alpha": float("nan"), "chi": float("nan")} for _ in xys]
    else:
        x_p = jnp.asarray([xy[0] for xy in xys], dtype=jnp.float64)
        y_p = jnp.asarray([xy[1] for xy in xys], dtype=jnp.float64)
        out = _full_output_near_puncture_jit(fn, p, ws, we, t_val, x_p, y_p, _PUNCTURE_TRACK_EPS)
        diags = [
            {
                "beta_x": float(out[i, BETA_SLICE.start]),
                "beta_y": float(out[i, BETA_SLICE.start + 1]),
                "alpha":  float(out[i, ALPHA_IDX]),
                "chi":    float(out[i, CHI_IDX]),
            }
            for i in range(len(xys))
        ]
    _scan_track_state = [
        {"x": xy[0], "y": xy[1], "beta_x": d["beta_x"], "beta_y": d["beta_y"], "t": t_val}
        for xy, d in zip(xys, diags)
    ]
    return [(x, y, 0.0) for x, y in xys], diags


_network_diverged_warned = False


def _warn_network_diverged_once():
    """Print the network-diverged warning exactly once per run (not once
    per call -- find_bh_centers_lapse/locate_chi_minima can be called many
    times per window, and once training has actually diverged it stays
    diverged for the rest of the run, so repeating the same warning every
    call would just spam stdout for no benefit)."""
    global _network_diverged_warned
    if not _network_diverged_warned:
        print("WARNING: network output is non-finite (NaN/Inf) -- training has "
              "diverged (check the loss history for a blowup; a common cause is "
              "an unguarded division in r_ccz4_evolution -- see chi_safe/"
              "chi_safe there). Skipping the chi/lapse grid scan for this and "
              "every subsequent call this run (falls back to the last known "
              "hint/analytic position instead of a meaningless argmin-of-NaN "
              "result) to avoid repeatedly paying for an expensive scan that "
              "cannot produce a real answer. This message prints only once.")
        _network_diverged_warned = True


def _diverged_fallback_centers(search_centers, include_z):
    """Shared fallback center list for find_bh_centers_lapse/
    locate_chi_minima when _state_is_finite reports a diverged network (no
    scan can produce a meaningful answer -- see _warn_network_diverged_once):
    reuse the caller's own search hints if given (already exactly what's
    needed), else the static Bowen-York initial x-centers with y=0.
    include_z picks the two callers' differing return shape: True appends a
    z=0.0 (find_bh_centers_lapse's 3-tuple), False stays a plain (x, y)
    2-tuple (locate_chi_minima's)."""
    if search_centers is not None:
        if include_z:
            return [(cx, cy, 0.0) for cx, cy in search_centers]
        return list(search_centers)
    if include_z:
        return [(c[0], 0.0, 0.0) for c in _bh_center_list(BL_masses)]
    return [(c[0], 0.0) for c in _bh_center_list(BL_masses)]


def _network_state_is_finite(fn, p, ws, we, t_val):
    """Cheap ONE-POINT probe of whether `fn`'s output is even finite,
    before paying for a full box-search or whole-plane grid scan (tens to
    thousands of points): once training has diverged, EVERY point of a
    chi/alpha scan is NaN too, and np.argmin on an all-NaN array
    deterministically returns index 0 -- the grid's very first corner
    (comparisons against NaN are always False, so the first element is
    never beaten) -- which used to be silently reported as if it were a
    real puncture location (e.g. chi_min_x pinned at exactly -R_domain)
    while STILL paying full price for the scan that produced it, every
    single window for the rest of the run (see find_bh_centers_lapse/
    locate_chi_minima's search_centers fallback logic: a NaN hint makes
    the box degenerate to the whole domain, its edge-check never fires
    since a NaN-vs-boundary comparison is also always False, so it
    'succeeds' at an all-NaN whole-domain scan instead of ever reporting
    failure). A single evaluated point turns that wasted O(n_scan) (or
    O(n_axis^2)) cost into O(1) whenever the state is already known-bad;
    once params are NaN, NaN propagates through the entire network
    output deterministically, so one point is exactly as conclusive as
    scanning the whole domain would be."""
    out = _full_output_near_puncture_jit(
        fn, p, ws, we, t_val,
        jnp.zeros(1, dtype=jnp.float64), jnp.zeros(1, dtype=jnp.float64),
        _PUNCTURE_TRACK_EPS)
    return bool(np.all(np.isfinite(np.asarray(out))))


def _state_is_finite(state, fn, p, ws, we, t_val):
    """Cached, per-state wrapper around _network_state_is_finite (see
    there) -- find_bh_centers_lapse/locate_chi_minima/
    advance_shift_tracked_centers can all be called MULTIPLE times per
    window on the SAME state (e.g. locate_chi_minima's fine-grid cross-
    check calls it up to 5 times), and the SAME state object is also often
    `prev_state` for the NEXT window's find_bh_centers_lapse/
    locate_chi_minima call -- probing the network only ONCE per state
    (memoized on the state dict itself, the same mutable-dict pattern
    already used elsewhere in this file, e.g. window_state["traj_tt"])
    instead of every caller independently reprobing it avoids a redundant
    network evaluation on every one of those repeat calls, AND is the
    single place all three callers' divergence handling now goes through,
    instead of three independently copy-pasted _network_state_is_finite(
    ...) call sites that a future fourth caller could easily forget to add.

    Uses whichever (fn,p,ws,we) the FIRST caller to check this particular
    state happened to pass (e.g. locate_chi_minima's --chi_bg_mode moving
    wrapped view vs. advance_shift_tracked_centers' raw view) -- these are
    equivalent for THIS purpose: if the underlying network params are
    genuinely NaN, both views are NaN too (NaN propagates through the
    moving-background wrapper's additive correction just the same); this
    is a proxy for 'has training diverged', not a claim that every
    possible view of the state is independently verified finite."""
    if "_is_finite" not in state:
        state["_is_finite"] = _network_state_is_finite(fn, p, ws, we, t_val)
    return state["_is_finite"]


def _n_axis_from_budget(n_budget):
    """Per-axis point count for a 2D (x-y plane) grid whose total point
    budget matches a previous 1D x-axis-only scan's n_scan/n_fine point
    count (e.g. 2000 -> 45x45=2025) -- keeps the 2D generalization's
    computational cost in the same ballpark as the original 1D scan
    instead of literally squaring it (see find_bh_centers_lapse/
    locate_chi_minima's own docstrings for the resulting per-axis-
    resolution trade-off)."""
    return max(int(round(math.sqrt(n_budget))), 5)


def _search_halfwidth(betas_xy, dt, min_halfwidth=0.2, safety=4.0):
    """Starting half-width for find_bh_centers_lapse/locate_chi_minima's
    local box search: max over punctures of |beta|*dt*safety, floored at
    min_halfwidth -- a physically-motivated estimate of how far a
    puncture could plausibly have moved since its last known position
    (the same d(x,y)/dt=-beta ODE the shift-driven tracker integrates),
    with a safety margin since it's only a STARTING guess, not a bound:
    the box-search's own widen-on-edge-and-retry escalation (see there)
    corrects an underestimate automatically. betas_xy: iterable of
    (beta_x,beta_y) pairs, one per puncture, at each one's own last-known
    position (already computed as a side effect of locating it -- no
    extra network evaluation needed here). Guards against a non-finite
    result (e.g. a diagnostic evaluated at a still-spurious position) and
    against a pathologically large one (clamped to R_domain, beyond which
    the box may as well be the whole domain)."""
    speeds = [math.hypot(bx, by) for bx, by in betas_xy]
    hw = max(speeds) * max(dt, 0.0) * safety if speeds else 0.0
    if not math.isfinite(hw):
        hw = min_halfwidth
    return min(max(hw, min_halfwidth), R_domain)


def _scan_grid_2d(x_lo, x_hi, y_lo, y_hi, z_val, t_val, nx, ny):
    """(x_lin, y_lin (numpy 1D axes)), x_in, y_in, z_in, t_in (jax arrays,
    flattened nx*ny grid) for an nx-by-ny 2D (x-y plane) scan at fixed
    (z_val, t_val) -- the 2D (x-y plane) generalization of this file's
    original x-axis-only scan, needed once punctures can move off the
    x-axis (--BL_momenta_y). Shared query-grid boilerplate for
    find_bh_centers_lapse and _chi_scan_2d (they extract DIFFERENT
    physical quantities from the resulting full_output call -- alpha via a
    full ADM reconstruction vs. chi read directly -- so only the grid
    construction is shared, not the field extraction)."""
    x_lin = np.linspace(x_lo, x_hi, nx)
    y_lin = np.linspace(y_lo, y_hi, ny)
    XX, YY = np.meshgrid(x_lin, y_lin, indexing="ij")
    n = nx * ny
    x_in = jnp.asarray(XX.ravel()).reshape(-1, 1)
    y_in = jnp.asarray(YY.ravel()).reshape(-1, 1)
    z_in = jnp.full((n, 1), z_val, dtype=jnp.float64)
    t_in = jnp.full((n, 1), t_val, dtype=jnp.float64)
    return x_lin, y_lin, x_in, y_in, z_in, t_in


def _chi_scan_2d(fn, p, ws, we, t_val, x_lo, x_hi, y_lo, y_hi, nx, ny):
    """chi(x,y,z=_PUNCTURE_TRACK_EPS,t_val) over an nx-by-ny grid in
    [x_lo,x_hi]x[y_lo,y_hi]. Returns (x_lin, y_lin, chi_grid (nx,ny),
    grid_spacing) -- spacing is the LARGER of the two axis spacings (sizes
    the next zoom stage, same role as the original 1D scan's single
    spacing value)."""
    x_lin, y_lin, x_in, y_in, z_in, t_in = _scan_grid_2d(
        x_lo, x_hi, y_lo, y_hi, _PUNCTURE_TRACK_EPS, t_val, nx, ny)
    out = np.asarray(full_output(fn, p, t_in, x_in, y_in, z_in, ws, we))
    chi = out[:, CHI_IDX].reshape(nx, ny)
    spacing = max((x_hi - x_lo) / max(nx - 1, 1), (y_hi - y_lo) / max(ny - 1, 1))
    return x_lin, y_lin, chi, spacing


def _chi_argmin_on_grid_2d(fn, p, ws, we, t_val, x_lo, x_hi, y_lo, y_hi, nx, ny):
    """One coarse 2D scan, returning just (argmin_x, argmin_y, grid_
    spacing) -- used by locate_chi_minima's per-candidate refinement zoom,
    once a candidate (x,y) has already been picked."""
    x_lin, y_lin, chi, spacing = _chi_scan_2d(fn, p, ws, we, t_val, x_lo, x_hi, y_lo, y_hi, nx, ny)
    i0, j0 = np.unravel_index(int(np.argmin(chi)), chi.shape)
    return x_lin[i0], y_lin[j0], spacing


def locate_chi_minima(trained_state, t_val, n_scan=2000, n_fine=2000, n_refine=2,
                       traj_tt=None, traj_xx=None, min_center_separation=min_center_separation,
                       search_centers=None, search_halfwidth=None):
    """Independent field diagnostic, analogous to bbh_headon's
    locate_chi_minimum: location(s) of the minimum of chi in the x-y plane
    (one per basin for --n_stars 2, global for --n_stars 1). When called as
    shift_driven's cross-check, only its x-component is compared against
    the tracked position (see the main loop) -- the search itself still
    covers the full plane, so it correctly finds an off-axis puncture's
    basin even though the cross-check currently only reports x. When
    called for --bh_center_mode chi_min, this IS the sampling-focus
    driver, using the full (x,y) result. Uses the SAME off-axis offset
    (_PUNCTURE_TRACK_EPS) as _beta_xy_near_puncture's RK4 integration (an
    earlier mismatched offset, 0.025 here vs 0.03 there, guaranteed a
    persistent apples-to-oranges gap against the tracked trajectory
    independent of training quality).

    Generalized from an x-axis-only line scan to a genuine 2D (x-y plane)
    grid scan + candidate selection + zoom refinement, the same way
    find_bh_centers_lapse is (see there for the n_scan/n_fine "total point
    budget, split across both axes" convention and the resulting
    per-axis-resolution trade-off). Returns a list of (x,y) tuples (NOT
    bare x floats anymore).

    Under --chi_bg_mode moving, traj_tt/traj_xx (this window's own final
    trajectory, from advance_shift_tracked_centers) MUST be passed so the
    physical chi field can be reconstructed (see _network_apply_for_state) --
    without them this silently scans the raw regular-remainder field
    instead of physical chi, whose minimum has no reason to track the
    puncture at all (this was the reported bug: chi-min appeared to
    diverge here even though the shift-tracked position and the
    properly-wrapped in-training field snapshots both correctly showed the
    punctures approaching).

    MULTI-STAGE REFINEMENT: a single coarse n_scan-budget grid over the
    whole [-R_domain,R_domain]^2 plane is far coarser than the puncture
    displacements (~1e-3 to 1e-2 per window) this is used to cross-check/
    steer sampling with. Each successive stage after the first re-scans an
    n_fine-budget grid only a few PREVIOUS-stage cells wide, centered on
    the previous stage's result -- n_refine stages of a zoom each reach
    much finer resolution than the initial coarse grid without ever paying
    for a fine scan across the whole plane.

    CANDIDATE SELECTION: shares _pick_bh_centers_2d with find_bh_centers_
    lapse -- find ALL local minima of chi across the WHOLE scanned plane,
    rank by depth, greedily keep up to n_stars subject to
    min_center_separation (a genuine Euclidean distance now), falling back
    to the single deepest minimum for every remaining slot if fewer than
    n_stars distinct minima are found (post-merger). Each selected
    candidate is then refined independently via the existing zoom stages.

    search_centers/search_halfwidth (both None by default): if given
    (search_centers a list of n_stars (x,y) hints -- e.g. each puncture's
    own last-known position, search_halfwidth its half-width, e.g. from
    _search_halfwidth's |beta|*dt*safety estimate), SKIP the expensive
    whole-plane coarse scan + candidate selection above entirely and seed
    the zoom-refinement directly at each hint with search_halfwidth as the
    starting box size -- much cheaper (no coarse scan at all) and reuses
    100% of the existing widen-on-edge-and-retry refinement below
    unchanged, which is exactly what makes this safe: if a hint is stale
    (the puncture moved further than the estimate), the SAME escalation
    that already handles a shallow/broad chi basin widens the box (up to
    4^6x) until it finds a genuine non-edge-pinned minimum, so this never
    silently reports a wrong position -- it just costs more (approaching
    the whole-plane scan's cost) on the rare occasions the estimate is
    bad, instead of paying that cost every single call."""
    fn, p, ws, we = _network_apply_for_state(trained_state, traj_tt, traj_xx)

    if not _state_is_finite(trained_state, fn, p, ws, we, t_val):
        _warn_network_diverged_once()
        return _diverged_fallback_centers(search_centers, include_z=False)

    n_fine_axis = _n_axis_from_budget(n_fine)

    def _refine(candidates):
        locs = []
        for x0, y0, spacing in candidates:
            for _ in range(n_refine):
                zoom_width = 2.0
                center_x, center_y = x0, y0
                for _widen_attempt in range(6):
                    half_width = zoom_width * spacing
                    xlo, xhi = max(x_min, center_x - half_width), min(x_max, center_x + half_width)
                    ylo, yhi = max(y_min, center_y - half_width), min(y_max, center_y + half_width)
                    new_x0, new_y0, new_spacing = _chi_argmin_on_grid_2d(
                        fn, p, ws, we, t_val, xlo, xhi, ylo, yhi, n_fine_axis, n_fine_axis)
                    # If the argmin landed on (or within half a fine cell of)
                    # any of the scan window's 4 edges, the true minimum may
                    # lie OUTSIDE this window -- the window was centered on the
                    # previous, coarser stage's result, which for a shallow/
                    # broad/noisy chi basin can be more than zoom_width*spacing
                    # away from the true minimum. Widen the window (re-centered
                    # on the SAME starting point each retry, not on the still-
                    # pinned result -- that would just drift along the edge
                    # instead of actually widening the search) and re-scan
                    # THIS stage, instead of silently accepting a result pinned
                    # at an arbitrary edge.
                    at_edge = (
                        (abs(new_x0 - xlo) < 0.5 * new_spacing and xlo > x_min) or
                        (abs(new_x0 - xhi) < 0.5 * new_spacing and xhi < x_max) or
                        (abs(new_y0 - ylo) < 0.5 * new_spacing and ylo > y_min) or
                        (abs(new_y0 - yhi) < 0.5 * new_spacing and yhi < y_max)
                    )
                    if not at_edge:
                        x0, y0, spacing = new_x0, new_y0, new_spacing
                        break
                    zoom_width *= 4.0
                else:
                    x0, y0, spacing = new_x0, new_y0, new_spacing
                    print(f"  WARNING: locate_chi_minima's refinement zoom around (x={x0:.6f}, "
                          f"y={y0:.6f}) (t={t_val:.4f}) kept landing on its search-window edge "
                          f"even after widening -- reported position may be unreliable (e.g. a "
                          f"very shallow or broad chi basin).")
            locs.append((float(x0), float(y0)))
        return locs

    if search_centers is not None:
        locs = _refine([(cx, cy, search_halfwidth) for cx, cy in search_centers])
        # Reject two independently-refined boxes that converged onto (near)
        # the SAME point -- near merger, or if the two hints/boxes overlap,
        # one box's refinement can drift onto the OTHER puncture's (possibly
        # deeper) minimum entirely INSIDE its own box, never triggering the
        # per-candidate edge-hit check above. Falling back to the whole-
        # plane scan here restores min_center_separation's usual guarantee
        # for that case.
        if n_stars != 2 or math.hypot(locs[0][0] - locs[1][0], locs[0][1] - locs[1][1]) >= min_center_separation:
            return locs
        # else: fall through to the whole-plane scan below.

    n_scan_axis = _n_axis_from_budget(n_scan)
    x_lin, y_lin, chi, spacing0 = _chi_scan_2d(
        fn, p, ws, we, t_val, x_min, x_max, y_min, y_max, n_scan_axis, n_scan_axis)
    picked = _pick_bh_centers_2d(x_lin, y_lin, chi, n_stars, min_center_separation)
    return _refine([(x0, y0, spacing0) for x0, y0 in picked])

# --------------------------------------------------------------------------------
# MOVING CHI BACKGROUND  (--chi_bg_mode moving)
#
# Ported from bbh_headon_bssn_pinn_shift_driven_puncture.py's moving_bl_chi /
# solution(): under the DEFAULT --chi_bg_mode physical, the network's CHI_IDX
# output IS the physical conformal factor chi, exactly as in EEccz4_BH.py.
# Under --chi_bg_mode moving, CHI_IDX instead represents the SMOOTH REGULAR
# DEVIATION from the analytic Brill-Lindquist background evaluated at each
# puncture's live shift-driven-tracked position; physical chi (used
# everywhere -- residual, BC, diagnostics) is background + regular,
# reconstructed by _wrap_with_moving_chi_bg around the underlying apply_fn.
# Generalized from bbh_headon's single symmetric z-axis pair to independent
# per-puncture tracking on the x-axis (unequal --BL_masses/--BL_momenta,
# --n_stars 1 or 2), and -- unlike bbh_headon, which only has a hard-IC
# ansatz to bake this into -- works under EITHER --ic_mode, since it's
# implemented as a wrapper around whatever apply_fn (hard-wrapped or plain)
# is already in use, rather than as part of the hard-IC construction itself.
#
# The trajectory used for the LIVE background is refreshed periodically
# DURING training (every resampling point in the Adam loop, and between
# every BFGS outer restart -- see train_window), each time re-integrated
# from the window's start using the CURRENT params, rather than estimated
# once before training and left fixed for the whole window: this lets the
# trajectory keep improving as the network itself converges, instead of
# training against a single stale extrapolation.
# --------------------------------------------------------------------------------
def _translated_psi(x, y, z, positions):
    """Batched (N,1)-array Brill-Lindquist conformal factor psi, with the
    n_stars punctures (masses BL_masses) placed at the given (x,y)
    positions -- positions: [(cx_A,cy_A), ...] -- instead of the fixed
    (x_star_A/B, y=0) -- the moving analogue of _bl_psi_jax, generalized to
    n_stars=1|2, unequal masses (bbh_headon's moving_bl_chi only ever
    handled a symmetric equal-mass pair), and a genuine (x,y) position per
    puncture (not x-only), so it follows the puncture's full trajectory,
    including any orbit a --BL_momenta_y tangential kick produces."""
    psi = jnp.ones_like(x)
    for (cx, cy), mA in zip(positions, BL_masses):
        r = jnp.maximum(jnp.sqrt((x - cx)**2 + (y - cy)**2 + z**2), 1.0e-10)
        psi = psi + mA / (2.0 * r)
    return psi


def _chi_bg_from_traj(t, x, y, z, traj_tt, traj_xx):
    """chi = psi^-4 background at the puncture positions interpolated (via
    jnp.interp, JAX-differentiable) from a per-puncture trajectory grid:
    traj_tt shape (n_grid,), traj_xx shape (n_stars, 2, n_grid) (axis 1 is
    (x,y) -- see _integrate_puncture_trajectory). t,x,y,z: (N,1) batched
    query points, as used throughout this file. Returns (N,1).

    Interpolates BOTH (x,y) components now, so the background follows the
    puncture's actual tracked position, not just its x-coordinate -- exact
    reduction to the original x-axis-only behavior whenever --BL_momenta_y
    is all-zero (traj_xx[:,1] stays identically 0 in that case, same as
    the original head-on-collision behavior)."""
    t_flat = t[:, 0]
    positions = [
        (jnp.interp(t_flat, traj_tt, traj_xx[k, 0]).reshape(-1, 1),
         jnp.interp(t_flat, traj_tt, traj_xx[k, 1]).reshape(-1, 1))
        for k in range(traj_xx.shape[0])
    ]
    return _translated_psi(x, y, z, positions) ** -4


# Keyed on the raw_apply_fn object ITSELF (function objects are hashable
# by identity by default), not id(raw_apply_fn) as an int: using the
# object as the key makes _IdentityLRUCache's internal dict hold a live
# reference to it for as long as the entry survives, so id() can never be
# silently recycled by another object while this cache still thinks it
# owns that key -- the usual hazard with id()-keyed caches. Same shared
# cache class/bound as _hard_ic_apply_fn_cache (see _PERWINDOW_CACHE_MAXSIZE)
# so this doesn't itself become an unbounded-growth leak.
_moving_chi_bg_wrap_cache = _IdentityLRUCache(_PERWINDOW_CACHE_MAXSIZE)


def _wrap_with_moving_chi_bg(raw_apply_fn):
    """Wrap an apply_fn (operating on plain net params) so its params
    argument becomes a bundle {"net":.., "traj_tt":.., "traj_xx":..} and
    its CHI_IDX output channel is the PHYSICAL chi = moving background
    (from the trajectory) + raw_apply_fn's own CHI_IDX output (the regular
    deviation). traj_tt/traj_xx are stop-gradiented so they can never
    become optimized/trainable quantities: Adam's diagonal update rule
    alone would already leave an exactly-zero-gradient parameter
    untouched, but stop_gradient also protects the BFGS path, whose
    quasi-Newton Hessian approximation could in principle still couple a
    zero-gradient parameter to others through off-diagonal terms.

    Memoized by raw_apply_fn's own identity (see _moving_chi_bg_wrap_cache):
    physical_apply_fn reads traj_tt/traj_xx from its `params` argument at
    CALL time, not from anything closed over at WRAP time, so two calls
    wrapping the SAME raw_apply_fn always produce behaviorally identical
    closures regardless of which (traj_tt, traj_xx) values later flow
    through them -- returning the SAME cached object is exact, not an
    approximation. This matters because raw_apply_fn is passed as a STATIC
    jax.jit argument in several functions downstream (e.g.
    _full_output_near_puncture_jit): rebuilding a fresh, unequal-by-
    identity closure on every call (the previous behavior) made every one
    of those calls under --chi_bg_mode moving a guaranteed JIT cache MISS,
    forcing a full recompile of the whole network graph on every single
    diagnostic call (_network_apply_for_state calls this on every
    advance_shift_tracked_centers/_locate_centers_and_diags/locate_chi_
    minima invocation -- several times per window) instead of reusing one
    compiled executable across all of them, same as every other mode
    already gets for free."""
    def _build():
        def physical_apply_fn(params, t, x, y, z, window_start, window_end):
            net_params = params["net"]
            traj_tt = jax.lax.stop_gradient(params["traj_tt"])
            traj_xx = jax.lax.stop_gradient(params["traj_xx"])
            raw = raw_apply_fn(net_params, t, x, y, z, window_start, window_end)
            bg  = _chi_bg_from_traj(t, x, y, z, traj_tt, traj_xx)
            return raw.at[:, CHI_IDX].add(bg[:, 0])
        if _TAYLOR_KIND.get(raw_apply_fn) == "plain":
            _TAYLOR_KIND[physical_apply_fn] = "moving"
        return physical_apply_fn

    return _moving_chi_bg_wrap_cache.get_or_build(raw_apply_fn, _build)


def _bundle_params(net_params, traj_tt, traj_xx):
    return {"net": net_params, "traj_tt": traj_tt, "traj_xx": traj_xx}


# --------------------------------------------------------------------------------
# CROSS-WINDOW JIT-CACHE REUSE
#
# flatten_util.ravel_pytree(params) is called fresh every train_window
# invocation (once for BFGS's initial_weights/unflatten_func), and the
# unflatten_func it returns is passed as a STATIC jax.jit argument into
# bfgs_step/bfgs_loss_and_grad/bfgs_val_and_grad. A fresh closure object
# every window -- even though its actual unflattening LOGIC is identical
# across windows, since the network architecture (N_LAYERS/HIDDEN/FEATURES)
# never changes mid-run -- is exactly the same "unequal-by-identity forces
# a JIT cache miss" pattern already fixed for _wrap_with_moving_chi_bg (see
# there): every window recompiles bfgs_step's whole graph from scratch
# instead of reusing the previous window's compiled executable.
#
# Cached here keyed on the params pytree's STRUCTURE *and* each leaf's
# (shape, dtype) -- NOT on tree_structure() alone: verified directly that
# jax.tree_util.tree_structure() only captures container/key nesting, NOT
# leaf shape or dtype (two dict pytrees with identical keys but
# differently-shaped arrays compare tree_structure-EQUAL and hash equal),
# so a structure-only key could silently return a stale unflatten_func
# built for the WRONG shapes if params' shapes ever changed while its
# key/container layout stayed the same -- a real gap in an earlier version
# of this cache, not merely theoretical (confirmed: unflatten_func from a
# 14-element pytree raises a reshape TypeError when called on a
# 102-element flat vector of otherwise-identical structure, rather than
# being caught by a structure-only cache key). Including each leaf's own
# (shape, dtype) closes this exactly: any two params pytrees with the same
# combined key are truly interchangeable for unflattening, regardless of
# --reinit_mode ("full" re-initializes via the same model.init(...) call
# every time, "warm"/"last_layer" only ever change VALUES, never keys/
# shapes, so this is 1 key for the entire run in every normal case) --
# and self-correcting if that assumption is ever violated (a genuinely
# different structure OR shape/dtype anywhere just misses the cache and
# builds a fresh, still-correct unflatten_func instead of reusing an
# incompatible one).
# Same shared cache class as _hard_ic_apply_fn_cache/_moving_chi_bg_wrap_cache
# (see _IdentityLRUCache) -- the key space here is the number of DISTINCT
# (structure, shapes) combinations ever seen in one run, which is 1 for the
# entire run in every normal case (fixed architecture), so this bound is
# never expected to actually evict anything; it exists purely so all three
# of this file's memoization caches share one eviction discipline instead
# of this one being the sole unbounded exception.
_unflatten_func_cache = _IdentityLRUCache(_PERWINDOW_CACHE_MAXSIZE)


def _params_shape_key(params):
    """Hashable key capturing params' pytree structure AND every leaf's
    (shape, dtype) -- see module comment above for why tree_structure()
    alone is NOT sufficient."""
    struct = jax.tree_util.tree_structure(params)
    leaf_shapes = tuple((leaf.shape, leaf.dtype) for leaf in jax.tree_util.tree_leaves(params))
    return (struct, leaf_shapes)


def _get_stable_unflatten(params):
    """unflatten_func for `params`' pytree structure+shapes, reused across
    windows whenever that combination repeats (see module comment above)."""
    key = _params_shape_key(params)
    return _unflatten_func_cache.get_or_build(key, lambda: flatten_util.ravel_pytree(params)[1])


# --------------------------------------------------------------------------------  # [GPUS]
# [GPUS] MULTI-GPU DATA PARALLELISM (automatic sharding; PINN_LESSONS §14).
# The only hook into the rest of the code is _resample_train_data, which passes
# every freshly built train_data tuple through _shard_train_data. Per-point arrays
# (leading axis = points) are split over the GPUs, scalars are replicated, and
# jax.jit/GSPMD partitions the UNCHANGED loss: each GPU evaluates the residuals of
# its own points and XLA inserts the all-reduces of the sums (means, causal
# bucket sums) and of the gradient. Parameters and the inverse Hessian stay
# uncommitted/replicated, so the optimizer code is untouched. With --n_gpus 1 this
# block does nothing and the code path is the original one.
# --------------------------------------------------------------------------------  # [GPUS]
n_gpus = int(args.n_gpus)  # [GPUS]
if n_gpus > 1:  # [GPUS]
    from jax.sharding import Mesh as _Mesh, NamedSharding as _NS, PartitionSpec as _PS  # [GPUS]
    _gpu_devs = jax.devices()  # [GPUS]
    if len(_gpu_devs) < n_gpus:  # [GPUS]
        raise ValueError(f"--n_gpus {n_gpus} but only {len(_gpu_devs)} devices visible: {_gpu_devs}")  # [GPUS]
    _GPU_MESH = _Mesh(np.array(_gpu_devs[:n_gpus]), ("d",))  # [GPUS]
    _SH_POINTS = _NS(_GPU_MESH, _PS("d"))  # [GPUS]
    _SH_REPL   = _NS(_GPU_MESH, _PS())  # [GPUS]
    print(f"[GPUS] data parallelism over {n_gpus} devices: {_gpu_devs[:n_gpus]}")  # [GPUS]


profile_restarts = bool(args.profile_restarts)  # [GPUS]
fast_hessian     = bool(args.fast_hessian)  # [GPUS]
loss_plot_every  = max(1, int(args.loss_plot_every))  # [GPUS]
if fast_hessian:  # [GPUS]
    # minimize() looks _ssbroyden_update up as a module global at call time, so  # [GPUS]
    # rebinding it here swaps in the compiled version without touching Crunch/.  # [GPUS]
    _minimize_mod._ssbroyden_update = jax.jit(_minimize_mod._ssbroyden_update)  # [GPUS]
    print("[GPUS] --fast_hessian: SSBroyden inverse-Hessian update compiled with jax.jit")  # [GPUS]
_PROF_ROW  = collections.defaultdict(float)  # [GPUS] current restart: component -> seconds / counts
_PROF_ROWS = []  # [GPUS] finished restarts of the current window
_PROF_WIN  = {}  # [GPUS] window start time


def _prof_add(name, t0, *block):  # [GPUS]
    """[GPUS] --profile_restarts: wait for `block`, add the time since t0 to
    component `name` of the current restart, return the new start time."""
    if not profile_restarts:
        return t0
    if block:
        jax.block_until_ready(block)
    t = time.perf_counter()
    _PROF_ROW[name] += t - t0
    return t


def _prof_wrap(fn, name):  # [GPUS]
    """[GPUS] Time and count every call of a loss function handed to minimize()."""
    def wrapped(w):
        t0 = time.perf_counter()
        r = fn(w)
        jax.block_until_ready(r)
        _PROF_ROW[name] += time.perf_counter() - t0
        _PROF_ROW["n_" + name] += 1
        return r
    return wrapped


def _prof_end_restart(t_start):  # [GPUS]
    if not profile_restarts:
        return
    _PROF_ROW["restart_total"] = time.perf_counter() - t_start
    _PROF_ROWS.append(dict(_PROF_ROW))
    _PROF_ROW.clear()


def _prof_window_summary(window_idx, skip=2):  # [GPUS]
    """[GPUS] Mean per restart over the window's restarts, skipping the first
    `skip` (compilation, long first line searches)."""
    if not profile_restarts or not _PROF_ROWS:
        return
    rows = _PROF_ROWS[skip:] if len(_PROF_ROWS) > skip else _PROF_ROWS
    _all = sum(r.get("restart_total", 0.0) for r in _PROF_ROWS)
    _first = sum(r.get("restart_total", 0.0) for r in _PROF_ROWS[:skip])
    _el = time.perf_counter() - _PROF_WIN.get("t0", time.perf_counter())
    print(f"\n  [PROFILE] window {window_idx}: {len(_PROF_ROWS)} restarts took {_all:.1f} s "
          f"(first {skip}: {_first:.1f} s); window elapsed so far {_el:.1f} s, so "
          f"{_el - _all:.1f} s before/between restarts (setup, IC targets, initial snapshots)")
    keys = sorted({k for r in rows for k in r})
    mean = {k: sum(r.get(k, 0.0) for r in rows) / len(rows) for k in keys}
    tot = mean.get("restart_total", float("nan"))
    t_f = mean.get("f_val", 0.0) + mean.get("f_valgrad", 0.0)
    opt_over = mean.get("bfgs_step", 0.0) - t_f - mean.get("hess_post", 0.0)
    print(f"  [PROFILE] window {window_idx}, n_gpus={n_gpus}: mean over {len(rows)} restarts "
          f"(first {skip} skipped of {len(_PROF_ROWS)}), seconds per restart")
    for k, v in [("restart_total", tot),
                 ("  bfgs_step (whole minimize call)", mean.get("bfgs_step", 0.0)),
                 ("    loss-only evals   (f_val)", mean.get("f_val", 0.0)),
                 ("    loss+grad evals   (f_valgrad)", mean.get("f_valgrad", 0.0)),
                 ("    optimizer overhead (H update, Python, line-search logic)", opt_over),
                 ("    Hessian post-processing (symmetrize, Cholesky)", mean.get("hess_post", 0.0)),
                 ("  trajectory refresh", mean.get("traj_refresh", 0.0)),
                 ("  logging evaluation (apply_model)", mean.get("apply_model", 0.0)),
                 ("  loss history (CSV, loss plot, snapshots)", mean.get("loss_log", 0.0)),
                 ("    loss plot (save_loss_plot)", mean.get("loss_plot", 0.0)),
                 ("    2D snapshots (every --snapshot_save_every steps)", mean.get("snapshots", 0.0)),
                 ("  resample (build_train_data + sharding)", mean.get("resample", 0.0)),
                 ("  other", tot - sum(mean.get(c, 0.0) for c in
                     ("bfgs_step", "traj_refresh", "apply_model", "loss_log", "resample")))]:
        print(f"  [PROFILE] {k:<62s} {v:8.3f}  ({100 * v / tot:5.1f}%)")
    print(f"  [PROFILE] evals per restart: f_val {mean.get('n_f_val', 0):.1f}, "
          f"f_valgrad {mean.get('n_f_valgrad', 0):.1f}; per eval: "
          f"f_val {mean.get('f_val', 0) / max(mean.get('n_f_val', 0), 1e-9) * 1e3:.1f} ms, "
          f"f_valgrad {mean.get('f_valgrad', 0) / max(mean.get('n_f_valgrad', 0), 1e-9) * 1e3:.1f} ms")
    _PROF_ROWS.clear()


def _shard_train_data(data):  # [GPUS]
    """[GPUS] Place a train_data tuple on the GPU mesh: arrays with a leading
    point axis are sharded along it, 0-d arrays (window_start/end) replicated.
    Identity for --n_gpus 1."""
    if n_gpus <= 1:
        return data
    out = []
    for i, a in enumerate(data):
        a = jnp.asarray(a)
        if a.ndim == 0:
            out.append(jax.device_put(a, _SH_REPL))
        else:
            if a.shape[0] % n_gpus:
                raise ValueError(f"[GPUS] train_data[{i}] has {a.shape[0]} points, not divisible by "
                                 f"--n_gpus {n_gpus}: adjust --Nint/--NI/--Nb/--core_points/--core_ic_points.")
            out.append(jax.device_put(a, _SH_POINTS))
    return tuple(out)


# --------------------------------------------------------------------------------
# BUILD TRAINING DATA
# train_data tuple: (tc,xc,yc,zc, tf,xf,yf,zf, tb,xb,yb,zb,gb,
#                    ti,xi,yi,zi, ic_ccz4, w_int,w_tapa,w_ic, ws,we)
# --------------------------------------------------------------------------------
def build_train_data(window_start, window_end, key, prev_state=None, star_centers=None, sigma_time=None,
                     follow=None):
    X_c, key   = generate_interior(Nint, key, window_start, window_end, centers=star_centers,
                                   sigma_time=sigma_time, follow=follow)
    tc = X_c[:, 0:1].astype(jnp.float64)
    xc = X_c[:, 1:2].astype(jnp.float64)
    yc = X_c[:, 2:3].astype(jnp.float64)
    zc = X_c[:, 3:4].astype(jnp.float64)

    X_f, key   = generate_ic_surface(N0, key, window_end, centers=star_centers)
    if follow is not None and star_centers is not None:
        # terminal (tapa) surface sits at t=window_end: focus it on where the
        # punctures are THEN, not at window_start (see --follow_puncture_sampling)
        xyz_f = _follow_puncture_shift(X_f[:, 1:4], X_f[:, 0:1], star_centers, *follow, ic_focus_radius)
        X_f = jnp.hstack((X_f[:, 0:1], xyz_f))
    tf = X_f[:, 0:1].astype(jnp.float64)
    xf = X_f[:, 1:2].astype(jnp.float64)
    yf = X_f[:, 2:3].astype(jnp.float64)
    zf = X_f[:, 3:4].astype(jnp.float64)

    X_b, key   = generate_boundary(Nb, key, window_start, window_end)
    tb = X_b[:, 0:1].astype(jnp.float64)
    xb = X_b[:, 1:2].astype(jnp.float64)
    yb = X_b[:, 2:3].astype(jnp.float64)
    zb = X_b[:, 3:4].astype(jnp.float64)
    if metric_bc_mode == "dirichlet":
        gb = compute_ccz4_ic(xb[:, 0], yb[:, 0], zb[:, 0])
        gb = gb.astype(jnp.float64)
    else:
        # gb (analytic metric IC at the boundary) is only read by the
        # dirichlet branch of apply_model/compute_bfgs_loss; under the
        # default sommerfeld BC it's unused, so skip compute_ccz4_ic's
        # expensive metric inversions + jvp-based Christoffel computation
        # (otherwise re-run on all Nb points every resampling for the
        # entire run) and substitute a cheap placeholder of the same shape.
        gb = jnp.zeros((Nb, CCZ4_SLICE.stop - CCZ4_SLICE.start), dtype=jnp.float64)

    X0, key    = generate_ic_surface(N0, key, window_start, centers=star_centers)
    ti = X0[:, 0:1].astype(jnp.float64)
    xi = X0[:, 1:2].astype(jnp.float64)
    yi = X0[:, 2:3].astype(jnp.float64)
    zi = X0[:, 3:4].astype(jnp.float64)

    if prev_state is None:
        ic_ccz4 = compute_ccz4_ic(xi[:, 0], yi[:, 0], zi[:, 0])
    else:
        # Window k>0: IC comes from the previous window's trained network.
        # Only VALUES are needed for every field: every field is
        # first-order-in-time, so its d/dt is determined by the PDE
        # residual itself, not prescribed from an external IC.
        prev_p = prev_state["params"]
        _ps = jnp.array(prev_state["t_start"], dtype=jnp.float64)
        _pe = jnp.array(prev_state["t_end"],   dtype=jnp.float64)
        if prev_state.get("ic_mode", "hard") == "hard":
            prev_fn = make_hard_ic_apply_fn(
                model.apply, float(_ps), float(_pe),
                ic_history=prev_state.get("ic_history"))
        else:
            prev_fn = apply_fn

        x_in = xi[:, 0].reshape(-1, 1)
        y_in = yi[:, 0].reshape(-1, 1)
        z_in = zi[:, 0].reshape(-1, 1)
        t_in = jnp.full((x_in.shape[0], 1), window_start, dtype=jnp.float64)
        _out    = full_output(prev_fn, prev_p, t_in, x_in, y_in, z_in, _ps, _pe)
        ic_ccz4 = _out[:, CCZ4_SLICE]
        if chi_bg_mode == "moving":
            # prev_fn is the RAW (unwrapped) apply function, so its CHI_IDX
            # output is the regular deviation, not physical chi -- add back
            # the background at the previous window's own final tracked
            # position (_shift_track_xy, already advanced to that value by
            # advance_shift_tracked_centers before this window's
            # build_train_data call) to get the physical IC target used by
            # loss_ics/compute_bfgs_loss.
            chi_bg_prev = _translated_psi(x_in, y_in, z_in, _shift_track_xy) ** -4
            ic_ccz4 = ic_ccz4.at[:, CHI_IDX].add(chi_bg_prev[:, 0])

    ic_ccz4 = ic_ccz4.astype(jnp.float64)

    # No excision in this file (see _lapse_weight) -- always 1, kept only for
    # interface compatibility with _wmean/apply_model/compute_bfgs_loss.
    w_int  = _lapse_weight(xc[:, 0], yc[:, 0], zc[:, 0])
    w_tapa = _lapse_weight(xf[:, 0], yf[:, 0], zf[:, 0])
    w_ic   = _lapse_weight(xi[:, 0], yi[:, 0], zi[:, 0])

    ws_arr = jnp.array(window_start, dtype=jnp.float64)
    we_arr = jnp.array(window_end,   dtype=jnp.float64)
    return (tc, xc, yc, zc, tf, xf, yf, zf, tb, xb, yb, zb, gb,
            ti, xi, yi, zi, ic_ccz4,
            w_int, w_tapa, w_ic,
            ws_arr, we_arr), key


# --------------------------------------------------------------------------------
# EARLY STOPPING
# --------------------------------------------------------------------------------
class EarlyStopping:
    def __init__(self, patience: int, min_delta: float = 1e-6, min_loss: float = 0.0):
        self.patience    = patience
        self.min_delta   = min_delta
        self.min_loss    = min_loss      # > 0: stop only once best_loss <= min_loss
        self.best_loss   = float("inf")
        self.counter     = 0
        self.should_stop = False

    def update(self, loss: float) -> bool:
        if self.patience <= 0:
            return False
        if self.best_loss == float("inf"):
            # No finite baseline established yet -- covers BOTH the true
            # first call AND any later call where every loss so far has
            # been non-finite (sustained divergence). A relative-
            # improvement ratio against an infinite baseline is
            # (finite-inf)/inf = NaN regardless of the new loss (confirmed:
            # IEEE-754, not a hypothetical), and `NaN > min_delta` is
            # ALWAYS False in Python -- so best_loss could never escape
            # inf via the normal path below, and a window whose loss goes
            # inf then RECOVERS to a good, improving finite value would
            # still be forced to stop after patience more calls, treating
            # "recovering nicely" identically to "still diverging." Here,
            # any finite loss is trivially an infinite improvement over
            # "no finite baseline yet", so accept it unconditionally as
            # the new best regardless of how many prior calls were
            # non-finite; a non-finite loss just counts as another
            # non-improving step, so sustained non-finite still eventually
            # stops via patience, as intended.
            if math.isfinite(loss):
                self.best_loss = loss
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.should_stop = True
            return self.should_stop
        rel_improvement = (self.best_loss - loss) / (abs(self.best_loss) + 1e-12)
        if rel_improvement > self.min_delta:
            self.best_loss = loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience and (self.min_loss <= 0.0 or self.best_loss <= self.min_loss):
                self.should_stop = True
        return self.should_stop

    def reset(self, patience: int = None) -> None:
        """Start a fresh probation window after an external recovery action
        (e.g. resetting the BFGS Hessian to identity). Deliberately keeps
        best_loss as-is rather than reverting to inf: a recovery attempt
        should only count as a genuine improvement if it beats the true
        best-so-far, not merely any finite loss measured against a fresh
        infinite baseline."""
        if patience is not None:
            self.patience = patience
        self.counter = 0
        self.should_stop = False

    def __repr__(self):
        return (f"EarlyStopping(patience={self.patience}, min_delta={self.min_delta}, "
                f"best={self.best_loss:.3e}, counter={self.counter}/{self.patience})")


# --------------------------------------------------------------------------------
# METRIC INVERSION HELPER
# --------------------------------------------------------------------------------
def _invert_contra_10(g_contra_10):
    return np.asarray(jnp.linalg.inv(unpack_metric_components(jnp.asarray(g_contra_10))))


def _clamp_ylim(ax, margin_frac=0.05):
    lines = ax.get_lines()
    if not lines:
        return
    yd = np.asarray(lines[0].get_ydata(), dtype=float)
    finite = yd[np.isfinite(yd)]
    if len(finite) < 2:
        return
    ylo = float(finite.min())
    yhi = float(finite.max())
    if yhi <= ylo:
        return
    margin = margin_frac * (yhi - ylo)
    ax.set_ylim(ylo - margin, yhi + margin)


def _safe_log_yscale(ax, *arrays):
    """Set a log y-scale only if at least one finite, positive value exists
    across the given array(s); otherwise leave the axis on its default
    linear scale.

    matplotlib's log-scale tick locator raises a hard ValueError ("Data
    cannot be log-scaled because all values are <= 0") when handed an
    all-NaN/all-non-positive array -- which happens whenever a constraint
    quantity is EXACTLY zero everywhere it's sampled (e.g. Theta/the IC
    residual at t=window_start under --ic_mode hard, where the hard-IC
    wrapper pins the value algebraically) or a diverged run blows up to NaN
    everywhere. Guarding here lets the run (and the plot) survive instead of
    crashing on what is otherwise just a diagnostic figure."""
    for arr in arrays:
        a = np.asarray(arr, dtype=float)
        if np.any(np.isfinite(a) & (a > 0)):
            ax.set_yscale("log")
            return
    ax.set_yscale("linear")


def _is_divergent(arr, rel_threshold):
    """Boolean mask: non-finite, or larger in magnitude than rel_threshold
    times arr's OWN typical (median-of-finite) scale. Shared by
    _mask_divergent and _apply_plot_nan so the two never drift apart on
    what counts as 'divergent' -- they previously duplicated this exact
    three-line test verbatim."""
    arr = np.asarray(arr, dtype=float)
    finite = arr[np.isfinite(arr)]
    typical_scale = np.median(np.abs(finite)) if finite.size else 0.0
    return ~np.isfinite(arr) | (np.abs(arr) > rel_threshold * max(typical_scale, 1e-12))


def _mask_divergent(arr, rel_threshold=mask_rel_threshold):
    """Mask (to NaN) values _is_divergent flags. Pure value-based -- no
    positional/geometric information needed, unlike _apply_plot_nan below.
    Use this (not _apply_plot_nan) for any diagnostic spanning MULTIPLE
    times/windows (e.g. the final diagnostics_1D_cut.png/diagnostics_2D_t*.png,
    which plot t=0 through t=tfinal in one pass): _apply_plot_nan's
    geometric pre-filter is anchored to _bh_display_mask_zones' state as of
    the LAST window, so under a moving-puncture mode it would be centered
    at the wrong location for earlier time slices -- potentially masking
    ordinary far-field pixels at the stale position while leaving the
    ACTUAL (moved) divergence completely unmasked. This has no such
    staleness concern since it never needs to know WHERE a puncture
    currently is."""
    arr = np.asarray(arr, dtype=float)
    return np.where(_is_divergent(arr, rel_threshold), np.nan, arr)


def _apply_plot_nan(arr, x_arr, y_arr=None, z_arr=None, rel_threshold=mask_rel_threshold, zones=None):
    """Mask (to NaN) only points that are BOTH (a) within a near-puncture
    display-mask zone (see _bh_display_mask_zones) AND (b) actually
    divergent THERE FOR THIS FIELD (see _mask_divergent's criterion).
    Previously masked EVERY point in the geometric zone unconditionally,
    regardless of whether that specific field actually diverges there --
    e.g. alpha often stays well-behaved right at a puncture after
    training, while g_xx genuinely blows up (chi -> 0) there; the old
    version masked alpha too, purely because it happened to be evaluated
    at the same x location. This only touches points that are an actual
    problem for THIS array, leaving well-behaved fields completely
    unmasked even directly at a puncture.

    Appropriate for per-window training snapshots, evaluated at a time
    WITHIN that window whose puncture position IS current -- see
    _mask_divergent for diagnostics spanning multiple times/windows,
    where this function's geometric pre-filter would be stale.

    zones (default None = the module-global _bh_display_mask_zones, set once
    per window at window_start): pass an explicit list (see
    _display_mask_zones_for) to mask around the punctures' positions at a
    specific LATER time instead -- the punctures drift ~0.1-0.2 units per
    window, more than the ~0.06 mask radius, so window_start's zones no longer
    cover them by the time of the later snapshot columns / window_end."""
    arr = np.asarray(arr, dtype=float)
    if zones is None:
        zones = _bh_display_mask_zones
    if not zones:
        return arr
    x_np  = np.asarray(x_arr).ravel()
    y_np  = np.zeros_like(x_np) if y_arr is None else np.asarray(y_arr).ravel()
    z_np  = np.zeros_like(x_np) if z_arr is None else np.asarray(z_arr).ravel()
    near_puncture = np.zeros(x_np.shape, dtype=bool)
    for cx, cy, cz, r_min in zones:
        r = np.sqrt((x_np - cx)**2 + (y_np - cy)**2 + (z_np - cz)**2)
        near_puncture |= (r < r_min)
    out = arr.copy()
    out[near_puncture & _is_divergent(arr, rel_threshold)] = np.nan
    return out


def _bh_adm_from_out(out):
    """alpha, beta^x, covariant g_xx from the network's 24-field CCZ4 output.
    alpha and beta^x are direct network outputs now (no need to reconstruct
    g^{ab} and re-derive them, unlike the old g^{ab}-only representation);
    g_xx = gamma-tilde_xx / chi is the direct physical-metric relation."""
    alpha   = np.asarray(out[:, ALPHA_IDX])
    beta_x  = np.asarray(out[:, BETA_SLICE.start])
    chi     = np.asarray(out[:, CHI_IDX])
    gt_xx   = np.asarray(out[:, GT_SLICE])[:, 0]   # xx is the first packed component
    # chi is only soft-penalized (relu(-chi)**2) during training, not hard-
    # clamped, so it can go near/below zero close to a puncture -- floor it
    # the same way _bh_adm_from_contra10 floors -g_tt, to avoid Inf/NaN/
    # sign-flipped g_xx feeding into plot axis limits.
    gxx_cov = gt_xx / np.maximum(chi, 1e-30)
    return alpha, beta_x, gxx_cov


def _bh_adm_from_contra10(g_contra_10):
    g_tt   = g_contra_10[:, 0]
    g_tx   = g_contra_10[:, 1]
    alpha  = 1.0 / np.sqrt(np.maximum(-g_tt, 1e-30))
    beta_x = -g_tx / np.where(g_tt < -1e-30, g_tt, np.full_like(g_tt, -1e-30))
    g_cov  = _invert_contra_10(g_contra_10)
    return alpha, beta_x, g_cov[:, 1, 1]


# --------------------------------------------------------------------------------
# TRAIN WINDOW
# --------------------------------------------------------------------------------
def train_window(window_idx, window_start, window_end, key, prev_state=None):
    print("\n" + "=" * 80)
    print(f"Training window {window_idx+1}/{N_window}: [{window_start:.6f}, {window_end:.6f}]  ic_mode={ic_mode}")
    _PROF_WIN["t0"] = time.perf_counter()  # [GPUS]
    print("=" * 80)

    if bh_center_mode == "shift_driven":
        star_centers = find_bh_centers_shift_driven()
        print(f"  BH centers (shift-driven track): "
              f"{[(f'{c[0]:.3f}', f'{c[1]:.3f}', f'{c[2]:.3f}') for c in star_centers]}")
    elif bh_center_mode in ("lapse_min", "chi_min") and prev_state is not None:
        # Local box hint from _scan_track_state's last-known position (set
        # by _locate_centers_and_diags at the end of the PREVIOUS window --
        # see there and _search_halfwidth) -- None until the first
        # successful locate, in which case both helpers fall back to their
        # original whole-plane scan.
        if all(h is not None for h in _scan_track_state):
            _search_centers = [(h["x"], h["y"]) for h in _scan_track_state]
            _dt = window_start - _scan_track_state[0]["t"]
            _halfwidth = _search_halfwidth([(h["beta_x"], h["beta_y"]) for h in _scan_track_state], _dt)
        else:
            _search_centers, _halfwidth = None, None

        if bh_center_mode == "lapse_min":
            star_centers = find_bh_centers_lapse(prev_state, window_start,
                                                  search_centers=_search_centers, search_halfwidth=_halfwidth)
            print(f"  BH centers from lapse minima: "
                  f"{[(f'{c[0]:.3f}', f'{c[1]:.3f}', f'{c[2]:.3f}') for c in star_centers]}")
        else:
            # Reuses locate_chi_minima -- the same multi-stage refinement/
            # candidate-selection already used as shift_driven's post-hoc
            # cross-check -- as the actual sampling-focus driver, rather
            # than only a diagnostic. No traj_tt/traj_xx: chi_min is
            # incompatible with --chi_bg_mode moving (see --bh_center_
            # mode's help), which is the only case those would be needed
            # for (see locate_chi_minima's docstring). Returns (x,y) pairs
            # (a genuine 2D x-y plane scan) -- wrap to match
            # find_bh_centers_lapse/find_bh_centers_shift_driven's (x,y,z)
            # convention.
            star_centers = [(x, y, 0.0) for x, y in locate_chi_minima(
                prev_state, window_start, search_centers=_search_centers, search_halfwidth=_halfwidth)]
            print(f"  BH centers from chi minima: "
                  f"{[(f'{c[0]:.3f}', f'{c[1]:.3f}', f'{c[2]:.3f}') for c in star_centers]}")
    else:
        star_centers = [(c[0], 0.0, 0.0) for c in _bh_center_list(BL_masses)]

    # Keep the near-puncture plot-masking zones (see _apply_plot_nan, used
    # by every snapshot plotter below) centered on THIS window's actual
    # tracked positions, not the static t=0 ones computed at module load.
    _refresh_display_mask_zones(star_centers)

    # Exponential-to-uniform time-distribution schedule.
    _SIGMA_MAX_MULT = 20.0
    _win_width      = window_end - window_start
    _n_adam_resamp  = EPOCHS_ADAM // Nchange if EPOCHS_ADAM > 0 else 0
    _n_bfgs_resamp  = (max(1, EPOCHS_QN // Nchange) if EPOCHS_QN > 0 else 0)
    _total_resamp   = _n_adam_resamp + _n_bfgs_resamp
    _k_uniform      = f_uniform * _total_resamp

    def _sigma_time_at(resample_count):
        if time_dist == "uniform":
            return None
        if _k_uniform <= 0 or resample_count >= _k_uniform:
            return None
        frac      = resample_count / _k_uniform
        sigma_mul = sigma_time_0 + (_SIGMA_MAX_MULT - sigma_time_0) * frac
        return sigma_mul * _win_width

    _resample_count = 0

    # Declared here (not down with the chi_bg_mode block that first assigns
    # it) because _resample_train_data below reads it via closure and is
    # called once immediately, before that block runs -- with no trajectory
    # yet, so the first batch of a window is drawn unshifted.
    traj_tt, traj_xx = None, None
    # Bumped on every _refresh_traj: keys _snap_grid_at's memo. NOT id(traj_xx)
    # -- CPython recycles the ids of freed arrays, and _refresh_traj rebinds
    # traj_xx every resample, so a later trajectory could hit an older one's
    # cache entry (reproduced: ids repeat with period ~3).
    _traj_version = [0]

    def _resample_train_data():
        # Shared by the initial draw below and both the Adam/BFGS resample
        # blocks further down -- previously this 4-line call was triplicated
        # verbatim at all three sites, so a future change to its arguments
        # had three places to update in lockstep instead of one.
        nonlocal key
        data, key = build_train_data(window_start, window_end, key, prev_state=prev_state,
                                     star_centers=(_INITIAL_CENTERS if args.sampling_centers == "initial"
                                                   else star_centers),
                                     sigma_time=_sigma_time_at(_resample_count),
                                     follow=((traj_tt, traj_xx) if follow_puncture_sampling and traj_tt is not None else None))
        return _shard_train_data(data)  # [GPUS]

    train_data = _resample_train_data()

    if prev_state is None or ic_mode != "hard":
        ic_history = None
    else:
        print(f"  Building frozen IC from previous window at t={window_start:.6f}...")
        ic_history = extend_ic_history(prev_state)

    if ic_mode == "hard":
        train_apply_fn = make_hard_ic_apply_fn(model.apply, window_start, window_end,
                                                ic_history=ic_history)
    else:
        train_apply_fn = apply_fn

    _dummy_args = (jnp.ones((Nint, 1), dtype=jnp.float64),) * 4
    _dummy_ws   = jnp.array(0.0,    dtype=jnp.float64)
    _dummy_we   = jnp.array(tfinal, dtype=jnp.float64)

    if prev_state is None or reinit_mode == "full":
        params = model.init(key, *_dummy_args, _dummy_ws, _dummy_we)
        params = jax.tree_util.tree_map(lambda x: x.astype(jnp.float64), params)
    elif reinit_mode == "last_layer":
        fresh = model.init(key, *_dummy_args, _dummy_ws, _dummy_we)
        fresh = jax.tree_util.tree_map(lambda x: x.astype(jnp.float64), fresh)
        last_key = f"Dense_{N_LAYERS}"
        params = flax.core.unfreeze(prev_state["params"])
        params['params']['MLP'][last_key] = flax.core.unfreeze(fresh)['params']['MLP'][last_key]
    else:
        params = prev_state["params"]

    _nx_snap  = _SNAP_NX_SHARED
    # The snapshot cut line follows the puncture(s)' CURRENT actual position
    # (see _snap_cut_line): it is built per snapshot column by _snap_grid_at,
    # not once here at window_start, because the punctures keep moving within
    # the window.

    # save_field_snapshot's "BL analytic" overlay is evaluated at the SAME
    # (x, y(x)) points as the PINN curve it's compared against, for a
    # pointwise-comparable overlay -- the analytic formula itself has no
    # notion of the orbital motion (it's the static initial-data solution),
    # so this isn't "more correct" physically, only a fair like-for-like
    # comparison at whatever point the PINN is actually being read at. Now
    # computed PER SNAPSHOT COLUMN there (see _snap_grid_at), since each
    # column's own y(x) can differ once the cut line tracks the puncture's
    # within-window motion instead of freezing at window_start.

    _ws_snap = jnp.array(window_start, dtype=jnp.float64)
    _we_snap = jnp.array(window_end,   dtype=jnp.float64)

    if chi_bg_mode == "moving":
        # train_apply_fn built above is the RAW (regular-chi-output)
        # function; keep a handle to it for trajectory integration (beta^x
        # is an unaffected channel, so no circularity with the physical
        # wrapper). train_apply_fn itself is then rebuilt ONCE (here) as
        # the physical wrapper and NEVER rebuilt again for the rest of the
        # window -- refreshing the trajectory only ever updates the plain
        # traj_tt/traj_xx array VALUES (of fixed shape, threaded through
        # apply_model/compute_bfgs_loss as ordinary dynamic arguments and
        # bundled with net params right before calling apply_fn), which
        # triggers no recompilation. An earlier version of this feature
        # instead rebuilt apply_fn itself (a new Python closure) on every
        # refresh; since apply_fn is a static jax.jit argument in several
        # functions, that meant a brand-new compiled executable accumulating
        # in jax's (never-evicted) jit cache at every refresh -- harmless
        # for a short smoke test, but unbounded memory growth (eventually
        # OOM) over a long run with many BFGS outer restarts. See
        # _wrap_with_moving_chi_bg / apply_model's total_loss.
        raw_train_apply_fn = train_apply_fn
        train_apply_fn = _wrap_with_moving_chi_bg(raw_train_apply_fn)

        def _refresh_traj(p):
            # moving_bg_refresh_steps (cheap), NOT track_steps (expensive) --
            # this is an unjitted, Python-looped RK4 integration re-run from
            # scratch on every resample/BFGS-restart, so its cost multiplies
            # by however many times training calls it; the background only
            # needs to track the puncture's approximate current position
            # between refreshes, not the high-precision trajectory that gets
            # logged (that still uses the full track_steps, once per window,
            # in advance_shift_tracked_centers -- see --moving_bg_refresh_steps).
            nonlocal traj_tt, traj_xx
            traj_tt, traj_xx = _integrate_puncture_trajectory(
                raw_train_apply_fn, p, _ws_snap, _we_snap,
                window_start, window_end, _shift_track_xy, moving_bg_refresh_steps)
            _traj_version[0] += 1

        _refresh_traj(params)
        if follow_puncture_sampling:
            # The batch drawn at the top of train_window predates any
            # trajectory (so it was unshifted); redraw it now that one exists
            # -- informative under --reinit_mode warm, where params (and so the
            # trajectory) come from the previous window.
            train_data = _resample_train_data()

    def _snap_params(p):
        """p (plain net params) -> whatever train_apply_fn actually expects
        (a {"net":..,"traj_tt":..,"traj_xx":..} bundle under --chi_bg_mode
        moving, using the CURRENT traj_tt/traj_xx; p itself unchanged
        otherwise) -- used wherever a snapshot/diagnostic function is about
        to call train_apply_fn directly with a net-only params value."""
        if chi_bg_mode != "moving":
            return p
        return _bundle_params(p, traj_tt, traj_xx)

    def _star_centers_at(t_val):
        """Best estimate of each puncture's (x,y,0) position AT t_val, for
        the snapshot cut line (see _snap_cut_line). star_centers itself is
        only ever computed ONCE, at window_start (top of train_window) --
        reusing it unchanged for every snapshot column up to window_end
        left the cut line stale by the time a puncture has actually moved
        within the window (measured directly on a real run: ~0.09 units of
        drift over a single window near peak orbital angular velocity,
        bigger than the puncture's own near-zone mask radius). When the
        within-window RK4 trajectory (traj_tt/traj_xx, refreshed
        periodically during training -- see _refresh_traj above) is
        available, interpolate along it instead so later columns track the
        puncture's actual current position. Falls back to the frozen
        window-start star_centers when no such trajectory exists --
        chi_min/lapse_min modes have no sub-window trajectory at all
        (they re-locate from scratch each window, see
        _locate_centers_and_diags), and a shift_driven run under
        --chi_bg_mode physical never builds traj_tt/traj_xx either (see
        the chi_bg_mode=='moving' gate above) -- window-start is the best
        (only) estimate available in either case."""
        if traj_tt is None:
            return star_centers
        out = []
        for k in range(len(star_centers)):
            xk, yk = _interp_trajectory(traj_tt, traj_xx, k, t_val)
            out.append((float(xk), float(yk), 0.0))
        return out

    _snap_cut_cache = {}

    def _snap_grid_at(t_val):
        """The _SnapCut (cut line + display zones + IC reference) for THIS
        t_val's own current puncture positions (see _star_centers_at).
        Memoized on (t_val, trajectory refresh counter): the four 1D snapshot
        functions run back to back for the same columns with the same
        trajectory, so this builds each column once per snapshot event (and
        the t=0 IC reference on it is computed once, not once per plot)."""
        key = (float(t_val), _traj_version[0])
        cut = _snap_cut_cache.get(key)
        if cut is None:
            if len(_snap_cut_cache) > 32:
                _snap_cut_cache.clear()
            cut = _snap_cut_cache[key] = _SnapCut(_star_centers_at(t_val), _nx_snap)
        return cut

    def _eval_fields(p, t_val):
        g = _snap_grid_at(t_val)
        t_in = jnp.full((_nx_snap, 1), t_val, dtype=jnp.float64)
        out  = np.asarray(full_output(train_apply_fn, p, t_in, g.x_in, g.y_in, g.z_in, _ws_snap, _we_snap))
        return _bh_adm_from_out(out)

    def save_field_snapshot(p, step):
        """4-row x 5-col: alpha, beta^x, g_xx (vs. BL analytic reference),
        and |H| (Hamiltonian constraint)."""
        snap_times = np.linspace(window_start, window_end, 5)
        gs = [_snap_grid_at(float(t)) for t in snap_times]
        cols = [_eval_fields(p, float(t)) for t in snap_times]
        ref_cols = [_bh_adm_from_contra10(_bl_contra_np(gs[i].x, gs[i].y, np.zeros(_nx_snap)))
                    for i in range(len(snap_times))]
        n_rows = 4

        fig, axes = plt.subplots(n_rows, 5, figsize=(20, 3 * n_rows))
        fig.suptitle(f"Window {window_idx}  —  step {step:06d}", fontsize=13)

        row_info = [
            (0, r"$\alpha$",   "tab:blue"),
            (1, r"$\beta^x$",  "tab:purple"),
            (2, r"$g_{xx}$",   "tab:orange"),
        ]
        for row, title, color in row_info:
            for col, (alpha_v, betax_v, gxx_v) in enumerate(cols):
                # pinn_v is plotted FIRST (lines[0]), which is what
                # _clamp_ylim actually reads -- masking only ref_v (as
                # before) left a near-puncture divergent PINN value free to
                # dominate the axis scale on the beta^x/g_xx rows.
                pinn_v = _apply_plot_nan(
                    np.asarray([alpha_v, betax_v, gxx_v][row], dtype=float), gs[col].x, gs[col].y,
                    zones=gs[col].zones)
                ref_v  = _apply_plot_nan(
                    np.asarray(ref_cols[col][row], dtype=float), gs[col].x, gs[col].y, zones=gs[col].zones)
                ax = axes[row, col]
                ax.plot(gs[col].s, pinn_v, color=color, label="PINN")
                ax.plot(gs[col].s, ref_v, "--", color="tab:red", label="BL analytic")
                ax.set_title(f"{title},  $t={snap_times[col]:.4f}$", fontsize=9)
                ax.set_xlabel("s (along line through punctures)"); ax.grid(True, alpha=0.3)
                if col == 0:
                    ax.legend(fontsize=7)
                if row == 0:
                    ax.set_ylim(-0.05, 1.1)
                else:
                    _clamp_ylim(ax)
        # Row 3: Hamiltonian constraint — |H| in log scale
        for col, t_val in enumerate(snap_times):
            g = gs[col]
            tin   = jnp.full((_nx_snap, 1), float(t_val), dtype=jnp.float64)
            H_raw = np.asarray(hamiltonian_constraint(train_apply_fn, p, tin, g.x_in, g.y_in, g.z_in, _ws_snap, _we_snap))
            H_abs = np.where(H_raw != 0.0, np.abs(H_raw), np.nan)
            H_abs = _apply_plot_nan(H_abs, g.x, g.y, zones=g.zones)
            ax  = axes[3, col]
            ax.plot(g.s, H_abs, color="tab:red")
            for _sh in g.horizon_s:
                ax.axvline(_sh, color="k", linewidth=0.8, linestyle="--", alpha=0.7)
            _safe_log_yscale(ax, H_abs)
            ax.set_title(rf"$|H|$,  $t={t_val:.4f}$", fontsize=9)
            ax.set_xlabel("s (along line through punctures)"); ax.grid(True, alpha=0.3)
            if col == 0:
                ax.set_ylabel(r"$|H|$")
            _clamp_ylim(ax)
        # Rug = training-point density projected onto the window_start cut
        # line's direction (s coordinate), so it lines up with the s axis.
        _n_rug  = min(200, int(train_data[1].shape[0]))
        _rug_i  = np.random.choice(int(train_data[1].shape[0]), _n_rug, replace=False)
        _g0     = _snap_grid_at(float(window_start))
        _rug_x  = ((np.asarray(train_data[1]).ravel()[_rug_i] - _g0.mid[0]) * _g0.u[0]
                   + (np.asarray(train_data[2]).ravel()[_rug_i] - _g0.mid[1]) * _g0.u[1])
        for _r in range(n_rows):
            for _c in range(5):
                axes[_r, _c].plot(_rug_x, np.zeros(_n_rug), '|',
                                  color='dimgray', alpha=0.30,
                                  markersize=8, markeredgewidth=0.7,
                                  transform=axes[_r, _c].get_xaxis_transform(),
                                  clip_on=True, zorder=0)
        fig.tight_layout()
        _atomic_savefig(fig, os.path.join(file_path, f"fields_w{window_idx}_step{step:06d}.png"), dpi=120)
        plt.close(fig)

    def save_residual_snapshot(p, step):
        """Plot pointwise CCZ4/gauge/IC residuals along the 1-D cut through the punctures at 5 time slices."""
        snap_times = np.linspace(window_start, window_end, 5)

        fig, axes = plt.subplots(3, 5, figsize=(20, 10))
        fig.suptitle(f"Window {window_idx}  —  residuals — step {step:06d}", fontsize=13)

        for col, t_val in enumerate(snap_times):
            g = _snap_grid_at(float(t_val))
            t_in = jnp.full((_nx_snap, 1), float(t_val), dtype=jnp.float64)

            full_res = np.asarray(r_ccz4_evolution(
                train_apply_fn, p, t_in, g.x_in, g.y_in, g.z_in,
                _ws_snap, _we_snap))                                 # (N, 22)

            # Row 0: CCZ4 evolution residual — RMS over all 22 metric-sector fields
            ein_mag = np.sqrt(np.mean(full_res**2, axis=1))

            # Row 1: momentum-constraint violation |Z^i|
            gv = np.asarray(gauge_source_vector(
                train_apply_fn, p, t_in, g.x_in, g.y_in, g.z_in,
                _ws_snap, _we_snap))                                 # (N, 3): Z^i
            gauge_mag = np.sqrt(np.sum(gv**2, axis=1))

            # Row 2: IC residual at t ≈ t0 (only meaningful at col=0)
            out_ic = np.asarray(full_output(
                train_apply_fn, p, t_in, g.x_in, g.y_in, g.z_in, _ws_snap, _we_snap))
            ic_mag  = np.sqrt(np.mean((out_ic[:, CCZ4_SLICE] - g.ic())**2, axis=1))

            row_data = [
                (0, ein_mag,   r"$|\text{CCZ4 residual}|$",         "tab:orange"),
                (1, gauge_mag, r"$|Z^i|$",                          "tab:purple"),
                (2, ic_mag,    r"$|\text{IC residual}|$",           "tab:green"),
            ]
            for row, data, ylabel, color in row_data:
                ax = axes[row, col]
                # Mask near-puncture points (same convention as
                # save_field_snapshot's reference curve) before the
                # positive-only filter: these are magnitudes, so a
                # divergence near a puncture shows up as a large FINITE
                # spike, not necessarily zero/Inf -- np.where(data>0,...)
                # alone wouldn't catch it, and _clamp_ylim's isfinite check
                # would then let that spike dominate the axis scale.
                data = _apply_plot_nan(data, g.x, g.y, zones=g.zones)
                data_pos = np.where(data > 0, data, np.nan)
                ax.plot(g.s, data_pos, color=color, linewidth=1.0)
                _safe_log_yscale(ax, data_pos)
                ax.set_title(f"$t={t_val:.4f}$", fontsize=9)
                ax.set_xlabel("s (along line through punctures)"); ax.grid(True, alpha=0.3)
                if col == 0:
                    ax.set_ylabel(ylabel, fontsize=8)
                _clamp_ylim(ax)

        fig.tight_layout()
        _atomic_savefig(fig,
            os.path.join(file_path, f"residuals_w{window_idx}_step{step:06d}.png"),
            dpi=100, bbox_inches="tight")
        plt.close(fig)

    def save_gauge_metric_snapshot(p, step):
        """4-row x 5-col: alpha, beta^x, Theta, Z^x along the 1-D cut through the punctures.
        Rows 0-1 (lapse/shift, now genuine dynamical CCZ4 fields): solid =
        network value, dashed = t=0 IC value. Rows 2-3 (constraint
        violations): solid = network value, dashed = 0 (their target for a
        constraint-satisfying solution). No GHG target-tracking anymore --
        gauge is a real evolution equation now, see r_ccz4_evolution.
        """
        snap_times = np.linspace(window_start, window_end, 5)
        colors_t   = plt.cm.plasma(np.linspace(0.0, 0.85, 5))

        fig, axes = plt.subplots(4, 5, figsize=(20, 12))
        fig.suptitle(
            f"Window {window_idx}  —  gauge & constraints — step {step:06d}\n"
            r"rows 0-1: solid = network, dashed = t=0 IC.  "
            r"rows 2-3: solid = network, dashed = 0 (constraint-satisfied target)",
            fontsize=11)

        for col, t_val in enumerate(snap_times):
            g = _snap_grid_at(float(t_val))
            alpha_ic = g.ic()[:, ALPHA_IDX]
            betax_ic = g.ic()[:, BETA_SLICE.start]
            t_in = jnp.full((_nx_snap, 1), float(t_val), dtype=jnp.float64)
            out  = np.asarray(full_output(
                train_apply_fn, p, t_in, g.x_in, g.y_in, g.z_in, _ws_snap, _we_snap))
            zv   = np.asarray(gauge_source_vector(
                train_apply_fn, p, t_in, g.x_in, g.y_in, g.z_in, _ws_snap, _we_snap))   # (N,3): Z^i

            clr = colors_t[col]
            ttl = f"$t={t_val:.4f}$"

            # Row 0: alpha
            ax = axes[0, col]
            ax.plot(g.s, _apply_plot_nan(out[:, ALPHA_IDX], g.x, g.y, zones=g.zones), color=clr, linewidth=1.2)
            ax.plot(g.s, alpha_ic, "k--", linewidth=0.8, alpha=0.7)
            ax.set_title(ttl, fontsize=9); ax.set_xlabel("s (along line through punctures)"); ax.grid(True, alpha=0.3)
            if col == 0:
                ax.set_ylabel(r"$\alpha$", fontsize=9)
                ax.plot([], [], "k--", linewidth=0.8, label="IC data")
                ax.legend(fontsize=7)
            _clamp_ylim(ax)

            # Row 1: beta^x
            ax = axes[1, col]
            ax.plot(g.s, _apply_plot_nan(out[:, BETA_SLICE.start], g.x, g.y, zones=g.zones), color=clr, linewidth=1.2)
            ax.plot(g.s, betax_ic, "k--", linewidth=0.8, alpha=0.7)
            ax.set_title(ttl, fontsize=9); ax.set_xlabel("s (along line through punctures)"); ax.grid(True, alpha=0.3)
            if col == 0:
                ax.set_ylabel(r"$\beta^x$", fontsize=9)
            _clamp_ylim(ax)

            # Row 2: Theta (Hamiltonian-constraint violation)
            ax = axes[2, col]
            ax.plot(g.s, _apply_plot_nan(out[:, THETA_IDX], g.x, g.y, zones=g.zones), color=clr, linewidth=1.2,
                    label=r"$\Theta$" if col == 0 else None)
            ax.axhline(0.0, color="k", linestyle="--", linewidth=0.8, alpha=0.7)
            ax.set_title(ttl, fontsize=9); ax.set_xlabel("s (along line through punctures)"); ax.grid(True, alpha=0.3)
            if col == 0:
                ax.set_ylabel(r"$\Theta$", fontsize=9)
                ax.legend(fontsize=7)
            _clamp_ylim(ax)

            # Row 3: Z^x (momentum-constraint violation)
            ax = axes[3, col]
            ax.plot(g.s, _apply_plot_nan(zv[:, 0], g.x, g.y, zones=g.zones), color=clr, linewidth=1.2,
                    label=r"$Z^x$" if col == 0 else None)
            ax.axhline(0.0, color="k", linestyle="--", linewidth=0.8, alpha=0.7)
            ax.set_title(ttl, fontsize=9); ax.set_xlabel("s (along line through punctures)")
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.set_ylabel(r"$Z^x$", fontsize=9)
                ax.legend(fontsize=7)
            _clamp_ylim(ax)

        fig.tight_layout()
        _atomic_savefig(fig,
            os.path.join(file_path, f"gauge_metric_w{window_idx}_step{step:06d}.png"),
            dpi=100, bbox_inches="tight")
        plt.close(fig)

    def save_native_fields_snapshot(p, step):
        """4-row x 5-col: chi, Khat, alpha, beta^x along the 1-D cut through the punctures --
        the NATIVE evolved CCZ4 fields read directly from the network output
        (no adm_to_g4 reconstruction/blending), as opposed to the derived
        g^{tt}/g^{xx}-style quantities used elsewhere. Solid = network
        value, dashed = t=0 IC value (Khat_ic = K_ic since Theta_ic=0)."""
        snap_times = np.linspace(window_start, window_end, 5)
        colors_t   = plt.cm.plasma(np.linspace(0.0, 0.85, 5))

        fig, axes = plt.subplots(4, 5, figsize=(20, 12))
        fig.suptitle(
            f"Window {window_idx}  —  native fields — step {step:06d}\n"
            r"solid = network, dashed = t=0 IC",
            fontsize=11)

        rows = [
            (0, CHI_IDX,           r"$\chi$"),
            (1, KHAT_IDX,          r"$\hat{K}$"),
            (2, ALPHA_IDX,         r"$\alpha$"),
            (3, BETA_SLICE.start,  r"$\beta^x$"),
        ]

        for col, t_val in enumerate(snap_times):
            g = _snap_grid_at(float(t_val))
            t_in = jnp.full((_nx_snap, 1), float(t_val), dtype=jnp.float64)
            out  = np.asarray(full_output(
                train_apply_fn, p, t_in, g.x_in, g.y_in, g.z_in, _ws_snap, _we_snap))

            clr = colors_t[col]
            ttl = f"$t={t_val:.4f}$"

            for row, idx, label in rows:
                ax = axes[row, col]
                ax.plot(g.s, _apply_plot_nan(out[:, idx], g.x, g.y, zones=g.zones), color=clr, linewidth=1.2)
                ax.plot(g.s, g.ic()[:, idx], "k--", linewidth=0.8, alpha=0.7)
                ax.set_title(ttl, fontsize=9); ax.set_xlabel("s (along line through punctures)"); ax.grid(True, alpha=0.3)
                if col == 0:
                    ax.set_ylabel(label, fontsize=9)
                    if row == 0:
                        ax.plot([], [], "k--", linewidth=0.8, label="IC data")
                        ax.legend(fontsize=7)
                _clamp_ylim(ax)

        fig.tight_layout()
        _atomic_savefig(fig,
            os.path.join(file_path, f"native_fields_w{window_idx}_step{step:06d}.png"),
            dpi=100, bbox_inches="tight")
        plt.close(fig)

    def _render_2d_slice(p, grid, out_path, extra_title):
        """2D (x,y) slice at z=0, t=window_end of alpha/beta^x/g_xx/|H| over
        `grid` (a _Snap2DGrid) -- this window's own just-trained state.
        Shared by save_2d_slice_snapshot's full-domain and zoomed-in
        (--snapshot_2d_zoom_halfwidth) calls below, which differ only in
        which grid/output path they pass."""
        n_pts = grid.Xg.size
        # Mask zones at the punctures' window_END positions (this slice is at
        # t=window_end): the module-global zones are set once at window_start
        # and the punctures drift ~0.1-0.2 units per window, more than the
        # ~0.06 mask radius, so they would no longer cover them here.
        zones_end, _ = _display_mask_zones_for(_star_centers_at(float(window_end)))
        t_in  = jnp.full((n_pts, 1), float(window_end), dtype=jnp.float64)
        x_in2 = jnp.asarray(grid.xr).reshape(-1, 1)
        y_in2 = jnp.asarray(grid.yr).reshape(-1, 1)
        z_in2 = jnp.asarray(grid.zr).reshape(-1, 1)
        out   = np.asarray(full_output(train_apply_fn, p, t_in, x_in2, y_in2, z_in2, _ws_snap, _we_snap))
        alpha_2d, betax_2d, gxx_2d = _bh_adm_from_out(out)
        H_raw = np.asarray(hamiltonian_constraint(train_apply_fn, p, t_in, x_in2, y_in2, z_in2, _ws_snap, _we_snap))
        H_abs_2d = np.where(H_raw != 0.0, np.abs(H_raw), np.nan)

        ic_alpha_2d, ic_betax_2d, ic_gxx_2d = _bh_adm_from_out(grid.ic_shared)
        pinn_cols = [alpha_2d, betax_2d, gxx_2d, H_abs_2d]
        ic_cols   = [ic_alpha_2d, ic_betax_2d, ic_gxx_2d, np.zeros(n_pts)]

        extent = [grid.Xg[0, 0], grid.Xg[0, -1], grid.Yg[0, 0], grid.Yg[-1, 0]]
        fig2, axes2 = plt.subplots(2, len(_Q_LABELS_2D), figsize=(5 * len(_Q_LABELS_2D), 10),
                                    constrained_layout=True)
        for col, (qname, pinn_data, ic_data) in enumerate(zip(_Q_LABELS_2D, pinn_cols, ic_cols)):
            for row, (img, row_label) in enumerate([(pinn_data, "PINN"), (ic_data, "IC (t=0)")]):
                ax = axes2[row, col]
                # _apply_plot_nan (not _mask_divergent): this snapshot is a
                # single time (window_end), so geometric masking is valid --
                # with zones built for THAT time's puncture positions (see
                # zones_end above) -- unlike the end-of-run block, which spans
                # multiple times and must use the position-agnostic
                # _mask_divergent instead.
                _img = _apply_plot_nan(img, grid.xr, grid.yr, zones=zones_end).reshape(grid.nxy, grid.nxy).copy()
                _img[grid.outside_domain] = np.nan
                im = ax.imshow(_img, origin="lower", extent=extent, aspect="equal")
                ax.set_facecolor("0.85")
                fig2.colorbar(im, ax=ax)
                ax.set_title(f"{row_label}  {qname}", fontsize=10)
                ax.set_xlabel("x"); ax.set_ylabel("y")
        fig2.suptitle(f"Window {window_idx}  —  2D slice $z=0$, $t={window_end:.4f}${extra_title}", fontsize=16)
        _atomic_savefig(fig2, out_path, dpi=120, bbox_inches="tight")
        plt.close(fig2)

    def save_2d_slice_snapshot(p):
        """Two 2D (x,y) slices at z=0, t=window_end: the full simulation
        domain (diagnostics_2D_w{window}.png) and a zoomed-in
        --snapshot_2d_zoom_halfwidth box around the binary
        (diagnostics_2D_zoom_w{window}.png) -- the full-domain view dilutes
        the near-binary dynamics across a much larger plot area than the
        punctures actually occupy, so the zoomed one (same point COUNT
        over a much smaller area -- see _snap_grid_2d_zoom -- so also
        higher-resolution, not just cropped) is what actually shows the BH
        dynamics clearly. Both complement the four 1D line-cut snapshots
        above, all of which only sample ALONG the line through the
        puncture centers (see _snap_cut_line): a quasi-circular orbit
        (--BL_momenta_y != 0) can develop off-that-line structure (e.g. an
        asymmetric shift/field distortion perpendicular to the
        instantaneous line joining the punctures) that no 1D cut, however
        well it tracks the punctures, can ever show. Called once at the
        true end of each window's training (not on the snapshot_save_every
        cadence the 1D snapshots use above) -- a full x-y grid + |H| (needs
        second derivatives) is materially more expensive per call, so 2*
        N_window calls total (two per window) instead of N_window*(steps/
        snapshot_save_every) keeps the added cost bounded and comparable to
        the existing end-of-run '2D SLICES' block's fixed 3 calls (see
        there), rather than multiplying it by the snapshot cadence."""
        _render_2d_slice(p, _snap_grid_2d_full, os.path.join(file_path, f"diagnostics_2D_w{window_idx}.png"), "")
        _render_2d_slice(p, _snap_grid_2d_zoom, os.path.join(file_path, f"diagnostics_2D_zoom_w{window_idx}.png"),
                          f" (zoom $\\pm{snapshot_2d_zoom_halfwidth:g}M$)")

    loss_plot_path = os.path.join(file_path, f"loss_window{window_idx}.png")
    loss_csv_path  = os.path.join(file_path, f"loss_window{window_idx}.csv")
    loss_history   = []
    _csv_written   = [False]

    _OPT_STYLE = {"adam": ("Adam", "#eda100"), "bfgs": ("QN (SSBroyden)", "#2a78d6"),
                  "dsgnar": ("DSGNAR", "#1baf7a")}

    def _shade_optimizer_spans(ax, label_top=True):
        """Shade each contiguous stretch of loss_history by the optimizer that produced it
        and name it at the top of the panel, so a loss curve shows which part belongs to
        which optimizer (Adam, QN, DSGNAR -- in a DSGNAR run before or after the QN restarts)."""
        spans, cur = [], None
        for r in loss_history:
            o = r["optimizer"]
            if cur is None or cur[0] != o:
                cur = [o, r["step"], r["step"]]
                spans.append(cur)
            else:
                cur[2] = r["step"]
        # step 0 is logged before any optimizer runs (named "bfgs" when there is no Adam):
        # a single-point leading span is the starting point, not an optimizer stretch
        if len(spans) > 1 and spans[0][1] == spans[0][2]:
            spans[1][1] = spans[0][1]
            spans.pop(0)
        for k, (o, a, b) in enumerate(spans):
            if k > 0:
                a = spans[k - 1][2]            # starts where the previous optimizer's last step ended
            name, col = _OPT_STYLE.get(o, (o, "#888888"))
            ax.axvspan(a, b, color=col, alpha=0.10, lw=0)
            if label_top and len(spans) > 1:
                ax.text(0.5 * (a + b), 0.97, name, transform=ax.get_xaxis_transform(),
                        ha="center", va="top", fontsize=8, color="#444444")

    def save_loss_plot():
        if len(loss_history) < 2:
            return
        steps  = [r["step"] for r in loss_history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))
        fig.suptitle(f"Window {window_idx}", fontsize=13)

        # Left panel: major loss components
        for fk, label, color in [
            ("loss_total",    "Total",    "black"),
            ("loss_einstein", "Einstein", "tab:orange"),
            ("loss_gauge",    "Gauge",    "tab:purple"),
            ("loss_initial",  "IC (sum)", "tab:green"),
            ("loss_boundary", "BC",       "tab:red"),
        ]:
            if ic_mode == "hard" and fk == "loss_initial":
                # Hard IC enforcement makes this residual identically ~0 (machine
                # precision, e.g. 1e-32) by construction -- not a trained quantity,
                # so it just squashes the log-scale axis without carrying information.
                continue
            vals = [r[fk] for r in loss_history]
            if any(v > 0 for v in vals):
                ax1.semilogy(steps, vals, label=label,
                             linewidth=1.5 if fk == "loss_total" else 1.0,
                             linestyle="-" if fk == "loss_total" else "--", color=color)
        _shade_optimizer_spans(ax1)
        ax1.set_xlabel("Step"); ax1.set_ylabel("Loss")
        ax1.set_title("Major components"); ax1.legend(fontsize=8, ncol=2)
        ax1.grid(True, which="both", alpha=0.3)

        # Right panel: IC sub-losses + gauge, with grad-norm on twin axis
        for fk, label, color, ls in [
            ("loss_g0",    r"$g$ val",          "tab:orange", "--"),
            ("loss_gt0",   r"$\partial_t g^{ij}$", "tab:red","--"),
            ("loss_gauge", "Gauge",             "tab:purple", "-"),
            ("loss_pos",   "Positivity",        "tab:brown",  "--"),
        ]:
            if ic_mode == "hard" and fk in ("loss_g0", "loss_gt0"):
                # Same as the left panel: hard-enforced, ~0 by construction.
                continue
            vals = [r[fk] for r in loss_history]
            if any(v > 0 for v in vals):
                ax2.semilogy(steps, vals, label=label, linewidth=1.0, linestyle=ls, color=color)
        gnorms = [r.get("grad_norm", float("nan")) for r in loss_history]
        has_term_gn = any(not np.isnan(r.get("gn_ein", float("nan"))) for r in loss_history)
        if any(not np.isnan(g) for g in gnorms) or has_term_gn:
            ax2b = ax2.twinx()
            if any(not np.isnan(g) for g in gnorms):
                ax2b.semilogy(steps, gnorms, color="dimgray", linewidth=0.8,
                              linestyle=":", alpha=0.6, label=r"$\|\nabla\mathcal{L}\|$ (total)")
            for fk, label, color in [
                ("gn_ein",   r"$\|\nabla\mathrm{Ein}\|$",   "tab:orange"),
                ("gn_gauge", r"$\|\nabla\mathrm{Gauge}\|$", "tab:purple"),
                ("gn_bc",    r"$\|\nabla\mathrm{BC}\|$",    "tab:red"),
                ("gn_ic",    r"$\|\nabla\mathrm{IC}\|$",    "tab:green"),
                ("gn_pos",   r"$\|\nabla\mathrm{Pos}\|$",   "tab:brown"),
            ]:
                vals = [r.get(fk, float("nan")) for r in loss_history]
                if any(not np.isnan(v) for v in vals):
                    ax2b.semilogy(steps, vals, label=label, linewidth=1.0,
                                  linestyle="-.", color=color, marker=".", markersize=3)
            ax2b.set_ylabel(r"Gradient norm", color="dimgray", fontsize=9)
            ax2b.tick_params(axis="y", labelcolor="dimgray")
            ax2b.legend(loc="lower right", fontsize=7, ncol=2)
        _shade_optimizer_spans(ax2, label_top=False)
        ax2.set_xlabel("Step"); ax2.set_ylabel("Loss")
        ax2.set_title("IC sub-losses & gauge"); ax2.legend(fontsize=8, ncol=2)
        ax2.grid(True, which="both", alpha=0.3)

        fig.tight_layout()
        _atomic_savefig(fig, loss_plot_path, dpi=120)
        plt.close(fig)

    _loss_plot_count = [-1]  # [GPUS] so the first save (step 0) is always drawn
    def append_loss_history(step, optimizer_name, all_loss_it, params_snap=None, grad_norm=None,
                             term_grad_norms=None):
        tgn = term_grad_norms or {}
        record = {
            "window_idx": window_idx, "step": int(step),
            "optimizer": optimizer_name,
            "loss_total":    float(all_loss_it["Loss"]),
            "loss_einstein": float(all_loss_it["loss_Einstein"]),
            "loss_boundary": float(all_loss_it["loss_BC"]),
            "loss_gauge":    float(all_loss_it["loss_gauge"]),
            "loss_initial":  float(all_loss_it["loss_IC"]),
            "loss_g0":  float(all_loss_it["loss_g0"]),
            "loss_ic_log": float(all_loss_it["loss_ic_log"]),
            "loss_gt0": float(all_loss_it["loss_gt0"]),
            "loss_pos": float(all_loss_it["loss_pos"]),
            "loss_constraint": float(all_loss_it.get("loss_constraint", 0.0)),
            "grad_norm": float(grad_norm) if grad_norm is not None else float("nan"),
            "gn_ein":   float(tgn["gn_ein"])   if "gn_ein"   in tgn else float("nan"),
            "gn_gauge": float(tgn["gn_gauge"]) if "gn_gauge" in tgn else float("nan"),
            "gn_bc":    float(tgn["gn_bc"])    if "gn_bc"    in tgn else float("nan"),
            "gn_ic":    float(tgn["gn_ic"])    if "gn_ic"    in tgn else float("nan"),
            "gn_pos":   float(tgn["gn_pos"])   if "gn_pos"   in tgn else float("nan"),
        }
        loss_history.append(record)
        # "w" (truncate) for the FIRST write of this train_window call, "a"
        # for every one after -- NOT unconditional "a": if this window was
        # previously attempted and crashed mid-training (no per-window
        # checkpoint got written, so --resume_checkpoint retrains it from
        # scratch), loss_window{window_idx}.csv from that crashed attempt
        # is still on disk, and _csv_written=[False] here has no memory of
        # it (it's reinitialized fresh every train_window call) -- opening
        # in "a" unconditionally would append a brand-new header partway
        # into the old file instead of replacing it, corrupting the CSV
        # (a header line in the middle, overlapping/duplicated step
        # numbers) for every downstream loader that treats it as one clean
        # file. Truncating on the first write of each train_window call
        # discards exactly that stale content while still using cheap
        # append (not a full rewrite) for every subsequent write within
        # this same call.
        with open(loss_csv_path, "w" if not _csv_written[0] else "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(record.keys()))
            if not _csv_written[0]:
                w.writeheader(); _csv_written[0] = True
            w.writerow(record)
        _tq = time.perf_counter()  # [GPUS]
        _loss_plot_count[0] += 1  # [GPUS]
        if _loss_plot_count[0] % loss_plot_every == 0:  # [GPUS]
            save_loss_plot()
        _tq = _prof_add("loss_plot", _tq)  # [GPUS]
        if params_snap is not None and step % snapshot_save_every == 0:
            save_field_snapshot(params_snap, step)
            save_residual_snapshot(params_snap, step)
            save_gauge_metric_snapshot(params_snap, step)
            save_native_fields_snapshot(params_snap, step)
        _prof_add("snapshots", _tq)  # [GPUS]

    # --- Adam -----------------------------------------------------------------
    # optimizer/update_model: module-level now (see their definition, right
    # after dummy_params) -- only opt_state (this window's own params) is
    # built here.
    opt_state  = optimizer.init(params)
    _params_pre_adam = params

    next_loss_step = loss_save_every

    def _log_step0(optimizer_name):
        """Log step 0 before the optimizer loop starts, so the loss plot
        has an initial point -- shared by the EPOCHS_ADAM>0 and ==0 cases
        below, which previously called this in lockstep with only the
        optimizer-name string differing."""
        all_loss_it_0, _ = apply_model(
            train_apply_fn, params,
            lam_bc, lamg0, w_pos, traj_tt, traj_xx,
            lam_einstein, lam_gauge, r_eps,
            *train_data)
        tgn_0 = per_term_grad_norms(
            train_apply_fn, params,
            lam_bc, lamg0, w_pos, traj_tt, traj_xx,
            lam_einstein, lam_gauge, r_eps,
            *train_data)
        append_loss_history(0, optimizer_name, all_loss_it_0, params_snap=_snap_params(params),
                            term_grad_norms=tgn_0)

    if EPOCHS_ADAM > 0:
        _log_step0("adam")

        pbar = tqdm(range(1, EPOCHS_ADAM + 1), desc=f"Adam w{window_idx+1}")
        # Cosmetic-only tqdm postfix cadence: refreshing it every step forces
        # a float()/host sync on the gradient norm and 5 loss terms every
        # single Adam step purely to redraw the progress bar, blocking async
        # dispatch between steps for no benefit -- none of these floats feed
        # back into training, and the ACTUAL loss-history save below already
        # has its own, separately correct throttle (next_loss_step). Refresh
        # at this stride (or whenever a save happens anyway, or on the very
        # first/last step so the bar isn't blank) instead; the values saved
        # to loss history are computed at exactly the same steps as before.
        _PBAR_REFRESH_EVERY = 10
        for e in pbar:
            if (e + 1) % Nchange == 0:
                _resample_count += 1
                if chi_bg_mode == "moving":
                    _refresh_traj(params)   # before the resample, so --follow_puncture_sampling uses it
                train_data = _resample_train_data()
            ramp = float(min(e / N_ramp, 1.0)) if N_ramp > 0 else 1.0
            all_loss_it, grads = apply_model(
                train_apply_fn, params,
                lam_bc, lamg0, w_pos, traj_tt, traj_xx,
                lam_einstein*ramp, lam_gauge, r_eps,
                *train_data)
            _refresh_pbar = (e % _PBAR_REFRESH_EVERY == 0 or e == 1
                              or e == EPOCHS_ADAM or e >= next_loss_step)
            if _refresh_pbar:
                flat_g, _ = flatten_util.ravel_pytree(grads)
                gn = float(jnp.linalg.norm(flat_g))
            if e >= next_loss_step:
                tgn = per_term_grad_norms(
                    train_apply_fn, params,
                    lam_bc, lamg0, w_pos, traj_tt, traj_xx,
                    lam_einstein*ramp, lam_gauge, r_eps,
                    *train_data)
                append_loss_history(next_loss_step, "adam", all_loss_it,
                                    params_snap=_snap_params(params), grad_norm=gn,
                                    term_grad_norms=tgn)
                next_loss_step += loss_save_every
            params, opt_state = update_model(params, opt_state, grads)
            if _refresh_pbar:
                pbar.set_postfix({
                    "Loss":  f"{float(all_loss_it['Loss']):.2e}",
                    "Ein":   f"{float(all_loss_it['loss_Einstein']):.2e}",
                    "Gauge": f"{float(all_loss_it['loss_gauge']):.2e}",
                    "IC":    f"{float(all_loss_it['loss_IC']):.2e}",
                    "Pos":   f"{float(all_loss_it['loss_pos']):.2e}",
                    "||g||": f"{gn:.2e}",
                    "ramp":  f"{ramp:.2f}",
                })
    else:
        # EPOCHS_ADAM == 0: log step 0 before BFGS so the plot has an initial point.
        _log_step0("bfgs")

    del opt_state
    import gc; gc.collect()
    # See _NEEDS_PERWINDOW_CACHE_CLEAR's module-level definition for why
    # this is (or isn't) still needed.
    if _NEEDS_PERWINDOW_CACHE_CLEAR:
        jax.clear_caches()

    # --- BFGS -----------------------------------------------------------------
    # unflatten_func: see _get_stable_unflatten -- reused across windows
    # (same params structure every window) instead of a fresh closure every
    # time, so bfgs_step's compiled executable can be reused too. Still
    # calling ravel_pytree fresh for initial_weights itself: the flat VALUES
    # genuinely are window-specific (this window's own params), only the
    # unflattening LOGIC is shared.
    if EPOCHS_ADAM > 0 and not all(bool(jnp.all(jnp.isfinite(leaf)))
                                   for leaf in jax.tree_util.tree_leaves(params)):
        # Adam diverged: seeding BFGS from NaN weights would make its NaN guard
        # "restore" the same NaN weights. Fall back to the pre-Adam params.
        print(f"\n  [Adam non-finite] window {window_idx}: parameters went NaN/Inf during Adam; "
              f"restoring the pre-Adam parameters before BFGS.")
        params = _params_pre_adam
    # --- DSGNAR (--dsgnar_steps > 0) ---------------------------------------------
    # step_base: the loss-history step after which the QN restarts count; it is
    # EPOCHS_ADAM unless DSGNAR runs first and adds its iterations.
    step_base = EPOCHS_ADAM

    def _dsgnar_phase(params_in):
        """DSGNAR on this window: --dsgnar_cycles cycles, each a fresh sample and a
        fresh trust region (Stationary's train.py dsgnar_qn_phase). Returns
        (params, iterations, final loss)."""
        nonlocal train_data, _resample_count, next_loss_step, step_base
        import dsgnar as _dsg_mod
        flat, unflat = flatten_util.ravel_pytree(params_in)
        ramp = float(min(step_base / N_ramp, 1.0)) if N_ramp > 0 else 1.0
        cur = [flat]

        def make_obj(p_flat):
            a_ = (lam_bc, lamg0, w_pos, traj_tt, traj_xx, lam_einstein * ramp, lam_gauge, r_eps, *train_data)
            return _DsgnarObjective(train_apply_fn, unflat, a_, p_flat, row_batch=int(args.dsgnar_row_batch))

        def new_sample():
            nonlocal train_data, _resample_count
            _resample_count += 1
            if chi_bg_mode == "moving":
                _refresh_traj(unflat(cur[0]))
            train_data = _resample_train_data()
            return make_obj(cur[0])

        def on_step(step, loss_now, flat_now):
            cur[0] = flat_now
            i = step - step_base
            if i > 0 and (i % max(1, int(args.dsgnar_log_every)) == 0):
                p_now = unflat(flat_now)
                all_loss_it, _ = apply_model(train_apply_fn, p_now, lam_bc, lamg0, w_pos, traj_tt, traj_xx,
                                             lam_einstein * ramp, lam_gauge, r_eps, *train_data)
                append_loss_history(step, "dsgnar", all_loss_it, params_snap=_snap_params(p_now))

        cycles = max(1, int(args.dsgnar_cycles))
        done, loss_end, t0_d = 0, float("nan"), time.perf_counter()
        obj = make_obj(flat)
        for cyc in range(cycles):
            if cyc:
                obj = new_sample()           # fresh sample; dsgnar_phase restarts the trust region
            cfg_c = _dsgnar_cfg(seed=SEED + 7919 * window_idx + 104729 * cyc)
            flat_new, hist, info = _dsg_mod.dsgnar_phase(
                obj, flat, cfg_c, verbose=True, callback=on_step, step_offset=step_base,
                resample=(new_sample if cfg_c.resample_every > 0 else None))
            if not bool(jnp.all(jnp.isfinite(flat_new))):
                print(f"  [DSGNAR] window {window_idx}, cycle {cyc + 1}: non-finite parameters; "
                      f"keeping the cycle's starting point and ending DSGNAR for this window.")
                break
            flat, cur[0] = flat_new, flat_new
            step_base += int(info["iterations"])
            done += int(info["iterations"])
            loss_end = float(info["loss_final"])
            print(f"  [DSGNAR] window {window_idx}, cycle {cyc + 1}/{cycles}: {info['iterations']} its, "
                  f"s={info['sketch']}, {info['accepted']} accepted / {info['rejected']} rejected, "
                  f"loss {hist[0]['loss']:.3e} -> {loss_end:.3e}, stopped: {info['stopped']} "
                  f"({info['wall']:.0f} s; JVPs {info['time_jvp']:.0f} s, probes {info['time_probe']:.0f} s)")
        next_loss_step = max(next_loss_step, (step_base // loss_save_every + 1) * loss_save_every)
        print(f"  [DSGNAR] window {window_idx}: {done} iterations in {time.perf_counter() - t0_d:.0f} s")
        return unflat(flat), done, loss_end

    if dsgnar_steps > 0 and args.dsgnar_order == "before_qn":
        params, _n_dsg, _ = _dsgnar_phase(params)

    initial_weights, _ = flatten_util.ravel_pytree(params)
    unflatten_func = _get_stable_unflatten(params)
    H0       = jnp.eye(len(initial_weights), dtype=jnp.float64)
    bfgs_key = jax.random.PRNGKey(SEED + 1000 + window_idx)

    static_options = {
        "maxiter": Nchange, "gtol": qn_gtol, "update_method": qn_update,
        "initial_scale": bool(h0_scaling), "alphamin": 1e-15, "max_iter": 100,
        "line_search": line_search, "c2": wolfe_c2,
    }
    if qn_scale_invariant:          # absent key = off: bitwise the pre-2026-10-19 call
        static_options["scale_invariant"] = True
    if qn_update == "score":
        static_options.update({"score_eta": score_eta, "score_mu_min": score_mu_min,
                               "score_mu_max": score_mu_max})
    static_options_tuple = tuple(static_options.items())
    qn_agg = {"n_iter": 0, "n_update": 0, "n_skip": 0, "n_first": 0, "mu_sum": 0.0, "last": {}}

    num_outer = max(1, EPOCHS_QN // Nchange) if EPOCHS_QN > 0 else 0
    loss_list, epochs_list, effective_steps = [], [], 0
    es_bfgs = EarlyStopping(patience=patience_bfgs, min_delta=min_delta, min_loss=es_min_loss)
    # If the window stagnates, try a fresh (identity) Hessian before giving
    # up -- the SSBroyden2 update can wander into a poorly-conditioned
    # direction that repeatedly collapses the line search to a
    # negligible step (unlike gowdy.py's lax.while_loop BFGS, this Python-loop
    # minimize() always accepts whatever step it lands on and keeps
    # incrementing nit, so it won't freeze outright -- but it can still burn
    # the whole window's budget taking steps too small to matter). A
    # steepest-descent direction (H0 = identity) is always a valid descent
    # direction, so it can break out of a bad quasi-Newton trajectory even
    # when the physics itself is fine. Each retry gets a shorter probation
    # (RESET_PATIENCE) than the original --patience_bfgs, so a reset that
    # doesn't help is detected quickly rather than re-burning the full budget.
    hessian_resets_left = 2
    RESET_PATIENCE = max(20, patience_bfgs // 10)

    # NaN/Inf guard. Once weights go non-finite nothing downstream can repair
    # them: every later BFGS call, resample and snapshot runs on NaN, so a
    # diverged window used to burn its whole remaining budget doing nothing
    # (seen on a real cluster run: ~1.5 h of walltime on all-NaN BFGS, then
    # the job hit its time limit). A non-finite outer iteration now restores
    # the last finite weights, resets the Hessian to identity (its history
    # may be what blew up), draws a fresh batch (a bad collocation point can
    # also produce a NaN) and tries again; after MAX_NAN_RETRIES consecutive
    # failures the window is ended with the last finite weights. A retry
    # consumes one of the window's num_outer iterations.
    MAX_NAN_RETRIES = 3
    nan_retries     = 0
    last_good_w     = initial_weights
    # Excursion guard (--excursion_ratio/_abs/_retries): a FINITE loss jump
    # that the NaN guard cannot see (on the boson-star runs, windows that
    # later failed started with restart-to-restart ratios of 500-1000; here
    # also chi crossing zero under --chi_floor, see NOTES.md). A restart
    # that trips it is discarded: the weights stay at their pre-restart
    # values, the inverse Hessian is reset and the batch redrawn at the
    # same sampling stage. Bounded per window; a retry consumes one of the
    # num_outer iterations.
    exc_retries = 0
    # Blow-up guard (--blowup_factor > 0): best restart of this window so far.
    best_restart_loss = math.inf
    best_restart_w    = initial_weights
    blowups           = 0

    qn_loss_scale = 1.0
    if qn_loss_norm and num_outer > 0:
        qn_loss_scale = float(bfgs_loss_and_grad(
            initial_weights, train_apply_fn, unflatten_func,
            lam_bc, lamg0, w_pos, traj_tt, traj_xx, lam_einstein, lam_gauge, r_eps, *train_data))
        if not (np.isfinite(qn_loss_scale) and qn_loss_scale > 0.0):
            qn_loss_scale = 1.0
        print(f"  --qn_loss_norm: QN minimizes loss / {qn_loss_scale:.3e} in window {window_idx}")

    if num_outer > 0:
        pbar = tqdm(range(num_outer), desc=f"BFGS w{window_idx+1}")
        for outer_idx in pbar:
            ramp_bfgs = float(min((step_base + effective_steps) / N_ramp, 1.0)) if N_ramp > 0 else 1.0
            _tp = _tp_restart = time.perf_counter()  # [GPUS]
            new_w, new_H0, bfgs_key, loss_val, nit = bfgs_step(
                initial_weights, H0, bfgs_key,
                train_data,
                lam_bc, lamg0, w_pos, traj_tt, traj_xx,
                lam_einstein*ramp_bfgs, lam_gauge, r_eps,
                train_apply_fn, unflatten_func, static_options_tuple,
                loss_scale=qn_loss_scale,
            )
            _tp = _prof_add("bfgs_step", _tp, new_w, new_H0)  # [GPUS]
            if not (bool(jnp.isfinite(loss_val)) and bool(jnp.all(jnp.isfinite(new_w)))):
                nan_retries += 1
                print(f"\n  [BFGS non-finite] window {window_idx}, outer iteration {outer_idx}: "
                      f"loss/weights went NaN/Inf; restoring the last finite weights, resetting the "
                      f"Hessian and resampling (failure {nan_retries}/{MAX_NAN_RETRIES}).")
                initial_weights = last_good_w
                H0              = jnp.eye(len(initial_weights), dtype=jnp.float64)
                _resample_count += 1
                train_data = _resample_train_data()
                if nan_retries >= MAX_NAN_RETRIES:
                    print(f"  [BFGS non-finite] window {window_idx}: {MAX_NAN_RETRIES} consecutive "
                          f"non-finite iterations -- ending this window with the last finite weights.")
                    break
                continue
            nan_retries = 0
            excursion = None
            # Inactive while a --N_ramp ramp is still raising the Einstein
            # weight between restarts (the loss then grows legitimately).
            if exc_retries < excursion_retries and ramp_bfgs >= 1.0:
                prev_ok = loss_list[-1] if loss_list else None
                if prev_ok is not None:
                    if excursion_ratio > 0 and float(loss_val) > excursion_ratio * prev_ok:
                        excursion = (f"loss {float(loss_val):.3e} > {excursion_ratio:g} x the previous "
                                     f"restart's {prev_ok:.3e}")
                elif prev_state is not None and excursion_abs > 0:
                    _prev_final = float((prev_state.get("loss_list") or [0.0])[-1])
                    if not np.isfinite(_prev_final):
                        _prev_final = 0.0
                    _thr = max(excursion_abs, excursion_ratio * _prev_final)
                    if float(loss_val) > _thr:
                        excursion = f"first-restart loss {float(loss_val):.3e} > {_thr:.3g}"
            if excursion is not None:
                exc_retries += 1
                print(f"\n  [excursion guard] window {window_idx}, outer iteration {outer_idx}: {excursion}; "
                      f"discarding this restart, resetting the Hessian and resampling "
                      f"(retry {exc_retries}/{excursion_retries}).")
                H0 = jnp.eye(len(initial_weights), dtype=jnp.float64)
                train_data = _resample_train_data()      # same sampling stage: no _resample_count bump
                continue
            # Blow-up guard (backstop once the excursion retries are used up): roll
            # back to the best restart of the window, not just the previous one.
            if blowup_factor > 0 and math.isfinite(best_restart_loss) \
                    and float(loss_val) > blowup_factor * best_restart_loss:
                blowups += 1
                print(f"\n  [BFGS blow-up] window {window_idx}, outer iteration {outer_idx}: loss "
                      f"{float(loss_val):.3e} > {blowup_factor:g} x best {best_restart_loss:.3e}; rolling back "
                      f"to the best weights, resetting the Hessian and resampling "
                      f"(rollback {blowups}/{blowup_max}).")
                initial_weights = best_restart_w
                last_good_w     = best_restart_w
                H0              = jnp.eye(len(initial_weights), dtype=jnp.float64)
                _resample_count += 1
                train_data = _resample_train_data()
                if blowups >= blowup_max:
                    print(f"  [BFGS blow-up] window {window_idx}: {blowup_max} rollbacks -- ending this "
                          f"window with the best weights (loss {best_restart_loss:.3e}).")
                    break
                continue
            if float(loss_val) < best_restart_loss:
                best_restart_loss = float(loss_val)
                best_restart_w    = new_w
            last_good_w = new_w
            if not bool(jnp.all(jnp.isfinite(new_H0))):
                new_H0 = jnp.eye(len(new_w), dtype=jnp.float64)
            effective_steps += int(nit)
            initial_weights  = new_w
            H0               = new_H0
            if _QN_LAST_STATS:
                for _k in ("n_iter", "n_update", "n_skip", "n_first"):
                    qn_agg[_k] += int(_QN_LAST_STATS.get(_k, 0))
                qn_agg["mu_sum"] += float(_QN_LAST_STATS.get("mu_mean", 0.0)) * int(_QN_LAST_STATS.get("n_iter", 0))
                qn_agg["last"] = dict(_QN_LAST_STATS)
            epochs_list.append(effective_steps)
            loss_list.append(float(loss_val))

            params_current = unflatten_func(initial_weights)
            if chi_bg_mode == "moving":
                _refresh_traj(params_current)
            _tp = _prof_add("traj_refresh", _tp, traj_xx)  # [GPUS]
            all_loss_it, _ = apply_model(
                train_apply_fn, params_current,
                lam_bc, lamg0, w_pos, traj_tt, traj_xx,
                lam_einstein*ramp_bfgs, lam_gauge, r_eps,
                *train_data)
            _tp = _prof_add("apply_model", _tp, all_loss_it)  # [GPUS]

            current_step = step_base + effective_steps
            if current_step >= next_loss_step:
                tgn = per_term_grad_norms(
                    train_apply_fn, params_current,
                    lam_bc, lamg0, w_pos, traj_tt, traj_xx,
                    lam_einstein*ramp_bfgs, lam_gauge, r_eps,
                    *train_data)
            while current_step >= next_loss_step:
                append_loss_history(next_loss_step, "bfgs", all_loss_it,
                                    params_snap=_snap_params(params_current),
                                    term_grad_norms=tgn)
                next_loss_step += loss_save_every
            _tp = _prof_add("loss_log", _tp)  # [GPUS]

            if not qn_fixed_batch:
                _resample_count += 1
                train_data = _resample_train_data()
            _tp = _prof_add("resample", _tp, train_data)  # [GPUS]
            _prof_end_restart(_tp_restart)  # [GPUS]
            pbar.set_postfix({
                "Loss":  f"{loss_val:.2e}",
                "Ein":   f"{float(all_loss_it['loss_Einstein']):.2e}",
                "IC":    f"{float(all_loss_it['loss_IC']):.2e}",
                "BC":    f"{float(all_loss_it['loss_BC']):.2e}",
                "Pos":   f"{float(all_loss_it['loss_pos']):.2e}",
                "ES":    f"{es_bfgs.counter}/{es_bfgs.patience}" if patience_bfgs > 0 else "off",
                "retry": f"{exc_retries}/{excursion_retries}",
            })
            if es_bfgs.update(float(loss_val)):
                if hessian_resets_left > 0:
                    hessian_resets_left -= 1
                    print(f"\n  [BFGS stall] window {window_idx}: no improvement > {min_delta:.1e} "
                          f"for {es_bfgs.patience} restarts (best={es_bfgs.best_loss:.3e}); "
                          f"resetting Hessian to identity and retrying "
                          f"({hessian_resets_left} reset(s) left, {RESET_PATIENCE}-call probation).")
                    H0 = jnp.eye(len(initial_weights), dtype=jnp.float64)
                    es_bfgs.reset(patience=RESET_PATIENCE)
                    continue
                print(f"\n  [BFGS early stop] window {window_idx}: "
                      f"no improvement > {min_delta:.1e} for {es_bfgs.patience} restarts "
                      f"(best={es_bfgs.best_loss:.3e}"
                      + (f" <= es_min_loss {es_min_loss:.1e}" if es_min_loss > 0 else "")
                      + "); giving up after 2 Hessian reset attempt(s)")
                break

    _prof_window_summary(window_idx)  # [GPUS]
    if loss_plot_every > 1:  # [GPUS] final state of this window's loss plot
        save_loss_plot()  # [GPUS]
    params = unflatten_func(initial_weights)
    if dsgnar_steps > 0 and args.dsgnar_order == "after_qn":
        step_base = step_base + effective_steps
        params, _n_dsg, _l_dsg = _dsgnar_phase(params)
        if _n_dsg and np.isfinite(_l_dsg):
            epochs_list.append(step_base)
            loss_list.append(_l_dsg)
    if snapshot_2d_every > 0 and (window_idx % snapshot_2d_every == 0 or window_idx == N_window - 1):
        save_2d_slice_snapshot(_snap_params(params))

    # ── end-of-window diagnostic summary ──────────────────────────────────────
    if loss_history:
        last = loss_history[-1]
        print(f"\n  ┌── Window {window_idx} final loss breakdown ──────────────────────")
        print(f"  │  Total:    {last['loss_total']:.4e}")
        print(f"  │  Einstein: {last['loss_einstein']:.4e}")
        print(f"  │  Gauge:    {last['loss_gauge']:.4e}")
        print(f"  │  BC:       {last['loss_boundary']:.4e}")
        print(f"  │  Positivity: {last['loss_pos']:.4e}")
        print(f"  │  IC total: {last['loss_initial']:.4e}")
        print(f"  │    g val:        {last['loss_g0']:.4e}")
        print(f"  │    ∂_t g^ij:     {last['loss_gt0']:.4e}")
        gn = last.get("grad_norm", float("nan"))
        if not np.isnan(gn):
            print(f"  │  Grad norm: {gn:.4e}")
        if qn_agg["n_iter"] > 0:
            _ql = qn_agg["last"]
            _qs = (f"  │  QN ({qn_update}): {qn_agg['n_iter']} its, {qn_agg['n_update']} updates, "
                   f"{qn_agg['n_skip']} skipped (y.s<=0), first trial accepted {qn_agg['n_first']}; "
                   f"last restart: decrement {_ql.get('lam_med', float('nan')):.2e}, "
                   f"y.s/s.s {_ql.get('curv_med', float('nan')):.2e}")
            if qn_update == "score":
                _qs += f", mean shift mu {qn_agg['mu_sum'] / qn_agg['n_iter']:.2e}"
            print(_qs)
        print(f"  └────────────────────────────────────────────────────────────")
    # ──────────────────────────────────────────────────────────────────────────

    return {
        "window_idx": window_idx,
        "t_start":    float(window_start),
        "t_end":      float(window_end),
        "params":     jax.tree_util.tree_map(np.asarray, params),
        "ic_mode":    ic_mode,
        "ic_history": ic_history,
        "loss_list":  loss_list,
        "epochs_list":  epochs_list,
        "loss_history": loss_history,
    }, key


# --------------------------------------------------------------------------------
# MAIN TRAINING LOOP
# --------------------------------------------------------------------------------
window_states = []
resume_start  = 0


def _arrays_mismatch(a, b):
    """True if a and b differ -- a plain np.allclose(a, b) raises (or
    silently broadcasts wrong) when a and b have different lengths, e.g.
    resuming a checkpoint saved with a different --n_stars, so shapes must
    be compared first."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return True
    return not np.allclose(a, b)


# Single source of truth for BOTH save_checkpoint's model_config contents
# AND resume_checkpoint's mismatch-check -- previously two independently
# hand-maintained structures (model_config's dict literal and a separate
# _RESUME_CHECKS list), so a field added to only one silently never made it
# into the other: added to model_config only -> never checked on resume;
# added to _RESUME_CHECKS only -> _cfg_get always returns None for it and
# `required=False` rows then just `continue` forever, silently no-op'ing
# the check even for BRAND NEW checkpoints, not just old-format ones (the
# backward-compat case that `required=False` was actually meant for).
# (label, dot-path into model_config, current value, "differs" predicate,
# required). required=True: always checked (present in every checkpoint
# format this file has ever written). required=False: only checked if
# present in the checkpoint, for backward-compat with older checkpoints
# saved before that field was added.
_CHECKPOINT_FIELDS = [
    ("N_LAYERS",       "N_LAYERS",       N_LAYERS,       lambda a, b: a != b,               True),
    ("HIDDEN",         "HIDDEN",         HIDDEN,         lambda a, b: a != b,               True),
    ("ic_mode",        "ic_mode",        ic_mode,        lambda a, b: a != b,               True),
    ("N_window",       "N_window",       N_window,       lambda a, b: a != b,               True),
    ("tfinal",         "domain.tfinal",  tfinal,         lambda a, b: a != b,               True),
    ("R_domain",       "domain.R",       R_domain,       lambda a, b: not np.isclose(a, b), False),
    ("n_stars",        "n_stars",        n_stars,        lambda a, b: a != b,               False),
    ("BL_masses",      "BL_masses",      BL_masses,      _arrays_mismatch,                  False),
    ("BL_momenta",     "BL_momenta",     BL_momenta,     _arrays_mismatch,                  False),
    ("BL_momenta_y",   "BL_momenta_y",   BL_momenta_y,   _arrays_mismatch,                  False),
    ("BL_spins",       "BL_spins",       BL_spins,       _arrays_mismatch,                  False),
    ("chi_bg_mode",    "chi_bg_mode",    chi_bg_mode,    lambda a, b: a != b,               False),
    ("bh_center_mode", "bh_center_mode", bh_center_mode, lambda a, b: a != b,               False),
    ("min_puncture_dist", "min_puncture_dist", min_puncture_dist, lambda a, b: not np.isclose(a, b), False),
    ("min_center_separation", "min_center_separation", min_center_separation, lambda a, b: not np.isclose(a, b), False),
    ("metric_bc",      "metric_bc",      metric_bc_mode, lambda a, b: a != b,               False),
    ("gauge",          "gauge",          gauge_mode,     lambda a, b: a != b,               False),
]


def _set_nested(d, dot_path, value):
    """d[a][b]...[z] = value for dot-path 'a.b...z', creating intermediate
    dicts as needed -- shared by save_checkpoint so model_config's nested
    "domain.tfinal"/"domain.R" entries are written at exactly the paths
    _CHECKPOINT_FIELDS/resume's _cfg_get expect, instead of a hand-written
    nested dict literal that could drift from the dot-paths above."""
    parts = dot_path.split(".")
    for part in parts[:-1]:
        d = d.setdefault(part, {})
    d[parts[-1]] = value


def _atomic_write(path, write_fn):
    """write_fn(tmp_path) should write the file to tmp_path; this then
    atomically (temp file + os.replace, same directory so POSIX-atomic)
    moves it into place at `path`. Shared primitive for every "rewritten
    from scratch every window/step, must never be left truncated/corrupted
    by a crash/OOM/preemption mid-write" output in this file: the
    checkpoint pickle, the puncture-track .txt files, and the diagnostic
    PNGs that accumulate multi-window/multi-step history.

    Temp name inserts ".tmp" BEFORE the real extension (foo.tmp.png), not
    after (foo.png.tmp): matplotlib's savefig infers output FORMAT from
    the file's extension, so a naive path+".tmp" suffix breaks it --
    confirmed directly: fig.savefig("foo.png.tmp") raises "ValueError:
    Format 'tmp' is not supported". Preserving the real extension as the
    suffix keeps format-sniffing working and is harmless for pickle/
    savetxt, which never cared about the extension either way."""
    root, ext = os.path.splitext(path)
    tmp_path = f"{root}.tmp{ext}"
    write_fn(tmp_path)
    os.replace(tmp_path, path)


def _atomic_savefig(fig, path, **kwargs):
    _atomic_write(path, lambda p: fig.savefig(p, **kwargs))


def _build_model_config():
    """model_config dict for save_checkpoint: 'features'/'window_edges' are
    structural (derived from other already-checked values, not independent
    physically-relevant settings), so they're written directly; everything
    else comes from _CHECKPOINT_FIELDS -- the single source of truth also
    used by resume_checkpoint's mismatch-check (see its definition above)."""
    cfg = {"features": feat_sizes, "window_edges": window_edges.tolist()}
    for _label, _key_path, _current, _differs, _required in _CHECKPOINT_FIELDS:
        _set_nested(cfg, _key_path, _current)
    return cfg


# --------------------------------------------------------------------------------
# CHECKPOINT
# --------------------------------------------------------------------------------
checkpoint_path = os.path.join(file_path, "pinn_checkpoint.pkl")


def _window_checkpoint_path(window_idx):
    return os.path.join(file_path, f"pinn_checkpoint_w{window_idx:04d}.pkl")


# Tracks which window indices THIS PROCESS has already written a checkpoint
# file for -- see save_checkpoint's "once written, never touched again"
# skip logic below. Deliberately NOT `os.path.exists(path)`: that treats
# "a file happens to already sit at this path" as proof it holds the
# current run's correct, complete data, which is false in two real cases --
# (1) a crash between writing window k's file and rewriting the manifest:
# on resume, window k gets retrained from scratch, but the STALE pre-crash
# file already exists on disk, so os.path.exists would skip overwriting it
# with the freshly retrained (and possibly different) params; (2) starting
# a fresh, non-resuming run in an output directory that still has leftover
# pinn_checkpoint_w####.pkl files from an older run -- os.path.exists would
# silently keep that old run's data instead of this run's. Checking
# in-memory set membership instead means only files THIS process actually
# wrote are ever treated as "already saved, skip" -- any pre-existing file
# for a window this process is about to (re)write gets overwritten with
# the current, correct in-memory state, exactly once.
_written_window_files = set()


def save_checkpoint():
    """Write one small per-window file (pinn_checkpoint_w####.pkl, WRITTEN
    ONCE and never touched again) for each completed window, plus a
    lightweight manifest (pinn_checkpoint.pkl) listing those filenames
    alongside puncture_track_history/model_config. Called once per window
    (not just once after the whole training loop finishes, which is where
    this used to live) so a crash/OOM/preemption partway through a long run
    -- the exact scenario --resume_checkpoint exists to recover from --
    doesn't lose every already-completed window; resuming picks up at the
    last window actually saved instead of restarting from scratch.

    Split into per-window files (previously: one single pickle holding
    ALL windows, entirely rewritten from scratch on every call) because
    window_states is append-only -- earlier windows' dicts are never
    mutated once appended -- so re-pickling every earlier window's full
    parameter pytree on every single later window's save was pure waste:
    total disk I/O across a run grew O(N_window^2) (window 0's params
    written N_window times, window 1's N_window-1 times, ...) purely to
    gain crash-resumability that writing each window's own (large,
    dominant-cost) data exactly once gives far more cheaply. Each
    per-window file is still atomically written (see _atomic_write) and,
    once it exists, is skipped on every later call -- the manifest itself
    (small: filenames + puncture_track_history + model_config, no
    parameter pytrees) is still rewritten every call, same as before, since
    its own size is comparatively negligible.

    Defined here (before the main loop) rather than after it, since the
    loop needs to call it; puncture_track_history is read as a global at
    call time, so it's fine that it isn't assigned until after this def --
    only its assignment before the loop actually STARTS matters."""
    for s in window_states:
        if s["window_idx"] in _written_window_files:
            continue
        _wpath = _window_checkpoint_path(s["window_idx"])
        _w = {
            "window_idx": s["window_idx"],
            "t_start":    s["t_start"],
            "t_end":      s["t_end"],
            "params":     s["params"],
            "ic_mode":    s["ic_mode"],
            "ic_history": s["ic_history"],
            "loss_list":  s["loss_list"],
            "epochs_list":s["epochs_list"],
            "loss_history":s["loss_history"],
            # Only present under --chi_bg_mode moving (see _get_eval_fn) --
            # this window's own final shift-driven trajectory, needed to
            # reconstruct physical chi for post-hoc whole-run diagnostics.
            **({"traj_tt": s["traj_tt"], "traj_xx": s["traj_xx"]} if chi_bg_mode == "moving" else {}),
        }
        def _dump_w(p, _w=_w):
            with open(p, "wb") as f:
                pickle.dump(_w, f)
        _atomic_write(_wpath, _dump_w)
        _written_window_files.add(s["window_idx"])

    manifest = {
        "window_files": [os.path.basename(_window_checkpoint_path(s["window_idx"])) for s in window_states],
        "puncture_track_history": puncture_track_history,
        "model_config": _build_model_config(),
    }
    # Atomic write (see _atomic_write): this is now called every window
    # specifically so a crash/OOM/preemption doesn't lose prior windows --
    # writing directly to checkpoint_path would defeat that if the crash
    # lands mid-write, truncating/corrupting the one file --resume_checkpoint
    # depends on.
    def _dump(p):
        with open(p, "wb") as f:
            pickle.dump(manifest, f)
    _atomic_write(checkpoint_path, _dump)


if resume_checkpoint is not None:
    print(f"\nLoading checkpoint: {resume_checkpoint}")
    with open(resume_checkpoint, "rb") as _f:
        _ckpt = pickle.load(_f)

    if "window_files" in _ckpt:
        # Current (per-window-file) format: the manifest only lists
        # filenames -- load each window's own file, relative to the
        # manifest's own directory (not the CWD) so a checkpoint dir moved
        # or referenced by relative path still resolves correctly.
        _ckpt_dir = os.path.dirname(os.path.abspath(resume_checkpoint))
        _windows = []
        for _wf in _ckpt["window_files"]:
            with open(os.path.join(_ckpt_dir, _wf), "rb") as _wfh:
                _windows.append(pickle.load(_wfh))
        _ckpt["windows"] = _windows
    # else: old-format single-file checkpoint already has "windows" inline
    # (pre-dating the per-window-file split) -- nothing to do, everything
    # below already just reads _ckpt["windows"].

    # The loaded windows' per-window files are only already "written" if they
    # sit in THIS run's output directory. When resuming into a different
    # --outdir (or from an old single-file checkpoint), they must be written
    # again there, or the new manifest would list files the new directory
    # does not contain.
    _resume_in_place = ("window_files" in _ckpt
                        and os.path.isdir(_ckpt_dir)
                        and os.path.samefile(_ckpt_dir, file_path))
    if not _resume_in_place:
        print(f"Resuming into a different output directory ({file_path}): the "
              f"{len(_ckpt['windows'])} loaded window(s) are written there again.")

    _cfg = _ckpt.get("model_config", {})

    def _cfg_get(cfg, key_path):
        """Nested dict lookup by dot-path, e.g. 'domain.tfinal'."""
        val = cfg
        for part in key_path.split("."):
            val = val.get(part) if isinstance(val, dict) else None
        return val

    # Uses _CHECKPOINT_FIELDS (defined above, shared with save_checkpoint's
    # _build_model_config) rather than a second hand-maintained list --
    # adding a new physically-relevant CLI argument now means adding ONE
    # row there, which is automatically both written to model_config AND
    # checked here, instead of two lists that could silently drift apart.
    _mismatches = []
    for _label, _key_path, _current, _differs, _required in _CHECKPOINT_FIELDS:
        _ckpt_val = _cfg_get(_cfg, _key_path)
        if _ckpt_val is None and not _required:
            continue
        if _differs(_ckpt_val, _current):
            _mismatches.append(f"{_label}: checkpoint={_ckpt_val} vs arg={_current}")
    if _mismatches:
        raise ValueError("Checkpoint / argument mismatch — cannot resume safely:\n  " +
                         "\n  ".join(_mismatches))

    # chi_bg_mode is only "required=False" above for backward-compat with
    # checkpoints saved before that field existed -- so an old-format
    # checkpoint is silently accepted for resume even under --chi_bg_mode
    # moving. But moving mode needs each window's traj_tt/traj_xx (see
    # _get_eval_fn) to reconstruct physical chi for post-hoc diagnostics;
    # without them those windows silently fall back to raw regular-
    # remainder chi with no error. Check explicitly rather than let that
    # happen quietly.
    if chi_bg_mode == "moving":
        _missing_traj = [w["window_idx"] for w in _ckpt["windows"] if "traj_tt" not in w]
        if _missing_traj:
            raise ValueError(
                f"Checkpoint / argument mismatch — cannot resume safely: --chi_bg_mode moving "
                f"requires every window's traj_tt/traj_xx, but window(s) {_missing_traj} in this "
                f"checkpoint don't have them (saved under a different chi_bg_mode or an older "
                f"format). Post-hoc diagnostics for those windows would silently use raw regular-"
                f"remainder chi instead of physical chi.")

    if args.resume_windows > 0:
        if args.resume_windows > len(_ckpt["windows"]):
            raise ValueError(f"--resume_windows {args.resume_windows} but the checkpoint has only "
                             f"{len(_ckpt['windows'])} window(s)")
        _ckpt["windows"] = _ckpt["windows"][:args.resume_windows]
        _ckpt["puncture_track_history"] = [e for e in _ckpt.get("puncture_track_history", [])
                                           if e.get("window_idx", 0) < args.resume_windows]
        print(f"--resume_windows {args.resume_windows}: forking after window {args.resume_windows}")
    for _w in _ckpt["windows"]:
        if "ic_history" not in _w:
            # Old checkpoint (pre flat-vmap-sum rewrite): migrate the nested
            # ic_frozen linked list into the new flat ic_history format.
            _w["ic_history"] = _migrate_ic_frozen_to_history(_w.get("ic_frozen"))
        window_states.append(_w)
        # When resuming in place, this window's per-window file already
        # exists in file_path (we just loaded it from there) and holds this
        # exact data -- mark it written so the next save_checkpoint() doesn't
        # re-serialize and rewrite every already-completed window's full
        # parameter pytree from scratch (see _written_window_files' docstring
        # above). Otherwise save_checkpoint() writes it into file_path.
        if _resume_in_place:
            _written_window_files.add(_w["window_idx"])

    resume_start = len(window_states)
    print(f"Resumed {resume_start} completed window(s); "
          f"last window t=[{window_states[-1]['t_start']:.4f}, "
          f"{window_states[-1]['t_end']:.4f}]")
    if resume_start < N_window:
        print(f"Training will start from window {resume_start} "
              f"(t={window_edges[resume_start]:.4f} → {window_edges[resume_start+1]:.4f})")
    if resume_start >= N_window:
        raise ValueError(f"Checkpoint already contains all {N_window} windows. "
                         f"Increase --N_window or use a different checkpoint.")

puncture_track_history = []


def _atomic_savetxt(path, *args, **kwargs):
    """np.savetxt via _atomic_write -- same atomicity rationale as
    save_checkpoint's write: these files are rewritten from scratch every
    window (not appended), so a crash/OOM/preemption mid-write would
    otherwise corrupt the ENTIRE accumulated multi-window trajectory
    history, not just the latest window's increment."""
    _atomic_write(path, lambda p: np.savetxt(p, *args, **kwargs))


@lru_cache(maxsize=1)
def _t0_puncture_reference():
    """Analytic t=0 reference (x, beta^x, alpha) at each star's own
    near-axis position -- a pure function of BL_masses (fixed for the whole
    run), so memoized rather than recomputed on every save_puncture_track_
    outputs call (previously: one compute_ccz4_ic call per window, all
    returning the identical result). See save_puncture_track_outputs for
    why this reference point is needed at all."""
    t0_x    = tuple(c[0] for c in _bh_center_list(BL_masses))
    t0_y    = jnp.full((len(t0_x), 1), _PUNCTURE_TRACK_EPS, dtype=jnp.float64)
    t0_z    = jnp.full((len(t0_x), 1), _PUNCTURE_TRACK_EPS, dtype=jnp.float64)
    t0_xarr = jnp.asarray(t0_x, dtype=jnp.float64).reshape(-1, 1)
    t0_out  = np.asarray(compute_ccz4_ic(t0_xarr[:, 0], t0_y[:, 0], t0_z[:, 0]))
    return t0_x, t0_out[:, BETA_SLICE.start], t0_out[:, ALPHA_IDX]


_STRENGTH_DISK_HALFWIDTH = 1.0
_STRENGTH_DISK_N = 15
_STRENGTH_TIME_FRACS = (0.0, 0.5, 1.0)


def _puncture_strength_rows(window_idx, window_state, t_start, t_end, centers, traj_tt, traj_xx):
    """Per-window "is the puncture still there?" diagnostic, written to
    puncture_strength.csv (see save_puncture_track_outputs). The training
    loss cannot answer this: flat spacetime is an exact vacuum solution
    with zero residual, and the puncture cores are a negligible fraction of
    the volume in a mean-reduced loss, so a network that has erased the
    punctures can show a LOWER loss than one that kept them (seen on a real
    run: total loss fell 5x while alpha at the punctures climbed 0.05 -> 1.0).
    For each puncture and each sampled time in the window, records the
    values at the tracked position (offset by _PUNCTURE_TRACK_EPS in z, like
    the other puncture diagnostics) plus min alpha / min chi / max g_xx over
    a small disk of half-width _STRENGTH_DISK_HALFWIDTH around it (g_xx is NaN
    where chi<=1e-6; chi_nonpos counts those disk points -- nonzero means the
    network's chi went non-positive there, NOT a strong puncture). A healthy
    puncture keeps alpha_min small, chi_min small and gxx_max large; erasure
    shows as alpha_min -> 1, chi_min -> 1, gxx_max -> ~1.

    traj_tt/traj_xx (this window's RK4 sub-grid, shift_driven only) let the
    positions follow the puncture at each sampled time and are required
    under --chi_bg_mode moving to recover physical chi; without them (the
    chi_min/lapse_min modes have no sub-window trajectory) only t_end is
    sampled, at the located centers."""
    fn, p, ws, we = _network_apply_for_state(window_state, traj_tt, traj_xx)
    n_st = len(centers)
    if traj_tt is not None:
        times = [t_start + f * (t_end - t_start) for f in _STRENGTH_TIME_FRACS]
        pos = lambda k, t: tuple(float(v) for v in _interp_trajectory(traj_tt, traj_xx, k, t))
    else:
        times = [t_end]
        pos = lambda k, t: (float(centers[k][0]), float(centers[k][1]))

    g = np.linspace(-_STRENGTH_DISK_HALFWIDTH, _STRENGTH_DISK_HALFWIDTH, _STRENGTH_DISK_N)
    GX, GY = np.meshgrid(g, g, indexing="xy")
    in_disk = (GX**2 + GY**2) <= _STRENGTH_DISK_HALFWIDTH**2
    offs = np.concatenate([[[0.0, 0.0]], np.stack([GX[in_disk], GY[in_disk]], axis=1)], axis=0)   # row 0 = center
    m = offs.shape[0]

    nan = float("nan")
    finite = _state_is_finite(window_state, fn, p, ws, we, t_end)
    if finite:
        xs, ys, ts = [], [], []
        for t in times:
            for k in range(n_st):
                cx, cy = pos(k, t)
                xs.append(cx + offs[:, 0]); ys.append(cy + offs[:, 1]); ts.append(np.full(m, t))
        x_in = jnp.asarray(np.concatenate(xs), dtype=jnp.float64).reshape(-1, 1)
        y_in = jnp.asarray(np.concatenate(ys), dtype=jnp.float64).reshape(-1, 1)
        t_in = jnp.asarray(np.concatenate(ts), dtype=jnp.float64).reshape(-1, 1)
        z_in = jnp.full_like(x_in, _PUNCTURE_TRACK_EPS)
        out = np.asarray(full_output(fn, p, t_in, x_in, y_in, z_in, ws, we))
        alpha = out[:, ALPHA_IDX]
        chi   = out[:, CHI_IDX]
        # NOT _bh_adm_from_out's gxx (it floors chi at 1e-30, so a transiently
        # non-positive network chi reports g_xx ~ 1e30), and NOT a clamped
        # value either (a clamp turns chi<=0 into a huge g_xx that reads as a
        # STRONG puncture). g_xx is NaN wherever chi <= 1e-6 (a genuine puncture
        # at the tracked offset has chi ~ 1e-4, g_xx ~ 1e4); the number of such
        # points is reported separately as chi_nonpos.
        chi_ok = chi > 1e-6
        gxx   = np.where(chi_ok, out[:, GT_SLICE][:, 0] / np.where(chi_ok, chi, 1.0), np.nan)
        alpha, gxx, chi = (a.reshape(len(times), n_st, m) for a in (alpha, gxx, chi))

    rows = []
    for ti, t in enumerate(times):
        for k in range(n_st):
            cx, cy = pos(k, t)
            row = {"window": window_idx, "t": float(t), "puncture": k, "x": cx, "y": cy}
            if finite:
                g_k = gxx[ti, k]
                row.update(alpha_center=float(alpha[ti, k, 0]), chi_center=float(chi[ti, k, 0]),
                           gxx_center=float(g_k[0]), alpha_min=float(np.nanmin(alpha[ti, k])),
                           chi_min=float(np.nanmin(chi[ti, k])),
                           gxx_max=float(np.nanmax(g_k)) if np.isfinite(g_k).any() else nan,
                           chi_nonpos=int((chi[ti, k] <= 1e-6).sum()))
            else:
                row.update(alpha_center=nan, chi_center=nan, gxx_center=nan,
                           alpha_min=nan, chi_min=nan, gxx_max=nan, chi_nonpos=nan)
            rows.append(row)
    return rows


def save_puncture_track_outputs():
    """Write diagnostics_puncture_track.png / puncture_track.txt /
    puncture_track_fine.txt from whatever is in puncture_track_history so
    far. Re-written from scratch (not appended) on every call, so calling it
    once per window -- instead of only once after the whole training loop
    finishes -- makes the trajectory visible window-by-window without
    waiting for the final window to complete."""
    if not (bh_center_mode in ("shift_driven", "chi_min", "lapse_min") and puncture_track_history):
        return
    # Window-boundary summary (one point per window -- kept for backward
    # compatibility with puncture_track.txt's existing columns). Prepend an
    # analytic t=0 reference point: puncture_track_history's first entry is
    # window 0's END, not t=0 itself, so without this the window-boundary
    # series (and diagnostics_gauge_at_puncture.png, which has no fine-grid
    # equivalent) start mid-flight. At t=0 everything is known exactly, no
    # trained network needed -- see _t0_puncture_reference (memoized: the
    # chi minimum is trivially each star's own analytic position, that's
    # where the conformal factor's puncture singularity sits, by
    # construction; beta^i/alpha come from the same cached analytic call).
    _t0_x, _t0_beta, _t0_alpha = _t0_puncture_reference()

    # y (and y_fine): 0 for every checkpoint saved before --BL_momenta_y
    # existed (.get default), and always exactly 0 at t=0 by construction
    # regardless -- the punctures start exactly on the x-axis (see
    # --BL_momenta_y's help), only a tangential momentum curves the
    # SUBSEQUENT tracked position off it.
    _t0_y = np.zeros_like(_t0_x)
    _pt_t  = np.concatenate([[0.0], [r["t"] for r in puncture_track_history]])
    _pt_x  = np.concatenate([[_t0_x], [r["x"] for r in puncture_track_history]])         # (n_windows+1, n_stars)
    _pt_y  = np.concatenate([[_t0_y], [r.get("y", [0.0] * n_stars) for r in puncture_track_history]])
    _pt_cm = np.concatenate([[_t0_x], [r["chi_min_x"] for r in puncture_track_history]])
    _pt_beta  = np.concatenate([[_t0_beta], [r["beta_x"] for r in puncture_track_history]])  # (n_windows+1, n_stars)
    # beta^y: 0 for both t=0 (zero shift by construction) and any checkpoint
    # saved before --BL_momenta_y existed (.get default) -- same rationale
    # as _pt_y's .get above. |beta| below is the physically meaningful
    # "shift at puncture" magnitude once an orbit gives beta^y a nonzero
    # value too -- beta^x alone silently drops that component instead of
    # just being 0 in the non-orbiting case, which is the whole reason a
    # magnitude (not one arbitrary component) belongs in this diagnostic.
    _pt_beta_y = np.concatenate([[np.zeros_like(_t0_beta)],
                                  [r.get("beta_y", [0.0] * n_stars) for r in puncture_track_history]])
    _pt_beta_mag = np.sqrt(_pt_beta**2 + _pt_beta_y**2)
    _pt_alpha = np.concatenate([[_t0_alpha], [r["alpha"]  for r in puncture_track_history]])  # (n_windows+1, n_stars)

    # Fine-grained, WITHIN-window trajectory: each window's RK4 sub-grid
    # (track_steps+1 points, not just its two endpoints) concatenated
    # across all windows, so the plot/txt show what the puncture does in
    # between window boundaries, not only at them.
    _fine_t  = np.concatenate([np.asarray(r["t_fine"]) for r in puncture_track_history])
    _fine_x  = np.concatenate([np.asarray(r["x_fine"]) for r in puncture_track_history], axis=1)  # (n_stars, n_total)
    _fine_y  = np.concatenate([np.asarray(r.get("y_fine", [[0.0] * len(r["t_fine"])] * n_stars))
                                for r in puncture_track_history], axis=1)                          # (n_stars, n_total)
    _cm_t    = np.concatenate([np.asarray(r["chi_min_t_fine"]) for r in puncture_track_history])
    _cm_x    = np.concatenate([np.asarray(r["chi_min_x_fine"]) for r in puncture_track_history], axis=0)  # (n_checks, n_stars)

    fig_p, axes_p = plt.subplots(2 if n_stars == 2 else 1, 1,
                                  figsize=(9, 8 if n_stars == 2 else 4.5),
                                  sharex=True, squeeze=False)
    axes_p = axes_p[:, 0]

    # NOTE for the reader: post-merger, _pick_bh_centers_2d collapses the
    # chi-min cross-check ("x" markers) onto a single repeated point (see
    # its docstring) since chi genuinely has only one basin left, while the
    # tracked trace (solid line) stays two visibly separate curves -- each
    # puncture's own independent tracking has no way to know they've
    # merged. That divergence between the two traces after merger is
    # EXPECTED, not a tracking bug.
    #
    # The solid line is a within-window RK4 sub-grid for shift_driven, but
    # for chi_min/lapse_min it's just the same window-boundary points
    # connected (those modes re-locate from scratch each window -- there is
    # no sub-grid to show), so the "x" cross-check marker legitimately
    # overlaps the line for chi_min (its own tracked position IS the chi
    # minimum, see the caller).
    _colors = ["tab:blue", "tab:orange"]
    ax = axes_p[0]
    for k in range(n_stars):
        clr = _colors[k % len(_colors)]
        ax.plot(_fine_t, _fine_x[k], "-", linewidth=1.2, color=clr,
                label=f"puncture {k} ({bh_center_mode})")
        ax.plot(_pt_t, _pt_x[:, k], "o", markersize=4, color=clr,
                label=f"puncture {k} (window boundary)" if k == 0 else None)
        ax.plot(_cm_t, _cm_x[:, k], "x", markersize=4, alpha=0.7, color=clr,
                label=f"puncture {k} (chi-min cross-check)")
    ax.set_ylabel("x")
    ax.set_title("Puncture trajectory (line = within-window RK4 sub-grid)"
                 if bh_center_mode == "shift_driven" else
                 "Puncture trajectory (line = window-boundary points)")
    ax.grid(True); ax.legend(fontsize=7, ncol=2)
    for _we in window_edges[1:-1]:
        ax.axvline(_we, color="gray", linestyle=":", linewidth=1)

    if n_stars == 2:
        ax = axes_p[1]
        # Euclidean distance, not the x-difference: once an orbit rotates the
        # pair, x1-x0 goes to 0 (and negative) while the punctures are still
        # far apart (seen on a real run: x-separation -0.18 at t=18 while the
        # true separation was 1.80).
        ax.plot(_fine_t, np.hypot(_fine_x[1] - _fine_x[0], _fine_y[1] - _fine_y[0]), "-", linewidth=1.2, color="tab:red")
        ax.plot(_pt_t, np.hypot(_pt_x[:, 1] - _pt_x[:, 0], _pt_y[:, 1] - _pt_y[:, 0]), "o", markersize=4, color="tab:red")
        ax.set_ylabel("separation")
        ax.set_xlabel("t")
        ax.grid(True)
        for _we in window_edges[1:-1]:
            ax.axvline(_we, color="gray", linestyle=":", linewidth=1)
    else:
        axes_p[0].set_xlabel("t")

    fig_p.suptitle(f"Puncture tracking ({bh_center_mode})")
    fig_p.tight_layout()
    _atomic_savefig(fig_p, os.path.join(file_path, "diagnostics_puncture_track.png"), dpi=120, bbox_inches="tight")
    plt.close(fig_p)

    # Gauge-at-puncture panels (separation, shift, lapse), laid out to
    # mirror bbh_headon_bssn_pinn_shift_driven_puncture.py's own reference
    # diagnostic plot panel-for-panel, so a run here can be overlaid directly
    # against that one instead of eyeballing pixel positions on a screenshot.
    # beta_x/alpha are only evaluated once per window (at window_end, in
    # advance_shift_tracked_centers), so these are window-boundary points,
    # not a within-window fine grid like the trajectory panel above.
    fig_g, axes_g = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes_g[0]
    if n_stars == 2:
        ax.plot(_pt_t, np.hypot(_pt_x[:, 1] - _pt_x[:, 0], _pt_y[:, 1] - _pt_y[:, 0]), "o-", markersize=4, color="tab:red")
    else:
        for k in range(n_stars):
            ax.plot(_pt_t, _pt_x[:, k], "o-", markersize=4, color=_colors[k % len(_colors)])
    ax.set_title("Puncture separation" if n_stars == 2 else "Puncture position")
    ax.set_ylabel("D" if n_stars == 2 else "x"); ax.set_xlabel("t"); ax.grid(True)

    ax = axes_g[1]
    for k in range(n_stars):
        ax.plot(_pt_t, _pt_beta_mag[:, k], "o-", markersize=4, color=_colors[k % len(_colors)],
                label=f"puncture {k}")
    ax.set_title("Shift at puncture"); ax.set_ylabel(r"$|\beta|$"); ax.set_xlabel("t")
    ax.grid(True); ax.legend(fontsize=8)

    ax = axes_g[2]
    for k in range(n_stars):
        ax.plot(_pt_t, _pt_alpha[:, k], "o-", markersize=4, color=_colors[k % len(_colors)])
    ax.set_title("Lapse at puncture"); ax.set_ylabel(r"$\alpha$"); ax.set_xlabel("t")
    ax.grid(True)

    fig_g.suptitle("Gauge fields at puncture (window-boundary samples)")
    fig_g.tight_layout()
    _atomic_savefig(fig_g, os.path.join(file_path, "diagnostics_gauge_at_puncture.png"), dpi=120, bbox_inches="tight")
    plt.close(fig_g)

    # Top-down (x,y) view of the tracked trajectory -- a flat horizontal
    # line at y=0 for the original x-axis-only head-on case (--BL_momenta_y
    # all-zero), an actual 2D path once a tangential momentum curves it
    # off-axis. Separate figure (not a third row on fig_p above) so the
    # existing 1-or-2-row layout there never needs conditional restructuring.
    fig_xy, ax = plt.subplots(figsize=(5.5, 5.5))
    for k in range(n_stars):
        clr = _colors[k % len(_colors)]
        ax.plot(_fine_x[k], _fine_y[k], "-", linewidth=1.2, color=clr, label=f"puncture {k}")
        ax.plot(_pt_x[:, k], _pt_y[:, k], "o", markersize=4, color=clr)
    ax.plot(0.0, 0.0, "+", color="black", markersize=8, label="origin")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(f"Puncture trajectory, top-down ({bh_center_mode})")
    ax.grid(True); ax.legend(fontsize=8)
    fig_xy.tight_layout()
    _atomic_savefig(fig_xy, os.path.join(file_path, "diagnostics_puncture_track_xy.png"), dpi=120, bbox_inches="tight")
    plt.close(fig_xy)

    _track_txt = os.path.join(file_path, "puncture_track.txt")
    _hdr_cols  = (["t"] + [f"x_{k}" for k in range(n_stars)] + [f"y_{k}" for k in range(n_stars)]
                  + [f"chi_min_x_{k}" for k in range(n_stars)]
                  + [f"beta_x_{k}" for k in range(n_stars)] + [f"beta_y_{k}" for k in range(n_stars)]
                  + [f"beta_mag_{k}" for k in range(n_stars)] + [f"alpha_{k}" for k in range(n_stars)])
    _atomic_savetxt(_track_txt,
               np.column_stack([_pt_t, _pt_x, _pt_y, _pt_cm, _pt_beta, _pt_beta_y, _pt_beta_mag, _pt_alpha]),
               header="  ".join(_hdr_cols) + "  (t=0 analytic row, then one row per window boundary; "
                      "y is always 0 unless --BL_momenta_y curves the tracked position off-axis; "
                      "beta_mag = sqrt(beta_x^2+beta_y^2))", fmt="%.18e")

    _strength_rows = [row for r in puncture_track_history for row in r.get("strength", [])]
    if _strength_rows:
        def _write_strength(path):
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(_strength_rows[0].keys()))
                w.writeheader()
                w.writerows(_strength_rows)
        _atomic_write(os.path.join(file_path, "puncture_strength.csv"), _write_strength)

    _track_fine_txt = os.path.join(file_path, "puncture_track_fine.txt")
    _hdr_fine_cols  = ["t"] + [f"x_{k}" for k in range(n_stars)] + [f"y_{k}" for k in range(n_stars)]
    _fine_desc = ("within-window RK4 sub-grid, all windows concatenated" if bh_center_mode == "shift_driven"
                  else "one re-located point per window (no sub-grid for this --bh_center_mode)")
    _atomic_savetxt(_track_fine_txt,
               np.column_stack([_fine_t, _fine_x.T, _fine_y.T]),
               header="  ".join(_hdr_fine_cols) + f"  ({_fine_desc})",
               fmt="%.18e")


if resume_checkpoint is not None and bh_center_mode in ("shift_driven", "chi_min", "lapse_min"):
    _ckpt_track = _ckpt.get("puncture_track_history", [])
    if _ckpt_track:
        puncture_track_history = _ckpt_track
        if bh_center_mode == "shift_driven":
            # .get("y", ...): a checkpoint from before --BL_momenta_y
            # existed has no "y" key at all -- default to 0 (exactly what
            # it was, implicitly, before this generalization).
            _resumed_x = _ckpt_track[-1]["x"]
            _resumed_y = _ckpt_track[-1].get("y", [0.0] * len(_resumed_x))
            _shift_track_xy = [list(xy) for xy in zip(_resumed_x, _resumed_y)]
            print(f"Resumed shift-driven puncture track: x={_resumed_x}  y={_resumed_y}")
        else:
            print(f"Resumed {bh_center_mode} puncture track: x={_ckpt_track[-1]['x']}")

for window_idx in range(resume_start, N_window):
    if args.stop_after_window > 0 and window_idx >= args.stop_after_window:
        print(f"--stop_after_window {args.stop_after_window}: stopping after window {window_idx}.")
        # The post-run diagnostics below sample t in [0, tfinal] and look windows up by
        # time; restrict them to the trained span.
        N_window     = len(window_states)
        window_edges = window_edges[:N_window + 1]
        tfinal       = float(window_edges[-1])
        break
    t_start    = float(window_edges[window_idx])
    t_end      = float(window_edges[window_idx + 1])
    if window_idx == 0:
        prev_state = constraint_ic_state
    else:
        prev_state = window_states[-1]
    window_state, subkey = train_window(window_idx, t_start, t_end, subkey, prev_state=prev_state)
    window_states.append(window_state)

    if bh_center_mode in ("shift_driven", "chi_min", "lapse_min"):
        if bh_center_mode == "shift_driven":
            centers, diags, (traj_tt_fine, traj_xx_fine) = advance_shift_tracked_centers(window_state, t_start, t_end)
            # traj_xx_fine: (n_stars, 2, n_steps+1), axis 1 = (x,y) -- see
            # _integrate_puncture_trajectory. locate_chi_minima's own scan
            # below covers the full x-y plane now, but the cross-check
            # comparison itself (this diagnostic column only, not used for
            # sampling/training) still just compares the x-component; pull
            # that out once here rather than re-slicing at every use site.
            traj_x_fine = traj_xx_fine[:, 0, :]
            traj_y_fine = traj_xx_fine[:, 1, :]

            # traj_tt_fine/traj_xx_fine are this window's own final trajectory --
            # required under --chi_bg_mode moving so locate_chi_minima scans
            # physical chi (background + regular remainder) instead of the raw
            # regular remainder alone (see _network_apply_for_state); a no-op
            # under the default --chi_bg_mode physical.
            # locate_chi_minima's output is left-to-right SORTED (no persistent
            # per-star identity, see _pick_bh_centers_2d), while `centers` preserves
            # the shift-driven tracker's fixed per-star identity -- match to
            # `centers` so the two are directly comparable index-for-index in
            # the puncture-track plot/table (see _match_to_reference).
            # Local box around the shift-tracked position itself (already
            # the best available estimate) instead of a whole-plane scan --
            # see find_bh_centers_lapse/locate_chi_minima's search_centers/
            # search_halfwidth params. Falls back to the whole-plane scan
            # automatically (via the widen-on-edge escalation) if chi's
            # true minimum isn't actually near the tracked position.
            _cross_halfwidth = _search_halfwidth([(d["beta_x"], d["beta_y"]) for d in diags], t_end - t_start)
            _chi_min_xy = locate_chi_minima(
                window_state, t_end, traj_tt=traj_tt_fine, traj_xx=traj_xx_fine,
                search_centers=[(c[0], c[1]) for c in centers], search_halfwidth=_cross_halfwidth)
            # Match on the full (x,y) position, not just x (see _xy_dist) --
            # two punctures with close/crossed x but well-separated y would
            # otherwise be matchable "correctly" by x alone yet actually
            # mismatched in identity.
            _perm = _match_permutation(_chi_min_xy, [(c[0], c[1]) for c in centers], dist=_xy_dist)
            chi_min_x = [_chi_min_xy[i][0] for i in _perm]

            if chi_bg_mode == "moving":
                # Persist onto the window's OWN state (and thus the checkpoint)
                # so _get_eval_fn can reconstruct physical chi for POST-hoc,
                # whole-run diagnostics too (eval_full_pinn/eval_H_pinn/
                # eval_gauge_source_vector, e.g. the final "Constraint summary"),
                # not just the two locate_chi_minima calls right here -- those
                # only had traj_tt_fine/traj_xx_fine available transiently in
                # this loop, with nowhere else to get it from afterward.
                window_state["traj_tt"] = np.asarray(traj_tt_fine)
                window_state["traj_xx"] = np.asarray(traj_xx_fine)

            # Chi-minimum cross-check at a handful of times spanning the WHOLE
            # window (not just t_end), same idea as the fine-grained
            # shift-driven trajectory below -- gives visibility into what the
            # puncture is doing in between window boundaries, not just at them.
            _n_chi_check   = min(5, traj_tt_fine.shape[0])
            _chi_check_idx = np.unique(np.linspace(0, traj_tt_fine.shape[0] - 1, _n_chi_check).round().astype(int))
            chi_min_t_fine = traj_tt_fine[_chi_check_idx]
            # Match each check-time's chi-min candidates to the shift-driven
            # tracker's OWN position at that same sub-grid index (traj_x_fine
            # preserves per-star identity; locate_chi_minima's output doesn't --
            # see _match_to_reference), so chi_min_x_fine[i][k] and
            # x_fine[k][_chi_check_idx[i]] are directly comparable.
            chi_min_x_fine = []
            for idx, tv in zip(_chi_check_idx, chi_min_t_fine):
                _ref_xy = [(float(traj_x_fine[k, idx]), float(traj_y_fine[k, idx]))
                           for k in range(n_stars)]
                _cand_xy = locate_chi_minima(
                    window_state, float(tv), traj_tt=traj_tt_fine, traj_xx=traj_xx_fine,
                    search_centers=_ref_xy,
                    search_halfwidth=_search_halfwidth(
                        [(d["beta_x"], d["beta_y"]) for d in diags], float(tv) - t_start))
                # Full (x,y) match (see _xy_dist), not x-only.
                _perm = _match_permutation(_cand_xy, _ref_xy, dist=_xy_dist)
                chi_min_x_fine.append([_cand_xy[i][0] for i in _perm])
            _track_label = "Shift-driven"
        else:
            # chi_min/lapse_min: no persistent trajectory to advance -- just
            # re-locate this window's own freshly-trained network at t_end
            # (see _locate_centers_and_diags), the same scan train_window
            # already does at window_start for the NEXT window's sampling
            # focus, repeated here at window_end for the diagnostic record.
            # chi_bg_mode=='moving' can't occur here (enforced at argparse
            # time to require --bh_center_mode shift_driven), so there is no
            # trajectory to persist onto window_state, unlike the branch above.
            centers, diags = _locate_centers_and_diags(bh_center_mode, window_state, t_end)
            if bh_center_mode == "chi_min":
                # centers ARE chi minima already -- no separate cross-check.
                chi_min_x = [c[0] for c in centers]
            else:
                # lapse_min: independent chi-min cross-check, same idea as
                # shift_driven's (are the lapse minima where chi says the
                # punctures are?). Local box around the lapse-located
                # position itself, same rationale as shift_driven's cross-
                # check above.
                _cross_halfwidth = _search_halfwidth([(d["beta_x"], d["beta_y"]) for d in diags], t_end - t_start)
                _chi_min_xy = locate_chi_minima(
                    window_state, t_end, search_centers=[(c[0], c[1]) for c in centers],
                    search_halfwidth=_cross_halfwidth)
                # Full (x,y) match (see _xy_dist), not x-only.
                _perm = _match_permutation(_chi_min_xy, [(c[0], c[1]) for c in centers], dist=_xy_dist)
                chi_min_x = [_chi_min_xy[i][0] for i in _perm]
            # No within-window sub-stepping exists for these modes (unlike
            # shift_driven's RK4 sub-grid) -- a single (t_end, x, y) point
            # per window is the truthful fine-grid entry, not a fabricated
            # line. y is real now (chi_min/lapse_min's own scan covers the
            # full x-y plane -- see locate_chi_minima/find_bh_centers_lapse),
            # not forced to 0 -- centers already carries it via
            # _locate_centers_and_diags.
            traj_tt_fine   = np.array([t_end])
            traj_x_fine   = np.asarray([[c[0]] for c in centers])       # (n_stars, 1)
            traj_y_fine   = np.asarray([[c[1]] for c in centers])       # (n_stars, 1)
            chi_min_t_fine = np.array([t_end])
            chi_min_x_fine = [chi_min_x]                                  # (1, n_stars)
            _track_label = "Chi-min" if bh_center_mode == "chi_min" else "Lapse-min"

        _strength = _puncture_strength_rows(
            window_idx, window_state, t_start, t_end, centers,
            traj_tt_fine if bh_center_mode == "shift_driven" else None,
            traj_xx_fine if bh_center_mode == "shift_driven" else None)
        puncture_track_history.append({
            "window_idx": window_idx,
            "strength":   _strength,
            "t":          t_end,
            "x":          [c[0] for c in centers],
            "y":          [c[1] for c in centers],
            "separation": float(np.hypot(centers[1][0] - centers[0][0], centers[1][1] - centers[0][1])) if n_stars == 2 else None,
            "beta_x":     [d["beta_x"] for d in diags],
            "beta_y":     [d.get("beta_y", 0.0) for d in diags],
            "alpha":      [d["alpha"]  for d in diags],
            "chi":        [d["chi"]    for d in diags],
            "chi_min_x":  chi_min_x,
            # Fine-grained, WITHIN-window trajectory: for shift_driven, the
            # RK4 sub-grid used to advance _shift_track_xy (track_steps+1
            # points spanning [t_start, t_end]); for chi_min/lapse_min, the
            # single re-located point at t_end (see above -- no sub-grid
            # exists for those modes).
            "t_fine":          traj_tt_fine.tolist(),
            "x_fine":          traj_x_fine.tolist(),          # (n_stars, n_grid)
            "y_fine":          traj_y_fine.tolist(),          # (n_stars, n_grid)
            "chi_min_t_fine":  chi_min_t_fine.tolist(),
            "chi_min_x_fine":  chi_min_x_fine,                 # (n_checks, n_stars)
        })
        _track_x_str = [f"{x:.4f}" for x in ([c[0] for c in centers])]
        # y only ever printed when actually nonzero (i.e. an orbit is
        # curving the tracked position off the x-axis) -- keeps the
        # console line identical to before for the original x-axis-only
        # head-on-collision case.
        _track_y = [c[1] for c in centers]
        _track_y_str = f"  y={[f'{y:.4f}' for y in _track_y]}" if any(_track_y) else ""
        if bh_center_mode == "chi_min":
            print(f"  {_track_label} puncture track @ t={t_end:.4f}: x={_track_x_str}{_track_y_str}")
        else:
            print(f"  {_track_label} puncture track @ t={t_end:.4f}: "
                  f"x={_track_x_str}{_track_y_str}  "
                  f"(chi-min cross-check: {[f'{x:.4f}' for x in chi_min_x]})")
        _s_end = [r for r in _strength if r["t"] == _strength[-1]["t"]]
        _fmt = lambda key, nd: [format(r[key], f".{nd}f") for r in _s_end]
        print(f"  Puncture strength @ t={t_end:.4f}: "
              f"alpha_min={_fmt('alpha_min', 3)}  chi_min={_fmt('chi_min', 3)}  "
              f"gxx_max={_fmt('gxx_max', 2)}  chi<=1e-6 pts={[r['chi_nonpos'] for r in _s_end]}")
        save_puncture_track_outputs()

    save_checkpoint()
    if profile_restarts and "t0" in _PROF_WIN:  # [GPUS]
        print(f"  [PROFILE] window wall time incl. end-of-window diagnostics and checkpoint: "  # [GPUS]
              f"{time.perf_counter() - _PROF_WIN['t0']:.1f} s")  # [GPUS]
    # See _NEEDS_PERWINDOW_CACHE_CLEAR's module-level definition -- same
    # flag as train_window's matching gate.
    if _NEEDS_PERWINDOW_CACHE_CLEAR:
        jax.clear_caches()
    import gc; gc.collect()

print(f"Checkpoint saved to {checkpoint_path}")


# ================================================================================
# DIAGNOSTICS
# ================================================================================

def to_host(x):
    return np.asarray(x)


def _get_eval_fn(state):
    """Returns (efn, params) -- under --chi_bg_mode moving, params must be
    the BUNDLED {"net","traj_tt","traj_xx"} dict once efn is wrapped with
    _wrap_with_moving_chi_bg (see _network_apply_for_state), so callers
    must use the returned params, not state["params"] directly, or the
    physical-chi reconstruction silently doesn't happen (the same bug
    locate_chi_minima had -- see its docstring)."""
    if state.get("ic_mode", "hard") == "hard":
        if "_eval_fn" not in state:
            state["_eval_fn"] = make_hard_ic_apply_fn(
                model.apply, state["t_start"], state["t_end"],
                ic_history=state.get("ic_history"))
        fn = state["_eval_fn"]
    else:
        fn = apply_fn
    params = state["params"]
    fn, params = _wrap_moving_chi_bg_if_needed(fn, params, state.get("traj_tt"), state.get("traj_xx"))
    return fn, params


def _state_ws_we(state):
    return (jnp.array(state["t_start"], dtype=jnp.float64),
            jnp.array(state["t_end"],   dtype=jnp.float64))


def _eval_field_pinn(field_fn, window_states, x_vals, y_vals, z_vals, t_val):
    """Shared setup for eval_full_pinn/eval_H_pinn/eval_gauge_source_pinn --
    look up the right window's (efn, eparams) (see _get_eval_fn's note on
    why eparams, not state["params"], must be used under --chi_bg_mode
    moving), build the query grid, and call field_fn. The three were
    previously byte-identical except for field_fn -- a fix to this shared
    setup (like the _get_eval_fn/bundled-params pattern that already
    caused a real bug once for locate_chi_minima) now only needs applying
    once instead of in lockstep across three near-duplicate bodies."""
    idx   = find_window_index(t_val)
    state = window_states[idx]
    efn, eparams = _get_eval_fn(state)
    ws, we = _state_ws_we(state)
    x_in  = jnp.asarray(x_vals).reshape(-1, 1)
    y_in  = jnp.asarray(y_vals).reshape(-1, 1)
    z_in  = jnp.asarray(z_vals).reshape(-1, 1)
    t_in  = jnp.full((x_in.shape[0], 1), t_val, dtype=jnp.float64)
    return to_host(field_fn(efn, eparams, t_in, x_in, y_in, z_in, ws, we))


def eval_full_pinn(window_states, x_vals, y_vals, z_vals, t_val):
    return _eval_field_pinn(full_output, window_states, x_vals, y_vals, z_vals, t_val)


def eval_H_pinn(window_states, x_vals, y_vals, z_vals, t_val):
    return _eval_field_pinn(hamiltonian_constraint, window_states, x_vals, y_vals, z_vals, t_val)


def eval_gauge_source_pinn(window_states, x_vals, y_vals, z_vals, t_val):
    """Returns Z^i = chi*(Gamma-hat^i - Gamma-tilde^i)/2 as (N,3) array."""
    return _eval_field_pinn(gauge_source_vector, window_states, x_vals, y_vals, z_vals, t_val)


def eval_metric_reference(x_vals, y_vals, z_vals):
    """Analytic Brill-Lindquist/Bowen-York t=0 (alpha, beta^x, g_xx) reference."""
    g0, _ = evaluate_metric_initial_data_xyz(x_vals, y_vals, z_vals)
    return _bh_adm_from_contra10(np.asarray(to_host(g0)))


# --------------------------------------------------------------------------------
# 1D CUT  y=0, z=0
# --------------------------------------------------------------------------------
nx     = 600
x_vals = np.linspace(x_min, x_max, nx)
y_cut  = np.zeros_like(x_vals)
z_cut  = np.zeros_like(x_vals)
t_values = [0.0, tfinal / 3.0, 2.0 * tfinal / 3.0, tfinal]

quantity_names = [r"$\alpha$", r"$\beta^x$", r"$g_{xx}$", r"$H$"]

fig, axes = plt.subplots(len(quantity_names), len(t_values),
                         figsize=(5*len(t_values), 3.2*len(quantity_names)), sharex=True)

_alpha_ic_1d, _betax_ic_1d, _gxx_ic_1d = eval_metric_reference(x_vals, y_cut, z_cut)
ic_v_1d = [_alpha_ic_1d, _betax_ic_1d, _gxx_ic_1d, np.zeros(len(x_vals))]

# Populated below, reused by the Constraint summary section further down
# instead of recomputing eval_H_pinn/eval_gauge_source_pinn for the same
# (x_vals, y_cut, z_cut, t_now) queries a second/third time.
_H_cache = {}
_Z_cache = {}

for j, t_now in enumerate(t_values):
    out_full_1d = eval_full_pinn(window_states, x_vals, y_cut, z_cut, t_now)
    alpha_pinn, betax_pinn, gxx_cov_pinn = _bh_adm_from_out(out_full_1d)
    H_pinn = eval_H_pinn(window_states, x_vals, y_cut, z_cut, t_now)
    _H_cache[t_now] = H_pinn  # reused by the Constraint summary below -- same (x_vals,y_cut,z_cut,t_now)
    pinn_v = [alpha_pinn, betax_pinn, gxx_cov_pinn, H_pinn]

    for i, qname in enumerate(quantity_names):
        ax = axes[i, j]
        pv    = np.asarray(pinn_v[i], dtype=float)
        _is_H = (i == 3)
        if _is_H:
            pv_plot = np.where(pv != 0.0, np.abs(pv), np.nan)
            pv_plot = _mask_divergent(pv_plot)
            ax.plot(x_vals, pv_plot, label="PINN")
            for _xh in _horizon_x_locs:
                ax.axvline(_xh, color="k", linewidth=0.8, linestyle="--", alpha=0.7)
            _safe_log_yscale(ax, pv_plot)
            ax.set_title(rf"$|H|$, t={t_now:.3f}")
        else:
            # pv is plotted FIRST (lines[0]), which is what _clamp_ylim
            # actually reads -- masking only the IC reference curve (as
            # before) left a near-puncture divergent PINN value free to
            # dominate the axis scale, the same bug already fixed in
            # save_field_snapshot but missed in this end-of-run sibling.
            # Uses _mask_divergent (pure value-based), not _apply_plot_nan:
            # this loop spans t=0..tfinal in one pass, but _apply_plot_nan's
            # geometric pre-filter is anchored to _bh_display_mask_zones'
            # LAST-window position -- stale for the earlier t_now values
            # under a moving-puncture mode.
            ax.plot(x_vals, _mask_divergent(pv), label="PINN")
            ax.plot(x_vals, _mask_divergent(ic_v_1d[i]),
                    "--", color="tab:orange", label="IC (t=0)")
            ax.set_title(f"{qname}, t={t_now:.3f}")
        ax.set_xlim(x_min, x_max); ax.grid(True)
        if i == 0:
            ax.set_ylim(-0.05, 1.1)
        elif not _is_H:
            _clamp_ylim(ax)
        if j == 0:
            ax.set_ylabel(qname)
        if i == len(quantity_names) - 1:
            ax.set_xlabel("x")
        if i == 0 and j == 0:
            ax.legend(loc="best")

fig.suptitle(r"1D cut: $y=0$, $z=0$", fontsize=18, y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.97])
_atomic_savefig(fig, os.path.join(file_path, "diagnostics_1D_cut.png"), dpi=120, bbox_inches="tight")
plt.close(fig)


# --------------------------------------------------------------------------------
# MOMENTUM-CONSTRAINT VIOLATION  Z^i = chi*(Gamma-hat^i - Gamma-tilde^i)/2
# (1D cut y=0, z=0)
# --------------------------------------------------------------------------------
_h_labels = [r"$Z^x$", r"$Z^y$", r"$Z^z$"]
fig_gs, axes_gs = plt.subplots(3, len(t_values),
                               figsize=(5*len(t_values), 9), sharex=True)
axes_gs = np.array(axes_gs).reshape(3, len(t_values))

for j, t_now in enumerate(t_values):
    Ha = eval_gauge_source_pinn(window_states, x_vals, y_cut, z_cut, t_now)   # (nx, 3): Z^i
    _Z_cache[t_now] = Ha  # reused by the Constraint summary below
    for i, label in enumerate(_h_labels):
        ax = axes_gs[i, j]
        vals = np.abs(np.asarray(Ha[:, i], dtype=float))
        vals_plot = np.where(vals > 0, vals, np.nan)
        ax.plot(x_vals, vals_plot)
        _safe_log_yscale(ax, vals_plot)
        ax.set_title(f"{label},  t={t_now:.3f}")
        ax.set_xlim(x_min, x_max)
        ax.grid(True, which="both", ls=":")
        if j == 0:
            ax.set_ylabel(label)
        if i == 2:
            ax.set_xlabel("x")

fig_gs.suptitle(
    r"Momentum-constraint violation $Z^i = \chi(\hat\Gamma^i-\tilde\Gamma^i)/2$,"
    r" 1D cut $y=z=0$",
    fontsize=14, y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.97])
_atomic_savefig(fig_gs, os.path.join(file_path, "diagnostics_gauge_source.png"), dpi=120, bbox_inches="tight")
plt.close(fig_gs)


# --------------------------------------------------------------------------------
# 2D SLICES  z=0
# --------------------------------------------------------------------------------
nxy    = 160
x_grid = np.linspace(x_min, x_max, nxy)
y_grid = np.linspace(y_min, y_max, nxy)
Xg, Yg = np.meshgrid(x_grid, y_grid, indexing="xy")
Zg     = np.zeros_like(Xg)
# Only the out-of-domain mask is geometric/permanent; the near-puncture
# divergence masking below uses _mask_divergent (pure value-based) per
# time slice instead of a fixed _bh_display_mask_zones-based circle --
# this loop spans t=0..tfinal, but that zone is anchored to the LAST
# window's tracked position, which is wrong for the earlier slice_times
# under a moving-puncture mode (would mask ordinary far-field pixels at
# the stale position while leaving the ACTUAL, moved divergence unmasked).
_outside_domain = (Xg**2 + Yg**2) > R_domain**2

q_labels_2d = [r"$\alpha$", r"$\beta^x$", r"$g_{xx}$", r"$H$"]
slice_times  = [0.0, tfinal / 2.0, tfinal]

xr, yr, zr = Xg.ravel(), Yg.ravel(), Zg.ravel()
_alpha_ic_2d, _betax_ic_2d, _gxx_ic_2d = eval_metric_reference(xr, yr, zr)
ic_cols_2d = [_alpha_ic_2d, _betax_ic_2d, _gxx_ic_2d, np.zeros(len(xr))]

for t_now in slice_times:
    out_full_2d = eval_full_pinn(window_states, xr, yr, zr, t_now)
    alpha_2d, betax_2d, gxx_cov_2d = _bh_adm_from_out(out_full_2d)
    H_2d = eval_H_pinn(window_states, xr, yr, zr, t_now)
    pinn_cols = [alpha_2d, betax_2d, gxx_cov_2d, H_2d]

    fig2, axes2 = plt.subplots(2, len(q_labels_2d),
                               figsize=(5 * len(q_labels_2d), 10), constrained_layout=True)
    for col, (qname, pinn_data, ic_data) in enumerate(zip(q_labels_2d, pinn_cols, ic_cols_2d)):
        for row, (img, row_label) in enumerate([(pinn_data, "PINN"), (ic_data, "IC (t=0)")]):
            ax = axes2[row, col]
            # Value-based divergence masking (see _mask_divergent) first,
            # THEN the permanent out-of-domain mask.
            _img = _mask_divergent(np.asarray(img, dtype=float)).reshape(nxy, nxy).copy()
            _img[_outside_domain] = np.nan
            im = ax.imshow(_img, origin="lower",
                           extent=[x_min, x_max, y_min, y_max], aspect="equal")
            ax.set_facecolor("0.85")
            fig2.colorbar(im, ax=ax)
            ax.set_title(f"{row_label}  {qname}", fontsize=10)
            ax.set_xlabel("x"); ax.set_ylabel("y")

    fig2.suptitle(rf"2D slice $z=0$, $t={t_now:.3f}$", fontsize=18)
    _atomic_savefig(fig2, os.path.join(file_path, f"diagnostics_2D_t{t_now:.4f}.png"),
                 dpi=120, bbox_inches="tight")
    plt.close(fig2)


# --------------------------------------------------------------------------------
# CONSTRAINT SUMMARY  (1D cut, L2 over x at each snapshot time)
# --------------------------------------------------------------------------------
print("\n" + "="*80)
print("Constraint summary — 1D cut y=0, z=0")
print("="*80)
for t_now in t_values:
    # Already computed above (1D-cut and momentum-constraint sections) for
    # this exact (x_vals, y_cut, z_cut, t_now) -- reuse instead of paying
    # for the expensive jacfwd-based CCZ4 residual machinery a second and
    # third time.
    H_now = _H_cache[t_now]
    Z_now = _Z_cache[t_now]
    H_l2  = float(np.sqrt(np.mean(np.asarray(H_now)**2)))
    Z_l2  = float(np.sqrt(np.mean(np.sum(np.asarray(Z_now)**2, axis=1))))
    print(f"  t={t_now:8.4f}   |H|_L2={H_l2:.6e}   |Z^i|_L2={Z_l2:.6e}")


# --------------------------------------------------------------------------------
# PUNCTURE TRAJECTORY  (shift_driven/chi_min/lapse_min tracks; shift_driven
# ported from bbh_headon_bssn_pinn_shift_driven_puncture.py's diagnostics.npz)
#
# Already written window-by-window (see save_puncture_track_outputs() call
# right after each window's puncture_track_history.append() above) so the
# files are viewable mid-training, not just at the end -- this final call
# is just to print the confirmation paths once training is done.
# --------------------------------------------------------------------------------
if bh_center_mode in ("shift_driven", "chi_min", "lapse_min") and puncture_track_history:
    save_puncture_track_outputs()
    print(f"\nPuncture track (window boundaries) saved to: {os.path.join(file_path, 'puncture_track.txt')}")
    print(f"Puncture track (fine, within-window) saved to: {os.path.join(file_path, 'puncture_track_fine.txt')}")
