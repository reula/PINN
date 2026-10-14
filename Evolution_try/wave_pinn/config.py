"""Every knob of a wave-equation PINN experiment, in one dataclass.

The project is deliberately scheme-oriented: the *equation*, the *hard-coded
initial data*, the *input feature map*, the *ansatz* and the *optimiser* are all
selected by name, so a new idea is a new branch in one of the small dispatch
functions rather than a new script.

Nothing here imports JAX, so a config can be inspected, serialised and reloaded
without paying for a JAX import.

Usage
-----
>>> cfg = Config(n_modes=8, optimizer="dsgnar")
>>> cfg = Config.from_json("runs/foo/config.json")
>>> cfg = apply_overrides(cfg, ["optimizer=adam", "n_coll=2048"])   # --set k=v
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------
# what each string-valued field is allowed to be; used by validate()
# --------------------------------------------------------------------------
CHOICES: Dict[str, tuple] = {
    "equation": ("wave2", "advection"),
    "u0": ("gaussian", "sin", "sin2", "sech2", "cosine_bump", "poly_bump"),
    "ansatz": ("t2", "t2sat", "t", "net"),
    "features": ("periodic", "periodic_ic", "fourier", "fourier_ic", "plain", "plain_ic"),
    "activation": ("tanh", "sin", "gelu", "relu", "softplus"),
    "init": ("glorot", "lecun", "siren", "zeros", "uniform"),
    "sampler": ("uniform", "random", "grid_random", "lhs"),
    "optimizer": ("adam", "ssbroyden", "adam+ssbroyden", "dsgnar", "adam+dsgnar",
                  "trustregion", "adam+trustregion", "jaxopt_broyden",
                  "jaxopt_bfgs", "jaxopt_lbfgs", "jaxopt_ncg", "optax_lbfgs"),
    "tr_hessian": ("exact", "gauss_newton"),
    "wolfe_method": ("bfgs", "lbfgs", "ncg"),
    "qn_linesearch": ("backtracking", "wolfe"),
    "dsgnar_pick": ("rho", "best"),
    "window_ic": ("hard", "soft", "soft_all"),
    "overwrite": ("archive", "wipe", "fail"),
    "precision": ("float64", "float32"),
    "scheduler": ("none", "plateau", "cosine"),
    "residual_norm": ("auto", "none"),
}


@dataclass
class Config:
    """A complete description of one run.

    The defaults are the reference configuration requested for this project:
    the second-order wave equation ``u_tt = c^2 u_xx`` on ``[-L, L]`` with
    ``L = 1``, ``c = 1``, integrated to ``T = 2``, periodic in ``x`` (enforced by
    cos/sin features), a hard-coded initial condition
    ``u = u0(x) + t v0(x) + t^2 N(t, x)`` with ``v0 = -c u0'``, and a
    6-layer x 20-neuron network.
    """

    # ---------------------------------------------------------------- physics
    L: float = 1.0                     # domain is x in [-L, L]
    c: float = 1.0                     # wave speed
    T: float = 2.0                     # final time
    equation: str = "wave2"            # "wave2" (u_tt - c^2 u_xx = 0) | "advection" (u_t + c u_x = 0)
    periodic: bool = True              # periodic in x, imposed by the feature map
    u0: str = "gaussian"               # initial profile name, see problem.initial_data
    u0_sigma: float = 0.2              # width parameter (gaussian / sech2 / cosine_bump)
    u0_amp: float = 1.0                # amplitude
    periodize_ic: bool = True          # replace u0 by its smooth 2L-periodic sum
    # ---------------------------------------------------------------- ansatz
    ansatz: str = "t2"                 # "t2": u0 + t v0 + t^2 N | "t2sat": u0 + t v0 + t^2/(tau^2+t^2) N | "t": u0 + t N
    ansatz_tau: float = 1.0            # the saturation scale in "t2sat"
    windows: int = 1                   # time slabs; 1 = one global solve on [0, T]
    window_ic: str = "hard"            # how a window inherits the previous one:
                                       # "hard": build it into the ansatz (depth one,
                                       #   needs the edge as a differentiable function)
                                       # "soft": a plain network plus a penalty pulling
                                       #   it to the stored edge values (window 1 always
                                       #   keeps the hard-coded physical IC)
    w_ic: float = 1.0e2                # weight of the soft initial-condition penalty
    rounds_loss_floor: float = 1.0e-16 # stop the converge/redraw rounds once the loss is at
                                       # the numerical floor: further rounds run on rounding
                                       # noise, where the Gauss-Newton model is degenerate
                                       # (measured: rho = -4.9e13, then a hang).  0 disables.
    window_retries: int = 2            # re-draw and retry a window whose optimiser stalled
    window_stall_factor: float = 1.0e3 # stalled = final loss above this times the best
                                       # final loss seen so far in the run
    ic_grid: int = 256                 # points on the edge grid for the window hand-over
    ic_modes: int = 64                 # Fourier modes kept in the edge representation
    t_scale: float = 0.0               # time-feature scale; 0 = the window width
                                       # (or T when there is a single window).  A positive
                                       # value overrides it -- for controlled comparisons.
    # ------------------------------------------------------------- features
    features: str = "periodic"         # input feature map, see features.py
    n_modes: int = 1                   # harmonics, used only by features="fourier"
    # ---------------------------------------------------------------- network
    n_layers: int = 6                  # hidden layers
    n_neurons: int = 20                # neurons per hidden layer
    activation: str = "tanh"
    init: str = "glorot"
    init_scale: float = 1.0            # multiplies the init bound; SIREN wants ~1.0 with sin
    # --------------------------------------------------------------- sampling
    sampler: str = "random"
    n_coll: int = 0                    # collocation points; 0 -> one per trainable parameter
    n_boundary: int = 128              # points per periodic boundary side (only for the penalty term)
    w_periodic: float = 0.0            # weight of an explicit periodic-BC penalty (0 = off)
    w_l2: float = 0.0                  # weight of an L2 penalty on the network weights
    residual_norm: str = "auto"        # "auto": divide the residual by its scale on the batch
                                       # (a conditioning device; the ratio DSGNAR uses is
                                       # scale invariant, and the error metrics are unaffected)
    resample_every: int = 250          # redraw the collocation points every N steps (0 = never)
    resample_rounds: int = 5           # converge fully, then redraw and converge again,
                                       # this many times.  The redraw happens BETWEEN
                                       # convergence runs, so the optimiser is never
                                       # chasing a moving target -- and whether the
                                       # solution moves at all between rounds is the
                                       # measurement of stability to resampling.
                                       # 1 = a single run, no redraws.
    resample_span: float = 0.15        # with min_resamples > 0: the fraction of the phase
                                       # by which the guaranteed redraws must be complete
                                       # (i.e. the assumed early-stopping point).
    min_resamples: int = 0             # 0 = resample_every is a literal fixed period.  A
                                       # positive value instead GUARANTEES that many redraws
                                       # per phase, front-loading them geometrically.
    resample_growth: float = 1.5       # interval multiplier between redraws (front-loaded)
    min_redraw_interval: int = 0       # 0 = redraw exactly as asked.  A positive value
                                       # suppresses redraws in phases too short to redraw
                                       # without the optimiser chasing its own sample.
    seed: int = 0
    init_from: str = ""                # path to a theta.npy (or a run directory) to warm start from
    # -------------------------------------------------------------- optimiser
    optimizer: str = "ssbroyden"       # "adam" | "ssbroyden" | "adam+ssbroyden" | "dsgnar" | "adam+dsgnar"
    steps: int = 4000                  # total first-phase steps (Adam when the optimiser has an Adam phase)
    lr: float = 1.0e-3                 # Adam learning rate
    scheduler: str = "none"            # Adam schedule: "none" | "cosine"
    qn_steps: int = 4000               # quasi-Newton / DSGNAR iteration budget
    qn_block: int = 250                # quasi-Newton iterations per block (0 = one single block)
    qn_gtol: float = 1.0e-12           # gradient tolerance handed to the line search
    qn_linesearch: str = "backtracking" # Crunch ships the SAME ssbroyden2 update twice,
                                       # differing only in the line search: vendored
                                       # backtracking, or JAX's strong-Wolfe with a zoom
                                       # phase (bfgs.py).  MEASURED and the Wolfe path loses:
                                       # T=2, n_coll=4096, 3000 iterations, same seed --
                                       # backtracking 2753 it, loss 8.263e-09, rel L2 6.702e-04;
                                       # wolfe 500 it, loss 1.229e-04, rel L2 2.218e-01, with every
                                       # block returning status 3 (zoom failed) against 11 of 12
                                       # succeeding the other way.  The default stays put.
    qn_initial_scale: bool = True      # SSBroyden tau_k^A first-step rescaling
    qn_rescale_after_adam: bool = True # use initial_scale on the first quasi-Newton block only
    plateau_tol: float = 0.0           # relative-improvement tolerance; 0 disables the plateau stop
    plateau_patience: int = 3          # consecutive blocks below plateau_tol before stopping
    qn_max_H_gb: float = 8.0           # refuse a dense inverse Hessian larger than this
    dsgnar_steps: int = 200            # DSGNAR iterations (each one rebuilds a sketched Jacobian)
    dsgnar_sketch: int = 0             # sketch size s; 0 -> max(1, round(d_theta/3)).
                                       # MEASURED, and this is the knob that matters here.
                                       # scripts/dsgnar_sketch_audit.py forms the exact
                                       # Jacobian (38.7 MB, under a second) and compares:
                                       #
                                       #   the exact J at T=2, n_coll=2201:
                                       #     singular values 6.997e+02 down to 4.557e-18,
                                       #     ratio 1.5e20; numerical rank 619 at 1e-8 of
                                       #     the largest, 914 at 1e-10, 1233 at 1e-12.
                                       #   cosine between the lifted DSGNAR step and the
                                       #   exact Gauss-Newton step, over sketch draws:
                                       #     s= 733 (d/3)  0.42 (lam=1e-3)   0.48 (lam=1)
                                       #     s= 900        0.53              0.56
                                       #     s=1100 (d/2)  0.62              0.64
                                       #     s=1500 (0.68d) 0.75             0.78
                                       #   so d/2 is NOT the optimum: the cosine is still
                                       #   climbing at 0.68 d, i.e. past the paper's range,
                                       #   and the cost is what stops it -- the SVD goes as
                                       #   s^3 (8.6x from 733 to 1500) and the JVPs as s.
                                       #   and K makes no difference: 0.315 (K=4) against
                                       #   0.303 (K=8) at lam=1e-8.
                                       #
                                       # So the limit is the COLUMN sketch, and s = d/3 sits
                                       # below the numerical rank of J -- the paper asks for s
                                       # to comfortably exceed it.  The predictions are already
                                       # honest (pred/exact 1.03 to 1.13), which is why the
                                       # calibration had nothing to fix.
                                       #
                                       # And it is not a cosmetic difference.  T=2, n_coll=2201,
                                       # K=4, same seed and same frozen batch, only s changed.
                                       # scripts/run_sketch_curve.sh, on the A30:
                                       #   s= 733 (d/3)   144 s   loss 2.283e-14   rel L2 4.580e-06
                                       #   s=1100 (d/2)   215 s   loss 2.897e-20   rel L2 4.167e-09
                                       #   s=1500 (0.68d) 276 s   loss 5.412e-23   rel L2 6.963e-10
                                       # On CPU the same two first points gave 6.756e-06 and
                                       # 4.116e-09: the LOSSES reproduce to 2%, the errors differ
                                       # by up to 1.5x because the error is far more sensitive
                                       # than the loss to GPU-versus-CPU reduction order.
                                       #
                                       # The returns collapse: d/3 -> d/2 buys 1099x for 1.49x
                                       # the time, and d/2 -> 0.68 d only 6x more for 1.28x.  The
                                       # paper's [d/3, d/2] range therefore spans three orders of
                                       # magnitude, and the bottom of it is the wrong end.  Past
                                       # d/2 the per-window solve stops being the limit -- in the
                                       # windowed chain the hand-over amplification (~1.5x per
                                       # window, ten windows) takes over, so a further 6x on each
                                       # window buys little at the end of the chain.
                                       # Run scripts/dsgnar_sketch_audit.py before choosing.
    dsgnar_stage1_ratio: float = 0.15  # target ratio rho in stage 1 (textbook Eq. 14 units)
    dsgnar_stage2_ratio: float = 0.5   # target ratio rho in stage 2
    dsgnar_delta0: float = 1.0         # initial trust-region radius
    dsgnar_delta_min: float = 1.0e-14  # termination radius
    dsgnar_omega: float = 1.0e-8       # regularisation floor in the ratio solve
    broyden_steps: int = 500           # jaxopt.Broyden: a ROOT finder on r(theta)=0
    broyden_tol: float = 1e-12         # stopping tolerance on |r|
    broyden_history: int = 0           # 0 = full Broyden memory (O(steps*n))
    broyden_stepsize: float = 0.0      # <=0 -> backtracking line search (jaxopt default)
    broyden_linesearch: str = "backtracking"
    broyden_maxls: int = 15            # line-search trials; jaxopt default
    broyden_decrease: float = 0.8      # line-search shrink per trial; jaxopt default
    dsgnar_hashes: int = 0              # K, CountSketch hashes; 0 = the module default (4).
                                       # The row sketch's variance falls like 1/K and the
                                       # predicted decrease is noise-limited at K=4, which
                                       # is what stalled every run of the sibling problem
                                       # (0.94% of the loss in 15 iterations against 87.8%)
    dsgnar_pick: str = "rho"            # how the final step is chosen among the 25 probes:
                                       # "rho" targets the ratio (the paper) and "best"
                                       # takes the lowest measured loss.  MEASURED and
                                       # rejected: "best" halves the loss and makes the
                                       # error worse (rel L2 4.249e-01 against 3.216e-01
                                       # over 45 iterations at T=2, s=733).
    dsgnar_calibrate: bool = False      # divide the model's predicted decrease by the
                                       # median measured/predicted ratio over the probes
                                       # (clamped >= 1).  OFF by default, and the reason is
                                       # measured: it fixes a 12x over-optimism on the
                                       # sibling problem, but HERE the factor is only
                                       # 1.0-3.3 and correcting a near-unity noisy factor
                                       # mostly adds variance to the radius choice.  At
                                       # s=733, K=8, 20 iterations: 0.021 calibrated against
                                       # 0.0052 not; the 30-iteration run at n_coll=2201 on
                                       # the same problem says the opposite (3.0e-03 against
                                       # 1.1e-02).  Mixed evidence does not earn a default.
    dsgnar_backtrack: float = 0.5       # scan 1, f, f^2 ... of the lifted step and take the
                                       # first length that lowers the loss
    dsgnar_backtrack_steps: int = -1    # extra fractions; NEGATIVE disables the safeguard
                                       # (the default), 4 gives 1/2 ... 1/16
    # ------------------------------------------------------------------ DSGNAR, ported
    # The three knobs below are a port from the sibling project, where an ablation
    # found the sketched model over-optimistic by a factor 12 and K = 4 the cause of
    # every stall (0.94% of the loss in 15 iterations against 87.8% at K = 8).  The
    # port is complete, tested and OFF BY DEFAULT, because measured on THIS problem
    # at s = 733 it does not transfer, and in two of four comparisons it is worse:
    #
    #   production size, cold start, to convergence (n_coll = 2201, s = 733)
    #     K=4 no-cal no-safe   186 it, loss 2.24e-14, rel L2 6.756e-06
    #     K=8 cal    safe      252 it, loss 2.35e-14, rel L2 9.277e-06
    #   warm-started from a converged theta (the rounds regime, loss 3.93e-13)
    #     K=4 no-cal no-safe   40 it, ratio 0.1413
    #     K=8 no-cal safe      25 it, ratio 0.2376, stops on "radius below threshold"
    #     K=4 cal    safe      40 it, ratio 0.1399
    #   30 iterations at n_coll = 2201, delta_min = 0 (their ablation, our problem)
    #     K=8 cal best (3.02e-03), K=4 cal worst (4.06e-02) -- their ordering
    #   20 iterations at s = 733, n_coll = 128
    #     K=8 no-cal best (0.0052), K=8 cal 0.0210 -- the opposite ordering
    #
    # The pattern that explains it: their over-optimism is a factor 12 at s = 128;
    # here it measures 1.0 to 3.3, because s = d/3 is a far milder compression of a
    # 2201-parameter problem.  Calibrating a near-unity noisy factor mostly adds
    # variance to the radius choice, and the safeguard's rule -- follow the ACCEPTED
    # length rather than delta_star -- shrinks the radius faster and reaches
    # delta_min sooner.  K = 8 costs double the sketch work for no measured gain.
    # "Mixed evidence does not earn a default" is the whole of the reasoning.
    # ------------------------------------------------- the Wolfe family
    # Measured negative control: a strong-Wolfe zoom line search fixes the
    # backtracking Broyden's explosion (x1e5-x1e9) and then stalls -- 300
    # iterations reach loss 3.3e-04 with a relative error of 0.42, against
    # DSGNAR's 2.2e-14 and 6.8e-06 in 186.  Explicit curvature is doing the
    # work on this problem; a gradient-only quasi-Newton method stops
    # somewhere else, not merely later.  See optim/wolfe.py.
    wolfe_method: str = "bfgs"          # bfgs | lbfgs | ncg (jaxopt, zoom line search)
    wolfe_steps: int = 300
    wolfe_memory: int = 10             # LBFGS history
    wolfe_tol: float = 1e-16           # gradient-norm tolerance
    wolfe_maxls: int = 30              # zoom line-search steps (optax path)
    dsgnar_chunk: int = 64             # tangents per batched JVP; 0 = all at once (GPU OOM)
    dsgnar_row_chunk: int = 0          # CountSketch rows per block; 0 = off (unneeded at M ~ 2e3)
    dsgnar_audit_stops: bool = False   # audit every 'radius below threshold' stop against the
                                       # exact Jacobian (optim/stall_audit.py, README 6.11);
                                       # appends to <outdir>/dsgnar_stop_audits.jsonl
    # ---- trust-region Newton with an exact dense Hessian (arXiv:2105.07552) ----
    tr_maxiter: int = 300              # outer iterations (the paper caps at 5000)
    tr_delta0: float = 1.0             # initial trust radius   (scipy default)
    tr_delta_max: float = 1.0e3        # maximum trust radius   (scipy default)
    tr_eta: float = 0.15               # accept iff rho > eta; scipy requires eta < 0.25
    tr_contract_below: float = 0.25    # rho below this  -> radius *= tr_contract
    tr_contract: float = 0.25
    tr_expand_above: float = 0.75      # rho above this *and* the step hit the boundary
    tr_expand: float = 2.0             #                  -> radius *= tr_expand
    tr_gtol: float = 1.0e-10           # stop when ||g||_2 < tr_gtol (scipy compares the 2-norm)
    tr_subproblem_maxiter: int = 25    # secular-equation Newton cap (scipy >= 1.17 default)
    tr_hessian: str = "exact"          # "exact": the paper's indefinite Hessian.
                                       # "gauss_newton": (2/M) J^T J -- a *different*
                                       # algorithm, kept only as an ablation.
    tr_chunk: int = 128                # forward-mode chunk size for the dense Hessian
    # ------------------------------------------------------------ bookkeeping
    precision: str = "float64"
    outdir: str = ""                   # empty -> runs/<timestamp>-<label>
    overwrite: str = "archive"         # what to do when the output directory already
                                       # holds a run: move it aside as <dir>_prev_<stamp>
                                       # (archive), delete it (wipe), or refuse (fail).
                                       # Reusing a label used to write over the old run
                                       # while leaving files the new one does not
                                       # produce -- a 10-window run's theta_window5..9
                                       # surviving into a 5-window run, for instance.
    label: str = "run"
    log_every: int = 25                # print/record every N optimiser steps
    eval_every: int = 0                # if >0, record the test error every N steps (needs the exact solution)
    snapshot_times: List[float] = field(default_factory=lambda: [0.0, 0.5, 1.0, 1.5, 2.0])
    n_test_x: int = 401                # spatial resolution of the error measurement
    threads: int = 0                   # 0 -> leave JAX's default

    # ------------------------------------------------------------------ utils
    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise KeyError(f"unknown config fields: {sorted(unknown)}")
        return cls(**d)

    @classmethod
    def from_json(cls, path: str) -> "Config":
        with open(path) as fh:
            return cls.from_dict(json.load(fh))

    def replace(self, **kw) -> "Config":
        return dataclasses.replace(self, **kw)

    def validate(self) -> None:
        for name, allowed in CHOICES.items():
            got = getattr(self, name)
            if got not in allowed:
                raise ValueError(f"{name}={got!r} not in {allowed}")
        if self.n_layers < 1 or self.n_neurons < 1:
            raise ValueError("n_layers and n_neurons must be >= 1")
        if self.n_coll < 0:
            raise ValueError("n_coll must be >= 0 (0 means one point per trainable parameter)")
        if self.resample_every < 0:
            raise ValueError("resample_every must be >= 0")
        if self.resample_every and self.sampler == "uniform":
            # The uniform sampler is a deterministic tensor grid; it ignores the
            # key, so "redrawing" would silently return exactly the same points.
            raise ValueError(
                "sampler='uniform' is deterministic, so resample_every would redraw the "
                "same points; use 'random', 'grid_random' or 'lhs' (or set resample_every=0)")
        if self.features.startswith("fourier") and self.n_modes < 1:
            raise ValueError("fourier features need n_modes >= 1")
        if self.features.startswith("plain") and self.periodic:
            raise ValueError(
                "features='plain' feeds a raw x, so the ansatz cannot be exactly periodic; "
                "use features='periodic' or set periodic=False")
        if self.equation == "advection" and not self.periodic:
            raise ValueError(
                "non-periodic advection needs an inflow boundary condition, which this "
                "project does not implement yet; set periodic=True")
        if self.ansatz == "t2" and self.equation == "advection":
            # the t^2 ansatz is tuned for the second-order equation; it still works, but warn
            pass
        if self.u0_sigma <= 0.0:
            raise ValueError("u0_sigma must be positive")

    # ------------------------------------------------------- derived quantities
    @property
    def n_hidden(self) -> int:
        return self.n_layers


def resample_schedule(cfg: "Config", total_steps: int) -> tuple:
    """``(first_redraw, growth)``; ``first_redraw == 0`` means "never redraw".

    ``resample_every`` means what it says: a **fixed period**, the first redraw
    after exactly that many iterations, and ``growth = 1``.  Set
    ``min_resamples > 0`` to ask instead for a *guaranteed number* of redraws
    within the phase, which front-loads them geometrically -- the interval is then
    ``min(resample_every, total/(8*min_resamples))`` growing by
    ``resample_growth``.

    The two modes are separate because fusing them was a mistake: with
    ``min_resamples`` on by default, ``resample_every=250`` on a 500-iteration
    phase redrew at step 13, not 250, which is not something a reader of the
    config can see.  If a phase is short enough that a fixed period never fires,
    the honest outcome is that no redraw happens.

    ``min_redraw_interval`` (default 0, off) additionally suppresses redraws when
    the interval would be shorter than it.  It was added on the strength of a
    comparison that turned out to be confounded -- see the note on `resample_span`
    -- so it is off by default and should be read as a knob, not a finding.
    """
    if cfg.resample_every <= 0 or total_steps <= 0:
        return 0.0, 1.0
    wanted = int(getattr(cfg, "min_resamples", 0) or 0)
    if wanted > 0:
        growth = max(1.0, float(cfg.resample_growth))
        # Redraws fall at t, t*g, ..., t*g^(m-1), so putting the last guaranteed one
        # at `resample_span` of the phase means t = span*total/g^(m-1).  The span is
        # where the assumption lives: a DSGNAR phase usually stops on its own radius
        # criterion well before its budget, so redraws have to be done early or they
        # never happen.  This replaces a hard-coded 8, which was 1.5^4 = 5.06 with
        # 1.6x of unexplained margin and quietly meant span = 0.13.
        span = float(getattr(cfg, "resample_span", 0.15))
        first = min(float(cfg.resample_every),
                    max(1.0, span * float(total_steps) / (growth ** (wanted - 1))))
    else:
        first = float(cfg.resample_every)
        growth = 1.0                                  # a fixed period
    floor = int(getattr(cfg, "min_redraw_interval", 0) or 0)
    if floor and first < floor:
        return 0.0, 1.0
    return max(1.0, first), growth


def resample_interval(cfg: "Config", total_steps: int) -> int:
    """The *first* redraw interval for a phase of ``total_steps`` (0 = never).

    Kept as a thin wrapper so callers that only want to report a number, and the
    tests, do not have to know about the geometric schedule.
    """
    first, _growth = resample_schedule(cfg, total_steps)
    return int(first)


def _coerce(text: str) -> Any:
    """Turn a command-line string into the most plausible python literal."""
    low = text.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("none", "null"):
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    if text.startswith("[") or text.startswith("{"):
        return json.loads(text)
    return text


def apply_overrides(cfg: Config, overrides: Optional[List[str]]) -> Config:
    """Apply ``key=value`` strings (a ``--set``) to a config, with type coercion."""
    if not overrides:
        return cfg
    updates: Dict[str, Any] = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override {item!r} is not of the form key=value")
        key, _, value = item.partition("=")
        key = key.strip()
        if key not in {f.name for f in dataclasses.fields(Config)}:
            raise KeyError(f"unknown config field in --set: {key!r}")
        updates[key] = _coerce(value.strip())
    return cfg.replace(**updates)


def prepare_outdir(cfg: Config, outdir: Optional[str] = None) -> Optional[str]:
    """Make the output directory ready, without ever mixing two runs.

    A run writes its artefacts into ``outdir`` and does not delete anything, so
    reusing a label used to leave whatever the previous run produced.  ``config.json``
    is written by every run and by nothing else, so its presence identifies a
    previous run here; when it is found the directory is moved aside (default),
    deleted, or refused, according to ``cfg.overwrite``.

    Returns the path the previous run was archived to, or None.
    """
    outdir = outdir or resolve_outdir(cfg)
    if not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)
        return None
    if "config.json" not in os.listdir(outdir):
        return None                        # nothing of ours here; leave it alone
    mode = getattr(cfg, "overwrite", "archive")
    if mode == "fail":
        raise SystemExit(
            f"{outdir} already holds a run.  Move it, or pass "
            f"--set overwrite=archive (move it aside) or --set overwrite=wipe.")
    if mode == "wipe":
        shutil.rmtree(outdir)
        os.makedirs(outdir, exist_ok=True)
        return None
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dest, n = f"{outdir}_prev_{stamp}", 1
    while os.path.exists(dest):
        dest, n = f"{outdir}_prev_{stamp}_{n}", n + 1
    shutil.move(outdir, dest)
    os.makedirs(outdir, exist_ok=True)
    return dest


def resolve_outdir(cfg: Config, root: Optional[str] = None) -> str:
    """Where the artefacts go; ``root`` defaults to ``<Evolution_try>/runs``."""
    if cfg.outdir:
        return os.path.abspath(cfg.outdir)
    root = root or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")
    return os.path.join(root, cfg.label)
