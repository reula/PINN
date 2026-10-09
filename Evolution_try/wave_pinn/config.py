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
                  "trustregion", "adam+trustregion"),
    "tr_hessian": ("exact", "gauss_newton"),
    "window_ic": ("hard", "soft"),
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
    min_resamples: int = 5             # ...but never fewer than this many redraws per phase
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
    qn_initial_scale: bool = True      # SSBroyden tau_k^A first-step rescaling
    qn_rescale_after_adam: bool = True # use initial_scale on the first quasi-Newton block only
    plateau_tol: float = 0.0           # relative-improvement tolerance; 0 disables the plateau stop
    plateau_patience: int = 3          # consecutive blocks below plateau_tol before stopping
    qn_max_H_gb: float = 8.0           # refuse a dense inverse Hessian larger than this
    dsgnar_steps: int = 200            # DSGNAR iterations (each one rebuilds a sketched Jacobian)
    dsgnar_sketch: int = 0             # sketch size s; 0 -> max(1, round(d_theta/3))
    dsgnar_stage1_ratio: float = 0.15  # target ratio rho in stage 1 (textbook Eq. 14 units)
    dsgnar_stage2_ratio: float = 0.5   # target ratio rho in stage 2
    dsgnar_delta0: float = 1.0         # initial trust-region radius
    dsgnar_delta_min: float = 1.0e-14  # termination radius
    dsgnar_omega: float = 1.0e-8       # regularisation floor in the ratio solve
    dsgnar_chunk: int = 64             # tangents per batched JVP; 0 = all at once (GPU OOM)
    dsgnar_row_chunk: int = 0          # CountSketch rows per block; 0 = off (unneeded at M ~ 2e3)
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

    The redraw interval grows geometrically instead of staying fixed, and the
    first redraw comes early.  Two reasons, both empirical:

    * DSGNAR converges in tens of iterations here, not thousands.  A fixed
      interval sized against its *budget* (``dsgnar_steps``) therefore fires once
      or twice before the trust region collapses and the run ends -- measured: a
      run that converged at iteration 69 under ``resample_every=25`` and a
      250-iteration budget got exactly two redraws.  Sizing the first interval
      against an eighth of the budget places ``min_resamples`` redraws inside
      the first eighth of the phase, which is where a converging DSGNAR run
      actually lives: measured here, runs that were allowed 250 iterations
      stopped at 55 and 69.
    * Early iterations move the solution a lot and later ones barely at all, so
      spending the redraws early is the better use of them.
    """
    if cfg.resample_every <= 0 or total_steps <= 0:
        return 0.0, 1.0
    wanted = max(1, int(cfg.min_resamples))
    first = min(float(cfg.resample_every), max(1.0, float(total_steps) / (8.0 * wanted)))
    # A phase too short to redraw without the optimiser chasing its own sample gets
    # no redraws at all.  This is a measured rule, not a preference: on this
    # problem, 200 DSGNAR iterations at n_coll = 2201 reach a loss of 1.05e-09 with
    # nine redraws and 4.48e-15 with none -- six orders, plus 4.5x the wall time in
    # recompiles, because each redraw rebuilds the jitted objective on the device.
    # The min_resamples rule below is sound for a long phase and destructive for a
    # short one, so length decides.
    floor = int(getattr(cfg, "min_redraw_interval", 0) or 0)
    if floor and first < floor:
        return 0.0, 1.0
    return max(1.0, first), max(1.0, float(cfg.resample_growth))


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


def resolve_outdir(cfg: Config, root: Optional[str] = None) -> str:
    """Where the artefacts go; ``root`` defaults to ``<Evolution_try>/runs``."""
    if cfg.outdir:
        return os.path.abspath(cfg.outdir)
    root = root or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")
    return os.path.join(root, cfg.label)
