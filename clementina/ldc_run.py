#!/usr/bin/env python
"""Lid-driven cavity: the paper's DSGNAR optimiser vs SSBroyden, as a batch script.

This is ``ldc/ldc.ipynb`` turned into a script so it can run under Slurm on
Clementina XXI instead of on a Jupyter kernel.  The numerics are the notebook's,
cell for cell:

  * steady incompressible Navier-Stokes, Re = 100, on [0,1]^2, lid u = 4x(1-x);
  * DSGNAR is the paper's own implementation vendored at
    ``Evolution_try/vendor/dsgnar_pinn`` (Webb, Jerad, Cartis, arXiv:2607.02194),
    driven step by step as its ``trainer64`` does: float32 pre-pass, then
    float64, 24 radius probes, target rho 0.075, accept iff rho > 0;
  * SSBroyden is the merged Crunch, on the same objective and point sets, from
    the same initial parameters, plus a polish run from DSGNAR's result;
  * every iterate is graded against ``ldc/reference_gen/ldc_reference.npz``
    (relative L2 per channel, not just the loss).

What changed versus the notebook (deliberately only plumbing):

  * headless (Agg backend, no ``plt.show``); figures go to ``<outdir>/figures/``;
  * outputs (PNGs, ``results.npz``, ``summary.json``) go to ``--outdir`` instead
    of the checkout, so nothing writes into the repo;
  * ``--mode`` / ``--dsgnar-steps`` / ``--qn-maxiter`` override the presets, and
    ``--allow-cpu`` exists only for a local pipeline check (see the guard below);
  * every progress line is flushed, so a killed job still leaves its log.

Usage:
    python ldc_run.py --mode a30 --outdir /data/.../ldc/ldc_a30
    python ldc_run.py --mode smoke --dsgnar-steps 3 --qn-maxiter 3 --allow-cpu --outdir /tmp/x

``--mode`` presets (from the notebook):
    smoke  40-40-40, 2^11/2^10 points, sketch 256,  batch 256,   40 DSGNAR steps
    a30    48-48-48, 2^13/2^12 points, sketch 1665, batch 1024, 200 DSGNAR steps
    paper  60-60-60-60, 2^15/2^14, sketch 4000, batch 4096,     200 steps, paper numbers
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time


# ---------------------------------------------------------------------------
# Arguments first: some of them decide environment variables that must be set
# before JAX is imported.
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("smoke", "a30", "paper"), default="a30",
                    help="size preset (default a30: fits one 48-64 GB tile comfortably)")
    ap.add_argument("--outdir", default=None,
                    help="where figures/results/summary go (default ./ldc_run)")
    ap.add_argument("--reference", default=None,
                    help="reference npz (default <repo>/ldc/reference_gen/ldc_reference.npz)")
    ap.add_argument("--repo", default=None, help="repo root (default: found automatically)")
    ap.add_argument("--label", default=None, help="label stored in summary.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dsgnar-steps", type=int, default=None,
                    help="override the preset's float64 DSGNAR step budget")
    ap.add_argument("--qn-maxiter", type=int, default=None,
                    help="override the preset's SSBroyden iteration budget")
    ap.add_argument("--float32-steps", type=int, default=None,
                    help="override the preset's float32 pre-pass length: on the Intel tile "
                         "the float32 pass overflows in the probe evaluation, so 0 skips it")
    ap.add_argument("--exact-residual", action="store_true",
                    help="skip the CountSketch on the residual rows and use the exact Gram "
                         "system (SBH/06fac11): the residual-sketch noise swamps the true "
                         "gradient near the floor and turns the trust-region step near-random")
    ap.add_argument("--skip-fields", action="store_true",
                    help="skip the per-channel field figures at the end")
    ap.add_argument("--autotune", action="store_true",
                    help="leave XLA GPU autotuning on (default off: it costs compile "
                         "time and temporary memory for little gain at these sizes)")
    ap.add_argument("--allow-cpu", action="store_true",
                    help="permit a CPU backend (only for a local pipeline check)")
    return ap.parse_args(argv)


ARGS = parse_args(argv=None)
LABEL = ARGS.label or f"ldc_{ARGS.mode}"
OUTDIR = pathlib.Path(ARGS.outdir or (pathlib.Path.cwd() / "ldc_run")).resolve()
FIGDIR = OUTDIR / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

# Environment before the first JAX import.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(OUTDIR / ".mpl"))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
if not ARGS.autotune:
    # The notebook set this for the hub's shared slice; a full tile does not need
    # autotune workspaces either, and it keeps the first trace cheap.
    os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")


def _repo_root(hint=None) -> pathlib.Path:
    """Repo root: contains Evolution_try/vendor/dsgnar_pinn/pinn/__init__.py."""
    marker = pathlib.Path("Evolution_try") / "vendor" / "dsgnar_pinn" / "pinn" / "__init__.py"
    starts = [hint, os.environ.get("CXXI_REPO"), str(pathlib.Path(__file__).resolve().parent),
              str(pathlib.Path.cwd()),
              str(pathlib.Path.home() / "Julia" / "PINN"),
              str(pathlib.Path.home() / "serafin" / "Julia" / "PINN")]
    for start in starts:
        if not start:
            continue
        base = pathlib.Path(start).resolve()
        for p in (base, *base.parents):
            if (p / marker).exists():
                return p
    raise RuntimeError("repo root not found: pass --repo or set CXXI_REPO")


REPO = _repo_root(ARGS.repo)
# The paper's own package (vendored, no reimplementation), and Crunch's SSBroyden.
sys.path.insert(0, str(REPO / "Evolution_try" / "vendor" / "dsgnar_pinn"))
sys.path.insert(0, str(REPO / "Jax"))

REFERENCE = pathlib.Path(ARGS.reference or (REPO / "ldc" / "reference_gen" / "ldc_reference.npz"))

import numpy as np                                            # noqa: E402
import jax                                                     # noqa: E402
import jax.numpy as jnp                                        # noqa: E402

try:
    import equinox as eqx                                      # noqa: E402
    import jaxtyping                                           # noqa: E402,F401  (pinn's annotations)
except ModuleNotFoundError as exc:                             # pragma: no cover
    sys.exit(f"[run ] FATAL: {exc.name!r} is missing from {sys.executable}.\n"
             f"        The vendored paper package needs equinox and jaxtyping.  Install with:\n"
             f"        uv pip install --python {sys.executable} equinox jaxtyping")

import matplotlib.pyplot as plt                                # noqa: E402
import pinn                                                    # noqa: E402
from pinn.optimiser import Optimiser                           # noqa: E402  paper's sketched TR-LM
from Crunch.Optimizers.minimize_backtracking import minimize as minimize_qn   # noqa: E402
from ldc_network_io import model_meta, save_network             # noqa: E402  params + architecture

jax.config.update("jax_enable_x64", True)
CACHE_DIR = pathlib.Path(os.environ.get("JAX_COMPILATION_CACHE_DIR")
                         or str(OUTDIR / ".jaxcache")).resolve()
CACHE_DIR.mkdir(parents=True, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", str(CACHE_DIR))

BACKEND = jax.default_backend()
DEVICES = [str(d) for d in jax.devices()]
print(f"[run ] label {LABEL}  mode {ARGS.mode}", flush=True)
print(f"[run ] repo {REPO}", flush=True)
print(f"[run ] outdir {OUTDIR}", flush=True)
print(f"[run ] jax {jax.__version__} | backend {BACKEND} | devices {DEVICES}", flush=True)
if BACKEND == "cpu" and not ARGS.allow_cpu:
    sys.exit("[run ] FATAL: JAX is on the CPU.  This job must run on an Intel GPU "
             "(CPU hours are scarce on this allocation); pass --allow-cpu only for a "
             "local pipeline check.")
if BACKEND == "cpu":
    print("[run ] WARNING: running on the CPU because --allow-cpu was given", flush=True)


# ---------------------------------------------------------------------------
# The problem (notebook cell 2)
# ---------------------------------------------------------------------------
op = pinn.operators


class LidDrivenCavityProblem(pinn.Problem):
    """Steady lid-driven cavity at Re = 100; outputs (u, v, p)."""

    x_min, x_max = 0.0, 1.0
    Re = 100.0
    problem_name = "LidDrivenCavity"
    ref_path = str(REFERENCE)

    def __init__(self, *, n_pde, n_bc, n_ic=0):
        self.n_pde, self.n_bc, self.n_ic = n_pde, n_bc, n_ic

    def residual_fns(self):
        return {"pde": self.pde_residual, "bc": self.bc_residual}

    def pde_residual(self, model, coords):
        u_ = lambda c: model(c)[0]
        v_ = lambda c: model(c)[1]
        p_ = lambda c: model(c)[2]
        u, v = u_(coords), v_(coords)
        inv_Re = 1.0 / self.Re
        res_u = (u * op.grad(u_, coords, 0) + v * op.grad(u_, coords, 1)
                 + op.grad(p_, coords, 0)
                 - inv_Re * (op.grad2(u_, coords, 0) + op.grad2(u_, coords, 1)))
        res_v = (u * op.grad(v_, coords, 0) + v * op.grad(v_, coords, 1)
                 + op.grad(p_, coords, 1)
                 - inv_Re * (op.grad2(v_, coords, 0) + op.grad2(v_, coords, 1)))
        res_div = op.grad(u_, coords, 0) + op.grad(v_, coords, 1)
        return jnp.array([res_u, res_v, 2.0 * res_div])

    def bc_residual(self, model, coords):
        u, v = model(coords)[0], model(coords)[1]
        u_target = jnp.where(jnp.isclose(coords[1], 1.0), 4.0 * coords[0] * (1.0 - coords[0]),
                             0.0)
        return jnp.array([u - u_target, v, 0.0])

    def samplers(self):
        return {"pde": self._sample_pde, "bc": self._sample_bc}

    def _sample_pde(self, key, n, *_):
        # 7/8 bulk, 1/16 near the moving lid, 1/16 in the top corners
        n_lid, n_corner = n // 16, n // 16
        n_bulk = n - n_lid - n_corner
        k_bulk, k_lx, k_ly, k_l, k_r = jax.random.split(key, 5)
        bulk = jax.random.uniform(k_bulk, (n_bulk, 2), minval=0.0, maxval=1.0)
        lid = jnp.stack([jax.random.uniform(k_lx, (n_lid,)),
                         jax.random.uniform(k_ly, (n_lid,), minval=0.9, maxval=1.0)], axis=1)
        half = n_corner // 2
        s1 = jax.random.uniform(k_l, (half, 2))
        left = jnp.stack([0.1 * s1[:, 0] ** 2, 1.0 - 0.1 * s1[:, 1] ** 2], axis=1)
        s2 = jax.random.uniform(k_r, (half, 2))
        right = jnp.stack([1.0 - 0.1 * s2[:, 0] ** 2, 1.0 - 0.1 * s2[:, 1] ** 2], axis=1)
        return jnp.concatenate([bulk, lid, left, right], axis=0)

    def _sample_bc(self, key, n, *_):
        n_corner = n // 16
        n_uniform = n - n_corner
        half = n_corner // 2
        k_pt, k_dim, k_face, k_l, k_r = jax.random.split(key, 5)
        pts = jax.random.uniform(k_pt, (n_uniform, 2), minval=0.0, maxval=1.0)
        face_dim = jax.random.randint(k_dim, (n_uniform,), 0, 2)
        face_val = jax.random.randint(k_face, (n_uniform,), 0, 2).astype(pts.dtype)
        pts = pts.at[jnp.arange(n_uniform), face_dim].set(face_val)
        left = jnp.stack([0.1 * jax.random.uniform(k_l, (half,)), jnp.ones((half,))], axis=1)
        right = jnp.stack([1.0 - 0.1 * jax.random.uniform(k_r, (half,)),
                           jnp.ones((half,))], axis=1)
        return jnp.concatenate([pts, left, right], axis=0)


sampling = pinn.sampling

# ---------------------------------------------------------------------------
# Size presets and budgets (notebook cell 3)
# ---------------------------------------------------------------------------
PRESETS = {
    "smoke": dict(hidden=(40, 40, 40), n_pde=2 ** 11, n_bc=2 ** 10, sketch=256,
                  batch=2 ** 8, probe=2 ** 7, float32_steps=2, dsgnar_steps=40,
                  qn_maxiter=200, min_radius=1e-2),
    "a30":   dict(hidden=(48, 48, 48), n_pde=2 ** 13, n_bc=2 ** 12, sketch=1665,
                  batch=1024, probe=128, float32_steps=4, dsgnar_steps=400,
                  qn_maxiter=10000, min_radius=1e-14),
    "paper": dict(hidden=(60, 60, 60, 60), n_pde=2 ** 15, n_bc=2 ** 14, sketch=4000,
                  batch=2 ** 12, probe=2 ** 8, float32_steps=10, dsgnar_steps=200,
                  qn_maxiter=1000, min_radius=1e-10),
}
P = dict(PRESETS[ARGS.mode])
for k, v in (("dsgnar_steps", ARGS.dsgnar_steps), ("qn_maxiter", ARGS.qn_maxiter),
             ("float32_steps", ARGS.float32_steps)):
    if v is not None:
        P[k] = v
HIDDEN, N_PDE, N_BC = P["hidden"], P["n_pde"], P["n_bc"]
SKETCH, BATCH, PROBE = P["sketch"], P["batch"], P["probe"]
FLOAT32_STEPS, DSGNAR_STEPS = P["float32_steps"], P["dsgnar_steps"]
QN_MAXITER, MIN_RADIUS = P["qn_maxiter"], P["min_radius"]
PDE_WEIGHT, TARGET_RHO = 1e-2, 0.075

assert N_PDE % BATCH == 0 and N_BC % BATCH == 0, (N_PDE, N_BC, BATCH)
assert N_PDE % PROBE == 0 and N_BC % PROBE == 0, (N_PDE, N_BC, PROBE)


def siren_params(in_dim, hidden, out_dim=3):
    n, prev = 0, in_dim
    for h in hidden:
        n += h * prev + h
        prev = h
    return n + out_dim * prev + out_dim


D_PARAMS = siren_params(2, HIDDEN)
print(f"[cfg ] mode {ARGS.mode}: hidden {HIDDEN} d={D_PARAMS} | n_pde {N_PDE} n_bc {N_BC} | "
      f"sketch {SKETCH} batch {BATCH} probe {PROBE}", flush=True)
print(f"[cfg ] DSGNAR {FLOAT32_STEPS} float32 + {DSGNAR_STEPS} float64 steps "
      f"(stop when radius < {10 * MIN_RADIUS:.0e}); SSBroyden {QN_MAXITER} iterations x2",
      flush=True)
print(f"[cfg ] dense Jacobian estimate {BATCH * 3 * D_PARAMS * 8 / 2 ** 30:.2f} GiB "
      f"+ sketch {SKETCH ** 2 * 8 / 2 ** 30:.3f} GiB per macro-batch", flush=True)

# ---------------------------------------------------------------------------
# Model, point sets, grading (notebook cell 4)
# ---------------------------------------------------------------------------
key = jax.random.PRNGKey(ARGS.seed)
problem = LidDrivenCavityProblem(n_pde=N_PDE, n_bc=N_BC)
model = pinn.SIREN(key, LidDrivenCavityProblem, time_dependent=False, hidden_dims=HIDDEN,
                   periodic_bc=False, n_inputs=2, n_outputs=3)
params, static = eqx.partition(model, eqx.is_array)
flat0, unflatten = jax.flatten_util.ravel_pytree(params)

fns, samplers = problem.residual_fns(), problem.samplers()
key, sk = jax.random.split(key)
CONDITIONS = tuple((fns[name], samplers[name](sk, n, 0.0, 0.0))
                   for name, n in (("pde", N_PDE), ("bc", N_BC)))
WEIGHTS = (PDE_WEIGHT, 1.0)


def paper_loss(flat_or_params, conditions=CONDITIONS, weights=WEIGHTS):
    m = flat_or_params if isinstance(flat_or_params, eqx.Module) else \
        eqx.combine(unflatten(flat_or_params), static)
    total = 0.0
    for (fn, pts), w in zip(conditions, weights):
        r = jax.vmap(lambda c: fn(m, c))(pts)
        total = total + w * jnp.mean(r ** 2)
    return total


paper_loss_jit = jax.jit(paper_loss)

ref = pinn.Reference.load(problem.ref_path)
REF_COORDS = jnp.asarray(ref.coords)
REF_SOL = np.asarray(ref.solution)
CHANNELS = [str(c) for c in ref.channels]


@jax.jit
def _predict(m, coords):
    return jax.vmap(lambda c: m(c))(coords)


def flat_of(model_params):
    f, _ = jax.flatten_util.ravel_pytree(model_params)
    return f


def rel_l2(flat_or_params):
    m = flat_or_params if isinstance(flat_or_params, eqx.Module) else \
        eqx.combine(unflatten(flat_or_params), static)
    coords = REF_COORDS.astype(jnp.float64 if jax.config.jax_enable_x64 else jnp.float32)
    pred = np.asarray(_predict(m, coords))
    out = {}
    for i, ch in enumerate(CHANNELS):
        a, b = pred[:, i], REF_SOL[:, i]
        if ch == "p":
            a, b = a - a.mean(), b - b.mean()
        out[ch] = float(np.linalg.norm(a - b) / np.linalg.norm(b))
    out["mean"] = float(np.mean([out[c] for c in CHANNELS]))
    return out


d = int(flat0.size)
try:
    limit = f"{jax.devices()[0].memory_stats()['bytes_limit'] / 2 ** 30:.1f} GiB"
except Exception:                                            # noqa: BLE001
    limit = "unknown"
print(f"[cfg ] {d} parameters | {BATCH * 3 * d * 8 / 2 ** 30:.2f} GiB dense Jacobian "
      f"| device limit {limit}", flush=True)
print(f"[ref ] {ref.describe()}", flush=True)
_initial = {"loss": float(paper_loss_jit(flat0)),
            "rel_l2": {k: round(v, 4) for k, v in rel_l2(flat0).items()}}
print(f"[ref ] initial: loss {_initial['loss']:.4e} | rel L2 "
      f"{ {k: f'{v:.3f}' for k, v in _initial['rel_l2'].items()} }", flush=True)

# ---------------------------------------------------------------------------
# DSGNAR -- the paper's own Optimiser, driven as its trainer64 does (cell 5)
# ---------------------------------------------------------------------------
solver = Optimiser(residual_sketch=SKETCH, parameter_sketch=SKETCH, n_hashes=4,
                   sub_batch_size=4, batch_size=BATCH, probe_batch_size=PROBE,
                   n_probes=24, window_scale=3.0, min_radius=MIN_RADIUS, max_radius=1e3,
                   pchip_grid=512, newton_iters=80,
                   exact_residual=ARGS.exact_residual)

ds_hist = []          # (iteration, paper loss, rel L2 mean, radius, rho, accepted, pass)
step_total = 0
key, k2 = jax.random.split(key)
opt_state = solver.init_state(k2, initial_radius=1.0)
t_dsgnar = time.perf_counter()

for bits, n_steps in ((32, FLOAT32_STEPS), (64, DSGNAR_STEPS)):
    jax.config.update("jax_enable_x64", bits == 64)          # the trainer's two passes
    dtype = jnp.float64 if bits == 64 else jnp.float32
    params = jax.tree_util.tree_map(
        lambda x: x.astype(dtype) if eqx.is_inexact_array(x) else x, params)
    key, sk = jax.random.split(key)
    conds = tuple((fns[name], samplers[name](sk, n, 0.0, 0.0))
                  for name, n in (("pde", N_PDE), ("bc", N_BC)))
    for k in range(n_steps):
        loss, params, opt_state, metrics = solver.step(
            params, static, opt_state, conds, TARGET_RHO, WEIGHTS)
        step_total += 1
        accepted = bool(metrics["accepted"])
        rl = rel_l2(params)["mean"]
        ds_hist.append((step_total, float(loss), rl, float(opt_state.radius),
                        float(metrics["rho"]), accepted, bits))
        if step_total % 5 == 0 or k == 0 or not accepted:
            print(f"[dsgnar] {step_total:4d} (float{bits})  loss {float(loss):.4e}  "
                  f"relL2 {rl:.4f}  radius {float(opt_state.radius):.2e}  "
                  f"rho {float(metrics['rho']):+.3f}  {'acc' if accepted else 'rej'}",
                  flush=True)
        if float(opt_state.radius) < 10 * MIN_RADIUS:
            print(f"[dsgnar] radius below threshold after {step_total} iterations", flush=True)
            break
    CONDITIONS_QN = conds          # the float64 pass's point sets: SSBroyden's objective
t_dsgnar = time.perf_counter() - t_dsgnar
jax.config.update("jax_enable_x64", True)
ds_params, ds_flat = params, flat_of(params)
np.save(OUTDIR / "dsgnar_flat.npy", np.asarray(ds_flat))
ds_final = {"loss": float(paper_loss_jit(ds_flat)), "rel_l2": rel_l2(ds_flat)}
print(f"[dsgnar] done: {step_total} iterations in {t_dsgnar:.1f}s | "
      f"paper loss {ds_final['loss']:.4e} | rel L2 {ds_final['rel_l2']['mean']:.4f}",
      flush=True)

# ── checkpoint the DSGNAR result NOW ────────────────────────────────────────
# The SSBroyden phase below is two runs of up to 10^4 iterations; a job killed
# there must not lose the DSGNAR solution (that is exactly how the paper-preset
# run 1548247 ended: 2 h of work and nothing on disk).
_net_meta_ds = model_meta(
    hidden_dims=HIDDEN, seed=ARGS.seed,
    driver=f"ldc.ipynb hand-driven Optimiser ({ARGS.mode} preset), DSGNAR phase only",
    config={"mode": ARGS.mode, "preset": dict(P), "n_pde": N_PDE, "n_bc": N_BC,
            "sketch": SKETCH, "batch": BATCH, "probe": PROBE,
            "pde_weight": PDE_WEIGHT, "target_rho": TARGET_RHO},
    backend=BACKEND, devices=DEVICES,
    extra={"stage": "dsgnar", "dsgnar_steps": step_total,
           "dsgnar_seconds": t_dsgnar,
           "paper_loss": ds_final["loss"],
           "rel_l2": {k: float(v) for k, v in ds_final["rel_l2"].items()}})
save_network(OUTDIR, eqx.combine(unflatten(ds_flat), static), _net_meta_ds,
             name="params_dsgnar_flat.npy", network_name="network_dsgnar.json")
print(f"[net ] DSGNAR checkpoint saved (before SSBroyden): {OUTDIR}/params_dsgnar_flat.npy",
      flush=True)

paper_loss_qn = jax.jit(lambda flat: paper_loss(flat, CONDITIONS_QN, WEIGHTS))
paper_loss_qn_grad = jax.jit(jax.value_and_grad(lambda flat: paper_loss(flat, CONDITIONS_QN,
                                                                         WEIGHTS)))

# ---------------------------------------------------------------------------
# SSBroyden -- the merged Crunch, same objective and same start (cell 6)
# ---------------------------------------------------------------------------
def run_ssbroyden(x_start, n_iter, tag):
    x, H, hist = jnp.asarray(x_start), None, []
    t0 = time.perf_counter()
    for k in range(n_iter):
        opts = {"maxiter": 1, "gtol": 1e-9, "update_method": "ssbroyden2",
                "line_search": "wolfe", "max_iter": 100, "scale_invariant": True,
                "alphamin": 1e-15, "initial_scale": (k == 0)}
        if H is not None:
            opts["initial_H"] = H
        r = minimize_qn(paper_loss_qn, paper_loss_qn_grad, x, opts)
        x, H = r.x, r.hess_inv
        loss_now = float(r.fun)
        hist.append((k + 1, loss_now, rel_l2(x)["mean"]))
        if (k + 1) % 20 == 0 or k == 0:
            print(f"[ssb:{tag}] {k + 1:4d}  loss {loss_now:.4e}  relL2 {hist[-1][2]:.4f}",
                  flush=True)
    print(f"[ssb:{tag}] {n_iter} iterations in {time.perf_counter() - t0:.1f}s | "
          f"loss {float(paper_loss_qn(x_start)):.4e} -> {hist[-1][1]:.4e} | "
          f"rel L2 {hist[-1][2]:.4f}", flush=True)
    return x, hist


qn_flat, qn_hist = run_ssbroyden(flat0, QN_MAXITER, "from x0")
qn_pol_flat, qn_pol_hist = run_ssbroyden(ds_flat, QN_MAXITER, "polish")

# Grade the candidates and keep the best by ACCURACY (the loss is not the metric).
cands = {"DSGNAR": ds_flat, "SSBroyden from x0": qn_flat, "SSBroyden polish": qn_pol_flat}
grades = {k: rel_l2(v)["mean"] for k, v in cands.items()}
losses = {k: float(paper_loss_jit(v)) for k, v in cands.items()}
best_name = min(grades, key=grades.get)
best = cands[best_name]
print("\nrel L2 (mean over channels) and paper loss:", flush=True)
for k in cands:
    print(f"  {k:20s} relL2 {grades[k]:.4f}   loss {losses[k]:.4e}", flush=True)
print(f"best by accuracy: {best_name}", flush=True)

# ---------------------------------------------------------------------------
# Figures (cells 7 and 8), headless
# ---------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.2))
it = [h[0] for h in ds_hist]
axs[0].semilogy(it, [h[1] for h in ds_hist], "-", lw=1.8, label="DSGNAR (paper code)")
axs[0].semilogy([h[0] for h in qn_hist], [h[1] for h in qn_hist], "-", lw=1.8,
                label="SSBroyden from x0")
axs[0].semilogy([h[0] for h in qn_pol_hist], [h[1] for h in qn_pol_hist], "--", lw=1.6,
                label="SSBroyden polish from DSGNAR")
axs[0].set(xlabel="iteration", ylabel="paper loss  $\\sum_i w_i\\,\\mathrm{mean}(r_i^2)$",
           title="loss")
axs[1].semilogy(it, [h[2] for h in ds_hist], "-", lw=1.8, label="DSGNAR (paper code)")
axs[1].semilogy([h[0] for h in qn_hist], [h[2] for h in qn_hist], "-", lw=1.8,
                label="SSBroyden from x0")
axs[1].semilogy([h[0] for h in qn_pol_hist], [h[2] for h in qn_pol_hist], "--", lw=1.6,
                label="SSBroyden polish from DSGNAR")
axs[1].set(xlabel="iteration", ylabel="rel $\\ell_2$ error vs reference", title="accuracy")
axs[1].axhline(grades["DSGNAR"], color="C0", ls=":", lw=1)
for a in axs:
    a.grid(alpha=0.3, which="both")
    a.legend(fontsize=8)
plt.tight_layout()
fig.savefig(FIGDIR / "optimizers_loss_and_relL2.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"[fig ] {FIGDIR / 'optimizers_loss_and_relL2.png'}", flush=True)

best_params = eqx.combine(unflatten(best), static)
x0, x1 = ref.plot_x0, ref.plot_x1
X, Y = np.meshgrid(x0, x1, indexing="ij")
grid_pts = np.stack([X.ravel(), Y.ravel()], 1)
pred_grid = np.asarray(_predict(best_params, jnp.asarray(grid_pts))).reshape(
    x0.size, x1.size, -1)
extent = [x0[0], x0[-1], x1[0], x1[-1]]

if not ARGS.skip_fields:
    for i, ch in enumerate(CHANNELS):
        p = pred_grid[..., i]
        e = np.array(ref.plot_grids[ch])
        if ch == "p":
            p, e = p - p.mean(), e - e.mean()
        err = np.abs(p - e)
        rl = np.linalg.norm(p - e) / np.linalg.norm(e)
        fig, axs = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
        for ax, data, title, cmap in zip(
                axs, [p, e, err],
                [f"PINN  ${ch}$", f"reference  ${ch}$",
                 f"abs error  (rel $\\ell_2$={rl:.2e})"],
                ["RdBu_r", "RdBu_r", "magma"]):
            im = ax.imshow(data.T, origin="lower", aspect="auto", extent=extent, cmap=cmap)
            ax.set(xlabel="$x$", ylabel="$y$", title=title)
            fig.colorbar(im, ax=ax)
        fig.savefig(FIGDIR / f"fields_{ch}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"[fig ] {FIGDIR / f'fields_{ch}.png'}  (rel L2 {rl:.2e})", flush=True)

# ---------------------------------------------------------------------------
# Everything that a later analysis needs, in the run directory
# ---------------------------------------------------------------------------
# ── parameters + the information needed to rebuild the network ───────────────
# A flat vector alone is useless: save also hidden_dims/seed/SIREN options/config.
_net_meta = model_meta(
    hidden_dims=HIDDEN,
    seed=ARGS.seed,
    driver=f"ldc.ipynb hand-driven Optimiser ({ARGS.mode} preset) + Crunch SSBroyden",
    config={"mode": ARGS.mode, "preset": dict(P), "n_pde": N_PDE, "n_bc": N_BC,
            "sketch": SKETCH, "batch": BATCH, "probe": PROBE,
            "pde_weight": PDE_WEIGHT, "target_rho": TARGET_RHO},
    backend=BACKEND, devices=DEVICES,
    extra={"best_by_accuracy": best_name,
           "rel_l2": {k: float(v) for k, v in rel_l2(best).items()},
           "paper_loss": float(paper_loss_jit(best)),
           "note": "best = the best-by-accuracy of DSGNAR / SSBroyden-from-x0 / "
                   "SSBroyden-polish; the other three vectors are saved next to it"})
save_network(OUTDIR, eqx.combine(unflatten(best), static), _net_meta)
for _name, _flat in (("dsgnar", ds_flat), ("ssb_x0", qn_flat), ("ssb_polish", qn_pol_flat)):
    np.save(OUTDIR / f"params_{_name}.npy", np.asarray(_flat))
print(f"[net ] saved parameters + network.json in {OUTDIR} "
      f"({int(np.asarray(best).size)} params)", flush=True)

np.savez_compressed(OUTDIR / "results.npz",
                    ds_flat=np.asarray(ds_flat), qn_flat=np.asarray(qn_flat),
                    qn_polish_flat=np.asarray(qn_pol_flat), best=np.asarray(best),
                    ref_coords=np.asarray(REF_COORDS), ref_solution=REF_SOL,
                    pred_grid=pred_grid, plot_x0=np.asarray(x0), plot_x1=np.asarray(x1))
hist = {"dsgnar": ds_hist, "ssb_from_x0": qn_hist, "ssb_polish": qn_pol_hist}
(OUTDIR / "history.json").write_text(json.dumps(hist, indent=1))
summary = {
    "label": LABEL, "mode": ARGS.mode, "preset": {k: v for k, v in P.items()},
    "overrides": {k: v for k, v in (("dsgnar_steps", ARGS.dsgnar_steps),
                                    ("qn_maxiter", ARGS.qn_maxiter),
                                    ("float32_steps", ARGS.float32_steps)) if v is not None},
    "backend": BACKEND, "devices": DEVICES, "jax": jax.__version__,
    "n_parameters": d, "grid": {"n_pde": N_PDE, "n_bc": N_BC, "sketch": SKETCH,
                                "batch": BATCH, "probe": PROBE},
    "initial": _initial, "dsgnar": {**ds_final, "steps": step_total,
                                    "seconds": t_dsgnar},
    "grades_rel_l2": grades, "losses": losses, "best_by_accuracy": best_name,
    "final": {"rel_l2": rel_l2(best), "loss": losses[best_name]},
    "outdir": str(OUTDIR),
    "exact_residual": ARGS.exact_residual,
}
(OUTDIR / "summary.json").write_text(json.dumps(summary, indent=2))

print("\n[final] " + json.dumps({"label": LABEL, "best": best_name,
                                 "rel_l2": {k: round(v, 4) for k, v in rel_l2(best).items()},
                                 "loss": f"{losses[best_name]:.4e}",
                                 "dsgnar_seconds": round(t_dsgnar, 1)}, indent=0),
      flush=True)
print(f"[done ] {OUTDIR}", flush=True)
