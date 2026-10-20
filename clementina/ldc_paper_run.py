#!/usr/bin/env python
"""Lid-driven cavity through the PAPER'S OWN trainer (``ldc_pinn.ipynb``), as a batch script.

This is a different beast from ``ldc_run.py``: that one ports ``ldc.ipynb``'s
hand-driven ``pinn.optimiser.Optimiser`` loop, while this one calls
``pinn.train64(problem, RunConfig)`` -- the reference implementation's own trainer,
with the paper's ``auto_adjust`` weight/lambda schedule, the two-pass
(float32 then float64) drive, its stopping rules and its own rel-L2 evaluation.

On the hub the SMOKE configuration of exactly this pipeline reached
``final rel error = 6.384e-02`` in 11.5 s of compute, which is the fastest good
approximation we have seen on this problem; hence this script.

Usage:
    python ldc_paper_run.py --mode smoke  --outdir /data/.../ldc_paper_smoke
    python ldc_paper_run.py --mode paper  --outdir /data/.../ldc_paper_paper     # 2^15 pts

Modes (from the notebook):
    smoke   2^11 PDE / 2^10 BC points, sketch 256,   batch 2^8,  40-40-40 net,
            float32_steps 2, min_radius 1e-2 (bounds the float64 pass)
    paper   2^15 PDE / 2^14 BC points, sketch 4000,  batch 2^12, 60-60-60-60 net,
            float32_steps 10, min_radius resolved to 1e-8 in double precision

Batch-only adaptations: Agg backend, the notebook widget display is disabled
(``Plotter.prime`` no-ops; the plotter's matplotlib thread is unaffected),
outputs go to ``--outdir``, and a CPU backend is refused unless ``--allow-cpu``.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("smoke", "paper"), default="paper")
    ap.add_argument("--outdir", default=None, help="default ./ldc_paper_run")
    ap.add_argument("--label", default=None)
    ap.add_argument("--repo", default=None)
    ap.add_argument("--reference", default=None,
                    help="reference npz (default <repo>/ldc/reference_gen/ldc_reference.npz)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--float32-steps", type=int, default=None,
                    help="override the preset's float32 pre-pass length.  0 makes the "
                         "trainer quit the pre-pass after its first step; note train64 "
                         "iterates [32, 64] hard-coded, so the float32 graph is still "
                         "traced/compiled.")
    ap.add_argument("--precompile", action="store_true",
                    help="warm the compile cache first (the notebook does this for a "
                         "production run; costs minutes and is pointless for smoke)")
    ap.add_argument("--skip-fields", action="store_true")
    ap.add_argument("--autotune", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true")
    return ap.parse_args(argv)


ARGS = parse_args()
LABEL = ARGS.label or f"ldc_paper_{ARGS.mode}"
OUTDIR = pathlib.Path(ARGS.outdir or (pathlib.Path.cwd() / "ldc_paper_run")).resolve()
FIGDIR = OUTDIR / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

# Environment before the first JAX import.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(OUTDIR / ".mpl"))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
if not ARGS.autotune:
    os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")


def _repo_root(hint=None) -> pathlib.Path:
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
sys.path.insert(0, str(REPO / "Evolution_try" / "vendor" / "dsgnar_pinn"))

REFERENCE = pathlib.Path(ARGS.reference or (REPO / "ldc" / "reference_gen" / "ldc_reference.npz"))

import numpy as np                                            # noqa: E402
import jax                                                     # noqa: E402
import jax.numpy as jnp                                        # noqa: E402

try:
    import equinox                                             # noqa: E402,F401  (paper package)
    import jaxtyping                                           # noqa: E402,F401
except ModuleNotFoundError as exc:                             # pragma: no cover
    sys.exit(f"[run ] FATAL: {exc.name!r} is missing from {sys.executable}.\\n"
             f"        uv pip install --python {sys.executable} equinox jaxtyping")

import matplotlib.pyplot as plt                                # noqa: E402
import pinn                                                    # noqa: E402
from pinn.plotting import Plotter                              # noqa: E402

# Batch context: the trainer calls plotter.prime(), whose only job is to display an
# ipywidgets.Image in Jupyter.  The plotter's matplotlib rendering happens on its own
# thread and does not need the widget, so make prime() a no-op instead of dragging
# IPython/ipywidgets into the cluster environment.
Plotter.prime = lambda self: None                              # noqa: E731

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
             "(CPU hours are scarce on this allocation); --allow-cpu is only for a "
             "local pipeline check.")
if BACKEND == "cpu":
    print("[run ] WARNING: running on the CPU because --allow-cpu was given", flush=True)


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
        x, y = coords[0], coords[1]
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
        x, y = coords[0], coords[1]
        u, v = model(coords)[0], model(coords)[1]
        u_target = jnp.where(jnp.isclose(y, 1.0), 4.0 * x * (1.0 - x), 0.0)
        return jnp.array([u - u_target, v, 0.0])

    def samplers(self):
        return {"pde": self._sample_pde, "bc": self._sample_bc}

    def _sample_pde(self, key, n, *_):
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


op = pinn.operators

SMOKE = ARGS.mode == "smoke"
cfg = pinn.RunConfig(
    network=lambda key: pinn.SIREN(
        key, LidDrivenCavityProblem, time_dependent=False,
        hidden_dims=(40, 40, 40) if SMOKE else (60, 60, 60, 60),
        periodic_bc=False, n_inputs=2, n_outputs=3,
    ),
    seed=ARGS.seed,
    # The paper package builds a dense Jacobian per macro-batch (jacrev), so batch_size and
    # the point count set peak memory: 2^12 x 3 x 11347 x 8 ~= 1.04 GiB plus temporaries.
    n_pde=2 ** 11 if SMOKE else 2 ** 15,
    n_bc=2 ** 10 if SMOKE else 2 ** 14,
    residual_sketch=256 if SMOKE else 4000,
    parameter_sketch=256 if SMOKE else 4000,
    batch_size=2 ** 8 if SMOKE else 2 ** 12,
    sub_batch_size=4,
    probe_batch_size=2 ** 7,
    cache_dir=str(CACHE_DIR),
    pde_weight=1e-2,
    min_radius=1e-2 if SMOKE else None,       # None -> 1e-8 in double precision
    float32_steps=(2 if SMOKE else 10) if ARGS.float32_steps is None else ARGS.float32_steps,
)
print(f"[cfg ] smoke={SMOKE}  n_pde={cfg.n_pde} n_bc={cfg.n_bc} sketch={cfg.residual_sketch} "
      f"batch={cfg.batch_size} float32_steps={cfg.float32_steps} min_radius={cfg.min_radius}",
      flush=True)
print(f"[cfg ] auto_adjust={cfg.auto_adjust} pde_weight={cfg.pde_weight} "
      f"target_rho={cfg.target_rho} alpha={cfg.alpha}", flush=True)
print(f"[cfg ] cache {CACHE_DIR}", flush=True)

if ARGS.precompile:
    print("[cfg ] precompiling the graphs (this is the notebook's production step)", flush=True)
    t_pc = time.perf_counter()
    pinn.precompile64(LidDrivenCavityProblem, cfg)
    print(f"[cfg ] precompile done in {time.perf_counter() - t_pc:.1f}s", flush=True)

t0 = time.perf_counter()
results = pinn.train64(LidDrivenCavityProblem, cfg)
elapsed = time.perf_counter() - t0

# ---------------------------------------------------------------------------
# Report and save (the notebook's cells 5 and 7)
# ---------------------------------------------------------------------------
ref = pinn.Reference.load(LidDrivenCavityProblem.ref_path)
x0, x1 = results["plot_x0"], results["plot_x1"]
lx, ly = results["plot_axes"]
extent = [x0[0], x0[-1], x1[0], x1[-1]]

channel_rel = {}
for ch in ref.channels:
    pred = np.array(results["u_pred_plot"][ch])
    exact = np.array(ref.plot_grids[ch])
    err = np.abs(pred - exact)
    rel = float(np.linalg.norm(pred - exact) / np.linalg.norm(exact))
    channel_rel[str(ch)] = rel
    if ARGS.skip_fields:
        continue
    fig, axs = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, data, title, cmap in zip(
            axs, [pred, exact, err],
            [f"PINN  ${ch}$", f"reference  ${ch}$",
             f"abs error  (rel $\\ell_2$={rel:.2e})"],
            ["RdBu_r", "RdBu_r", "magma"]):
        im = ax.imshow(data.T, origin="lower", aspect="auto", extent=extent, cmap=cmap)
        ax.set(xlabel=f"${lx}$", ylabel=f"${ly}$", title=title)
        fig.colorbar(im, ax=ax)
    fig.savefig(FIGDIR / f"fields_{ch}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig ] {FIGDIR / f'fields_{ch}.png'}  (rel L2 {rel:.3e})", flush=True)

np.savez_compressed(OUTDIR / "results.npz",
                    u_pred_plot=np.array([np.array(results["u_pred_plot"][c]) for c in ref.channels]),
                    plot_x0=np.asarray(x0), plot_x1=np.asarray(x1),
                    ref_solution=np.asarray(ref.solution), ref_coords=np.asarray(ref.coords),
                    u_pred_final=np.asarray(results["u_pred_final"]))
def _plain(v):
    """numpy/json-friendly copy of a slab field (the trainer stores per-channel dicts)."""
    if isinstance(v, dict):
        return {str(k): float(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)) and v and isinstance(v[0], (int, float, np.floating)):
        return [float(x) for x in v]
    if isinstance(v, (int, float, np.floating)):
        return float(v)
    return None


slabs = []
for slab in results["slabs"]:
    slabs.append({k: _plain(v) for k, v in slab.items()
                  if k in ("t0", "t1", "global_step", "wall_time", "rel_l2", "pure_l2",
                           "n_steps", "final_weights")})
final_rel = slabs[-1]["rel_l2"]                      # {channel: relative L2}
final_rel_mean = float(np.mean(list(final_rel.values())))
summary = {
    "label": LABEL, "mode": ARGS.mode, "smoke": SMOKE,
    "backend": BACKEND, "devices": DEVICES, "jax": jax.__version__,
    "config": {"n_pde": cfg.n_pde, "n_bc": cfg.n_bc, "residual_sketch": cfg.residual_sketch,
               "parameter_sketch": cfg.parameter_sketch, "batch_size": cfg.batch_size,
               "sub_batch_size": cfg.sub_batch_size, "probe_batch_size": cfg.probe_batch_size,
               "n_probes": cfg.n_probes, "window_scale": cfg.window_scale,
               "target_rho": cfg.target_rho, "pde_weight": cfg.pde_weight,
               "auto_adjust": cfg.auto_adjust, "alpha": cfg.alpha,
               "float32_steps": cfg.float32_steps, "min_radius": cfg.min_radius},
    "network": {"hidden_dims": list((40, 40, 40) if SMOKE else (60, 60, 60, 60))},
    "wall_seconds": elapsed,
    "slabs": slabs,
    "final_rel_l2_slab": final_rel,
    "final_rel_l2_slab_mean": final_rel_mean,
    "rel_l2_per_channel_from_plot_grid": channel_rel,
    "outdir": str(OUTDIR),
}
(OUTDIR / "summary.json").write_text(json.dumps(summary, indent=2))

print("\n[final] " + json.dumps({"label": LABEL, "mode": ARGS.mode,
                                 "trainer_final_rel_l2_mean": final_rel_mean,
                                 "trainer_final_rel_l2": final_rel,
                                 "per_channel": {k: round(v, 5) for k, v in channel_rel.items()},
                                 "wall_seconds": round(elapsed, 1)}, indent=0), flush=True)
print(f"[done ] {OUTDIR}", flush=True)
