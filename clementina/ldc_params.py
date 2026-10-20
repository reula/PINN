#!/usr/bin/env python
"""Score a saved LDC network (parameters + architecture) against the reference field.

    python ldc_params.py --run /data/.../ldc_paper_paper
    python ldc_params.py --params ldc/params/hub_a30_dsgnar_48x48x48.npy \
                         --network ldc/params/hub_a30_dsgnar_48x48x48.json
    python ldc_params.py --run <dir> --outdir <dir>      # also (re)make field figures

Prints the relative L2 per channel and the mean, and with --outdir writes
`eval_summary.json`, `eval_fields.npz` and `figures/fields_{u,v,p}.png`.

The network is rebuilt from `network.json` (hidden_dims, seed, SIREN options) and the
flat vector is checked against that architecture before it is used, so a mismatched
pair fails loudly instead of producing a plausible-looking wrong field.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

os.environ.setdefault("MPLBACKEND", "Agg")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None,
                    help="run directory holding network.json + params_flat.npy")
    ap.add_argument("--params", default=None, help="flat parameter vector (.npy)")
    ap.add_argument("--network", default=None, help="network.json for --params")
    ap.add_argument("--reference", default=None,
                    help="reference npz (default <repo>/ldc/reference_gen/ldc_reference.npz)")
    ap.add_argument("--outdir", default=None,
                    help="write eval_summary.json / eval_fields.npz / figures here")
    ap.add_argument("--skip-fields", action="store_true")
    return ap.parse_args(argv)


ARGS = parse_args()
if not ARGS.run and not (ARGS.params and ARGS.network):
    sys.exit("give --run DIR, or --params P.npy --network N.json")


def _repo_root() -> pathlib.Path:
    marker = pathlib.Path("Evolution_try") / "vendor" / "dsgnar_pinn" / "pinn" / "__init__.py"
    starts = [os.environ.get("CXXI_REPO"), str(pathlib.Path(__file__).resolve().parent),
              str(pathlib.Path.cwd())]
    for start in starts:
        if not start:
            continue
        base = pathlib.Path(start).resolve()
        for p in (base, *base.parents):
            if (p / marker).exists():
                return p
    raise RuntimeError("repo root not found: set CXXI_REPO")


REPO = _repo_root()
sys.path.insert(0, str(REPO / "Evolution_try" / "vendor" / "dsgnar_pinn"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np                                             # noqa: E402
import jax                                                     # noqa: E402
import jax.numpy as jnp                                        # noqa: E402
import equinox as eqx                                          # noqa: E402
import pinn                                                    # noqa: E402
from ldc_network_io import (build_model, load_network, model_meta,   # noqa: E402
                            rel_l2, LidDrivenCavityProblem, NETWORK_FILE)

jax.config.update("jax_enable_x64", True)

if ARGS.run:
    model, meta = load_network(ARGS.run)
    source = str(ARGS.run)
else:
    meta = json.loads(pathlib.Path(ARGS.network).read_text())
    flat = np.load(ARGS.params)
    model = build_model(meta)
    params, static = eqx.partition(model, eqx.is_array)
    ref_flat, unflatten = jax.flatten_util.ravel_pytree(params)
    if int(flat.size) != int(ref_flat.size):
        sys.exit(f"{ARGS.params}: {flat.size} parameters, but "
                 f"network {meta.get('hidden_dims')} needs {ref_flat.size}")
    model = eqx.combine(unflatten(jnp.asarray(flat, dtype=jnp.float64)), static)
    source = f"{ARGS.params} + {ARGS.network}"

ref_path = pathlib.Path(ARGS.reference or (REPO / "ldc" / "reference_gen" / "ldc_reference.npz"))
LidDrivenCavityProblem.ref_path = str(ref_path)
ref = pinn.Reference.load(ref_path)

scores = rel_l2(model, ref)
print(f"[params] {source}")
print(f"[net   ] {meta.get('driver', '?')} | hidden {meta.get('hidden_dims')} | "
      f"{meta.get('n_parameters', '?')} parameters | seed {meta.get('seed', '?')}")
print("[score ] " + json.dumps({k: round(v, 6) for k, v in scores.items()}), flush=True)

if ARGS.outdir:
    out = pathlib.Path(ARGS.outdir); out.mkdir(parents=True, exist_ok=True)
    figdir = out / "figures"; figdir.mkdir(exist_ok=True)
    x0, x1 = np.asarray(ref.plot_x0), np.asarray(ref.plot_x1)
    X, Y = np.meshgrid(x0, x1, indexing="ij")
    grid_pts = np.stack([X.ravel(), Y.ravel()], 1)
    pred = np.asarray(jax.jit(lambda m, c: jax.vmap(lambda x: m(x))(c))(
        model, jnp.asarray(grid_pts, dtype=jnp.float64))).reshape(x0.size, x1.size, -1)
    np.savez_compressed(out / "eval_fields.npz", pred_grid=pred,
                        plot_x0=x0, plot_x1=x1,
                        ref_solution=np.asarray(ref.solution),
                        ref_coords=np.asarray(ref.coords))
    (out / "eval_summary.json").write_text(json.dumps(
        {"source": source, "network": meta, "rel_l2": scores}, indent=2))

    if not ARGS.skip_fields:
        import matplotlib.pyplot as plt
        extent = [x0[0], x0[-1], x1[0], x1[-1]]
        for i, ch in enumerate([str(c) for c in ref.channels]):
            p = pred[..., i]
            e = np.array(ref.plot_grids[ch])
            if ch == "p":
                p, e = p - p.mean(), e - e.mean()
            err = np.abs(p - e)
            rel = np.linalg.norm(p - e) / np.linalg.norm(e)
            fig, axs = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
            for ax, data, title, cmap in zip(
                    axs, [p, e, err],
                    [f"PINN  ${ch}$", f"reference  ${ch}$",
                     f"abs error  (rel $\\ell_2$={rel:.2e})"],
                    ["RdBu_r", "RdBu_r", "magma"]):
                im = ax.imshow(data.T, origin="lower", aspect="auto", extent=extent, cmap=cmap)
                ax.set(xlabel="$x$", ylabel="$y$", title=title)
                fig.colorbar(im, ax=ax)
            fig.savefig(figdir / f"fields_{ch}.png", dpi=140, bbox_inches="tight")
            plt.close(fig)
            print(f"[fig   ] {figdir / f'fields_{ch}.png'}  (rel L2 {rel:.3e})", flush=True)
    print(f"[saved ] {out}", flush=True)
