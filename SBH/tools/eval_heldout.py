"""Grade finished runs on held-out samples: the same fresh points for every run.

    JAX_ENABLE_X64=1 python tools/eval_heldout.py --runs runs/a runs/b ... [--windows 15-24]
        [--seed 2026] [--out heldout.csv] -- <the runs' physics flags, e.g. the PRESET's>

Why: a run's logged loss is measured on the sample it has just trained on. With
--qn_fixed_batch (or after many iterations on one sample) that number can be far
below the loss on new points, so training losses cannot rank optimizers.

For every window (0-based indices, default: all windows present in every run) it
draws ONE sample per window with a fixed key and the sampling centres of the
FIRST run (so all runs see identical points), uniform in time, and evaluates
each run's network for that window:

  loss_total          compute_bfgs_loss as in training (causal weights, lam_*,
                      IC targets = this run's own previous window at t_start)
  ein_int, gauge_int  plain mean squares of the 18 gravity / 4 gauge residual rows
                      on the interior points (no causal or lam weights)
  ein_tapa            same at the window-end surface
  bc, ic              the boundary term and the IC fit (as in the loss, unweighted by lamg0)
  theta_rms, z_rms    rms of Theta and |Z^i| on the interior points (constraint monitors)

The moving chi background uses each window's stored (fine) puncture trajectory,
which is close to but not exactly the coarse one used during training.
"""
import argparse
import csv
import os
import pickle
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_defs import load

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--windows", default="", help="a-b (0-based, inclusive); default: common windows")
ap.add_argument("--seed", type=int, default=2026)
ap.add_argument("--out", default="heldout.csv")
a, rest = ap.parse_known_args()
if rest[:1] == ["--"]:
    rest = rest[1:]

ns = load(rest, outdir=tempfile.mkdtemp(prefix="eval_"))
jax, jnp, np = ns["jax"], ns["jnp"], ns["np"]
CHI = ns["CHI_IDX"]


def load_run(d):
    man = pickle.load(open(os.path.join(d, "pinn_checkpoint.pkl"), "rb"))
    out = {}
    for f in man["window_files"]:
        w = pickle.load(open(os.path.join(d, f), "rb"))
        w["params"] = jax.tree_util.tree_map(jnp.asarray, w["params"])
        out[w["window_idx"]] = w
    return out


runs = {d.rstrip("/"): load_run(d) for d in a.runs}
common = sorted(set.intersection(*[set(r) for r in runs.values()]))
if a.windows:
    lo, hi = (int(v) for v in a.windows.split("-"))
    common = [w for w in common if lo <= w <= hi]
first = runs[a.runs[0].rstrip("/")]
raw_fn = ns["apply_fn"]
fn = ns["_wrap_with_moving_chi_bg"](raw_fn) if ns["chi_bg_mode"] == "moving" else raw_fn


def ms(x):
    return float(jnp.mean(x ** 2))


rows = []
for wi in common:
    st0 = first[wi]
    ws, we = jnp.array(st0["t_start"]), jnp.array(st0["t_end"])
    # sampling centres: the first run's puncture position at t_start
    xy0 = np.asarray(st0["traj_xx"])[:, :, 0] if "traj_xx" in st0 else None
    centers = [(float(p[0]), float(p[1]), 0.0) for p in xy0] if xy0 is not None else None
    for name, run in runs.items():
        st = run[wi]
        prev = run.get(wi - 1)
        if xy0 is not None:            # IC target's moving background at t_start (build_train_data reads it)
            pxy = np.asarray(st["traj_xx"])[:, :, 0]
            ns["_shift_track_xy"] = [[float(p[0]), float(p[1])] for p in pxy]
        key = jax.random.PRNGKey(a.seed + 1000 * wi)
        td = ns["build_train_data"](ws, we, key, prev_state=prev, star_centers=centers, sigma_time=None)
        td = td[0] if isinstance(td[0], tuple) else td
        (t, x, y, z, tf, xf, yf, zf, tb, xb, yb, zb, gb, ti, xi, yi, zi, ic_ccz4, *_rest) = td
        p_net = st["params"]
        tt = jnp.asarray(st["traj_tt"]) if "traj_tt" in st else None
        xx = jnp.asarray(st["traj_xx"]) if "traj_xx" in st else None
        P = p_net if tt is None else ns["_bundle_params"](p_net, tt, xx)
        args = (ns["lam_bc"], ns["lamg0"], ns["w_pos"], tt, xx, ns["lam_einstein"], ns["lam_gauge"],
                ns["r_eps"], *td)
        total = float(ns["compute_bfgs_loss"](fn, p_net, *args))
        ri = ns["r_ccz4_evolution"](fn, P, t, x, y, z, ws, we)
        rf = ns["r_ccz4_evolution"](fn, P, tf, xf, yf, zf, ws, we)
        if ns["metric_bc_mode"] == "dirichlet":
            mb = ns["metric_dirichlet_residual"](fn, P, tb, xb, yb, zb, ws, we, gb, ns["lam_bc"])
        else:
            mb = ns["metric_sommerfeld_residual"](fn, P, tb, xb, yb, zb, ws, we, ns["c"], ns["lam_bc"], ns["r_eps"])
        ug = ns["full_output"](fn, P, ti, xi, yi, zi, ws, we)
        theta = ns["hamiltonian_constraint"](fn, P, t, x, y, z, ws, we)
        zvec = ns["gauge_source_vector"](fn, P, t, x, y, z, ws, we)
        row = {"run": os.path.basename(name), "window": wi, "t_end": float(we),
               "loss_total": total,
               "ein_int": ms(ri[:, :18]), "gauge_int": ms(ri[:, 18:22]), "ein_tapa": ms(rf[:, :18]),
               "bc": ms(mb), "ic": ms(ug[:, ns["CCZ4_SLICE"]] - ic_ccz4),
               "theta_rms": float(jnp.sqrt(jnp.mean(theta ** 2))),
               "z_rms": float(jnp.sqrt(jnp.mean(jnp.sum(jnp.reshape(zvec, (zvec.shape[0], -1)) ** 2, axis=1))))}
        rows.append(row)
        print(f"w{wi:3d} {row['run']:16s} total {total:.3e}  ein_int {row['ein_int']:.3e}  "
              f"ein_tapa {row['ein_tapa']:.3e}  ic {row['ic']:.3e}  theta {row['theta_rms']:.2e}  "
              f"Z {row['z_rms']:.2e}", flush=True)

with open(a.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"wrote {a.out} ({len(rows)} rows)")
