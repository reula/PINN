"""Score the fit-test network with the PINN's OWN loss (PINN_LESSONS 20.2, CORRECTION of 2026-10-18).

A value fit says how well the network matches the reference values, not how small a PDE residual
this network can reach. This evaluates the BH code's training loss (compute_bfgs_loss and its terms)
for the PINN's window-0 checkpoint and for the fit network (fit_window_bh.py output) on the SAME
collocation batches drawn by the code's own build_train_data.

  JAX_PLATFORMS=cpu python3 reference_1d/fit_residual_check.py RUN_DIR FIT_DIR \
      [--Nint 4000 --core_points 2400 --NI 400 --core_ic_points 240 --Nb 800 --batches 2]
"""
import argparse
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("fit_dir")
    p.add_argument("--Nint", type=int, default=4000)
    p.add_argument("--core_points", type=int, default=2400)
    p.add_argument("--NI", type=int, default=400)
    p.add_argument("--core_ic_points", type=int, default=240)
    p.add_argument("--Nb", type=int, default=800)
    p.add_argument("--batches", type=int, default=2)
    p.add_argument("--seed", type=int, default=2026)
    a = p.parse_args()

    import fit_window_bh as fw
    with open(os.path.join(a.run_dir, "pinn_checkpoint.pkl"), "rb") as f:
        cfg = pickle.load(f)["model_config"]
    with open(os.path.join(a.run_dir, "pinn_checkpoint_w0000.pkl"), "rb") as f:
        w0 = pickle.load(f)
    with open(os.path.join(a.fit_dir, "fit_params.pkl"), "rb") as f:
        fit = pickle.load(f)
    la = argparse.Namespace(Nint=a.Nint, NI=a.NI, Nb=a.Nb, core_points=a.core_points,
                            core_ic_points=a.core_ic_points, core_radius=0.5, N_change=250, steps=250,
                            seed=42, h0_scaling=1, h_rescale=1, scale_invariant=1,
                            out=os.path.join(a.fit_dir, "residual_check_tmp"))
    os.makedirs(la.out, exist_ok=True)
    ns = fw.load_code(cfg, la)
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)

    traj_tt = jnp.asarray(w0["traj_tt"])
    traj_xx = jnp.asarray(w0["traj_xx"])
    fit_traj_xx = jnp.asarray(fit["traj_xx"])
    phys = ns["_wrap_with_moving_chi_bg"](ns["apply_fn"])
    t0, t1 = float(w0["t_start"]), float(w0["t_end"])

    def terms(net_params, txx, data):
        params = ns["_bundle_params"](net_params, traj_tt, txx)
        (t, x, y, z, tf, xf, yf, zf, tb, xb, yb, zb, gb, ti, xi, yi, zi, ic_ccz4,
         w_int, w_tapa, w_ic, ws, we) = data
        rei = ns["r_ccz4_evolution"](phys, params, t, x, y, z, ws, we)
        ref = ns["r_ccz4_evolution"](phys, params, tf, xf, yf, zf, ws, we)
        r = np.sqrt(np.sum(np.asarray(x) ** 2 + np.asarray(y) ** 2 + np.asarray(z) ** 2, axis=1))
        ein_int = float(jnp.mean(rei[:, :18] ** 2))
        gauge_int = float(jnp.mean(rei[:, 18:22] ** 2))
        core = r < 0.5
        ein_core = float(jnp.mean(rei[core][:, :18] ** 2))
        ein_out = float(jnp.mean(rei[~core][:, :18] ** 2))
        theta_core = float(jnp.mean(rei[core][:, 14] ** 2)) if rei.shape[1] > 14 else np.nan
        ein_tapa = float(jnp.mean(ref[:, :18] ** 2))
        ug = ns["full_output"](phys, params, ti, xi, yi, zi, ws, we)
        ic_abs, _ = ns["_ic_fit"](ug, ic_ccz4, w_ic)
        if ns["metric_bc_mode"] == "dirichlet":
            met = ns["metric_dirichlet_residual"](phys, params, tb, xb, yb, zb, ws, we, gb, ns["lam_bc"])
        else:
            met = ns["metric_sommerfeld_residual"](phys, params, tb, xb, yb, zb, ws, we, ns["c"], ns["lam_bc"], ns["r_eps"])
        bc = float(jnp.mean(met ** 2))
        total = float(ns["compute_bfgs_loss"](phys, net_params, ns["lam_bc"], ns["lamg0"], ns["w_pos"],
                                              traj_tt, txx, ns["lam_einstein"], ns["lam_gauge"],
                                              ns["r_eps"], *data))
        return dict(total=total, ein_int=ein_int, ein_core=ein_core, ein_r_gt_0p5=ein_out,
                    gauge_int=gauge_int, ein_tapa=ein_tapa, ic_fit=float(ic_abs), bc=bc)

    pinn_p = jax.tree_util.tree_map(jnp.asarray, w0["params"])
    fit_p = jax.tree_util.tree_map(jnp.asarray, fit["params"])
    key = jax.random.PRNGKey(a.seed)
    rows = {"PINN w0": [], "fit": []}
    for b in range(a.batches):
        data, key = ns["build_train_data"](t0, t1, key, prev_state=None, star_centers=[(0.0, 0.0, 0.0)])
        rows["PINN w0"].append(terms(pinn_p, traj_xx, data))
        rows["fit"].append(terms(fit_p, fit_traj_xx, data))
        print(f"batch {b + 1}/{a.batches} done", flush=True)
    keys = list(rows["fit"][0])
    mean = {k: {kk: float(np.mean([r[kk] for r in v])) for kk in keys} for k, v in rows.items()}
    lines = [f"PINN loss terms on {a.batches} batches of the code's own sampler "
             f"(Nint {a.Nint} + core {a.core_points}, NI {a.NI} + core {a.core_ic_points}, Nb {a.Nb}), window 0",
             f"{'term':<14}{'PINN w0':>12}{'fit':>12}{'fit/PINN':>10}"]
    for kk in keys:
        lines.append(f"{kk:<14}{mean['PINN w0'][kk]:>12.3e}{mean['fit'][kk]:>12.3e}"
                     f"{mean['fit'][kk] / max(mean['PINN w0'][kk], 1e-300):>10.1f}")
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(a.fit_dir, "fit_residual_check.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
