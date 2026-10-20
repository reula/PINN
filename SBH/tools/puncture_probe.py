"""Where is the puncture, and what pushes it? Tracker-independent probe of saved windows.

    JAX_ENABLE_X64=1 python tools/puncture_probe.py RUN_DIR [RUN_DIR ...] [--half 0.2 --n 201]
        -- <the runs' physics flags, e.g. --n_stars 1 --BL_masses 0.5 --BL_momenta 0 --BL_momenta_y 0 --tfinal 10 --N_window 40>

For every window in each run's checkpoint, at the window's end time t_end:
  * beta^x, beta^y at the origin (z = the tracker's offset): the shift that drives the
    shift-driven tracker (dx/dt = -beta), so a nonzero value at the origin is a spurious push
    on a puncture that should stay there;
  * the chi minimum on an n x n grid of half-width `half` around the origin (same z): an
    independent position estimate, resolution 2*half/(n-1).
--chi_bg_mode is read from each run's code_md5.txt/log flags if present, else from the flags
given; under 'moving' the window's stored trajectory restores physical chi.
"""
import argparse
import os
import pickle
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_defs import load

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--half", type=float, default=0.2)
ap.add_argument("--n", type=int, default=201)
argv = sys.argv[1:]
rest = argv[argv.index("--") + 1:] if "--" in argv else []
a = ap.parse_args(argv[:argv.index("--")] if "--" in argv else argv)


def run_mode(d):
    """chi_bg_mode / bh_center_mode the run used (from the job's recorded EXTRA_ARGS)."""
    txt = ""
    for f in ("code_md5.txt",):
        p = os.path.join(d, f)
        if os.path.exists(p):
            txt += open(p).read()
    chi = "physical" if "--chi_bg_mode physical" in txt else "moving"
    return chi


cache = {}
for d in a.runs:
    chi_mode = run_mode(d)
    if chi_mode not in cache:
        cache[chi_mode] = load(rest + ["--chi_bg_mode", chi_mode]
                               + (["--bh_center_mode", "fixed"] if chi_mode == "physical" else []),
                               outdir=tempfile.mkdtemp(prefix="pp_"))
    ns = cache[chi_mode]
    jax, jnp, np = ns["jax"], ns["jnp"], ns["np"]
    eps = float(ns["_PUNCTURE_TRACK_EPS"])
    g = np.linspace(-a.half, a.half, a.n)
    GX, GY = np.meshgrid(g, g, indexing="xy")
    man = pickle.load(open(os.path.join(d, "pinn_checkpoint.pkl"), "rb"))
    print(f"== {os.path.basename(d.rstrip('/'))} (chi_bg_mode {chi_mode}); grid step {g[1] - g[0]:.4f}")
    print(f"   {'t_end':>6s} {'beta_x(0)':>11s} {'beta_y(0)':>11s} {'chi_min x':>10s} {'chi_min y':>10s} {'chi_min':>9s}")
    for f in man["window_files"]:
        st = pickle.load(open(os.path.join(d, f), "rb"))
        st["params"] = jax.tree_util.tree_map(jnp.asarray, st["params"])
        tt = jnp.asarray(st["traj_tt"]) if "traj_tt" in st else None
        xx = jnp.asarray(st["traj_xx"]) if "traj_xx" in st else None
        fn, p, ws, we = ns["_network_apply_for_state"](st, tt, xx)
        t1 = float(st["t_end"])
        o = np.asarray(ns["full_output"](fn, p, jnp.full((1, 1), t1), jnp.zeros((1, 1)), jnp.zeros((1, 1)),
                                         jnp.full((1, 1), eps), ws, we))[0]
        X = jnp.asarray(GX.reshape(-1, 1)); Y = jnp.asarray(GY.reshape(-1, 1))
        out = np.asarray(ns["full_output"](fn, p, jnp.full(X.shape, t1), X, Y, jnp.full(X.shape, eps), ws, we))
        chi = out[:, ns["CHI_IDX"]]
        k = int(np.argmin(chi))
        bx, by = o[ns["BETA_SLICE"].start], o[ns["BETA_SLICE"].start + 1]
        print(f"   {t1:6.3f} {bx:+11.3e} {by:+11.3e} {GX.reshape(-1)[k]:+10.4f} {GY.reshape(-1)[k]:+10.4f} {chi[k]:9.2e}",
              flush=True)
