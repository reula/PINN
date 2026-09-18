"""Notebook setup for the stationary-Einstein PINN.  Run this cell FIRST.

    %run notebook_setup.py          # from inside Stationary/
    pf, cfg = load("runs/m2R4_realrobin")
    plot_lambda_vs_rho(pf, cfg)

Why a cell is needed at all: `run_hub.sh` sets these variables for you when it launches
a *script*, but a Jupyter kernel inherits none of it.  Each setting below has to happen
before `import jax` / `import matplotlib` for the first time in the kernel, so if you have
already imported them in an earlier cell, restart the kernel first.
"""
import os
import sys

# ---------------------------------------------------------------- environment
# 1. JAX reserves 75% of the visible GPU at import time.  On a shared hub that either
#    fails outright ("cuBlas allocation failure") or squats on the card.  MUST be set
#    before jax is imported for the first time in this kernel.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.4")   # cap instead of nothing
# os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")               # pick a card

# 2. Matplotlib must not try to build its font cache in an unwritable directory.
os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(".mplcache"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

# 3. Make `import stationary` work from anywhere (notebook in a subdirectory, etc.).
_HERE = os.path.dirname(os.path.abspath(__file__ if "__file__" in globals() else "."))
if os.path.isdir(os.path.join(_HERE, "stationary")) and _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ---------------------------------------------------------------------- jax
import jax

jax.config.update("jax_enable_x64", True)          # post-processing is float64 by
                                                   # convention here (training is float32)

import jax.numpy as jnp

from stationary.evaluate import load_run            # noqa: E402
from stationary.model import point_fields           # noqa: E402

print(f"jax {jax.__version__} on {[str(d) for d in jax.devices()]}  "
      f"| x64={jax.config.jax_enable_x64} | MPLCONFIGDIR={os.environ['MPLCONFIGDIR']}")
if all(d.platform == "cpu" for d in jax.devices()):
    print("note: running on CPU (fine for analysis; training runs are another matter)")


# ------------------------------------------------------------------ helpers
def load(run_dir, params_file="params.pkl"):
    """Load a finished (or in-progress) run: returns (point_fields, cfg).

    params_file="ckpt.pkl" reads the live checkpoint of a run that is still going.
    """
    cfg, model, state = load_run(run_dir, params_file)
    return point_fields(model, state["net"]), cfg


def lambda_vs_rho(pf, cfg, n_rho=240, thetas=(0.0, 0.7, 1.5707963)):
    """lambda as a function of rho along the given polar angles."""
    rhos = jnp.geomspace(cfg.rho_in, cfg.rho_out, n_rho)
    curves = {}
    for th in thetas:
        n = jnp.array([jnp.sin(th), 0.0, jnp.cos(th)])
        curves[float(th)] = jax.vmap(lambda y: pf(y).lam)(rhos[:, None] * n[None, :])
    return rhos, curves


def plot_lambda_vs_rho(pf, cfg, thetas=(0.0, 0.7, 1.5707963), logx=True, save=None):
    import matplotlib.pyplot as plt
    rhos, curves = lambda_vs_rho(pf, cfg, thetas=thetas)
    for th, lam in curves.items():
        plt.plot(rhos, lam, label=fr"$\theta={th:.2f}$")
    lam_inf = cfg.lam_inf if cfg.lam_inf is not None else cfg.lam_inf_init
    plt.axhline(lam_inf, ls=":", c="grey", label=fr"$\lambda_\infty={lam_inf:g}$")
    plt.axhline(cfg.lam0, ls="-.", c="grey", alpha=0.6, label=fr"$\lambda_0={cfg.lam0:g}$")
    if logx:
        plt.xscale("log")
    plt.xlabel(r"$\rho$"); plt.ylabel(r"$\lambda$"); plt.legend(); plt.grid(alpha=0.3)
    if save:
        plt.savefig(save, dpi=120)
        print(f"wrote {save}")
    return rhos, curves
