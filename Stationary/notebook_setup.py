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
    cfg.run_dir = run_dir                      # so plots can name the run they came from
    return point_fields(model, state["net"]), cfg


def lambda_vs_rho(pf, cfg, n_rho=240, thetas=(0.0, 0.7, 1.5707963)):
    """lambda as a function of rho along the given polar angles."""
    rhos = jnp.geomspace(cfg.rho_in, cfg.rho_out, n_rho)
    curves = {}
    for th in thetas:
        n = jnp.array([jnp.sin(th), 0.0, jnp.cos(th)])
        curves[float(th)] = jax.vmap(lambda y: pf(y).lam)(rhos[:, None] * n[None, :])
    return rhos, curves


def plot_lambda_vs_rho(pf, cfg, thetas=(0.0, 0.7, 1.5707963), logx=True, save=None,
                       tol=1e-6, table=True):
    """lambda against rho, with a legend that says exactly what is drawn.

    Two things this does that a naive plot does not:

    * if the solution is spherically (or axisymmetrically) independent of the polar
      angle, the curves for different theta coincide -- drawing three identical curves
      and a five-entry legend is what makes such a plot unreadable, so it collapses to a
      single curve and says the angles agree;
    * otherwise it draws one curve per angle and adds the *envelope* (min/max over the
      angles) so the angular spread is visible rather than implied.

    The two horizontal guides are the asymptotic value lambda_inf and the imposed inner
    value lam0, labelled as such rather than by symbol alone.
    """
    import matplotlib.pyplot as plt

    plt.close("all")      # a notebook keeps every figure from earlier cells otherwise
    rhos, curves = lambda_vs_rho(pf, cfg, thetas=thetas)
    keys = list(curves)
    first = curves[keys[0]]
    scale = max(float(jnp.max(jnp.abs(first))), 1e-30)
    spread = max((float(jnp.max(jnp.abs(curves[k] - first))) for k in keys[1:]), default=0.0)
    degenerate = spread < tol * scale

    lam_inf = cfg.lam_inf if cfg.lam_inf is not None else cfg.lam_inf_init
    name = getattr(cfg, "run_dir", "?")
    fig, ax = plt.subplots(figsize=(9.5, 6))

    if degenerate:
        ax.plot(rhos, first, "C0-", lw=2,
                label=fr"$\lambda(\rho)$, same for every $\theta$ (spread < {spread:.1e})")
    else:
        for k in keys:
            ax.plot(rhos, curves[k], lw=1.4, label=fr"$\theta = {k:.2f}$ rad")
        lo = jnp.min(jnp.stack([curves[k] for k in keys]), axis=0)
        hi = jnp.max(jnp.stack([curves[k] for k in keys]), axis=0)
        ax.fill_between(rhos, lo, hi, color="C0", alpha=0.12,
                        label=fr"spread over $\theta$ (max {spread:.1e})")
    ax.axhline(lam_inf, color="grey", ls=":", lw=1.5,
               label=fr"asymptotic value $\lambda_\infty = {lam_inf:g}$")
    ax.axhline(cfg.lam0, color="grey", ls="-.", lw=1.5, alpha=0.8,
               label=fr"imposed inner value $\lambda_0 = {cfg.lam0:g}$")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(r"$\rho$   (harmonic radial coordinate)")
    ax.set_ylabel(r"$\lambda$")
    ax.set_title(f"$\\lambda$ vs $\\rho$  --  {name}\n"
                 f"inner data: $\\lambda_0={cfg.lam0:g}$, "
                 f"$\\bf S_1={cfg.lam_bc_S1:g}$, $S_2={cfg.lam_bc_S2:g}$   |   "
                 f"arch={cfg.arch}, Robin order {cfg.robin_orders or cfg.robin_order}",
                 fontsize=10)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=120)
        print(f"wrote {save}")

    if table:
        show = jnp.geomspace(cfg.rho_in, cfg.rho_out, 9)
        cols = [float(r) for r in show]
        print(f"{'rho':>10} " + " ".join(f"{'theta=%.2f' % k:>12}" for k in keys) +
              ("   (identical to 1e-6)" if degenerate else ""))
        for r in cols:
            vals = [float(jnp.interp(jnp.log(r), jnp.log(rhos), curves[k])) for k in keys]
            print(f"{r:10.4f} " + " ".join(f"{v:12.7f}" for v in vals))
    return rhos, curves


def show_run(run_dir, params_file="params.pkl", thetas=(0.0, 0.7), lmax=3, save=None):
    """ONE figure for a run: lambda(rho) and the multipole content of lambda at rho_out.

    Use this in a notebook instead of calling evaluate()/make_figures() there: those draw
    four separate multi-panel figures (15 axes in total) and a notebook displays every one
    of them.  Everything figure-producing in this module closes previous figures first, so
    a re-run replaces the plot rather than adding another one.

    The caption carries the run directory, arch and the inner boundary data
    (lambda_0, S1, S2) so a saved PNG identifies itself.
    """
    import matplotlib.pyplot as plt

    from stationary.multipoles import angular_profiles, lambda_multipoles  # absolute:
    # `%run notebook_setup.py` in a notebook has no package context

    plt.close("all")
    pf, cfg = load(run_dir, params_file)
    lam_inf = cfg.lam_inf if cfg.lam_inf is not None else cfg.lam_inf_init

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))

    # --- 1. lambda against rho
    rhos, curves = lambda_vs_rho(pf, cfg, thetas=thetas)
    keys = list(curves)
    first = curves[keys[0]]
    spread = max((float(jnp.max(jnp.abs(curves[k] - first))) for k in keys[1:]), default=0.0)
    degenerate = spread < 1e-6 * max(float(jnp.max(jnp.abs(first))), 1e-30)
    if degenerate:
        ax[0].plot(rhos, first, "C0-", lw=2,
                   label=fr"every $\theta$ agrees (spread {spread:.1e})")
    else:
        for k in keys:
            ax[0].plot(rhos, curves[k], lw=1.4, label=fr"$\theta = {k:.2f}$ rad")
        lo = jnp.min(jnp.stack([curves[k] for k in keys]), axis=0)
        hi = jnp.max(jnp.stack([curves[k] for k in keys]), axis=0)
        ax[0].fill_between(rhos, lo, hi, color="C0", alpha=0.12,
                           label=fr"angular spread (max {spread:.1e})")
    ax[0].axhline(lam_inf, color="grey", ls=":", lw=1.5,
                  label=fr"asymptotic $\lambda_\infty={lam_inf:g}$")
    ax[0].axhline(cfg.lam0, color="grey", ls="-.", lw=1.5, alpha=0.8,
                  label=fr"inner $\lambda_0={cfg.lam0:g}$")
    ax[0].set_xscale("log")
    ax[0].set_xlabel(r"$\rho$"); ax[0].set_ylabel(r"$\lambda$")
    ax[0].set_title(r"$\lambda$ vs $\rho$")
    ax[0].legend(fontsize=8)

    # --- 2 and 3. multipoles of lambda at the outer sphere
    coef, power, (mu_o, phi_o, vals_o) = lambda_multipoles(pf, cfg.rho_out, lmax)
    th_o, prof = angular_profiles(coef, lmax)
    ls = list(range(min(3, lmax + 1)))
    amps = [float(jnp.sqrt(power[l])) for l in ls]
    ax[1].bar([f"$l={l}$" for l in ls], amps)
    nz = [a for a in amps if a > 0]
    if nz and max(nz) / min(nz) > 1e3:
        ax[1].set_yscale("log")
        ax[1].set_ylim(min(nz) / 10, max(nz) * 10)
    ax[1].set_xlabel("multipole order")
    ax[1].set_ylabel("amplitude")
    ax[1].set_title(fr"multipoles of $\lambda$ at $\rho={cfg.rho_out:g}$")
    for l in ls:
        ax[2].plot(th_o, prof[l], label=fr"$l={l}$  (amp {amps[l]:.2e})")
    ax[2].set_xlabel(r"$\theta$"); ax[2].set_ylabel(r"$\lambda_l(\theta)$")
    ax[2].set_title("angular dependence of each multipole")
    ax[2].legend(fontsize=8)

    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle(f"{run_dir}   |   inner data: $\\lambda_0={cfg.lam0:g}$,  "
                 f"$\\bf S_1={cfg.lam_bc_S1:g}$,  $S_2={cfg.lam_bc_S2:g}$   |   "
                 f"arch={cfg.arch}, Robin order {cfg.robin_orders or cfg.robin_order}",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    if save:
        fig.savefig(save, dpi=120)
        print(f"wrote {save}")
    return fig
