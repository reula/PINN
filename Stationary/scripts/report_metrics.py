#!/usr/bin/env python
"""Every number the Stationary reports (REPORT_PIN_COM.md, REPORT_OPTIMISERS.md) quote, per run.

    .venv/bin/python scripts/report_metrics.py --runs-dir runs --ref <fine specref .npz> \
        --out plots/report_metrics.json RUN [RUN ...]

Per run, against ONE nonlinear reference (the fine one, l <= 10, 36 radial, loss 2.8e-11):
  * the run's own final loss and wall time (report.json) and its loss COMPOSITION -- the share of
    each residual group, evaluated with the run's own configuration on a fixed sample;
  * the centre-of-mass flux d_z = D_z / 4 pi on spheres rho = 1, 2, 10, 100 (COM_GAUGE.md 4b);
  * the error lambda_net - lambda_ref and max_ij |h_net - h_ref|: rms per radial region and the
    angular-rms profile against rho (with the perturbation's own size, reference - spherical);
  * the l = 1 and l = 3 Legendre coefficients of lambda_net - lambda_ref against rho (the odd part,
    which the data's z -> -z symmetry forbids);
  * the dimensionless residuals (rho^2 Ricci, rho^2 lambda equation) per region, and the inner data.
Results are cached per run in the output JSON; rerun with new runs to add them.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stationary.evaluate import load_run                        # noqa: E402
from stationary.geometry import apply_config, scaled_residuals_batch  # noqa: E402
from stationary.losses import com_flux, inner_bc_raw, residual_vector  # noqa: E402
from stationary.model import point_fields                       # noqa: E402
from stationary.problem import sample_shell, sample_sphere      # noqa: E402
from stationary.specref import SpectralAxisym, compare, profiles  # noqa: E402
from stationary.train import exact_asset, make_batch            # noqa: E402

BINS = [(1, 3), (3, 10), (10, 30), (30, 100)]
EXPS = dict(compat=1.0, ricci=2.0, gauge=1.0, lam_eq=2.0)       # dimensionless, for every run alike


def metrics(run_dir, ref_npz):
    cfg, model, state = load_run(run_dir, "params.pkl")[:3]
    apply_config(cfg, derives_gamma=getattr(model, "derives_gamma", None), want_compat=False)
    pf = point_fields(model, state["net"])
    ex = exact_asset(cfg)
    z = np.load(ref_npz)
    ref = SpectralAxisym(cfg, ex, ls=tuple(int(v) for v in z["ls"]),
                         n_radial=int(z["n_radial"])).fields_of(jnp.asarray(z["c"]))
    rep = json.load(open(os.path.join(run_dir, "report.json")))
    out = {"config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in {
        "qn_method": cfg.qn_method, "scale_exps": cfg.scale_exps,
        "pin_lam_potential": getattr(cfg, "pin_lam_potential", False),
        "com_radii": getattr(cfg, "com_radii", None), "width": cfg.width,
        "radial": cfg.radial, "outer_shell_fracs": cfg.outer_shell_fracs,
        "far_weight": getattr(cfg, "far_weight", None)}.items()},
        "final_loss": rep.get("final_loss"), "wall_s": rep.get("wall_seconds"),
        "qn_stopped_at": rep.get("qn_stopped_at")}

    # loss composition, the run's own objective, fixed sample
    c2 = cfg.__class__(**{**cfg.__dict__, "n_coll": 2000, "n_bnd": 300, "n_bnd_outer": 300})
    batch = make_batch(jax.random.PRNGKey(5), c2)
    blocks = []
    r = np.asarray(residual_vector(state, batch, c2, model, ex, lam_inf=jnp.asarray(1.0),
                                   _blocks=blocks))
    comp, i = {}, 0
    for lab, n in blocks:
        comp[lab] = float(np.sum(r[i:i + n] ** 2))
        i += n
    out["loss_composition"] = comp

    cc = cfg.__class__(**{**cfg.__dict__, "com_radii": (1.0, 2.0, 10.0, 100.0)})
    out["d_z"] = [float(v) for v in np.asarray(com_flux(pf, cc)).reshape(-1, 3)[:, 2]]

    cm = compare(pf, ref, ex, cfg, modes=False)
    out["error_bins"] = cm["bins"]
    prof = profiles(pf, ref, ex, cfg, n_rho=36)
    out["profile"] = {k: np.asarray(v).tolist() for k, v in prof.items()}

    mu, w = np.polynomial.legendre.leggauss(48)
    P1, P3 = mu, 0.5 * (5 * mu ** 3 - 3 * mu)
    odd = {"rho": [], "l1": [], "l3": []}
    f = jax.jit(jax.vmap(lambda x: pf(x).lam - ref(x).lam))
    for rr in np.geomspace(1.05, 100.0, 24):
        X = jnp.asarray(np.stack([rr * np.sqrt(1 - mu ** 2), 0 * mu, rr * mu], -1))
        d = np.asarray(f(X))
        odd["rho"].append(float(rr))
        odd["l1"].append(float(1.5 * np.sum(w * d * P1)))
        odd["l3"].append(float(3.5 * np.sum(w * d * P3)))
    out["odd"] = odd

    xs = sample_shell(jax.random.PRNGKey(11), 3000, cfg)
    rho = np.asarray(jnp.linalg.norm(xs, axis=-1))
    rs = scaled_residuals_batch(pf, xs, EXPS)
    ric = np.asarray(jnp.sqrt(jnp.mean(rs["ricci"].reshape(len(rho), -1) ** 2, axis=1)))
    lam = np.abs(np.asarray(rs["lam_eq"]).reshape(len(rho)))
    rb = lambda v: [float(np.sqrt(np.mean(v[(rho >= a) & (rho <= b)] ** 2))) for a, b in BINS]
    out["residual_bins"] = {"ricci": rb(ric), "lam_eq": rb(lam)}
    ib = inner_bc_raw(pf, sample_sphere(jax.random.PRNGKey(12), 400, cfg.rho_in), cfg, ex)
    out["inner_data_rms"] = {k: float(jnp.sqrt(jnp.mean(v ** 2))) for k, v in ib.items()}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", default="plots/report_metrics.json")
    ap.add_argument("--redo", action="store_true", help="recompute runs already in the JSON")
    a = ap.parse_args(argv)
    data = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for name in a.runs:
        if name in data and not a.redo:
            print(f"[metrics] {name}: cached")
            continue
        print(f"[metrics] {name}", flush=True)
        data[name] = metrics(os.path.join(a.runs_dir, name), a.ref)
        with open(a.out, "w") as fh:
            json.dump(data, fh)
    print(f"[metrics] wrote {a.out}")


if __name__ == "__main__":
    main()
