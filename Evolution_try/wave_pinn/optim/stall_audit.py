"""Audit a DSGNAR phase at the moment it stops on its radius rule.

README §6.11 leaves one defect open: a window's DSGNAR phase can walk its trust region
to ``delta_min`` while the loss sits orders of magnitude above the floor the same
configuration reaches elsewhere.  What is ruled out is listed there (stop rule, hash
count, the sketched prediction, the sketch size); what is missing is the trigger.
``scripts/dsgnar_sketch_audit.py`` measures the sketch against the exact Jacobian, but
only at a cold start and at a converged state -- never at a stall, because a stall
has to be caught while it happens.

This module is that catch.  With ``dsgnar_audit_stops`` on, every DSGNAR phase that
stops on ``radius below threshold`` is audited at its final parameters, on the very
objective (batch) it stopped on, and one JSON line is appended to
``<outdir>/dsgnar_stop_audits.jsonl``.  Converged stops (loss at the floor) are
audited too: they are the control the stalls are read against.

The question each record answers is: **could an exact Gauss-Newton step have gone
on?**

* if the exact LM step from the full Jacobian lowers the loss by orders of magnitude
  while the sketched step does not, the sketch is what stalls (and ``cos`` says how);
* if the exact step cannot lower it either, the point is a genuine (local)
  stationary point of this sampled objective, and no change to the optimiser's
  sketch will help -- the cure is elsewhere (the sample, the initialisation).

Everything here is cheap at this problem size: ``n ~ 2.2e3`` parameters, so the exact
Jacobian is ``n`` JVPs and one dense SVD.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np

# Set by the caller (windows.run_windows) so that a record says which window and
# attempt it came from; run_optimizer adds the round.
CONTEXT: Dict = {}

LAMS = (1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0)
FRACS = (1.0, 0.5, 0.1, 0.01)
N_KEYS = 4


def _model_decrease(r, J, p, inv_m):
    """``mean(r^2) - mean((r + J p)^2)``: the exact quadratic model's decrease."""
    rp = r + J @ p
    return inv_m * (float(r @ r) - float(rp @ rp))


def audit_stop(objective, flat, cfg, info: Dict, outdir: str,
               round_index: Optional[int] = None, verbose: bool = True) -> Dict:
    """Measure the exact and the sketched Gauss-Newton steps at ``flat``; append a record."""
    from .dsgnar import (N_HASHES, N_NEWTON, _dct_matrix, count_sketch, jvp_columns,
                         lift_srct, make_srct, solve_subproblems, srct_columns, step_and_pred)

    t0 = time.time()
    flat = jnp.asarray(flat)
    dtype = flat.dtype
    n = int(flat.shape[0])
    res_fn = objective._residual
    loss_fn = objective._loss
    r = res_fn(flat)
    M = int(r.shape[0])
    inv_m = 1.0 / M
    loss = float(loss_fn(flat))
    chunk = int(getattr(cfg, "dsgnar_chunk", 0) or 0)

    # ---- the exact Jacobian and its spectrum ------------------------------------------
    J = jvp_columns(res_fn, flat, jnp.eye(n, dtype=dtype), chunk).T      # (M, n)
    U, S, Vt = jnp.linalg.svd(J, full_matrices=False)
    S_np = np.asarray(S)
    ur = U.T @ r
    grad = 2.0 * inv_m * (J.T @ r)                       # d mean(r^2) / d theta
    # fraction of |r|^2 that lies outside range(J): the part NO step can remove
    r_in = float(ur @ ur)
    r_out_frac = max(0.0, 1.0 - r_in / max(float(r @ r), 1e-300))
    spectrum = {
        "sigma_max": float(S_np[0]),
        "sigma_min": float(S_np[-1]),
        "rank": {f"{t:.0e}": int((S_np > t * S_np[0]).sum()) for t in (1e-6, 1e-8, 1e-10, 1e-12)},
        "residual_outside_range_frac": r_out_frac,
    }

    # ---- exact LM steps ---------------------------------------------------------------
    exact = []
    exact_steps = {}
    for lam in LAMS:
        p = -(Vt.T @ ((S / (S * S + lam)) * ur))
        exact_steps[lam] = p
        dec = _model_decrease(r, J, p, inv_m)
        row = {"lam": lam, "norm": float(jnp.linalg.norm(p)), "model_decrease": dec}
        for f in FRACS:
            lf = float(loss_fn(flat + f * p))
            row[f"loss@{f:g}"] = lf
        best = min(row[f"loss@{f:g}"] for f in FRACS)
        row["best_ratio_to_loss"] = best / loss if loss > 0 else float("nan")
        exact.append(row)

    # ---- the sketched step DSGNAR would take, at the same lambdas -------------------
    s = int(info.get("sketch") or (cfg.dsgnar_sketch or max(1, n // 3)))
    K = int(info.get("hashes") or getattr(cfg, "dsgnar_hashes", 0) or N_HASHES)
    dct = _dct_matrix(n, dtype)
    sketched = []
    for lam in (1e-8, 1e-4, 1.0, float(info.get("final_lambda", 1e-8))):
        coss, preds, decs, losses = [], [], [], []
        for i in range(N_KEYS):
            key = jax.random.PRNGKey(4242 + 17 * i)
            k_srct, k_ck = jax.random.split(key)
            srct = make_srct(k_srct, n, s)
            B = srct_columns(srct, n, s, dtype, dct_matrix=dct)
            j_sk, r_t = count_sketch(J @ B.T, r, k_ck, s, k_hashes=K)
            u, sg, vt = jnp.linalg.svd(j_sk, full_matrices=False)
            g = sg * (u.T @ r_t)
            p_sk, pred = step_and_pred(sg, g, vt, lam)
            p = lift_srct(p_sk, srct, n)
            pe = exact_steps.get(lam)
            if pe is None:
                pe = -(Vt.T @ ((S / (S * S + lam)) * ur))
            den = float(jnp.linalg.norm(p) * jnp.linalg.norm(pe))
            coss.append(float(p @ pe) / den if den > 0 else float("nan"))
            preds.append(float(pred) * inv_m)
            decs.append(_model_decrease(r, J, p, inv_m))
            losses.append(float(loss_fn(flat + p)))
        sketched.append({"lam": lam, "cos_median": float(np.median(coss)),
                         "cos_min": float(np.min(coss)),
                         "pred_median": float(np.median(preds)),
                         "exact_model_decrease_median": float(np.median(decs)),
                         "loss_after_median": float(np.median(losses))})

    rec = {
        **CONTEXT,
        "round": round_index,
        "loss": loss,
        "grad_norm": float(jnp.linalg.norm(grad)),
        "n": n, "M": M, "s": s, "K": K,
        "phase": {k: info.get(k) for k in ("iterations", "stopped", "accepted", "rejected",
                                           "final_radius", "final_lambda", "final_rho",
                                           "stage2_reached")},
        "spectrum": spectrum,
        "exact_lm": exact,
        "sketched": sketched,
        "best_exact_ratio": min(row["best_ratio_to_loss"] for row in exact),
        "audit_wall": time.time() - t0,
    }
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "dsgnar_stop_audits.jsonl"), "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    if verbose:
        print(f"[audit] stop at loss {loss:.3e} (window {CONTEXT.get('window')}, attempt "
              f"{CONTEXT.get('attempt')}, round {round_index}): |grad| {rec['grad_norm']:.2e}, "
              f"rank(1e-10) {spectrum['rank']['1e-10']}/{n}, residual outside range(J) "
              f"{r_out_frac:.2e}; best exact LM step takes the loss to "
              f"{rec['best_exact_ratio']:.2e} of itself; sketched cos(lam=1e-8) "
              f"{sketched[0]['cos_median']:+.3f}  ({rec['audit_wall']:.1f}s)", flush=True)
    return rec
