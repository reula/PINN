"""The T = 20 diagnosis: where the training loss stops predicting the error.

Three panels, all read from finished runs:

1. the relative L2 error against the exact solution at t = 0, 2, ..., 20, for each
   T = 20 run --- the error grows with t rather than decaying;
2. the residual on the training sample against the residual on an *independent*
   sample, along each run --- the two separate, which is the mechanism;
3. that independent-sample residual against the true error --- the curve that is
   supposed to be monotone and is not.

    cd Evolution_try
    /Users/reula/jax_env/bin/python -m scripts.t20_diagnosis
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np

from wave_pinn.evaluate import _mpl


def load(path: str) -> Dict:
    with open(os.path.join(path, "history.json")) as fh:
        h = json.load(fh)
    with open(os.path.join(path, "config.json")) as fh:
        c = json.load(fh)
    f = np.load(os.path.join(path, "fields.npz"))
    return {"label": c.get("label", os.path.basename(path)), "cfg": c, "hist": h, "fields": f}


def main(argv=None) -> int:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+",
                   default=["T20_ssbroyden", "T20_dsgnar", "T20_bigbatch"])
    p.add_argument("--runs-dir", default=os.path.join(here, "runs"))
    p.add_argument("--out", default=os.path.join(here, "runs", "T20_diagnosis.png"))
    args = p.parse_args(argv)

    runs = [load(os.path.join(args.runs_dir, r) if not os.path.isdir(r) else r) for r in args.runs]
    plt = _mpl()
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.4))

    # ---- 1. error against t -------------------------------------------------
    for r in runs:
        f = r["fields"]
        times = np.asarray([float(t) for t in f["times"]])
        err = [np.sqrt(np.mean(f["abs_err"][i] ** 2)) / np.sqrt(np.mean(f["exact"][i] ** 2))
               for i in range(len(times))]
        axes[0].semilogy(times, err, "o-", label=r["label"])
    axes[0].axhline(1.0, color="k", lw=0.8, ls=":")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("relative L2 error")
    axes[0].set_title("error grows with t, it does not decay")
    axes[0].grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=8)

    # ---- 2. training sample against an independent sample --------------------
    for r in runs:
        th = [d for d in r["hist"].get("test_history", []) if "test_loss" in d]
        if not th:
            continue
        st = [d["step"] for d in th]
        axes[1].semilogy(st, [max(d["train_loss"], 1e-300) for d in th], "-",
                         label=f"{r['label']}: training sample")
        axes[1].semilogy(st, [max(d["test_loss"], 1e-300) for d in th], "--",
                         label=f"{r['label']}: independent sample")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("normalised residual")
    axes[1].set_title("the two samples separate")
    axes[1].grid(alpha=0.3, which="both")
    axes[1].legend(fontsize=7)

    # ---- 3. error against the independent-sample residual --------------------
    for r in runs:
        th = [d for d in r["hist"].get("test_history", []) if "rel_l2" in d]
        if not th:
            continue
        axes[2].loglog([max(d["test_loss"], 1e-300) for d in th],
                       [max(d["rel_l2"], 1e-300) for d in th], "o-", ms=4,
                       label=r["label"])
    axes[2].set_xlabel("residual on an independent sample")
    axes[2].set_ylabel("relative L2 error")
    axes[2].set_title("lower residual, larger error")
    axes[2].grid(alpha=0.3, which="both")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    plt.close(fig)
    print(f"written {args.out}")
    for r in runs:
        m = r["hist"].get("metrics", {})
        print(f"{r['label']:>18}  n_coll {r['cfg']['n_coll']:>6}  loss "
              f"{r['hist'].get('final_loss'):.2e}  rel L2 {m.get('rel_l2_space_time'):.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
