"""Bring a copy of the LDC notebook up to the current optimiser / GPU-kernel setup.

The notebook exists in several copies (``ldc.ipynb``, ``ldc_grad_p.ipynb``, ...).  This script
copies the *generic* cells from the maintained ``ldc.ipynb`` into another copy, leaving every
problem-specific cell (model, residual, reference, plots of the fields) untouched:

  * the setup cell gets the GPU guard (refuses to run on a CPU kernel on the hub),
  * the DSGNAR cell gets the paper rules + fresh-sample restart cycles (its own cycle
    budget, so a copy keeps whatever HIDDEN/N_PDE/N_BC the budget cell sets),
  * the loss-plot cell is updated for the cycles,
  * the SSBroyden cell gets the merged Crunch with the strong-Wolfe line search,
  * the notebook's kernelspec is pointed at "PINN (jax, CUDA)".

Cells are recognised by their content, not by index, and a cell that is already updated is left
alone, so this is safe to run twice.  A timestamped backup is written first and every code cell
is compiled afterwards; if a recogniser fails, nothing is written and the script says which cell
to update by hand.

Usage (from the directory holding both notebooks):
    python update_notebook.py ldc_grad_p.ipynb
    python update_notebook.py ldc_grad_p.ipynb --reference ldc.ipynb
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import shutil
import sys


def source_cells(path: pathlib.Path) -> dict[str, list[str]]:
    """The generic cells of the maintained notebook, as lists of source lines."""
    nb = json.loads(path.read_text())
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]

    def find(*needles: str, exclude: tuple[str, ...] = ()) -> list[str]:
        for src in cells:
            if all(n in src for n in needles) and not any(x in src for x in exclude):
                return src.splitlines(keepends=True)
        raise SystemExit(f"reference {path}: no cell contains {needles}")

    return {
        "setup": find("jax.config.update", "jax_enable_x64"),
        "dsgnar": find("dsgnar_phase("),
        "plot": find("history", "semilogy"),
        "ssbroyden": find("minimize", "ssbroyden2"),
        "kernelspec": nb["metadata"].get("kernelspec"),
    }


def target_index(cells: list[dict], kind: str) -> int | None:
    """Index of the CODE cell to replace, or None.  Markdown is never considered: the
    intro mentions ``minimize``/``ssbroyden2`` too, and matching it once clobbered it.
    Recognises both the old and the already-updated content."""
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if kind == "setup":
            if "jax.config.update" in src and "jax_enable_x64" in src:
                return i
        elif kind == "dsgnar":
            if "dsgnar_phase(" in src:
                return i
        elif kind == "plot":
            if "history" in src and "semilogy" in src:
                return i
        elif kind == "ssbroyden":
            if "minimize" in src and "ssbroyden2" in src:
                return i
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("notebook", help="the copy to update (e.g. ldc_grad_p.ipynb)")
    ap.add_argument("--reference", default="ldc.ipynb",
                    help="the maintained notebook the generic cells come from (default: ldc.ipynb)")
    args = ap.parse_args(argv)

    target = pathlib.Path(args.notebook)
    reference = pathlib.Path(args.reference)
    if not target.exists():
        raise SystemExit(f"no such notebook: {target}")
    if not reference.exists():
        raise SystemExit(f"no such reference notebook: {reference}  (use --reference)")
    if target.resolve() == reference.resolve():
        raise SystemExit("target and reference are the same file")

    ref = source_cells(reference)
    nb = json.loads(target.read_text())
    cells = nb["cells"]

    replaced, skipped = [], []
    for kind in ("setup", "dsgnar", "plot", "ssbroyden"):
        i = target_index(cells, kind)
        if i is None:
            skipped.append(kind)
            continue
        if kind == "ssbroyden" and "minimize_backtracking" in "".join(cells[i]["source"]):
            skipped.append(f"{kind} (already updated)")
            continue
        cells[i]["source"] = ref[kind]
        if cells[i]["cell_type"] == "code":
            cells[i]["outputs"] = []
            cells[i]["execution_count"] = None
        replaced.append(kind)

    if ref["kernelspec"]:
        nb["metadata"]["kernelspec"] = ref["kernelspec"]

    # the copied cells use these names; if a copy renamed them, stop rather than write a broken nb
    joined = "".join("".join(c["source"]) for c in cells)
    for name in ("flat0", "unflatten", "objective", "SMOKE"):
        if name not in joined:
            raise SystemExit(f"{target}: the notebook does not define {name!r}; "
                             f"this is not a copy of {reference}. Nothing written.")
    for i, c in enumerate(cells):
        if c["cell_type"] == "code":
            try:
                compile("".join(c["source"]), f"cell{i}", "exec")
            except SyntaxError as exc:
                raise SystemExit(f"{target}: cell {i} would not compile after the update: {exc}")

    text = json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
    if text == target.read_text():
        print(f"{target}: already up to date -- nothing written")
        return 0

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = target.with_suffix(target.suffix + f".bak-{stamp}")
    shutil.copy2(target, backup)
    target.write_text(text)

    print(f"updated {target}")
    print(f"  replaced : {', '.join(replaced) if replaced else '(nothing)'}")
    print(f"  left     : {', '.join(skipped) if skipped else '(nothing)'}")
    print(f"  backup   : {backup}")
    print(f"  kernel   : {nb['metadata']['kernelspec']['display_name']}")
    print("\nReload the notebook in JupyterLab (close the tab first -- an open tab would save its\n"
          "stale copy over this file), pick 'PINN (jax, CUDA)', and check that cell 1 prints\n"
          "'backend gpu'.  If the kernel is missing, run ldc/setup_gpu_kernel.sh once.")
    if skipped:
        print(f"\nWARNING: no cell matched {skipped}; those parts were not updated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
