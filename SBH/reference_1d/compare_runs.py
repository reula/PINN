"""Side-by-side comparison of two compare_pinn.py results (validation only).
    python3 compare_runs.py BASE_errors.csv NEW_errors.csv [--times 0.25,0.5,0.75,1] [--out table.md]
Prints, per time and field, the banded L2 errors (core [0.05,0.5), mid [0.5,5),
far [5,20)) of both runs at window ends and the ratio base/new (> 1 = new is better)."""
import argparse
import csv

p = argparse.ArgumentParser()
p.add_argument("base"); p.add_argument("new")
p.add_argument("--times", default="0.25,0.5,0.75,1")
p.add_argument("--fields", default="alpha,chi,Khat,Theta,a,b,c,d,g,B")
p.add_argument("--out", default=None)
a = p.parse_args()
def load(f):
    return {(float(r["t"]), r["side"], r["field"]): r for r in csv.DictReader(open(f))}
B, N = load(a.base), load(a.new)
lines = [f"base: {a.base}", f"new:  {a.new}", "",
         "| t | field | core base | core new | ratio | mid base | mid new | ratio | far base | far new | ratio |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
for t in [float(x) for x in a.times.split(",")]:
    for f in a.fields.split(","):
        kb, kn = (t, "end", f), (t, "end", f)
        if kb not in B or kn not in N:
            continue
        row = [f"{t:g}", f]
        for band in ("core", "mid", "far"):
            vb, vn = float(B[kb]["l2_" + band]), float(N[kn]["l2_" + band])
            row += [f"{vb:.2e}", f"{vn:.2e}", f"{vb / vn:.1f}" if vn > 0 else "-"]
        lines.append("| " + " | ".join(row) + " |")
txt = "\n".join(lines)
print(txt)
if a.out:
    open(a.out, "w").write(txt + "\n")
