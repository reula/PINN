# Synthesize the common ancestor A of B (bundle base, Oct 15) and T (top-level base, Oct 16).
# Hunks that are Oct-15 features (present in B only) -> A takes T's text; others (Oct-16 features) -> A takes B's text.
import difflib, sys
B = open(sys.argv[1]).read().splitlines(keepends=True)
T = open(sys.argv[2]).read().splitlines(keepends=True)
# 1-based B line starts of hunks that belong to the Oct-15 (bundle-only) side
B_RANGES = [(424,437),(507,517),(556,583),(605,606),(964,970),(1766,1831),(5352,5354),(5665,5680),(5730,5737)]
BOTH = [(5304,5307),(5335,5350)]   # mixed: blow-up guard (B) vs excursion guard (T) -> ancestor has neither
inr = lambda a, R: any(lo <= a <= hi for lo, hi in R)
A = []
for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, B, T, autojunk=False).get_opcodes():
    if tag == 'equal':
        A += B[i1:i2]; continue
    start = i1 + 1 if i2 > i1 else i1  # diff reports 'a' hunks at the preceding B line
    side = 'X' if inr(start, BOTH) else ('B' if inr(start, B_RANGES) else 'T')
    print(f"{tag} B{i1+1}-{i2} T{j1+1}-{j2} -> feature of {side}", file=sys.stderr)
    A += [] if side == 'X' else (T[j1:j2] if side == 'B' else B[i1:i2])
open(sys.argv[3], 'w').writelines(A)
