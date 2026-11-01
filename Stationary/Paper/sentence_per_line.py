#!/usr/bin/env python
"""Put each sentence of the paper's prose on its own line (whitespace changes only).

    python Paper/sentence_per_line.py Paper/paper.tex          # rewrites in place, then checks
    python Paper/sentence_per_line.py Paper/paper.tex --check  # only the check, nothing written

Why: one sentence per line makes `git diff` and `latexdiff` show exactly which sentence changed.
TeX reads a newline as a space, so nothing changes in the PDF.  What is left alone: the preamble,
equations, tables, figures, comments (lines with %), \\begin/\\end and section lines.  A sentence
ends at . ? ! (optionally followed by closing brackets or quotes) and a space, when the next word
starts with a capital letter, \\ or $; never inside $...$ or inside braces (so \\claude{...} and
\\caption{...} stay whole), and not after e.g. i.e. Eq. Sec. Fig. Ref. et al. cf. vs. or a lone
capital initial.  The check compares every paragraph of the old and the new file with all
whitespace collapsed: they must be identical.
"""
import re
import sys

SKIP_ENVS = {"equation", "equation*", "align", "align*", "gather", "gather*", "multline",
             "multline*", "eqnarray", "tabular", "tabular*", "ruledtabular", "table", "table*",
             "figure", "figure*", "verbatim", "thebibliography", "lstlisting"}
ABBREV = {"e.g.", "i.e.", "eq.", "eqs.", "sec.", "secs.", "fig.", "figs.", "ref.", "refs.", "tab.",
          "cf.", "vs.", "al.", "approx.", "resp.", "no.", "etc.", "dr.", "prof.", "ch."}
STANDALONE = re.compile(r"^\s*(\\begin|\\end|\\section|\\subsection|\\subsubsection|\\label|"
                        r"\\includegraphics|\\caption|\\bibliograph|\\newcommand|\\usepackage|"
                        r"\\hline|\\maketitle|\\title|\\author|\\affiliation|\\date|\\newif|"
                        r"\\notestrue|\\documentclass|\\input|\\appendix)")
ENDS = re.compile(r"[.?!][)\]}'\"”’]*(?=\s)")


def comment(line):
    return re.search(r"(?<!\\)%", line) is not None


def split_sentences(par):
    """Split one paragraph (a single line of text) at sentence ends."""
    out, start, depth, dollars, i = [], 0, 0, 0, 0
    marks = [m for m in ENDS.finditer(par)]
    # brace depth and $ parity at every position, computed once
    dep, dol, d, p = [], [], 0, 0
    for k, ch in enumerate(par):
        if ch == "{" and (k == 0 or par[k - 1] != "\\"):
            d += 1
        elif ch == "}" and (k == 0 or par[k - 1] != "\\"):
            d -= 1
        elif ch == "$" and (k == 0 or par[k - 1] != "\\"):
            p ^= 1
        dep.append(d)
        dol.append(p)
    for m in marks:
        end = m.end()
        k = m.start()
        if dep[k] != 0 or dol[k] != 0:
            continue
        rest = par[end:].lstrip()
        if not rest or not (rest[0].isupper() or rest[0] in "\\$"):
            continue
        word = par[start:end].split()[-1].lower() if par[start:end].split() else ""
        if word.rstrip(")]}'\"”’") in ABBREV:
            continue
        if re.fullmatch(r"\(?[A-Z]\.", par[start:end].split()[-1] if par[start:end].split() else ""):
            continue
        out.append(par[start:end].strip())
        start = end
    out.append(par[start:].strip())
    return [s for s in out if s]


def reflow(text):
    lines = text.split("\n")
    res, buf, indent, skip, in_doc = [], [], "", [], False

    def flush():
        nonlocal buf
        if buf:
            par = " ".join(s.strip() for s in buf)
            for s in split_sentences(par):
                res.append(indent + s)
            buf = []

    for line in lines:
        if not in_doc:
            res.append(line)
            if "\\begin{document}" in line:
                in_doc = True
            continue
        if skip:
            res.append(line)
            for m in re.finditer(r"\\(begin|end)\{([^}]*)\}", line):
                if m.group(1) == "begin" and m.group(2) in SKIP_ENVS:
                    skip.append(m.group(2))
                elif m.group(1) == "end" and skip and m.group(2) == skip[-1]:
                    skip.pop()
            continue
        m = re.search(r"\\begin\{([^}]*)\}", line)
        if m and m.group(1) in SKIP_ENVS:
            flush(); res.append(line)
            skip.append(m.group(1))
            if re.search(r"\\end\{%s\}" % re.escape(m.group(1)), line):
                skip.pop()
            continue
        if not line.strip() or comment(line) or STANDALONE.match(line):
            flush(); res.append(line)
            continue
        if line.lstrip().startswith("\\item"):
            flush()
        if not buf:
            indent = line[: len(line) - len(line.lstrip())]
        buf.append(line)
    flush()
    return "\n".join(res)


def paragraphs(text):
    return [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text)]


if __name__ == "__main__":
    path = sys.argv[1]
    old = open(path).read()
    new = old if "--check" in sys.argv else reflow(old)
    if paragraphs(old) != paragraphs(new):
        bad = [(a, b) for a, b in zip(paragraphs(old), paragraphs(new)) if a != b]
        print("CHECK FAILED: the text changed beyond whitespace;", len(bad), "paragraphs; nothing written")
        print(bad[0][0][:200]); print(bad[0][1][:200])
        sys.exit(1)
    if "--check" not in sys.argv:
        open(path, "w").write(new)
    print(f"ok: {len(old.splitlines())} -> {len(new.splitlines())} lines; every paragraph identical "
          f"after collapsing whitespace")
