"""The loss-history panel must DRAW THE CURVES.

pq_c200_vac's diagnostics.png came out with a titled, empty frame -- "loss history (weighted)
[37 rows from history.json]" over nothing -- because the plotting loop and the legend sat
inside the `else:` branch of the history loader, i.e. they ran only when history.json was
MISSING.  Both halves of the fix are pinned here: the panel function draws (this file), and
`_plots` calls it at the top level rather than inside a fallback branch (the structural test at
the bottom, which is what actually catches a re-indentation).
"""
from __future__ import annotations

import inspect

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stationary import evaluate


def _hist(n=5):
    return [{"step": 1 + i * 100, "loss": 4.0 / (i + 1), "pde_ricci": 1e-3 / (i + 1),
             "pde_gauge": 1e-4 / (i + 1), "pde_lam_eq": 1e-5 / (i + 1),
             "pde_compat": 0.0}                                        # identically zero: skipped
             for i in range(n)]


def test_every_group_present_in_the_history_is_drawn():
    fig, ax = plt.subplots()
    evaluate._draw_loss_history(ax, _hist(), "history.json", ("pde_ricci", "pde_gauge", "pde_lam_eq"),
                               {"loss": "total", "pde_ricci": "Ricci", "pde_gauge": "gauge",
                                "pde_lam_eq": "lam"})
    # total, Ricci, gauge, lam: four curves, and a legend that names them
    assert len(ax.lines) == 4, [l.get_label() for l in ax.lines]
    assert ax.get_legend() is not None
    assert "5 rows from history.json" in ax.get_title()
    # the y-range is the loss's own: the group that reaches 1e-5 must not set it to 5 decades
    lo, hi = ax.get_ylim()
    assert hi <= 8.0 + 1e-9 and lo >= 0.4, (lo, hi)
    plt.close(fig)


def test_an_identically_zero_group_is_not_drawn():
    fig, ax = plt.subplots()
    labels = {"loss": "total", "pde_compat": "compat"}
    evaluate._draw_loss_history(ax, _hist(), "history.json", ("pde_compat",), labels)
    assert [l.get_label() for l in ax.lines] == ["total"]
    plt.close(fig)


def test_one_row_is_still_drawn():
    fig, ax = plt.subplots()
    evaluate._draw_loss_history(ax, _hist(1), "history.json", ("pde_ricci",),
                                {"loss": "total", "pde_ricci": "Ricci"})
    assert len(ax.lines) == 2
    plt.close(fig)


def test_no_history_says_so_instead_of_drawing_an_empty_frame():
    fig, ax = plt.subplots()
    evaluate._draw_loss_history(ax, None, None, ("pde_ricci",), {"loss": "total"})
    assert len(ax.lines) == 0
    assert any("no history" in t.get_text() for t in ax.texts)
    plt.close(fig)


def test_the_panel_is_drawn_from_the_top_level_of_plots_not_a_fallback_branch():
    """The bug was indentation, so the indentation is what this asserts.

    `_plots` calls the panel with four spaces -- the body of the function.  Inside the `else:`
    (the checkpoint fallback) it would be eight or more, and for any run whose history.json
    exists -- every finished run -- the panel would be a frame again.
    """
    src = inspect.getsource(evaluate._plots)
    call = "_draw_loss_history(ax[0, 0], hist, hist_src, pde_keys, labels)"
    lines = [l for l in src.splitlines() if call in l and not l.lstrip().startswith("#")]
    assert len(lines) == 1, lines
    assert lines[0] == "    " + call, repr(lines[0])
