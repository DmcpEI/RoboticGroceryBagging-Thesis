#!/usr/bin/env python3
"""The two data plots of Chapter 4, from the committed CSVs.

  .venv/bin/python tools/figures/plots.py [out_dir]

fig5: bags vs rule violations over the soft-CP penalty sweep (runs/eval/soft_cp_b50_v3.json,
lambda >= 10; the last point is the hard-constrained solve).
fig6: macro-F1 per characteristic against identity, robot and retail catalogs (Chapter 4 table).
"""
import csv
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

DATA = Path(__file__).parent / "data"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "out"
TEAL, ORANGE, GREY, INK = "#0B6E6E", "#B4541A", "#5B6470", "#1F2328"

mpl.rcParams.update({"font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
                     "axes.formatter.use_mathtext": True, "font.size": 9, "axes.unicode_minus": False,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GREY,
                     "xtick.color": GREY, "ytick.color": GREY, "text.color": INK,
                     "axes.labelcolor": INK, "pdf.fonttype": 42})


def cost_of_safety():
    rows = list(csv.DictReader(open(DATA / "frontier.csv")))
    x = [int(r["bags"]) for r in rows]
    y = [int(r["violations"]) for r in rows]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(x, y, color=TEAL, lw=1.2, marker="o", ms=3.5, zorder=3)
    ax.grid(axis="y", color="#E3E6EA", lw=0.6)
    ax.set_xlabel("bags (150 scenes)")
    ax.set_ylabel("rule violations (pairs of items)")
    ax.set_ylim(-75, 400)
    ax.set_xlim(265, 400)
    notes = {(274, 364): ("fewest bags that fit", (8, -2), "left"),
             (285, 270): ("+11 bags, 94 fewer violations", (8, 4), "left"),
             (378, 12): ("12 violations left", (0, 12), "center"),
             (390, 0): ("every rule satisfied: +42% bags", (4, -14), "right")}
    for (bx, by), (s, off, ha) in notes.items():
        ax.annotate(s, (bx, by), xytext=off, textcoords="offset points", ha=ha, va="center",
                    fontsize=8, color=GREY)
    fig.tight_layout()
    return fig


def characteristics_vs_identity():
    rows = list(csv.DictReader(open(DATA / "scores.csv")))
    ident = next(r for r in rows if r["characteristic"] == "identity")
    rest = sorted((r for r in rows if r is not ident), key=lambda r: float(r["robot"]))
    order = rest + [ident]                                    # identity on top
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for i, r in enumerate(order):
        rb, rt = float(r["robot"]), float(r["retail"])
        ax.plot([rt, rb], [i, i], color="#C9CED6", lw=1, zorder=1)
        ax.scatter(rb, i, s=22, color=TEAL, zorder=3)
        hollow = bool(r["retail_note"])
        ax.scatter(rt, i, s=22, zorder=3, color="white" if hollow else ORANGE, edgecolors=ORANGE, lw=1)
    ax.axvline(float(ident["robot"]), color=TEAL, lw=0.7, ls=(0, (3, 2)), zorder=0)
    ax.axvline(float(ident["retail"]), color=ORANGE, lw=0.7, ls=(0, (3, 2)), zorder=0)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([r["characteristic"] + ("*" if r["retail_note"] else "") for r in order])
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.get_yticklabels()[-1].set_fontfamily("cmb10")
    ax.tick_params(axis="y", length=0, colors=INK)
    ax.set_xlim(0.28, 1.0)
    ax.set_xlabel("macro-F$_1$")
    ax.grid(axis="x", color="#E3E6EA", lw=0.6)
    ax.spines["left"].set_visible(False)
    ax.scatter([], [], s=22, color=TEAL, label="Robot, 32 products")
    ax.scatter([], [], s=22, color=ORANGE, label="Retail, 200 products")
    ax.legend(loc="center left", frameon=False, fontsize=8, handletextpad=0.2)
    fig.text(0.99, 0.01, "* no retail product is refrigerated", ha="right", fontsize=7, color=GREY)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, make in [("fig5_cost_of_safety", cost_of_safety),
                       ("fig6_characteristics_vs_identity", characteristics_vs_identity)]:
        fig = make()
        fig.savefig(OUT / f"{name}.png", dpi=300)
        fig.savefig(OUT / f"{name}.pdf")
        print(OUT / f"{name}.png")
