#!/usr/bin/env python3
"""Confusion matrix per characteristic on the 137 labelled regions
(runs/eval/region_characteristic_confusion_v18.json), one panel each.

  .venv/bin/python3.12 tools/figures/region_confusions.py [out_dir]
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from plots import GREY, INK, TEAL  # noqa: E402  (also applies the shared rcParams)

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "runs/eval/region_characteristic_confusion_v18.json"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "out"
MISS = "not found"
TITLES = {"group": "group", "packaging": "packaging", "weight_class": "weight_class",
          "rigidity": "rigidity", "crush_score": "crush_score", "cold_chain": "cold_chain",
          "spill_risk": "spill_risk", "edible": "edible"}


def order(vals):
    vals = [v for v in vals if v != MISS]
    if all(v.isdigit() for v in vals):
        return sorted(vals, key=int)
    if set(vals) <= {"true", "false"}:
        return ["false", "true"]
    if set(vals) <= {"light", "medium", "heavy"}:
        return [v for v in ("light", "medium", "heavy") if v in vals]
    if set(vals) <= {"soft", "semi", "rigid"}:
        return [v for v in ("soft", "semi", "rigid") if v in vals]
    return sorted(vals)


def panel(ax, m, title):
    rows = order(m)
    cols = order({c for r in m.values() for c in r} | set(rows)) + [MISS]
    a = np.array([[m.get(r, {}).get(c, 0) for c in cols] for r in rows])
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = a[i, j]
            if not v:
                continue
            diag = j < len(rows) and cols[j] == rows[i]
            colour = TEAL if diag else ("#9AA1AA" if cols[j] == MISS else "#B4541A")
            ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, color=colour,
                                       alpha=0.18 + 0.6 * v / a.max(), lw=0))
            ax.text(j, i, str(v), ha="center", va="center", fontsize=6.5, color=INK)
    ax.set_xlim(-.5, len(cols) - .5)
    ax.set_ylim(len(rows) - .5, -.5)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=60, ha="right", fontsize=6)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=6)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, len(cols)), minor=True)
    ax.set_yticks(np.arange(-.5, len(rows)), minor=True)
    ax.grid(which="minor", color="#E3E6EA", lw=0.5)
    ax.grid(which="major", visible=False)
    ax.set_title(title, fontsize=8, color=INK, family="monospace")


def main():
    d = json.loads(SRC.read_text())["matrices"]
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.6),
                             gridspec_kw={"width_ratios": [1.5, 1.4, 1, 1]})
    layout = ["group", "packaging", "weight_class", "rigidity",
              "crush_score", "cold_chain", "spill_risk", "edible"]
    for ax, f in zip(axes.flat, layout):
        panel(ax, d[f], TITLES[f])
    fig.supxlabel("value read from the catalog through the predicted name", fontsize=7.5, color=GREY)
    fig.supylabel("true value", fontsize=7.5, color=GREY)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_region_confusions.{ext}", dpi=300)
    print(OUT / "fig_region_confusions.png")


if __name__ == "__main__":
    main()
