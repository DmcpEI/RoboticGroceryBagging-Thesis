#!/usr/bin/env python3
"""Two example figures of Chapter 4: benchmark scenes by tier, and one scene as
the detector, Gemini and fine-tuned Qwen see it.

  .venv/bin/python3.12 tools/figures/examples.py [out_dir]

Each predicted name is coloured by what it selects in the catalog: the right
product, another product, or no product at all (the matcher of Chapter 3).
"""
import collections
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/eval"))
from run_gemini_robotics import full_catalog_match  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "out"
RGB = ROOT / "data/robot_lab/rgbd"
GT = ROOT / "datasets_seq_GT_b50"
RUNS = {"Detector": "runs/b50_detector_v18", "Gemini": "runs/b50_gemini_er",
        "Fine-tuned Qwen": "runs/b50_vlm32b_record_lookup"}
OK, WRONG, NONE, INK = "#1B7F5A", "#C2571A", "#B3261E", "#1F2328"
STYLE = {OK: "-", WRONG: "--", NONE: ":"}
mpl.rcParams.update({"font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
                     "axes.formatter.use_mathtext": True, "font.size": 9, "pdf.fonttype": 42})

TIERS = [("b50_easy_pantry_basics_001", "Sparse"), ("b50_med_bags_pantry_004", "Medium"),
         ("b50_hard_everything_000", "Dense")]
SCENE = "b50_med_boxes_cans_004"
CROP = (100, 105, 620, 470)  # the table, in pixels of the 640x480 frame


def items(run, scene):
    d = json.loads((ROOT / run / scene / "frames/frame_000.json").read_text())
    return d["items"]


def colours(pred, truth):
    """One colour per prediction: right product, wrong product, or no product."""
    left = collections.Counter((full_catalog_match(i["name"]) or i["name"]).lower() for i in truth
                               for _ in range(int(i.get("quantity", 1) or 1)))
    out = []
    for i in pred:
        p = full_catalog_match(i["name"])
        if p is None:
            out.append(NONE)
        elif left[p.lower()] > 0:
            left[p.lower()] -= 1
            out.append(OK)
        else:
            out.append(WRONG)
    return out


def tiers():
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 1.95))
    for ax, (scene, label) in zip(axes, TIERS):
        n = sum(int(i.get("quantity", 1) or 1) for i in items(GT.relative_to(ROOT), scene))
        ax.imshow(Image.open(RGB / scene / "color.png").crop(CROP))
        ax.set_title(f"{label}: {n} objects", fontsize=9)
        ax.axis("off")
    fig.tight_layout(pad=0.3)
    return fig


def comparison():
    truth = items(GT.relative_to(ROOT), SCENE)
    img = Image.open(RGB / SCENE / "color.png").crop(CROP)
    cx, cy = CROP[:2]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), gridspec_kw={"width_ratios": [1, 1, 0.62]})
    for ax, name in zip(axes[:2], ["Detector", "Gemini"]):
        pred = items(RUNS[name], SCENE)
        ax.imshow(img)
        placed = []
        for it, c in zip(pred, colours(pred, truth)):
            x1, y1, x2, y2 = (v - o for v, o in zip(it["bbox_2d"], (cx, cy, cx, cy)))
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=c, lw=1.4,
                                       ls=STYLE[c]))
            lab = ax.text(x1 + 2, y1 - 2, it["name"], color="white", fontsize=6, va="bottom",
                          bbox=dict(fc=c, ec="none", pad=0.8, alpha=0.92))
            # above the box, one row higher, inside its top edge, else below it: the
            # first spot whose rendered label overlaps no label already placed
            for ly, va in ((y1 - 2, "bottom"), (y1 - 17, "bottom"), (y1 + 2, "top"),
                           (y2 + 2, "top")):
                lab.set_y(ly)
                lab.set_va(va)
                b = lab.get_window_extent(fig.canvas.get_renderer()).transformed(
                    ax.transData.inverted())
                r = (min(b.x0, b.x1), min(b.y0, b.y1), max(b.x0, b.x1), max(b.y0, b.y1))
                if not any(r[0] < q[2] and q[0] < r[2] and r[1] < q[3] and q[1] < r[3]
                           for q in placed):
                    break
            placed.append(r)
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    ax = axes[2]
    pred = items(RUNS["Fine-tuned Qwen"], SCENE)
    flat = [i for i in pred for _ in range(int(i.get("quantity", 1) or 1))]
    ax.set_title("Fine-tuned Qwen", fontsize=9)
    ax.axis("off")
    for k, (it, c) in enumerate(zip(flat, colours(flat, truth))):
        ax.text(0.02, 0.95 - k * 0.105, it["name"], color=c, fontsize=8, transform=ax.transAxes,
                va="top")
    ax.text(0.02, 0.95 - len(flat) * 0.105 - 0.06, "(names only, no regions)", color="#5B6470",
            fontsize=7, transform=ax.transAxes, va="top", style="italic")
    handles = [plt.Line2D([], [], color=c, ls=STYLE[c], lw=1.6) for c in (OK, WRONG, NONE)]
    fig.legend(handles, ["product in the scene", "product not in the scene", "no product"],
               loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(pad=0.3, rect=(0, 0.07, 1, 1))
    return fig


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, make in [("fig_benchmark_scenes", tiers), ("fig_method_comparison", comparison)]:
        fig = make()
        fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
        fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
        print(OUT / f"{name}.png")
