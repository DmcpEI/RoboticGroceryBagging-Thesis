#!/usr/bin/env python3
"""Run the fine-tuned vision-language model with the record prompt.

Whole frames (the VLM rows):
  python tools/eval/run_vlm_record.py --scenes datasets_perception/b50 \
      --adapter runs/fine_tuning/lora_32b_record --out runs/b50_vlm32b_record

Detector regions (the hybrid rows), one crop per detection:
  python tools/eval/run_vlm_record.py --crops data/robot_lab/yolo_crops_b50_v18/crop_manifest.json \
      --adapter runs/fine_tuning/lora_32b_record --out runs/b50_crops_vlm_record

Each output is <out>/<id>/frames/frame_000.json holding the model's own record for
every object; catalog lookup through the predicted names is applied afterwards.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/pipeline"))
from record_prompt import CROP_SUFFIX, FIELDS, GROUPS, PACKAGING, PROMPT  # noqa: E402

ENUMS = {
    "group": {g.strip() for g in GROUPS.split(",")},
    "packaging": {p.strip() for p in PACKAGING.split(",")},
    "weight_class": {"light", "medium", "heavy"},
    "rigidity": {"soft", "semi", "rigid"},
}


def parse(text: str) -> list[dict] | None:
    """First JSON object in the reply -> validated items, or None if unparseable."""
    i = text.find("{")
    if i < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[i:])
    except json.JSONDecodeError:
        return None
    items = []
    for it in obj.get("items") or []:
        if not isinstance(it, dict) or not it.get("name"):
            continue
        r = {"name": str(it["name"]).strip()}
        for k in ("group", "packaging", "weight_class", "rigidity"):
            v = str(it.get(k, "")).strip().lower()
            r[k] = v if v in ENUMS[k] else None
        for k in ("cold_chain", "spill_risk"):
            r[k] = it.get(k) if isinstance(it.get(k), bool) else None
        try:
            r["quantity"] = max(1, int(it.get("quantity", 1)))
        except (TypeError, ValueError):
            r["quantity"] = 1
        items.append({k: r[k] for k in FIELDS})
    return items


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--scenes", type=Path, help="datasets_perception root: <scene>/frames/frame_000.png")
    src.add_argument("--crops", type=Path, help="crop_manifest.json of detector regions")
    ap.add_argument("--model", default="/workspace/models/qwen3-vl-32b-instruct")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-new-tokens", type=int, default=1536)
    ap.add_argument("--image-max-side", type=int, default=768)
    args = ap.parse_args()

    if args.scenes:
        jobs = [(d.name, d / "frames/frame_000.png", PROMPT)
                for d in sorted(args.scenes.iterdir()) if (d / "frames/frame_000.png").exists()]
    else:
        base = args.crops.parent
        jobs = [(c["crop"][:-4], base / c["crop"], PROMPT + CROP_SUFFIX)
                for c in json.loads(args.crops.read_text())]
    todo = [j for j in jobs if not (args.out / j[0] / "frames/frame_000.json").exists()]
    print(f"{len(jobs)} images, {len(todo)} to run", flush=True)

    import torch
    from peft import PeftModel
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, args.adapter).eval()

    for n, (jid, img_path, prompt) in enumerate(todo, 1):
        image = Image.open(img_path).convert("RGB")
        if max(image.size) > args.image_max_side:
            image.thumbnail((args.image_max_side, args.image_max_side), Image.LANCZOS)
        messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                                 {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        reply = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                       skip_special_tokens=True)[0]
        items = parse(reply)
        dst = args.out / jid / "frames/frame_000.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps({
            "items": items or [],
            "metadata": {"parse_failed": items is None, "raw": reply if items is None else None,
                         "prompt_sha1": hashlib.sha1(prompt.encode()).hexdigest()[:12],
                         "adapter": str(args.adapter)}}, indent=1))
        if n % 25 == 0:
            print(f"  {n}/{len(todo)}", flush=True)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
