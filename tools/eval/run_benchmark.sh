#!/usr/bin/env bash
# The static benchmark of Chapter 4 (150 scenes): detector, and with --vlm the
# fine-tuned vision-language model and the hybrid, each scored for whole record
# (Table 4.1), identity (Table 4.2) and characteristic recovery (Table 4.4).
#
#   bash tools/eval/run_benchmark.sh          # detector only
#   bash tools/eval/run_benchmark.sh --vlm    # + VLM and hybrid (needs the adapter)
set -e
cd "$(dirname "$0")/../.."
PY=${PY:-python}
W=${W:-runs/detect/runs/detect/runs/detector/syn_v18/weights/best.pt}
C=data/robot_lab/synthetic_det_v18/classes.json
AD=${AD:-runs/fine_tuning/lora_32b_record}
GT=datasets_seq_GT_b50
VLM=0; [ "${1:-}" = "--vlm" ] && VLM=1

echo "=== evaluation sets from the captures ==="
$PY tools/eval/build_b50_eval_sets.py

echo "=== closed-set detector ==="
rm -rf runs/b50_detector
$PY tools/pipeline/detector_inventory.py --weights $W --classes $C \
    --pattern 'b50_*' --min-conf 0.40 --seq-layout --out runs/b50_detector
# detector_inventory writes the planner record; the perception record is read
# from the catalog through the predicted names.
$PY tools/eval/reattach_gemini_sku_attrs.py --run runs/b50_detector | tail -1
RUNS="--run detector=runs/b50_detector"
IDS="--run detector=runs/b50_detector"

if [ "$VLM" = "1" ]; then
  echo "=== vision-language model, whole frame ==="
  $PY tools/eval/run_vlm_record.py --scenes datasets_perception/b50 --adapter $AD \
      --out runs/b50_vlm_own
  rm -rf runs/b50_vlm_lookup && cp -r runs/b50_vlm_own runs/b50_vlm_lookup
  $PY tools/eval/reattach_gemini_sku_attrs.py --run runs/b50_vlm_lookup | tail -1

  echo "=== hybrid: detector regions named by the model ==="
  # every box the detector reports, its unknown objects (0.15-0.4) included
  $PY tools/eval/yolo_crops_for_vlm.py --weights $W --classes $C \
      --pattern 'b50_*' --min-conf 0.15 --out data/robot_lab/yolo_crops_b50
  M=data/robot_lab/yolo_crops_b50/crop_manifest.json
  $PY tools/eval/run_vlm_record.py --crops $M --adapter $AD --out runs/b50_crops_vlm
  $PY tools/eval/assemble_hybrid_run.py --manifest $M --vlm runs/b50_crops_vlm \
      --out runs/b50_hybrid_lookup
  $PY tools/eval/assemble_hybrid_run.py --manifest $M --vlm runs/b50_crops_vlm \
      --out runs/b50_hybrid_own --own-chars
  RUNS="$RUNS --run hybrid_lookup=runs/b50_hybrid_lookup --run hybrid_own=runs/b50_hybrid_own"
  RUNS="$RUNS --run vlm_lookup=runs/b50_vlm_lookup --run vlm_own=runs/b50_vlm_own"
  IDS="$IDS --run hybrid=runs/b50_hybrid_lookup --run vlm=runs/b50_vlm_lookup"
fi

echo "=== scores ==="
$PY tools/eval/score_full_record.py --gt $GT $RUNS --out runs/eval/b50_full_record.json
$PY tools/eval/score_identity_rules.py --gt $GT $IDS --out runs/eval/b50_identity_rules.json
$PY tools/eval/score_characteristics.py --gt $GT --pred runs/b50_detector --label Robot \
    --out runs/eval/b50_chars_detector.json
