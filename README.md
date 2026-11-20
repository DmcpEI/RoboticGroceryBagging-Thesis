# Robotic Grocery Bagging: Structured Perception for Packing-Relevant Inventory Generation

Code of the MSc dissertation of Diogo Paixão, Instituto Superior Técnico,
Universidade de Lisboa, 2026.

A camera above a work table photographs the groceries to be bagged. The system
identifies every object and describes it by the characteristics a bagging planner
needs (group, packaging, quantity, weight class, rigidity, cold chain, spill risk,
load tolerance, edibility). A planner assigns the objects to bags so that no
safety rule is broken, and a Baxter robot packs them.

Two methods compute the characteristics:

- **Catalog lookup** (the deployed method): a closed-set detector, trained only on
  scenes composed from photographs of single products, names each object, and its
  characteristics are read from a product catalog.
- **Direct prediction**: a fine-tuned vision–language model (Qwen3-VL-32B with a
  LoRA adapter) states the characteristics of each object from the image.

The risk-aware planner (ERM-MCTS) used on the robot and in Chapter 5 is the work
of Jacopo Silvestrin: `robot/algos/`, `robot/envs/`,
`robot/simulate_fixed_items_interface.py` and its guide `robot/README.md`, and his
experiment bundle `external/risk_aware_mcts/`. It is not a contribution of the
dissertation and is included so that the robot and Chapter 5 can be run.
`tools/planner/binpacking_env_v3_v18.py` is his environment with only the
perception-error table replaced, written by `tools/eval/build_planner_noise_env.py`.

Images, benchmark scenes, training runs and model weights are not included. The
scripts expect the benchmark under `datasets_perception/b50` with its ground truth
under `datasets_seq_GT_b50`, the captures under `data/robot_lab/rgbd`, and the RPC
dataset under `data/rpc`.

## Layout

| Folder | Contents |
|---|---|
| `tools/pipeline/` | Detector inventory (`detector_inventory.py`), catalog matching of names, the VLM prompt (`record_prompt.py`) |
| `tools/eval/` | Training-data builders (cutouts, composed scenes, for our catalog and the RPC catalog), the VLM runner, the baselines, and every scorer and analysis of Chapters 3 to 5 |
| `tools/finetune/` | Fine-tuning data and LoRA training of the VLM |
| `tools/planner/` | CP-SAT bagging solver, safety audit, off-the-shelf model as planner, runners for the risk-aware planner |
| `tools/figures/` | Data plots and example figures of the dissertation |
| `robot/` | Baxter deployment: pick computation from depth (`robot_pc_package/`), execution, calibration |
| `robot/algos/`, `robot/envs/`, `robot/simulate_fixed_items_interface.py` | Risk-aware planner (ERM-MCTS) used on the robot, by Jacopo Silvestrin |
| `external/risk_aware_mcts/` | His planner experiments (simulation, benchmark scenes, baselines), run by `tools/planner/run_risk_aware_v18.py` |
| `data/` | Product catalog of the 32 products and annotations of the RPC products |

## Where the results come from

Table and figure numbers are those of the final version of the dissertation.

| Result | Script |
|---|---|
| Corrupting each characteristic (Table 3.2) | `tools/eval/characteristic_perturbation.py` |
| Cutouts and composed training scenes (Sec. 3.4, Fig. 3.2) | `tools/eval/build_cutout_library.py --adaptive-height`, `tools/eval/build_plate_cutouts_rgb.py`, `tools/eval/build_synthetic_scenes.py` |
| VLM fine-tuning (Sec. 3.5, Table C.2) | `tools/finetune/build_record_ft_data.py`, `tools/finetune/fine_tune_qwen3vl.py --target-field-set record`, `tools/eval/run_vlm_record.py` |
| Whole record, identity, characteristics on the benchmark (Tables 4.2, 4.3, 4.4, 4.6 Robot) | `tools/eval/run_benchmark.sh --vlm`; Gemini: `tools/eval/run_gemini_robotics.py`, `tools/eval/reattach_gemini_sku_attrs.py`; hybrid: `tools/eval/yolo_crops_for_vlm.py`, `tools/eval/assemble_hybrid_run.py` |
| Benchmark scenes and one scene as three methods see it (Figs. 4.1, 4.2) | `tools/figures/examples.py` |
| Objects found and named (Table 4.5) | `tools/eval/removal_diff_boxes.py`, `tools/eval/localisation_recall.py` |
| Characteristic confusions on the located objects (Fig. 4.3) | `tools/eval/region_characteristic_confusion.py`, `tools/figures/region_confusions.py` |
| Cost of a wrong name (Sec. 4.4.1) | `tools/eval/confusion_cost.py` |
| Retail catalog (Table 4.6 Retail, Sec. 4.5) | `tools/eval/build_rpc_cutouts.py`, `build_rpc_synthetic_scenes.py`, `rpc_detector_inventory.py`, `eval_rpc_detector.py`, `score_characteristics.py`; robot side: `robot_identity_comparison.py` |
| Removal sequences (Sec. 4.6) | `tools/eval/removal_union_recall.py` |
| Cost of safety (Fig. 4.4) | `tools/planner/run_soft_cp_sweep.py --catalog-weights data/robot_lab/robot_item_catalog_perception.json`, `tools/figures/plots.py` |
| Bagging outcomes (Table 4.7) | `tools/planner/run_soft_cp_sweep.py`, `tools/planner/run_llm_planner_robotlab.py`, `tools/eval/run_gemini_full_pipeline.py`, `tools/eval/reaudit_cached_bags.py` |
| Products subject to a separation rule, each product (Table 4.8, Table D.1) | `tools/eval/compare_detectors_seeds.py` |
| Risk-aware planner with the deployed detector's errors (Tables 5.2, 5.3) | `tools/eval/perception_uncertainty.py`, `tools/eval/build_planner_noise_env.py`, `tools/eval/build_planner_inputs_b50.py`, `tools/planner/run_risk_aware_v18.py`, `tools/planner/summarize_risk_aware.py` |
| Robot trials (Chapter 5) | `robot/end_to_end_pipeline.py`, `tools/eval/robot_runs_ch4_audit.py` |

## Setup

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env   # only for the Gemini baselines
```

Checks that run without data:

```bash
python tools/planner/test_planner_contract.py
python tools/eval/test_catalog_match_container_names.py
python robot/test_planner_env_contract.py
```

## Licence

MIT, see `LICENSE`.
