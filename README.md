# Robotic Grocery Bagging: Structured Perception for Packing-Relevant Inventory Generation

Code of the MSc dissertation of Diogo Paixão (Instituto Superior Técnico, 2026).

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

## Layout

| Folder | Contents |
|---|---|
| `tools/pipeline/` | Detector inventory (`detector_inventory.py`), catalog matching of names, the VLM prompt (`record_prompt.py`) |
| `tools/eval/` | Training-data builders (cutouts, composed scenes, for our catalog and the RPC catalog), the VLM runner, the baselines, and every scorer and analysis of Chapter 4 |
| `tools/finetune/` | Fine-tuning data and LoRA training of the VLM |
| `tools/planner/` | CP-SAT bagging solver, safety audit, off-the-shelf model as planner |
| `tools/figures/` | Data plots of the dissertation |
| `robot/` | Baxter deployment: pick computation from depth (`robot_pc_package/`), execution, calibration |
| `robot/algos/`, `robot/envs/`, `robot/simulate_fixed_items_interface.py` | Risk-aware planner (ERM-MCTS) used on the robot, by Jacopo Silvestrin |
| `data/` | Product catalog of the 32 products and annotations of the RPC products |

## Where the results come from

| Result | Script |
|---|---|
| Cutouts and composed training scenes (Sec. 3.3) | `tools/eval/build_cutout_library.py --adaptive-height`, `tools/eval/build_synthetic_scenes.py` |
| Whole record, identity, characteristics (Tables 4.1, 4.2, 4.4) | `tools/eval/run_benchmark.sh --vlm` |
| VLM fine-tuning (Sec. 3.4) | `tools/finetune/build_record_ft_data.py`, `tools/finetune/fine_tune_qwen3vl.py --target-field-set record` |
| Localisation and naming (Table 4.3) | `tools/eval/removal_diff_boxes.py`, `tools/eval/localisation_recall.py` |
| Cost of a wrong name (Sec. 4.4.1) | `tools/eval/confusion_cost.py` |
| Retail catalog: detector, retrieval, characteristics (Tables 4.4, 4.6, 4.7) | `tools/eval/build_rpc_cutouts.py`, `build_rpc_synthetic_scenes.py`, `rpc_detector_inventory.py`, `rpc_build_exemplar_index.py`, `rpc_retrieval_eval.py`, `score_characteristics.py` |
| Robot catalog retrieval (Table 4.6) | `tools/eval/robot_identity_comparison.py` |
| Removal sequences (Sec. 4.6) | `tools/eval/removal_union_recall.py` |
| Cost of safety, bagging outcomes (Fig. 4.1, Table 4.8) | `tools/planner/run_soft_cp_sweep.py`, `tools/eval/run_gemini_full_pipeline.py`, `tools/planner/run_llm_planner_robotlab.py`, `tools/eval/reaudit_cached_bags.py` |
| Robot trials (Chapter 5) | `robot/end_to_end_pipeline.py`, `tools/eval/robot_runs_ch4_audit.py` |

## Setup

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env   # only for the Gemini baselines
```

Images, benchmark scenes, training runs and model weights are not included in the
repository. The scripts expect the benchmark under `datasets_perception/b50` and
its ground truth under `datasets_seq_GT_b50`, the captures under
`data/robot_lab/rgbd`, and the RPC dataset under `data/rpc`.

Checks that run without data:

```bash
python tools/planner/test_planner_contract.py
python tools/eval/test_catalog_match_container_names.py
python robot/test_planner_env_contract.py
```
