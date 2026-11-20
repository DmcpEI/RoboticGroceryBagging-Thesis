# Robotic Grocery Bagging: benchmark, captures and models

Data of the MSc dissertation *Robotic Grocery Bagging: Structured Perception for
Packing-Relevant Inventory Generation* (Diogo Paixão, Instituto Superior Técnico,
Universidade de Lisboa, 2026). Code: https://github.com/DmcpEI/RoboticGroceryBagging-Thesis

A RealSense D435 on the ceiling looks straight down at a work table holding
grocery and household products. This record holds the photographs and ground
truth used to evaluate the perception system, the photographs of single products
its detector was trained from, the deployed models, every method's output on the
benchmark, and the logs of the robot trials. Table and section numbers refer to
the dissertation.

## Contents

| Folder | Contents |
|---|---|
| `catalog/products.json` | The 32 products, with every characteristic, weight in grams and volume in cm³ (Table A.1) |
| `camera/` | `intrinsics.json` of the colour image, to which depth is aligned; `empty_table/`: photographs of the empty table, the background of the synthetic training scenes |
| `benchmark/` | The static benchmark: 150 photographs of 31 arrangements, 1,345 objects (Sec. 4.1.1). `scenes.csv` lists them |
| `removal_sequences/` | 20 sequences, 215 frames, 195 objects: an arrangement photographed after each object is removed (Sec. 4.1.1, 4.6) |
| `products/` | 1,100 photographs of single products, one product per photograph, the material of the synthetic training scenes (Sec. 3.4.2) |
| `predictions/` | Each method's output on the benchmark, behind Tables 4.2 to 4.4 |
| `robot_trials/trials.json` | The 60 robot trials of Chapter 5 |
| `models/` | The deployed detector and the low-rank adapter of fine-tuned Qwen |
| `id_map.csv` | Capture ids used in the code repository → ids in this record |

## Formats

Every photograph is a folder with `color.png` (640×480 RGB) and `depth.png`
(640×480, 16-bit, **millimetres** from the camera, aligned to the colour image;
0 = no reading). All boxes are pixels on the 640×480 image, `[x1, y1, x2, y2]`.

**`benchmark/<scene_id>/objects.json`** — the products in the scene and how many
of each. The benchmark records which products are present, not where. Scene ids
are `<density>_<arrangement>_<shot>`; density is `sparse` (4–6 objects, none
touching), `medium` (7–10) or `dense` (9–16, touching, overlapping or stacked).
All shots of one arrangement hold the same products.

**`removal_sequences/<sequence_id>/sequence.json`** — for each step, the object
removed since the previous step and the objects on the table. `step_00` is the
full arrangement and the last step the empty table. The three sequences named
`sparse_*` have no object touching another. `located_objects.json` gives, for
137 of the 195 removals, the region the removed object occupied in the frame
before its removal, found by comparing the depth images before and after; it
locates only the object that left, so it measures recall, not precision.

**`products/<product>/<capture>/`** — `meta.json` names the product. Produce bags
were photographed holding 1 to 5 fruits (`fruit_count`, capture ids `nK_###`).
Captures `early_1`/`early_2` come from the first capture session.

**`catalog/products.json`** — `group`, `packaging`, `weight_class`, `rigidity`,
`cold_chain`, `spill_risk`, `edible`, `crush_score` (load tolerance, 1 bears any
load, 10 bears almost none; derived from packaging, rigidity and group), and
`weight_g`, `volume_cm3` with their sources (the YCB object set where the product
belongs to it, otherwise published figures; produce bags from the weight of one
fruit and the number in the bag).

**`predictions/<method>/<scene_id>.json`** — `{"items": [...]}`, one entry per
product named, with `name`, `quantity` (when the method gives one) and the
characteristics. `_catalog` runs carry characteristics read from the catalog
through the predicted name; `_own` runs carry those the model stated itself.

| Method | Description |
|---|---|
| `detector` | Closed-set detector, characteristics from the catalog. `unknown_object` entries are detections of confidence 0.15–0.4 |
| `hybrid_*` | Detector regions named one at a time by fine-tuned Qwen; `bbox_2d` is the region given to the model |
| `finetuned_qwen_*` | Qwen3-VL-32B-Instruct with the adapter in `models/`, whole image |
| `base_qwen_*` | The same model without the adapter |
| `gemini_*` | Gemini Robotics-ER 2; `gemini_catalog` was asked for names only, `gemini_own` for names and characteristics |

**`robot_trials/trials.json`** — one entry per trial: product set (1–3),
arrangement (`spaced`/`clustered`), the planner's `beta`, repetition, `events`
(`[item, bag, outcome]` per pick, outcome `SUCCESS`, `LOST_PICKUP`,
`LOST_TRANSIT` or `BLOCKED`), the items placed and the bags they went to, the
products perceived during the trial, whether the trial was stopped before the
table was empty, and the operator's notes on what the log could not show.

**`models/`** — `detector_yolo11s_robot32.pt` (Ultralytics YOLO11s, 640 px input,
confidence threshold 0.4) with its class names in `detector_classes.json`;
`qwen3vl32b_lora_record/` (LoRA rank 16 on the query, key and value projections
of `Qwen/Qwen3-VL-32B-Instruct`). Training parameters are in Appendix C.

## Licence

Creative Commons Attribution 4.0 International (CC BY 4.0),
https://creativecommons.org/licenses/by/4.0/. Please cite the dissertation.
