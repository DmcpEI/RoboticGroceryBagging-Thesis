#!/usr/bin/env python3
"""
Like simulate_fixed_items.py, but the fixed OBSERVED sequence for each episode
comes from a real benchmark scene (~/Downloads/benchmark/benchmark/
planner_input_our_perception/*.json) instead of a hand-picked list of
catalog ids -- and instead of a single algo, this script sweeps THREE
planners over the same scenes: erm-mcts (via envs.binpacking_env_v3's
partial-observability/sliding-window MCTS rollout), gemini-planner (a single
upfront Gemini call bagging the whole scene), and rule-based-planner (a
CP-SAT solver bagging the whole scene under fixed safety rules).

Each planner_input_our_perception/<scene_id>.json file is the OBSERVED/
perceived item list for that scene (a real, noisy detector's actual output).
The REAL episode -- the sequence of placements actually run through the
physics, whose bags/violations get measured -- is backed by real ground
truth from the sibling planner_input_ground_truth/<scene_id>.json, paired to
the observed items via _pair_observed_with_ground_truth() (see below): this
is NOT a positional zip (verified empirically: only 1/53 same-count scenes
even has matching file order between the two files -- 30/53 have the exact
same item multiset in scrambled order, resolved with zero ambiguity by exact
display_name matching; the rest fall back to an optimal likelihood-based
assignment, see below). The agent itself (and MCTS's internal rollouts) never
sees ground truth -- only envs.binpacking_env_v3's posterior P(true|observed)
(_sample_true_item), same as every other simulate_*.py script in this repo.
That posterior is now used for exactly one thing here: ERM-MCTS's internal
`.learn()` simulated lookahead (never for the real episode).

H (horizon = num_items_to_pack) is NOT fixed like every other script's
CONFIG["H"] -- it equals the number of OBSERVED items in that specific scene
(3-17 across the 150 planner_input_our_perception scenes), read directly from
load_scene()'s output. Ground-truth items with no matching observation are
simply never perceived/placed (excluded from the episode, don't affect H).
Observed items with no ground-truth counterpart (hallucinated detections) are
paired with PHANTOM_TRUE_ITEM (zero weight/volume/risk -- "nothing is really
there") -- the agent still perceives and "places" it (H unchanged, matching
a real deployment that doesn't know it hallucinated), but it can never
trigger a cost or a violation, and doesn't count as occupying a bag.

erm-mcts sees the WHOLE scene at once (n_visible_items=H, no sliding-window
partial reveal) -- FixedScenePerceptionEnv.sample_initial_state() overrides
the base class's Poisson(5)-sampled initial reveal to deterministically show
every item from t=0. gemini-planner/rule-based-planner already effectively
saw the whole scene at once regardless (they plan the entire episode upfront
from the full observed list, never adapting to incremental reveal).

For erm-mcts, num_bags must be decided upfront (MCTS can't discover it
dynamically the way a whole-plan solver can) -- default
max(5, ceil(H/3)) rather than the flat 5 every other script uses, since scenes
here can have up to 17 items and 5 fixed-capacity bags would systematically
inflate illegal-move cost for large scenes as a capacity artifact, not a
genuine planning failure. Override via CONFIG["num_bags"]. gemini-planner and
rule-based-planner size their own bag count from their own plan output, same
as their standalone scripts.

Bags-used and three safety-violation counts are tracked per scene per planner
(count_violations, computed on the final TRUE bag contents -- physical
reality, not the agent's belief -- for all three planners identically):
  - fragile_weight_on_top: a crush_score>=8 item with >=1 item placed after
    it in the same bag (mirrors the env's own crush-risk accrual trigger,
    which fires on ANY later item, not just "heavy" ones)
  - spill_risk_weight_on_top: a spill_risk item with >=1 item placed after it
    in the same bag (mirrors the env's own spill-risk accrual trigger)
  - spill_vulnerable_with_spill_risk: a spill_vulnerable item sharing a bag
    with >=1 spill_risk item, regardless of order (mirrors the contamination
    cost mechanic)

rule-based-planner needs `ortools`, which must NEVER be installed into the
shared conda base env (see simulate_rule_based_planner.py's own warning) --
run this script via `venv/bin/python3` whenever "rule-based-planner" is in
CONFIG["algos"]. The import is deferred (inside run_scene_rule_based) so
`--algos erm-mcts` / `--algos gemini-planner` alone still work under the bare
interpreter.

Usage:
  python3 simulate_benchmark_scenes.py                       # CONFIG defaults
  python3 simulate_benchmark_scenes.py <seed> <data_folder>    # SLURM-style
  python3 simulate_benchmark_scenes.py --dry-run               # gemini-planner
                                                                 # uses the
                                                                 # round-robin
                                                                 # stub bagger,
                                                                 # no live API
  venv/bin/python3 simulate_benchmark_scenes.py                # required
                                                                 # whenever
                                                                 # rule-based-
                                                                 # planner is
                                                                 # selected
"""
from __future__ import annotations

import os
import sys
import json
import math
import random
import time
import pathlib
import multiprocessing as mp
from collections import deque, defaultdict
from datetime import datetime

import numpy as np
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

from algos.erm_mcts import ERMMCTS
from envs.binpacking_env_v3 import (
    BinPackingEnv, Bag, Item, ITEMS_DATA, PLACEHOLDER_ITEM,
    _sample_true_item, _PERCEIVED_CANDIDATES,
)

# gemini-planner's pure helper functions are imported lazily (inside
# run_scene_gemini) rather than at module level: simulate_gemini_planner.py
# itself imports `dotenv` at its own module level, which is only installed in
# this repo's venv/ (see simulate_rule_based_planner.py's isolation notes),
# not the shared conda base env -- deferring keeps `--algos erm-mcts` runnable
# under the bare interpreter, same rationale as rule-based-planner's ortools
# deferral below.

DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent) + '/data/'
DEBUG = False

FRAGILE_THRESHOLD = 8  # matches simulate_rule_based_planner.py's constant and v3's
                        # own crush-risk trigger condition (crush_score >= 8)

# Every scene item's own real catalog id, looked up by display_name -- NOT 0.
# available_actions() filters visible_items by `item.id != 0` (the
# PLACEHOLDER_ITEM sentinel check), so a scene item built with id=0 would be
# silently treated as "not yet revealed" and excluded from every action list.
# All 32 catalog names are covered (verified this session), so this lookup
# never misses for planner_input_our_perception's display_names.
_NAME_TO_ID: dict[str, int] = {item.name: item.id for item in ITEMS_DATA}

# Stand-in true item for a hallucinated observation (no ground-truth item left
# to pair it with): zero weight/volume/crush_score, no spill risk/vulnerability
# -- physically nothing is there, so it can never trigger a cost or a
# violation. id=-1: not 0 (PLACEHOLDER_ITEM's sentinel, which
# available_actions() filters on for OBSERVED items -- irrelevant here since
# this is only ever used as a TRUE item, never observed) and not a real
# catalog id 1-32.
PHANTOM_NAME = "(hallucinated)"
PHANTOM_TRUE_ITEM = Item(-1, PHANTOM_NAME, 0, 0, 0, "Household", "Ambient", False, False, False)

CONFIG = {
    "N": 150,                # scenes to process (bounded by len(selected scenes))
    "num_processors": 8,     # used by erm-mcts and rule-based-planner only --
                              # gemini-planner stays sequential (rate-limit pacing)
    "env": "binpacking",
    "algos": ["erm-mcts"],#, "rule-based-planner"],  # subset ok
    "gamma": 0.99,

    "scenes_dir": str(pathlib.Path(__file__).parent / "benchmark/planner_input_our_perception"),
    "ground_truth_dir": str(pathlib.Path(__file__).parent / "benchmark/planner_input_ground_truth"),
    "scene_limit": None,     # cap for smoke tests; None = all scenes found (up to N)
    "scene_ids": None,       # explicit override list of scene_id strings; None = all

    # --- erm-mcts only ---
    "erm_betas": [5],
    "n_iter_per_timestep": 5000,
    "num_bags": None,        # None -> max(5, ceil(H/3)) per scene; int overrides
    # n_visible_items is NOT configurable anymore -- erm-mcts always sees the
    # whole scene at once (n_visible_items=H, forced in run_scene_erm_mcts).

    # --- gemini-planner only ---
    "gemini_model": "gemini-robotics-er-2-preview",
    "sleep_between_calls": 4.0,
    "max_retries": 6,

    # --- rule-based-planner only ---
    "solver_time_limit_s": 10.0,
}


# ---------------------------------------------------------------------------
# Scene loading + ground-truth pairing
# ---------------------------------------------------------------------------

def _pair_observed_with_ground_truth(obs_fields: list, gt_fields: list) -> list:
    """Pairs each observed item (in its own file order -- preserved as the
    agent's perceived arrival order) with a ground-truth item, in two passes.
    Positional (zip) pairing is WRONG here -- verified empirically against
    the real files: only 1/53 same-count scenes has matching file order; 30
    have the exact same item multiset in scrambled order. There is no shared
    instance id between the two files, so exact-name matching (pass 1) is the
    strongest signal available, and a likelihood-based optimal assignment
    (pass 2) is the best that can be done for whatever's left.

    Pass 1 -- exact display_name match: consume gt_fields into a name-keyed
    pool, claim a match for each obs item wherever one remains.

    Pass 2 -- for observed items pass 1 couldn't resolve, and ground-truth
    items pass 1 didn't consume: solve an optimal assignment maximizing total
    P(obs|true)*P(true) using envs.binpacking_env_v3._PERCEIVED_CANDIDATES --
    the SAME (unnormalized Bayes-posterior) weights _sample_true_item draws
    from elsewhere in this codebase, not a fresh re-derivation. This is a
    proper bipartite assignment (scipy.optimize.linear_sum_assignment), not a
    greedy per-item pick, so two observed items can't both "claim" the same
    ground-truth item.

    Excess unresolved observed items (more observed than ground truth left)
    are hallucinations -> paired with None (caller substitutes
    PHANTOM_TRUE_ITEM). Excess unclaimed ground-truth items (more ground
    truth than observed left) are missed detections -> simply dropped, never
    referenced again.

    Returns a list of (obs_fields_dict, gt_fields_dict_or_None), one entry per
    observed item, in observed's original order.
    """
    gt_by_name = defaultdict(list)
    for g in gt_fields:
        gt_by_name[g["display_name"]].append(g)

    pairs = [None] * len(obs_fields)
    unresolved = []
    for i, o in enumerate(obs_fields):
        bucket = gt_by_name.get(o["display_name"])
        if bucket:
            pairs[i] = (o, bucket.pop(0))
        else:
            unresolved.append(i)
    leftover_gt = [g for bucket in gt_by_name.values() for g in bucket]

    if unresolved and leftover_gt:
        W = np.zeros((len(unresolved), len(leftover_gt)))
        for r, i in enumerate(unresolved):
            candidates = dict(_PERCEIVED_CANDIDATES.get(obs_fields[i]["display_name"], []))
            for c, g in enumerate(leftover_gt):
                W[r, c] = candidates.get(_NAME_TO_ID[g["display_name"]], 0.0)
        row_idx, col_idx = linear_sum_assignment(W, maximize=True)
        matched_rows = set()
        for r, c in zip(row_idx, col_idx):
            pairs[unresolved[r]] = (obs_fields[unresolved[r]], leftover_gt[c])
            matched_rows.add(r)
        still_unresolved = [unresolved[r] for r in range(len(unresolved)) if r not in matched_rows]
    else:
        still_unresolved = unresolved

    for i in still_unresolved:
        pairs[i] = (obs_fields[i], None)

    return pairs


def _make_item(fields: dict) -> Item:
    name = fields["display_name"]
    return Item(
        _NAME_TO_ID[name], name,
        fields["est_weight_g"], fields["est_volume_cc"], fields["crush_score"],
        fields["category"], fields["temperature"],
        fields["spill_risk"], fields["spill_vulnerable"],
        False,  # orientation_sensitive -- no source field, matches v3's
                # catalog-wide convention (see envs/binpacking_env_v3.py)
    )


def load_scene(op_path: pathlib.Path, gt_path: pathlib.Path) -> tuple[str, list]:
    """Reads one planner_input_our_perception/*.json file (observed items)
    and its sibling planner_input_ground_truth/*.json file (true items),
    pairs them via _pair_observed_with_ground_truth, and returns
    (scene_id, list of (observed_Item, true_Item) pairs) in the observed
    file's original order -- the arrival order for both erm-mcts and the
    whole-plan solvers' arrival-order safety rules."""
    with open(op_path) as f:
        op_data = json.load(f)
    with open(gt_path) as f:
        gt_data = json.load(f)

    raw_pairs = _pair_observed_with_ground_truth(
        list(op_data["items_data"].values()), list(gt_data["items_data"].values())
    )
    pairs = [
        (_make_item(o), _make_item(g) if g is not None else PHANTOM_TRUE_ITEM.copy())
        for o, g in raw_pairs
    ]
    return op_data["scene_id"], pairs


def load_all_scenes(scenes_dir: str, ground_truth_dir: str,
                     scene_limit: int = None, scene_ids: list = None):
    paths = sorted(pathlib.Path(scenes_dir).glob("*.json"))
    scenes = [
        load_scene(p, pathlib.Path(ground_truth_dir) / p.name)
        for p in paths
    ]
    if scene_ids is not None:
        wanted = set(scene_ids)
        scenes = [s for s in scenes if s[0] in wanted]
    if scene_limit is not None:
        scenes = scenes[:scene_limit]
    return scenes


# ---------------------------------------------------------------------------
# Bags-used and violation auditing -- shared across all three planners,
# computed once from whichever run_scene_* returns as the final TRUE bags.
# ---------------------------------------------------------------------------

def count_violations(bags) -> dict:
    fragile_weight_on_top = 0
    spill_risk_weight_on_top = 0
    spill_vulnerable_with_spill_risk = 0
    for bag in bags:
        items = bag.items  # placement order, bottom to top; may include
                            # PHANTOM_TRUE_ITEM copies (hallucinated
                            # detections) -- these carry zero physical
                            # presence, so they're excluded both as the
                            # potential SOURCE of a violation (trivial, since
                            # all their risk-relevant fields are False/0) and
                            # as "weight on top" of a real item below them
                            # (nothing is really stacked there)
        for i, item in enumerate(items):
            if item.name == PHANTOM_NAME:
                continue
            later_real = any(it.name != PHANTOM_NAME for it in items[i + 1:])
            if item.crush_score >= FRAGILE_THRESHOLD and later_real:
                fragile_weight_on_top += 1
            if item.spill_risk and later_real:
                spill_risk_weight_on_top += 1
        real_items = [it for it in items if it.name != PHANTOM_NAME]
        if any(it.spill_risk for it in real_items):
            spill_vulnerable_with_spill_risk += sum(1 for it in real_items if it.spill_vulnerable)
    return {
        "fragile_weight_on_top": fragile_weight_on_top,
        "spill_risk_weight_on_top": spill_risk_weight_on_top,
        "spill_vulnerable_with_spill_risk": spill_vulnerable_with_spill_risk,
    }


def n_bags_used(bags) -> int:
    # A bag containing only phantom (hallucinated) items is not "used" --
    # nothing physically sits there.
    return sum(1 for b in bags if any(it.name != PHANTOM_NAME for it in b.items))


VIOLATION_TYPES = ("fragile_weight_on_top", "spill_risk_weight_on_top", "spill_vulnerable_with_spill_risk")


def _scenes_affected_stats(violations_list: list) -> dict:
    """Per-type and combined count of scenes with >=1 violation of that kind,
    out of len(violations_list) scenes -- "combined"/"any" means a scene with
    >=1 violation of ANY of the three types, not summed across types (a scene
    with 3 fragile violations and 0 of anything else still counts once)."""
    n = len(violations_list)
    by_type = {k: sum(1 for v in violations_list if v[k] > 0) for k in VIOLATION_TYPES}
    affected_any = sum(1 for v in violations_list if any(v[k] > 0 for k in VIOLATION_TYPES))
    return {
        "n_scenes": n,
        "scenes_affected_any": affected_any,
        "fraction_scenes_affected_any": (affected_any / n) if n else 0.0,
        "scenes_affected_by_type": by_type,
    }


def _default_num_bags(H: int, cfg: dict) -> int:
    if cfg.get("num_bags") is not None:
        return cfg["num_bags"]
    return max(5, math.ceil(H / 3))


# ---------------------------------------------------------------------------
# erm-mcts branch
# ---------------------------------------------------------------------------

class FixedScenePerceptionEnv(BinPackingEnv):
    """BinPackingEnv (v3) where the OBSERVED items shown to the agent during
    the REAL episode are consumed from a scene's fixed, ground-truth-paired
    item list (see load_scene/_pair_observed_with_ground_truth) instead of
    being drawn noisily. Both the observed and true item in each scripted
    pair are already resolved (ground-truth-backed, or PHANTOM_TRUE_ITEM for
    a hallucinated detection) -- nothing is sampled for the real episode.
    MCTS's internal .learn() rollouts never touch this queue (scripted_mode
    is only True around the real sample_initial_state()/step() calls), so
    planning stays randomly sampled from NOISY_ITEMS via
    envs.binpacking_env_v3's own posterior, unaffected by any of this."""

    def __init__(self, pairs: list, **kwargs):
        super().__init__(**kwargs)
        n = self.num_items_to_pack
        if len(pairs) != n:
            raise ValueError(
                f"pairs must have length {n} (= num_items_to_pack = H), got {len(pairs)}."
            )
        self._pairs = list(pairs)  # list of (observed_Item, true_Item)
        self._observed_queue: list = []
        self._scripted_mode = False
        self._draw_log: list = []

    def set_scripted_mode(self, enabled: bool) -> None:
        self._scripted_mode = enabled

    def reset_scripted_queue(self) -> None:
        self._observed_queue = list(self._pairs)
        self._draw_log = []

    def get_draw_log(self) -> list:
        return list(self._draw_log)

    def _draw_item(self, slot_idx: int, t: int):
        if self._scripted_mode and self._observed_queue:
            observed_item, true_item = self._observed_queue.pop(0)
            self._draw_log.append((observed_item.name, true_item.name))
            return true_item, observed_item
        return super()._draw_item(slot_idx, t)  # MCTS internal planning stays random

    def sample_initial_state(self):
        # Everything visible from the start (deterministic n_initial = H),
        # not the base class's Poisson(5)-sampled partial reveal -- this
        # scene is fully perceived upfront, no sliding-window discovery.
        # Mirrors the base sample_initial_state() body exactly except
        # n_initial's computation (always H here, never sampled/clipped).
        bags = tuple(Bag() for _ in range(self.num_bags))
        observed_bags = tuple(Bag() for _ in range(self.num_bags))
        n_initial = self.num_items_to_pack  # = H, all of it
        true_items, visible_items = [], []
        for i in range(self.n_visible_items):
            if i < n_initial:
                true_item, observed_item = self._draw_item(i, 0)
            else:
                true_item, observed_item = PLACEHOLDER_ITEM, PLACEHOLDER_ITEM
            true_items.append(true_item)
            visible_items.append(observed_item)
        true_items, visible_items = tuple(true_items), tuple(visible_items)
        n_detected = sum(1 for it in visible_items if it.id != 0)
        return {
            "state": self._get_rl_vector(observed_bags, visible_items),
            "bags": bags,
            "observed_bags": observed_bags,
            "visible_items": visible_items,
            "true_items": true_items,
            "items_remaining": self.num_items_to_pack - n_detected,
            "t": 0,
        }


def simulate_erm_mcts_scene(env: FixedScenePerceptionEnv, H: int,
                             erm_beta: float, n_iter_per_timestep: int) -> tuple:
    K_ucb = np.sqrt(2)

    env.reset_scripted_queue()
    env.set_scripted_mode(True)
    extended_state = env.sample_initial_state()
    env.set_scripted_mode(False)

    mcts = ERMMCTS(initial_state=extended_state, env=env, K_ucb=K_ucb,
                   erm_beta=erm_beta, rollout_policy=None, root_depth=0)

    cumulative_discounted_cost = 0.0
    for t in range(H):
        mcts.learn(n_iters=n_iter_per_timestep)
        selected_action = mcts.best_action()

        env.set_scripted_mode(True)
        extended_state, cost, terminated = env.step(extended_state, selected_action)
        env.set_scripted_mode(False)

        cumulative_discounted_cost += cost * env.gamma ** t

        if DEBUG:
            print(f"t={t}  action={selected_action}  cost={cost:.4f}")

        updated_root = mcts.update_root_node(selected_action, extended_state)
        if updated_root:
            mcts.set_root_depth(t + 1)
        else:
            mcts = ERMMCTS(initial_state=extended_state, env=env, K_ucb=K_ucb,
                           erm_beta=erm_beta, rollout_policy=None, root_depth=t + 1)

    return cumulative_discounted_cost, extended_state["bags"]


def run_scene_erm_mcts(cfg: dict, scene_id: str, pairs: list, erm_beta: float, seed: int):
    np.random.seed(seed)
    random.seed(seed)
    H = len(pairs)
    num_bags = _default_num_bags(H, cfg)
    env = FixedScenePerceptionEnv(
        pairs=pairs, num_items_to_pack=H, num_bags=num_bags,
        n_visible_items=H,  # everything visible from the start, see class docstring
        gamma=cfg.get("gamma", 0.99),
    )
    cost, bags = simulate_erm_mcts_scene(env, H, erm_beta, cfg["n_iter_per_timestep"])
    return cost, bags, env.num_bags, env.get_draw_log()


# ---------------------------------------------------------------------------
# gemini-planner / rule-based-planner shared replay env + driver
# ---------------------------------------------------------------------------

class ScriptedOrderBinPackingEnv(BinPackingEnv):
    """Shared by gemini-planner and rule-based-planner -- both plan the WHOLE
    scene upfront (no MCTS branching), so this replays a pre-built
    (true_item, observed_item) queue in placement order. Byte-for-byte port
    of GeminiScriptedBinPackingEnv/RuleBasedScriptedBinPackingEnv onto v3."""

    def __init__(self, draw_queue: deque, **kwargs):
        super().__init__(**kwargs)
        self._draw_queue = draw_queue
        self._slot_queue_pos: dict = {}
        self._next_queue_pos = 0

    def _draw_item(self, slot_idx, t):
        true_item, obs_item = self._draw_queue.popleft()
        self._slot_queue_pos[slot_idx] = self._next_queue_pos
        self._next_queue_pos += 1
        return true_item, obs_item


def _replay_bags_by_pos(bags_by_pos: list, pairs: list, H: int, gamma: float) -> tuple:
    num_bags = max(len(bags_by_pos), 1)
    queue_order = [pos for bag in bags_by_pos for pos in bag]
    queue_bag_idx = [b for b, bag in enumerate(bags_by_pos) for _ in bag]
    draw_queue = deque(pairs[p] for p in queue_order)

    env = ScriptedOrderBinPackingEnv(
        draw_queue=draw_queue, num_bags=num_bags, num_items_to_pack=H, gamma=gamma,
    )
    state = env.sample_initial_state()
    cumulative_discounted_cost = 0.0
    for next_to_place in range(H):
        slot_idx = next(s for s, pos in env._slot_queue_pos.items() if pos == next_to_place)
        action = slot_idx * num_bags + queue_bag_idx[next_to_place]
        state, cost, _ = env.step(state, action)
        cumulative_discounted_cost += cost * env.gamma ** next_to_place
    return cumulative_discounted_cost, state["bags"], num_bags


# ---------------------------------------------------------------------------
# gemini-planner branch
# ---------------------------------------------------------------------------

def run_scene_gemini(cfg: dict, scene_id: str, pairs: list, seed: int,
                      model=None, dry_run: bool = False):
    # Deferred import -- see module-level comment near the top of this file.
    from simulate_gemini_planner import call_gemini_with_backoff, _dry_run_bagger, PACK_PROMPT, parse_bags

    np.random.seed(seed)
    random.seed(seed)
    H = len(pairs)

    # pairs is already ground-truth-backed (observed_Item, true_Item), from
    # load_scene -- no posterior sampling here at all (gemini-planner has no
    # rollout/lookahead phase, so there's nothing to preserve sampling for).
    names = [obs.name for obs, _ in pairs]
    draw_log = [(obs.name, true.name) for obs, true in pairs]
    replay_pairs = [(true, obs) for obs, true in pairs]  # _replay_bags_by_pos wants (true, obs)

    if dry_run:
        reply_text = _dry_run_bagger(names)
    else:
        listing = "\n".join(f"{i + 1}. {nm}" for i, nm in enumerate(names))
        reply_text = call_gemini_with_backoff(
            model, PACK_PROMPT.format(items=listing), cfg["max_retries"]
        )
    bags_by_pos, n_dropped = parse_bags(reply_text, H)

    cost, bags, num_bags = _replay_bags_by_pos(bags_by_pos, replay_pairs, H, cfg.get("gamma", 1))
    return cost, bags, num_bags, draw_log, n_dropped


# ---------------------------------------------------------------------------
# rule-based-planner branch
# ---------------------------------------------------------------------------

def run_scene_rule_based(cfg: dict, scene_id: str, pairs: list, seed: int):
    # Deferred import: keeps the ortools dependency (and the venv/-only
    # isolation rule -- see simulate_rule_based_planner.py) off any
    # invocation that doesn't select rule-based-planner.
    from simulate_rule_based_planner import solve_bagging_rules, _item_to_planner_fields

    np.random.seed(seed)
    random.seed(seed)
    H = len(pairs)

    # pairs is already ground-truth-backed (observed_Item, true_Item), from
    # load_scene -- no posterior sampling here (same rationale as gemini above).
    draw_log = [(obs.name, true.name) for obs, true in pairs]
    replay_pairs = [(true, obs) for obs, true in pairs]  # _replay_bags_by_pos wants (true, obs)

    item_features = [_item_to_planner_fields(obs) for obs, _ in pairs]
    bags_by_pos = solve_bagging_rules(item_features, cfg.get("solver_time_limit_s", 10.0))

    cost, bags, num_bags = _replay_bags_by_pos(bags_by_pos, replay_pairs, H, cfg.get("gamma", 1))
    return cost, bags, num_bags, draw_log


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def run_scene(cfg: dict, algo: str, scene_id: str, pairs: list, seed: int,
              erm_beta: float = None, model=None, dry_run: bool = False):
    if algo == "erm-mcts":
        cost, bags, _capacity, draw_log = run_scene_erm_mcts(cfg, scene_id, pairs, erm_beta, seed)
        n_dropped = 0
    elif algo == "gemini-planner":
        cost, bags, _capacity, draw_log, n_dropped = run_scene_gemini(
            cfg, scene_id, pairs, seed, model=model, dry_run=dry_run
        )
    elif algo == "rule-based-planner":
        cost, bags, _capacity, draw_log = run_scene_rule_based(cfg, scene_id, pairs, seed)
        n_dropped = 0
    else:
        raise ValueError(f"unknown algo {algo!r}")
    # "bags used" always means bags that actually ended up non-empty, computed
    # uniformly from the final true bag contents -- NOT each branch's own
    # `_capacity` value, which means different things per algo: erm-mcts's is
    # the fixed capacity handed to the env upfront (max(5, ceil(H/3)), almost
    # always > bags actually filled), while gemini/rule-based's is the
    # planner's own intended bag count (usually matches, but can diverge from
    # the real final contents if an illegal move during replay left a planned
    # bag empty). n_bags_used(bags) is correct and consistent in every case.
    num_bags = n_bags_used(bags)
    # scene_id is echoed back first so results can be attributed correctly
    # even when they arrive out of dispatch order (mp.Pool.imap_unordered).
    return scene_id, cost, num_bags, bags, draw_log, n_dropped


def _run_scene_star(args) -> tuple:
    """imap_unordered needs a single-argument callable; unpacks the same
    positional args starmap would have taken."""
    return run_scene(*args)


# ---------------------------------------------------------------------------
# Allocation printer
# ---------------------------------------------------------------------------

def print_allocation(run_idx: int, scene_id: str, algo: str, erm_beta: float,
                      cost: float, num_bags: int, violations: dict, bags,
                      draw_log: list = None) -> None:
    label = algo + (f" β={erm_beta}" if erm_beta is not None else "")
    header = f"Run {run_idx} | scene={scene_id} | {label} | cost={cost:.4f} | bags_used={num_bags}"
    print(f"\n── {header} {'─' * max(0, 76 - len(header))}")
    if draw_log:
        print("  Perceived → ground-truth item:")
        for perceived_name, true_name in draw_log:
            marker = "" if perceived_name == true_name else "  (misclassified)"
            print(f"    {perceived_name:<28} → {true_name}{marker}")
    any_used = False
    for i, bag in enumerate(bags):
        if bag.total_items == 0:
            continue
        any_used = True
        print(f"  Bag {i}  [vol={bag.volume_used}/{Bag.MAX_VOLUME}  "
              f"wt={bag.weight_used}/{Bag.MAX_WEIGHT}  items={bag.total_items}]")
        for item in bag.items:
            flags = []
            if item.crush_score >= FRAGILE_THRESHOLD:
                flags.append("fragile")
            if item.spill_risk:
                flags.append("spill_risk")
            if item.spill_vulnerable:
                flags.append("vulnerable")
            outcomes = []
            if item.is_crushed:
                outcomes.append("CRUSHED")
            if item.is_spilled:
                outcomes.append("SPILLED")
            flag_str = f"  [{', '.join(flags)}]" if flags else ""
            outcome_str = f"  ← {', '.join(outcomes)}" if outcomes else ""
            print(f"    - {item.name:<30}{flag_str}{outcome_str}")
    if not any_used:
        print("  (all bags empty)")
    print(f"  Violations: {violations}")


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def create_exp_name(args: dict) -> str:
    beta_part = f"_beta_{args['erm_beta']}" if args.get("erm_beta") is not None else ""
    return (
        args["env"] + "_benchmark-scenes_" + args["algo"]
        + "_gamma_" + str(args["gamma"]) + beta_part
        + "_" + datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
        + str(args["seed"])
    )


def main_single_algo(cfg: dict, algo: str, erm_beta, scenes: list, arg_seed: int = 0,
                      data_folder_path: str = None, dry_run: bool = False) -> str:
    if data_folder_path is None:
        data_folder_path = DATA_FOLDER_PATH

    exp_name = create_exp_name({
        "env": cfg["env"], "algo": algo, "gamma": cfg.get("gamma", 0.99),
        "erm_beta": erm_beta, "seed": arg_seed,
    })
    exp_path = data_folder_path + exp_name
    os.makedirs(exp_path, exist_ok=True)
    print("\nExperiment ID:", exp_name)
    print("Config:", cfg)
    print(f"Algo: {algo}" + (f"  beta={erm_beta}" if erm_beta is not None else ""))
    print(f"Scenes: {len(scenes)}")

    model = None
    if algo == "gemini-planner" and not dry_run:
        # Load .env BEFORE checking for the key -- simulate_gemini_planner.py
        # normally does this itself at its own module-import time, but that
        # import is deferred until run_scene_gemini (per-scene, not here), so
        # without this the key check below would always fail even with a
        # valid .env present (this was a real bug: caught when a user with a
        # working .env still got "no API key" trying to run gemini-planner).
        from dotenv import load_dotenv
        load_dotenv(pathlib.Path(__file__).parent / ".env")

        import google.generativeai as genai
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            sys.exit("no API key: set the GEMINI_API_KEY env var")
        genai.configure(api_key=key)
        model = genai.GenerativeModel(cfg["gemini_model"])

    # Accumulated as results complete. For the mp.Pool branch these arrive
    # out of dispatch order (imap_unordered) -- each result is self-describing
    # via its own scene_id (see run_scene), so index alignment across these
    # parallel lists just reflects arrival order, which is fine for every
    # downstream use (aggregate stats, or looking a scene up by its id).
    scene_ids: list = []
    costs: list = []
    num_bags_list: list = []
    violations_list: list = []
    draw_logs: list = []
    n_dropped_list: list = []

    def _save() -> None:
        # Overwrites exp_data.json after every completed scene, not just at
        # the very end -- matches simulate_gemini_planner.py's own
        # per-episode checkpointing precedent (now applied to every algo
        # here), so an interrupted or still-running multi-hour sweep never
        # loses completed work and can be inspected mid-run.
        exp_data = {
            "config": cfg,
            "algo": algo,
            "erm_beta": erm_beta,
            "scene_ids": list(scene_ids),
            "f_vals": np.array(costs),
            "n_bags_used": list(num_bags_list),
            "fragile_weight_on_top": [v["fragile_weight_on_top"] for v in violations_list],
            "spill_risk_weight_on_top": [v["spill_risk_weight_on_top"] for v in violations_list],
            "spill_vulnerable_with_spill_risk": [v["spill_vulnerable_with_spill_risk"] for v in violations_list],
            "scenes_affected": _scenes_affected_stats(violations_list),
            "draw_logs": list(draw_logs),
            "env": cfg["env"],
        }
        with open(exp_path + "/exp_data.json", "w") as f:
            json.dump(json.dumps(exp_data, cls=NumpyEncoder), f)

    def _record(i: int, scene_id: str, cost: float, num_bags: int, bags,
                draw_log: list, n_dropped: int) -> None:
        violations = count_violations(bags)
        scene_ids.append(scene_id)
        costs.append(cost)
        num_bags_list.append(num_bags)
        violations_list.append(violations)
        draw_logs.append(draw_log)
        n_dropped_list.append(n_dropped)
        print_allocation(i, scene_id, algo, erm_beta, cost, num_bags, violations, bags, draw_log)
        if n_dropped:
            print(f"  note: {n_dropped} item(s) missing from Gemini's reply were repaired")
        _save()

    if algo == "gemini-planner":
        # Sequential, rate-limit-paced -- matches simulate_gemini_planner.py's
        # own precedent (it never uses mp.Pool; its backoff/pacing design
        # assumes serial API calls).
        for t, (scene_id, pairs) in enumerate(tqdm(scenes)):
            seed = 1_000 * arg_seed + t
            result = run_scene(cfg, algo, scene_id, pairs, seed, model=model, dry_run=dry_run)
            _record(t, *result)
            if not dry_run and t < len(scenes) - 1:
                time.sleep(cfg["sleep_between_calls"])
    else:
        args_list = [
            (cfg, algo, scene_id, pairs, 1_000 * arg_seed + t, erm_beta)
            for t, (scene_id, pairs) in enumerate(scenes)
        ]
        # imap_unordered (not starmap) so results stream in as each scene
        # finishes, driving both the tqdm bar and the incremental save --
        # starmap blocks silently until EVERY scene in the batch is done,
        # which is what made a long run look stuck with zero feedback.
        # unordered (rather than imap) also means a slow straggler (scenes
        # vary a lot in cost, H ranges 3-17) never blocks reporting on
        # faster scenes that finish later in dispatch order.
        with mp.Pool(processes=cfg["num_processors"]) as pool:
            for i, result in enumerate(
                tqdm(pool.imap_unordered(_run_scene_star, args_list), total=len(args_list))
            ):
                _record(i, *result)

    f_vals = np.array(costs)
    total_violations = {k: sum(v[k] for v in violations_list) for k in VIOLATION_TYPES}
    scene_stats = _scenes_affected_stats(violations_list)
    print(f"\n=== Summary: {algo}" + (f" beta={erm_beta}" if erm_beta is not None else "") + " ===")
    print(f"  scenes processed: {len(scenes)}")
    print(f"  mean cost: {f_vals.mean():.4f}")
    print(f"  total bags used (across all scenes): {sum(num_bags_list)}")
    print(f"  total violations (across all scenes): {total_violations}"
          f"  (grand total: {sum(total_violations.values())})")
    print(f"  scenes affected by >=1 violation: {scene_stats['scenes_affected_any']}/{scene_stats['n_scenes']}"
          f" ({scene_stats['fraction_scenes_affected_any']:.1%})")
    print(f"  scenes affected by violation type: {scene_stats['scenes_affected_by_type']}")

    print(exp_name)
    return exp_name


def main(cfg: dict, arg_seed: int = 0, data_folder_path: str = None, dry_run: bool = False) -> dict:
    scenes = load_all_scenes(
        cfg["scenes_dir"], cfg["ground_truth_dir"], cfg.get("scene_limit"), cfg.get("scene_ids")
    )
    scenes = scenes[: cfg["N"]]
    print(f"Loaded {len(scenes)} scenes from {cfg['scenes_dir']}"
          f" (ground truth: {cfg['ground_truth_dir']})")

    exp_names = {}
    for algo in cfg["algos"]:
        betas = cfg["erm_betas"] if algo == "erm-mcts" else [None]
        for erm_beta in betas:
            print(f"\n{'=' * 60}\nRunning algo={algo}"
                  + (f" beta={erm_beta}" if erm_beta is not None else "") + f"\n{'=' * 60}")
            exp_names[(algo, erm_beta)] = main_single_algo(
                cfg, algo, erm_beta, scenes, arg_seed, data_folder_path, dry_run
            )

    print("\nAll experiments completed:")
    for (algo, beta), name in exp_names.items():
        print(f"  {algo}" + (f" beta={beta}" if beta is not None else "") + f": {name}")
    return exp_names


if __name__ == "__main__":
    argv = sys.argv[1:]
    dry_run = "--dry-run" in argv
    argv = [a for a in argv if a != "--dry-run"]

    if len(argv) > 1:
        main(CONFIG, int(argv[0]), argv[1], dry_run=dry_run)
    elif len(argv) == 1:
        main(CONFIG, int(argv[0]), dry_run=dry_run)
    else:
        main(CONFIG, dry_run=dry_run)
