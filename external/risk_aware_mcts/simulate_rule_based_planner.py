#!/usr/bin/env python3
"""
Like simulate_gemini_planner.py, but instead of an LLM deciding bag
allocation with zero rules, a small CP-SAT (Google OR-Tools) constraint
solver decides it subject to a fixed set of hard physical-safety rules,
then the plan is replayed through the same real envs.binpacking_env_v2.
BinPackingEnv physics for a comparable cumulative discounted cost.

This is a trimmed, from-scratch reimplementation of the rule-based bagging
solver found in the legacy
scratch/packPlan-main (1)/packPlan-main/rewardEllicitation/or_approach.py
/ or_approach_extended.py (kept only for reference -- not imported here,
since that path has spaces/parens that make it awkward to import from, and
the kept rule set differs from either legacy version anyway).

Rules KEPT (see solve_bagging_rules docstring for the exact CP-SAT
formulation):
  1. every item assigned to exactly one bag
  2. bag-usage linkage
  3. bag capacity -- unlike the legacy solver, uses THIS env's real
     Bag.MAX_VOLUME/Bag.MAX_WEIGHT (5000/5000), not the legacy file's
     unrelated 7000g/20000cc constants
  4. crush protection by arrival order (fragile item can't be loaded below
     a heavy item)
  5. orientation protection by arrival order (an orientation-sensitive/
     must-stay-upright item can't be loaded below a heavy item -- this is
     what keeps it from spilling)
  6. incompatibility separation: spill_risk vs spill_vulnerable
  7. spill protection by arrival order (a spill_risk item can't be loaded
     below a heavy item -- a heavy item pressing/jostling a spill-risk
     container is what triggers a spill in the first place)

Rules DROPPED vs the legacy solver: raw-meat separation, temperature
separation, cleaning-products vs food separation.

The planner solves the WHOLE episode's bag assignment upfront from the
observed (perceived) item features -- the same noisy features an ERM-MCTS
agent would see -- then this script drives BinPackingEnv.step() one
placement at a time, exactly like simulate_gemini_planner.py does, so the
same risk-sensitive cost model ERM-MCTS/acc-mcts/mcts/gemini-planner are all
scored against applies here too. See that file's module docstring for the
general mechanism (pre-sample items upfront + pin true identity, scripted
draw-order subclass, bag-major placement-ordered replay) -- reused here
unchanged.

Usage:
  python3 simulate_rule_based_planner.py                      # use CONFIG defaults
  python3 simulate_rule_based_planner.py <seed>                # SLURM-style
  python3 simulate_rule_based_planner.py <seed> <data_folder>  # SLURM-style
"""
from __future__ import annotations

import os
import sys
import json
import random
import pathlib
import multiprocessing as mp
from collections import deque
from datetime import datetime

import numpy as np
from ortools.sat.python import cp_model

from envs.binpacking_env_v3 import BinPackingEnv, Bag, _sample_true_item
from violation_metrics import count_violations, VIOLATION_TYPES, violations_affected_stats

DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent) + '/data/'

CONFIG = {
    "N": 500,
    "num_processors": 8,
    "env": "binpacking",
    "algo": "rule-based-planner",
    "H": 5,
    "gamma": 1,
    "solver_time_limit_s": 10.0,
}

FRAGILE_THRESHOLD = 8  # crush_score >= this -> fragile (can't go below a heavy item)
HEAVY_THRESHOLD = 2    # crush_score <= this -> heavy (can't go on top of a fragile/
                       # orientation-sensitive item)


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
    return (
        args["env"] + "_rule-based-planner_gamma_" + str(args["gamma"]) + "_"
        + datetime.today().strftime("%Y-%m-%d-%H-%M-%S") + str(args["seed"])
    )


def _item_to_planner_fields(item) -> dict:
    """Maps a binpacking_env_v2 Item to the solver's field names. Item's
    constructor params are named est_weight_g/est_volume_cc but the stored
    attributes are .weight/.volume -- that's the only naming translation
    needed, everything else below is already named the same on Item."""
    return {
        "est_weight_g": item.weight,
        "est_volume_cc": item.volume,
        "crush_score": item.crush_score,
        "category": item.category,
        "spill_risk": bool(item.spill_risk),
        "spill_vulnerable": bool(item.spill_vulnerable),
        "orientation_sensitive": bool(item.orientation_sensitive),
    }


def solve_bagging_rules(item_features: list, time_limit_s: float = 10.0) -> list:
    """
    Finds a bag assignment for `item_features` (indexed by arrival position --
    item_features[i] arrives/is loaded before item_features[k] for i < k)
    minimizing the number of bags used, subject to:

      1. every item assigned to exactly one bag (AddExactlyOne)
      2. bag-usage linkage (x[i,j] <= y[j])
      3. bag capacity: sum(weight) <= Bag.MAX_WEIGHT, sum(volume) <=
         Bag.MAX_VOLUME per used bag (this env's real capacity)
      4. crush protection by arrival order: a fragile item (crush_score >=
         FRAGILE_THRESHOLD) can't share a bag with a heavy item
         (crush_score <= HEAVY_THRESHOLD) that arrives after it
      5. orientation protection by arrival order: an orientation_sensitive
         item can't share a bag with a heavy item that arrives after it
         (this is what keeps it from tipping over and spilling)
      6. incompatibility separation (order-independent): spill_risk can't
         share a bag with spill_vulnerable
      7. spill protection by arrival order: a spill_risk item can't share a
         bag with a heavy item that arrives after it (a heavy item
         pressing/jostling a spill-risk container is what triggers a spill)

    Returns a list of bags, each a list of item indices sorted by arrival
    order (== bottom-to-top load order within that bag).
    """
    n = len(item_features)
    if n == 0:
        return []

    model = cp_model.CpModel()
    num_bags = n  # upper bound: one bag per item, always feasible

    x = {(i, j): model.NewBoolVar(f"x_{i}_{j}") for i in range(n) for j in range(num_bags)}
    y = [model.NewBoolVar(f"y_{j}") for j in range(num_bags)]

    # 1. every item in exactly one bag
    for i in range(n):
        model.AddExactlyOne(x[i, j] for j in range(num_bags))

    # 2. bag-usage linkage
    for i in range(n):
        for j in range(num_bags):
            model.Add(x[i, j] <= y[j])

    # 3. capacity (this env's real Bag constants, not the legacy solver's)
    for j in range(num_bags):
        bag_weight = sum(int(round(item_features[i]["est_weight_g"])) * x[i, j] for i in range(n))
        model.Add(bag_weight <= Bag.MAX_WEIGHT * y[j])
        bag_volume = sum(int(round(item_features[i]["est_volume_cc"])) * x[i, j] for i in range(n))
        model.Add(bag_volume <= Bag.MAX_VOLUME * y[j])

    # 4, 5 & 7. arrival-order protection (crush, orientation, spill risk)
    for j in range(num_bags):
        for i in range(n):
            for k in range(i + 1, n):
                bottom, top = item_features[i], item_features[k]
                top_is_heavy = top["crush_score"] <= HEAVY_THRESHOLD
                if bottom["crush_score"] >= FRAGILE_THRESHOLD and top_is_heavy:
                    model.AddBoolOr([x[i, j].Not(), x[k, j].Not()])
                if bottom["orientation_sensitive"] and top_is_heavy:
                    model.AddBoolOr([x[i, j].Not(), x[k, j].Not()])
                if bottom["spill_risk"] and top_is_heavy:
                    model.AddBoolOr([x[i, j].Not(), x[k, j].Not()])

    # 6. incompatibility separation (order-independent)
    for j in range(num_bags):
        for i in range(n):
            for k in range(i + 1, n):
                p1, p2 = item_features[i], item_features[k]
                spill_pair = (
                    (p1["spill_risk"] and p2["spill_vulnerable"]) or
                    (p2["spill_risk"] and p1["spill_vulnerable"])
                )
                if spill_pair:
                    model.AddBoolOr([x[i, j].Not(), x[k, j].Not()])

    model.Minimize(sum(y))

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1  # episodes are already parallelized via mp.Pool
    solver.parameters.max_time_in_seconds = time_limit_s
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # Should not happen -- one item per bag always satisfies every constraint
        # above. Fall back defensively rather than crash a whole sweep.
        return [[i] for i in range(n)]

    bags = []
    for j in range(num_bags):
        if solver.Value(y[j]) == 1:
            bags.append(sorted(i for i in range(n) if solver.Value(x[i, j]) == 1))
    return bags


class RuleBasedScriptedBinPackingEnv(BinPackingEnv):
    """Identical mechanism to simulate_gemini_planner.py's
    GeminiScriptedBinPackingEnv -- see that file's module docstring for the
    full explanation. Item draws are consumed, in order, from a pre-built
    (true_item, observed_item) queue instead of being sampled randomly."""

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


def run_episode(cfg: dict, seed: int):
    np.random.seed(seed)
    random.seed(seed)
    H = cfg["H"]

    # 1. Pre-sample H (true, observed) item pairs -- same approach as
    #    simulate_gemini_planner.py's run_gemini_episode: resolve+pin the
    #    true item now via the Bayes posterior step() would otherwise use
    #    lazily, since this is one fixed straight-line episode with no
    #    branching/revisiting to benefit from laziness.
    sampler = BinPackingEnv(num_bags=1, num_items_to_pack=H)
    pairs = []
    for _ in range(H):
        _, obs_item = sampler._draw_item(0, 0)
        pairs.append((_sample_true_item(obs_item), obs_item))

    # 2. Solve the whole episode's bag assignment upfront from the OBSERVED
    #    (perceived) item features -- the same noisy features an ERM-MCTS
    #    agent would see.
    item_features = [_item_to_planner_fields(obs) for _, obs in pairs]
    bags_by_pos = solve_bagging_rules(item_features, cfg.get("solver_time_limit_s", 10.0))

    # 3. Build a placement-ordered draw queue (bag-major, bottom-to-top
    #    within each bag) + a parallel bag-index list.
    num_bags = max(len(bags_by_pos), 1)
    queue_order = [pos for bag in bags_by_pos for pos in bag]
    queue_bag_idx = [b for b, bag in enumerate(bags_by_pos) for _ in bag]
    draw_queue = deque(pairs[p] for p in queue_order)

    # 4. Replay through the real env physics, one step per queued item.
    env = RuleBasedScriptedBinPackingEnv(
        draw_queue=draw_queue, num_bags=num_bags, num_items_to_pack=H,
        gamma=cfg.get("gamma", 1),
    )
    state = env.sample_initial_state()
    cumulative_discounted_cost = 0.0
    for next_to_place in range(H):
        slot_idx = next(s for s, pos in env._slot_queue_pos.items() if pos == next_to_place)
        action = slot_idx * num_bags + queue_bag_idx[next_to_place]
        state, cost, _ = env.step(state, action)
        cumulative_discounted_cost += cost * env.gamma ** next_to_place

    violations = count_violations(state["bags"])
    return cumulative_discounted_cost, num_bags, violations


def main(cfg: dict, arg_seed: int = 0, data_folder_path: str = None) -> str:
    if data_folder_path is None:
        data_folder_path = DATA_FOLDER_PATH

    exp_name = create_exp_name({"env": cfg["env"], "gamma": cfg["gamma"], "seed": arg_seed})
    exp_path = data_folder_path + exp_name
    os.makedirs(exp_path, exist_ok=True)
    print("\nExperiment ID:", exp_name)
    print("Config:", cfg)

    print("\nSimulating...")
    with mp.Pool(processes=cfg["num_processors"]) as pool:
        results = pool.starmap(run_episode, [(cfg, 1_000 * arg_seed + t) for t in range(cfg["N"])])
        pool.close()
        pool.join()

    f_vals = np.array([r[0] for r in results])
    n_bags_used = [r[1] for r in results]
    violations_list = [r[2] for r in results]

    exp_data = {
        "config": cfg,
        "f_vals": f_vals,
        "env": cfg["env"],
        "n_bags_used": n_bags_used,
    }
    for k in VIOLATION_TYPES:
        exp_data[k] = [v[k] for v in violations_list]
    exp_data["violations_affected"] = violations_affected_stats(violations_list)

    with open(exp_path + "/exp_data.json", "w") as f:
        json.dump(json.dumps(exp_data, cls=NumpyEncoder), f)

    stats = exp_data["violations_affected"]
    total_violations = {k: sum(v[k] for v in violations_list) for k in VIOLATION_TYPES}
    print("\n=== Violations summary ===")
    print(f"  total violations: {total_violations}  (grand total: {sum(total_violations.values())})")
    print(f"  episodes affected by >=1 violation: {stats['episodes_affected_any']}/{stats['n_episodes']}"
          f" ({stats['fraction_episodes_affected_any']:.1%})")
    print(f"  episodes affected by violation type: {stats['episodes_affected_by_type']}")

    print(exp_name)
    return exp_name


if __name__ == "__main__":
    if len(sys.argv) > 2:
        main(CONFIG, int(sys.argv[1]), sys.argv[2])
    elif len(sys.argv) == 2:
        main(CONFIG, int(sys.argv[1]))
    else:
        main(CONFIG)
