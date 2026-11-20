#!/usr/bin/env python3
"""
Like simulate_mcts_multiple_betas.py, but instead of ERM-MCTS/acc-mcts choosing
actions via search, each episode's full item->bag allocation is decided in a
single call to Gemini-planning/gemini_planner_run.py's no-rules bagger, then
replayed through the real envs.binpacking_env_v2.BinPackingEnv physics
(crush/spill/packing-reward) to get a comparable cumulative discounted cost.

Gemini plans the WHOLE episode upfront from the observed (perceived) item names
-- the same noisy names an ERM-MCTS agent would see -- with no capacity limits,
no attributes, and no safety rules given to it (see gemini_planner_run.py's
PACK_PROMPT). This script then drives BinPackingEnv.step() one placement at a
time so the same risk-sensitive cost model ERM-MCTS/acc-mcts are scored against
applies here too.

v2 (unlike v1) resolves the hidden true item lazily: its own _draw_item only
returns an observed item, leaving the true item as None until step() resolves
it -- either from a pinned value (if the caller already fixed one) or, absent
that, freshly from the Bayes-inverted posterior P(true|observed) via
_sample_true_item. Since this script replays one fixed straight-line episode
(no MCTS branching/revisits), there's no benefit to leaving resolution lazy --
run_gemini_episode resolves+pins each item's true identity once, upfront,
right after drawing its observation, via the same _sample_true_item helper.

BinPackingEnv reveals items through a stochastic sliding visible-item window
(n_visible_items/p_double_draw/p_zero_draw), not a fixed upfront list, and its
action is item_idx*num_bags+bag_idx where item_idx is a WINDOW SLOT, not a
fixed item id. GeminiScriptedBinPackingEnv below closes that gap by controlling
the DRAW ORDER: the draw queue is built by walking Gemini's bags in order and
concatenating each bag's items bottom-to-top, so draw order == placement order
-- whichever item the env reveals next is always exactly the next one Gemini's
plan calls for. Combined with BinPackingEnv's own anti-deadlock guarantee (at
least one real item is always visible while items remain), this makes the
replay always legal without ever needing search/backtracking.

Usage:
  export GEMINI_API_KEY=...
  python3 simulate_gemini_planner.py                       # use CONFIG defaults
  python3 simulate_gemini_planner.py <seed> <data_folder>   # SLURM-style
  python3 simulate_gemini_planner.py --dry-run              # skip live Gemini calls,
                                                              # use a round-robin stub bagger
  python3 simulate_gemini_planner.py --resume=<exp_name>    # continue an interrupted run
                                                              # (e.g. after Ctrl+C on quota
                                                              # exhaustion) -- picks up at
                                                              # episode len(f_vals), reusing
                                                              # that run's own config/seed
                                                              # from its exp_data.json
"""
from __future__ import annotations

import os
import sys
import json
import random
import time
import pathlib
import re
from collections import deque
from datetime import datetime

import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

from envs.binpacking_env_v3 import BinPackingEnv, Bag, _sample_true_item
from violation_metrics import count_violations, VIOLATION_TYPES, violations_affected_stats

DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent) + '/data/'

# Gemini-planning/ has a hyphen, so it isn't an importable package -- add it to
# sys.path directly and import the two pieces we reuse verbatim.
sys.path.insert(0, str(pathlib.Path(__file__).parent / "Gemini-planning"))
from gemini_planner_run import PACK_PROMPT, parse_bags  # noqa: E402

# Reads GEMINI_API_KEY (and anything else) from a .env file next to this
# script, if present, without overriding a value already set in the shell.
load_dotenv(pathlib.Path(__file__).parent / ".env")

CONFIG = {
    "N": 500,
    "env": "binpacking",
    "algo": "gemini-planner",
    "H": 5,
    "gamma": 1,
    "gemini_model": "gemini-robotics-er-2-preview",
    "sleep_between_calls": 4.0,
    "max_retries": 6,
}


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
        args["env"] + "_gemini-planner_gamma_" + str(args["gamma"]) + "_"
        + datetime.today().strftime("%Y-%m-%d-%H-%M-%S") + str(args["seed"])
    )


class GeminiScriptedBinPackingEnv(BinPackingEnv):
    """BinPackingEnv whose item draws are consumed, in order, from a
    pre-built (true_item, observed_item) queue instead of being sampled
    randomly. Both items in each pair are already resolved/pinned (see
    run_gemini_episode), so step()'s pinned_true branch uses them directly
    instead of falling back to v2's lazy Bayes-posterior resampling. Draw
    order == placement order (see run_gemini_episode), so _draw_item can
    ignore slot_idx/t entirely; it just pops the queue and records which slot
    each queue position landed in, so the driver can look up "which slot
    currently holds the item I need to place next"."""

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


QUOTA_WAIT_NO_HINT = 300.0  # fallback sleep when the API gives no retry_delay hint -- usually
                             # means a daily/credits quota wall rather than a per-minute rate
                             # limit, so there's nothing better to do than keep waiting for it
                             # to reset (or for you to add credits) and try again later.
QUOTA_WAIT_CAP = 600.0      # never sleep longer than this in one call, so progress prints
                             # (and Ctrl+C) stay responsive even if the API reports a huge delay.


def call_gemini_with_backoff(model, prompt: str, max_retries: int) -> str:
    """Retries quota/rate-limit errors (429) forever -- that's just waiting, and the
    process should sit there until credits/quota come back rather than give up and lose
    the run. Any other error (bad request, auth, etc.) still fails fast after
    `max_retries`, since no amount of waiting fixes those."""
    last_err = None
    quota_attempt = 0
    other_attempt = 0
    while True:
        try:
            return model.generate_content(prompt).text
        except Exception as e:  # noqa: BLE001
            last_err = e
            msg = str(e)
            if "429" in msg or "quota" in msg.lower():
                quota_attempt += 1
                wait_match = re.search(r"retry in ([\d.]+)s", msg)
                wait = min(float(wait_match.group(1)) + 2, QUOTA_WAIT_CAP) if wait_match else QUOTA_WAIT_NO_HINT
                print(f"  quota/rate-limited (attempt {quota_attempt}), waiting {wait:.0f}s "
                      f"-- Ctrl+C to abort, already-completed episodes are saved", flush=True)
                time.sleep(wait)
                continue
            other_attempt += 1
            if other_attempt >= max_retries:
                raise RuntimeError(f"Gemini call failed after {other_attempt} non-quota attempts: {last_err}")
            time.sleep(2 * other_attempt)


def _dry_run_bagger(names: list) -> str:
    """Stub bagger for --dry-run: round-robins items into 2 bags, no API call."""
    bags: list = [[], []]
    for i in range(len(names)):
        bags[i % 2].append(i + 1)
    return json.dumps({"bags": bags})


def run_gemini_episode(cfg: dict, seed: int, model=None, dry_run: bool = False):
    np.random.seed(seed)
    random.seed(seed)
    H = cfg["H"]

    # 1. Pre-sample H (true, observed) item pairs, independent of window mechanics.
    #    v2's own _draw_item leaves the true item unresolved (None) for lazy,
    #    per-rollout resampling -- irrelevant here since there's no branching/
    #    revisiting, so resolve+pin the true item now via the same Bayes
    #    posterior step() would otherwise use, and carry the pinned pair
    #    through Gemini's plan and the replay below.
    sampler = BinPackingEnv(num_bags=1, num_items_to_pack=H)
    pairs = []
    for _ in range(H):
        _, obs_item = sampler._draw_item(0, 0)
        pairs.append((_sample_true_item(obs_item), obs_item))
    names = [obs.name for _, obs in pairs]
    draw_log = [(obs.name, true.name) for true, obs in pairs]

    # 2. Ask Gemini (or the dry-run stub) to bag the OBSERVED names -- the same
    #    noisy names an ERM-MCTS agent would see, per BinPackingEnv's perception model.
    if dry_run:
        reply_text = _dry_run_bagger(names)
    else:
        listing = "\n".join(f"{i + 1}. {nm}" for i, nm in enumerate(names))
        reply_text = call_gemini_with_backoff(
            model, PACK_PROMPT.format(items=listing), cfg["max_retries"]
        )
    bags_by_pos, n_dropped = parse_bags(reply_text, H)

    # 3. Build a placement-ordered draw queue (bag-major, bottom-to-top within
    #    each bag) + a parallel bag-index list, and size num_bags to fit Gemini's answer.
    num_bags = max(len(bags_by_pos), 1)
    queue_order = [pos for bag in bags_by_pos for pos in bag]
    queue_bag_idx = [b for b, bag in enumerate(bags_by_pos) for _ in bag]
    draw_queue = deque(pairs[p] for p in queue_order)

    # 4. Replay through the real env physics, one step per queued item.
    env = GeminiScriptedBinPackingEnv(
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
    return cumulative_discounted_cost, num_bags, state["bags"], names, n_dropped, draw_log, violations


def print_allocation(run_idx: int, cost: float, num_bags: int, bags, draw_log: list = None,
                      violations: dict = None) -> None:
    """Print the final bag assignment for one run (mirrors simulate_fixed_items.py's format)."""
    header = f"Run {run_idx} | cost={cost:.4f} | bags_used={num_bags}"
    print(f"\n── {header} {'─' * max(0, 56 - len(header))}")
    if draw_log:
        print("  Perceived → sampled true item:")
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
            if item.crush_score >= 8:
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
            print(f"    - {item.name:<30} (id={item.id:2d}){flag_str}{outcome_str}")
    if not any_used:
        print("  (all bags empty)")
    if violations is not None:
        print(f"  Violations: {violations}")


def main(cfg: dict, arg_seed: int = 0, data_folder_path: str = None,
         dry_run: bool = False, resume_exp_name: str = None) -> str:
    if data_folder_path is None:
        data_folder_path = DATA_FOLDER_PATH

    f_vals, n_bags_list, violations_list = [], [], []
    start_t = 0

    if resume_exp_name:
        # Reuse the ORIGINAL run's exp_name (no new timestamp), config, and arg_seed --
        # seeds are per-episode independent (seed = 1000*arg_seed + t), so as long as
        # arg_seed matches, resuming at t=len(f_vals) reproduces exactly the episodes
        # that would've run next, without needing to replay anything already done.
        exp_name = resume_exp_name
        exp_path = data_folder_path + exp_name
        data_file = exp_path + "/exp_data.json"
        if not os.path.exists(data_file):
            sys.exit(f"--resume: no exp_data.json found at {data_file}")
        with open(data_file) as f:
            saved = json.loads(json.load(f))
        if "arg_seed" not in saved:
            sys.exit("--resume: this exp folder predates checkpointed arg_seed -- can't "
                      "safely resume it (re-run from scratch instead)")
        cfg = saved["config"]
        arg_seed = saved["arg_seed"]
        f_vals = list(saved["f_vals"])
        n_bags_list = list(saved.get("n_bags_used", []))
        start_t = len(f_vals)
        if all(k in saved for k in VIOLATION_TYPES):
            violations_list = [
                {k: saved[k][i] for k in VIOLATION_TYPES} for i in range(start_t)
            ]
        else:
            # Pre-existing run from before violation tracking was added -- can't
            # retroactively know those episodes' violations, so backfill zeros
            # rather than crash on the missing keys (only affects the aggregate
            # summary/saved lists for episodes already done before this resume).
            print(f"  note: resuming a run saved before violation tracking existed -- "
                  f"backfilling {start_t} historical episode(s) with zero violations")
            violations_list = [{k: 0 for k in VIOLATION_TYPES} for _ in range(start_t)]
        print(f"\nResuming {exp_name}: {start_t}/{cfg['N']} episodes already done "
              f"(arg_seed={arg_seed})")
        if start_t >= cfg["N"]:
            print("Nothing left to do.")
            return exp_name
    else:
        exp_name = create_exp_name({"env": cfg["env"], "gamma": cfg["gamma"], "seed": arg_seed})
        exp_path = data_folder_path + exp_name
        os.makedirs(exp_path, exist_ok=True)

    print("\nExperiment ID:", exp_name)
    print("Config:", cfg)

    model = None
    if not dry_run:
        import google.generativeai as genai
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            sys.exit("no API key: set the GEMINI_API_KEY env var")
        genai.configure(api_key=key)
        model = genai.GenerativeModel(cfg["gemini_model"])

    def _save():
        # Overwrites exp_data.json after every episode (not just at the end), so a run
        # that's interrupted mid-way -- whether by Ctrl+C or by exhausting quota for good
        # -- never loses episodes already completed, and can be picked up via --resume.
        exp_data = {
            "config": cfg,
            "f_vals": np.array(f_vals),
            "env": cfg["env"],
            "n_bags_used": n_bags_list,
            "arg_seed": arg_seed,
        }
        for k in VIOLATION_TYPES:
            exp_data[k] = [v[k] for v in violations_list]
        exp_data["violations_affected"] = violations_affected_stats(violations_list)
        with open(exp_path + "/exp_data.json", "w") as f:
            json.dump(json.dumps(exp_data, cls=NumpyEncoder), f)

    for t in tqdm(range(start_t, cfg["N"]), initial=start_t, total=cfg["N"]):
        seed = 1_000 * arg_seed + t
        cost, num_bags, bags, names, n_dropped, draw_log, violations = run_gemini_episode(
            cfg, seed, model=model, dry_run=dry_run
        )
        print_allocation(t, cost, num_bags, bags, draw_log, violations)
        if n_dropped:
            print(f"  note: {n_dropped} item(s) missing from Gemini's reply were repaired")
        f_vals.append(cost)
        n_bags_list.append(num_bags)
        violations_list.append(violations)
        _save()
        if not dry_run and t < cfg["N"] - 1:
            time.sleep(cfg["sleep_between_calls"])

    if violations_list:
        stats = violations_affected_stats(violations_list)
        total_violations = {k: sum(v[k] for v in violations_list) for k in VIOLATION_TYPES}
        print("\n=== Violations summary ===")
        print(f"  total violations: {total_violations}  (grand total: {sum(total_violations.values())})")
        print(f"  episodes affected by >=1 violation: {stats['episodes_affected_any']}/{stats['n_episodes']}"
              f" ({stats['fraction_episodes_affected_any']:.1%})")
        print(f"  episodes affected by violation type: {stats['episodes_affected_by_type']}")

    print(exp_name)
    return exp_name


if __name__ == "__main__":
    argv = sys.argv[1:]
    dry_run = "--dry-run" in argv
    argv = [a for a in argv if a != "--dry-run"]
    resume_exp_name = next((a.split("=", 1)[1] for a in argv if a.startswith("--resume=")), None)
    argv = [a for a in argv if not a.startswith("--resume=")]

    if resume_exp_name:
        data_folder = argv[0] if argv else None
        main(CONFIG, data_folder_path=data_folder, dry_run=dry_run, resume_exp_name=resume_exp_name)
    elif len(argv) > 1:
        main(CONFIG, int(argv[0]), argv[1], dry_run=dry_run)
    elif len(argv) == 1:
        main(CONFIG, int(argv[0]), dry_run=dry_run)
    else:
        main(CONFIG, dry_run=dry_run)
