"""
Real-robot ERM-MCTS planning interface for the grocery bin-packing environment.

Each real timestep, an external caller (the robot's VLM perception pipeline)
reports the currently visible CANDIDATE items (not yet placed). This module
plans exactly ONE action and returns it. It never calls env.step() to
fabricate what happens next, since the real robot supplies that by physically
executing the action and reporting the next observation itself.

Bag state (bags/observed_bags) is tracked internally by RobotPlanner: each
call first replays the action chosen on the PREVIOUS call into the tracked
bags via Bag.place_observed_item (envs/binpacking_env_v2.py) — the exact same
accumulation logic BinPackingEnv.step() uses for its observed-bag update. The
caller never needs to re-report already-placed items, and placement order
(which matters for the order-dependent risk-accrual features) is exactly
known rather than inferred from reported order.

There is no true-vs-observed duality in the real world: the VLM pipeline only
ever produces one feature estimate per item. So candidate items' true_items
entries are always left None, letting env.step()'s _sample_true_item()
posterior-sampling supply plausible physics only inside MCTS's own internal
simulated rollouts/expansion — never touching the real reported state.

Two call patterns are supported on top of the same RobotPlanner.plan_action:
  - Long-running process: construct one RobotPlanner, call plan_action(...)
    repeatedly within the same Python process. Enables in-memory ERMMCTS tree
    reuse across timesteps via update_root_node.
  - Fresh process per decision: see _cli_main / --planner-state-file below,
    which persists the full RobotPlanner (bags, t, and the MCTS tree itself)
    to a pickle file so tree reuse survives across process restarts too.

"""

import argparse
import json
import os
import pickle
import random
import sys

import numpy as np

from algos.erm_mcts import ERMMCTS
from envs.binpacking_env_v2 import (
    BinPackingEnv, Bag, Item, PLACEHOLDER_ITEM, CATEGORY_MAP, TEMPERATURE_MAP,
)


# MCTS budget per decision -- the one place to change it. Higher = better,
# slower. Override per run with PLANNER_N_ITER=2000, or per planner by passing
# n_iter_per_timestep=...
DEFAULT_N_ITER_PER_TIMESTEP = int(os.environ.get("PLANNER_N_ITER", 500))


_REQUIRED_FIELDS = (
    "est_weight_g", "est_volume_cc", "crush_score", "category",
    "temperature", "spill_risk", "spill_vulnerable",
)


def _to_item(obj: dict, item_id: int) -> Item:
    """Translate one SCHEMA.md-style observed-object dict into an Item."""
    missing = [f for f in _REQUIRED_FIELDS if f not in obj]
    if missing:
        raise ValueError(f"observed object {obj.get('name', '?')!r} missing fields: {missing}")
    if obj["category"] not in CATEGORY_MAP:
        raise ValueError(f"unknown category {obj['category']!r}; expected one of {sorted(CATEGORY_MAP)}")
    if obj["temperature"] not in TEMPERATURE_MAP:
        raise ValueError(f"unknown temperature {obj['temperature']!r}; expected one of {sorted(TEMPERATURE_MAP)}")
    return Item(
        id=item_id, name=obj.get("name", f"item_{item_id}"),
        est_weight_g=obj["est_weight_g"], est_volume_cc=obj["est_volume_cc"],
        crush_score=obj["crush_score"], category=obj["category"], temperature=obj["temperature"],
        spill_risk=bool(obj["spill_risk"]), spill_vulnerable=bool(obj["spill_vulnerable"]),
        # Jacopo's Item still takes this argument; nothing in his cost function
        # reads it and nothing on our side sets it, so it is passed as False
        # rather than pretending perception produces it.
        orientation_sensitive=False,
    )


class RobotPlanner:
    """Plans one ERM-MCTS bin-packing action at a time for a real robot.

    Bag state (bags/observed_bags) is tracked internally: each call first
    replays the action chosen on the PREVIOUS call into the tracked bags via
    Bag.place_observed_item (mirrors BinPackingEnv.step()'s own bag-update
    logic), then plans the next action from the caller-supplied list of newly
    visible candidate items. The caller never needs to re-report already-
    placed items.

    If the previous action's grasp physically failed, pass
    previous_action_failed=True instead: the pending placement is discarded
    rather than applied, so t/bags/observed_bags stay unchanged and the same
    timestep is effectively replanned from scratch.
    """

    def __init__(self, num_bags=5, num_items_to_pack=5, erm_beta=1.0,
                 K_ucb=np.sqrt(2), n_iter_per_timestep=None, gamma=0.99,
                 n_visible_items=5, env_kwargs=None):
        self.env = BinPackingEnv(num_bags=num_bags, num_items_to_pack=num_items_to_pack,
                                  gamma=gamma, n_visible_items=n_visible_items,
                                  **(env_kwargs or {}))
        self.num_bags = num_bags
        self.H = num_items_to_pack
        self.erm_beta = erm_beta
        self.K_ucb = K_ucb
        self.n_iter_per_timestep = (DEFAULT_N_ITER_PER_TIMESTEP
                                    if n_iter_per_timestep is None else n_iter_per_timestep)
        self.reset()

    def reset(self):
        """Start a new bagging job/episode: empty bags, t=0, no pending placement, no tree."""
        self.bags = tuple(Bag() for _ in range(self.num_bags))
        self.observed_bags = tuple(Bag() for _ in range(self.num_bags))
        self.t = 0
        self._pending = None       # (item, bag_idx) chosen last call, applied at start of next call
        self.mcts = None
        self._last_action = None

    def cancel_pending(self):
        """The last chosen item never reached its bag -- forget the placement.

        `plan_action` does not place its choice immediately; it holds it and
        applies it at the start of the next call, once the arm has had its
        turn. If the arm reports the item was dropped, never gripped, or that
        the descent was blocked, that placement must not happen: the bag is
        emptier than the planner thinks, and the item is still out there (back
        on the table, or gone).

        The search tree is dropped with it. Its root was advanced on the
        assumption that the action was taken, and reusing it would plan from a
        state that never happened.
        """
        if self._pending is None:
            return False
        self._pending = None
        self.mcts = None
        self._last_action = None
        return True

    def _apply_pending_placement(self):
        if self._pending is None:
            return
        item, bag_idx = self._pending
        self.bags[bag_idx].place_observed_item(item)
        self.observed_bags[bag_idx].place_observed_item(item)
        self.t += 1
        self._pending = None

    def plan_action(self, observed_candidates: list, previous_action_failed: bool = False) -> dict:
        """observed_candidates: flat list of SCHEMA.md-style feature dicts for the
        CURRENTLY VISIBLE (not yet placed) candidate items only.

        previous_action_failed: set True when the action returned by the LAST
        plan_action call physically failed (e.g. a dropped/missed grasp) --
        instead of applying it, it is discarded so t/bags/observed_bags stay
        exactly as they were, and this call effectively replans the same
        timestep from scratch given fresh candidates."""
        if previous_action_failed:
            self._pending = None  # grasp failed -- nothing was actually placed, forget it
        else:
            self._apply_pending_placement()

        # items_remaining means "items not yet revealed" (matches BinPackingEnv's own
        # sample_initial_state(): H - n_detected) -- NOT "placements not yet made".
        # Every item revealed so far is either already placed (self.t) or still sitting
        # in the currently-visible window (len(observed_candidates)); subtracting both
        # from H is what env.step()'s internal resampling (inside ERMMCTS's rollouts)
        # expects, so it stops inventing new hypothetical items once the real total is
        # accounted for -- e.g. if all H items are already visible in the first frame,
        # this is 0, not H, even though self.t is still 0.
        items_remaining = max(self.H - self.t - len(observed_candidates), 0)
        n_visible = self.env.n_visible_items
        if len(observed_candidates) > n_visible:
            raise ValueError(
                f"{len(observed_candidates)} candidate items reported but only "
                f"n_visible_items={n_visible} planner slots available."
            )

        visible_items, true_items, slot_map = [], [], []
        for i, obj in enumerate(observed_candidates):
            visible_items.append(_to_item(obj, item_id=i + 1))
            true_items.append(None)              # unresolved -> env.step() samples posterior
            slot_map.append(obj)                 # only inside MCTS's internal rollouts
        while len(visible_items) < n_visible:
            visible_items.append(PLACEHOLDER_ITEM)
            true_items.append(PLACEHOLDER_ITEM)  # matches sample_initial_state()'s convention
            slot_map.append(None)
        visible_items, true_items = tuple(visible_items), tuple(true_items)

        state = {
            "state": self.env._get_rl_vector(self.observed_bags, visible_items),
            "bags": self.bags, "observed_bags": self.observed_bags,
            "visible_items": visible_items, "true_items": true_items,
            "items_remaining": items_remaining, "t": self.t,
        }

        if not self.env.available_actions(state):
            return {"action": None, "done": True, "t": self.t, "items_remaining": items_remaining,
                    "reason": "no candidate items available / episode complete"}

        reused = False
        if self.mcts is not None and self._last_action is not None:
            reused = self.mcts.update_root_node(self._last_action, state)
            if reused:
                self.mcts.set_root_depth(self.t)
        if not reused:
            self.mcts = ERMMCTS(initial_state=state, env=self.env, K_ucb=self.K_ucb,
                                 erm_beta=self.erm_beta, rollout_policy=None, root_depth=self.t)

        self.mcts.learn(n_iters=self.n_iter_per_timestep)
        action = self.mcts.best_action()
        item_idx, bag_idx = action // self.num_bags, action % self.num_bags
        chosen_item = visible_items[item_idx]

        self._last_action = action
        self._pending = (chosen_item, bag_idx)  # applied at the start of the NEXT call

        return {
            "action": int(action), "item_slot_index": item_idx, "bag_index": bag_idx,
            "chosen_item": slot_map[item_idx], "tree_reused": reused,
            "t": self.t, "items_remaining": items_remaining, "done": False,
        }

    def save(self, path):
        """Persist full planner state (bags, t, MCTS tree) via pickle, so the
        fresh-process CLI mode can reuse the tree/bag-state across invocations."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)


def _cli_main():
    parser = argparse.ArgumentParser(description="Fresh-process-per-decision ERM-MCTS robot planner.")
    parser.add_argument("--state-file", type=str, default=None,
                         help="JSON file with {'observed_candidates': [...]}; if omitted, read JSON from stdin")
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--planner-state-file", type=str, default=None,
                         help="Optional path to persist RobotPlanner state (bags, t, MCTS tree) "
                              "across fresh-process invocations. Loaded if present, saved after planning.")
    parser.add_argument("--new-episode", action="store_true",
                         help="Ignore/overwrite any existing --planner-state-file and start fresh.")
    parser.add_argument("--num-bags", type=int, default=5)
    parser.add_argument("--num-items-to-pack", type=int, default=5)
    parser.add_argument("--n-visible-items", type=int, default=5)
    parser.add_argument("--erm-beta", type=float, default=1.0)
    parser.add_argument("--k-ucb", type=float, default=np.sqrt(2))
    parser.add_argument("--n-iter-per-timestep", type=int, default=DEFAULT_N_ITER_PER_TIMESTEP)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    if args.planner_state_file and os.path.exists(args.planner_state_file) and not args.new_episode:
        planner = RobotPlanner.load(args.planner_state_file)
    else:
        planner = RobotPlanner(
            num_bags=args.num_bags, num_items_to_pack=args.num_items_to_pack,
            erm_beta=args.erm_beta, K_ucb=args.k_ucb,
            n_iter_per_timestep=args.n_iter_per_timestep, gamma=args.gamma,
            n_visible_items=args.n_visible_items,
        )

    payload = json.load(open(args.state_file)) if args.state_file else json.load(sys.stdin)
    result = planner.plan_action(
        payload["observed_candidates"],
        previous_action_failed=payload.get("previous_action_failed", False),
    )

    if args.planner_state_file:
        planner.save(args.planner_state_file)

    out = json.dumps(result, indent=2, default=str)
    if args.output_file:
        with open(args.output_file, "w") as f:
            f.write(out)
    else:
        print(out)


if __name__ == "__main__":
    _cli_main()
