# Real-robot planning guide — `RobotPlanner`

This is a handout for using `simulate_fixed_items_interface.py`'s `RobotPlanner` to plan
grocery bin-packing actions for the real robot with ERM-MCTS. It plans **one action at a
time**: you give it the currently visible candidate items, it gives you back which item to
place in which bag. It never simulates what happens next — the robot executes the action
and you report the next real observation on the following call.

## The one hard constraint

**`RobotPlanner` assumes the robot executes the action it recommends, unless you say
otherwise.** It tracks bag contents internally by replaying its own previous recommendation —
it does not re-derive bag state from anything you report. So if the robot places a *different*
item or bag than the one recommended, and you say nothing, its bag state silently diverges
from physical reality.

**When the pick fails, call `cancel_pending()` before the next `plan_action()`.** The choice
is not placed when it is made: it is held and applied at the start of the next call, and
`cancel_pending()` takes it back — the bag is never credited, `t` does not advance, and the
search tree is dropped with it (its root was advanced assuming the action happened, so
reusing it would plan from a state that never existed). Returns `True` if there was something
to cancel. Calling it twice is a no-op. **This replaces the earlier advice to `reset()` and
restart the episode**, which threw away every correct placement to undo one bad one.

The robot side already reports what happened: `baxter_execution.py` returns `SUCCESS`,
`FAILED`, `LOST_PICKUP` (nothing in the jaws after the lift), `LOST_TRANSIT` (dropped on the
way to the bag) or `BLOCKED` (the descent was stopped by the object), and
`end_to_end_pipeline.py` calls `cancel_pending()` on anything but `SUCCESS`. A lost item is
simply seen again by the next perception pass if it landed on the table, and is absent if it
did not; either way the planner's books are correct.

`GeminiPlanner` does the same through `plan_action(..., previous_action_failed=True)` on the
following call.

## Item schema

Every candidate item you report is a dict with these fields (matches the VLM perception
pipeline's own output schema, `envs/SCHEMA.md`):

```python
{
    "name": "Whole Milk (1)",       # optional, for your own logging/debugging only
    "est_weight_g": 1000,           # int, grams
    "est_volume_cc": 950,           # int, cubic centimeters
    "crush_score": 4,               # int 1-10: how much load may rest ON TOP of this item.
                                     #   HIGH (>=8) = put nothing on it (produce bruises, open
                                     #   drinkware breaks); LOW (<=2) = stack anything on it
                                     #   (cans, bottles). This is LOAD TOLERANCE, not
                                     #   breakability: a sealed glass bottle stacks fine.
    "category": "Dairy",            # one of: Dairy, Bakery, Produce, Snacks, Pantry, Frozen,
                                     #   Raw Meat, Cleaning, Household, Hardware
    "temperature": "Refrigerated",  # one of: Ambient, Refrigerated, Frozen
    "spill_risk": True,             # bool — will contents escape if the bag is tipped or the
                                     #   item is laid on its side? Mechanical, not "contains
                                     #   liquid": a sealed can of soup is False.
    "spill_vulnerable": False,      # bool — is this item damaged by moisture?
}

# `orientation_sensitive` was removed from the schema: nothing populated it, so it
# arrived as None on every item and became False. `Item.__init__` still accepts the
# argument and the adapter passes False explicitly.
```

`category`/`temperature` must be one of the listed values exactly (case-sensitive) or the
call raises `ValueError`. `name` can be anything — it does not need to match any known
catalog; unrecognized names are handled automatically (see Known limitations below).

Bag capacity is currently modeled as `MAX_VOLUME = MAX_WEIGHT = 5000` (same units as
`est_weight_g`/`est_volume_cc` above — so 5000g / 5000cc per bag by default). If your real
bags have different capacity, this is a constant in `envs/binpacking_env_v2.py`
(`Bag.MAX_VOLUME`, `Bag.MAX_WEIGHT`) — let us know if it needs to be configurable instead of
hardcoded.

## Usage — long-running process (recommended if your robot control code is one long-lived Python process)

```python
from simulate_fixed_items_interface import RobotPlanner

planner = RobotPlanner(
    num_bags=5,               # how many bags are available
    num_items_to_pack=10,     # total items expected this episode
    erm_beta=1.0,             # risk sensitivity: >0 risk-averse, ~0 risk-neutral, <0 risk-seeking
    n_iter_per_timestep=1000, # MCTS iterations per decision -- higher = better action, slower
    n_visible_items=5,        # how many candidate items you can report at once
)

# Start of episode: nothing to do, RobotPlanner starts fresh by default.

while not episode_done:
    candidates = get_visible_items_from_perception()  # your code: list of item dicts, see schema above
    result = planner.plan_action(candidates)

    if result["done"]:
        break  # episode complete, or no valid candidates left

    print(result)
    # {
    #   "action": 3, "item_slot_index": 1, "bag_index": 3,
    #   "chosen_item": {...the exact dict you passed in for that item...},
    #   "tree_reused": False, "t": 2, "items_remaining": 7, "done": False,
    # }

    execute_on_robot(item=result["chosen_item"], bag_index=result["bag_index"])
    # Physically place result["chosen_item"] into bag result["bag_index"].
    # Do NOT report this item again on the next call -- the planner already
    # accounts for it internally on the following plan_action() call.

# Starting a new episode (new bagging job): 
planner.reset()
```

`result["chosen_item"]` is exactly the dict you passed in for that candidate — use it to
identify which physical item to grab.

## Usage — fresh process per decision (CLI mode)

If instead each decision is a separate process invocation (e.g. called from a non-Python
robot stack via subprocess), use the CLI. Pass `--planner-state-file` to persist bag state
and the MCTS tree across invocations (otherwise every call starts a brand-new episode):

```bash
# First call of a new episode:
echo '{"observed_candidates": [ {...item1...}, {...item2...} ]}' | \
python simulate_fixed_items_interface.py \
    --num-bags 5 --num-items-to-pack 10 --n-visible-items 5 \
    --erm-beta 1.0 --n-iter-per-timestep 1000 \
    --planner-state-file /tmp/planner_state.pkl --new-episode

# Subsequent calls (same episode, state file already exists):
echo '{"observed_candidates": [ {...new candidates...} ]}' | \
python simulate_fixed_items_interface.py \
    --planner-state-file /tmp/planner_state.pkl
```

Output is JSON printed to stdout (same fields as `plan_action`'s return dict above); use
`--output-file PATH` to write it to a file instead. You can also pass `--state-file PATH`
instead of piping JSON via stdin. Delete `/tmp/planner_state.pkl` (or pass `--new-episode`
again) to start a fresh episode.

## Tuning

- **`erm_beta`**: risk sensitivity. Positive values are risk-averse (avoids high-variance
  outcomes like crush/spill even at some average-cost expense); values near `1e-6` are close
  to risk-neutral (plans for average cost); negative values are risk-seeking. Try a few values
  to see how much it changes bag routing.
- **`n_iter_per_timestep`**: planning budget per decision. Higher = better/more consistent
  recommendations, slower. Defaults to `DEFAULT_N_ITER_PER_TIMESTEP` in
  `simulate_fixed_items_interface.py` (currently 500) for every caller that does not pass it
  explicitly; override for one run without editing code with `PLANNER_N_ITER=2000 python3 ...`.
- **`gamma`**: discount factor (default `0.99`), rarely needs changing.

## Known limitations

- **No hard packing-capacity safety filter.** The planner learns to avoid overfilling bags
  through a soft cost penalty during search, but nothing hard-blocks it from recommending a
  placement that would overflow a bag's capacity. If this matters for your setup, flag it —
  a real-world safety check can be added, it just isn't currently.
- **Unrecognized item names are handled gracefully but with a caveat.** If a reported item's
  `name` doesn't match one of our ~17-item internal catalog, the planner's internal lookahead
  (used only to reason about hypothetical future items during search, not the item you
  actually report) assumes that item's *true* physical properties equal exactly what you
  reported (no simulated perception noise for that item). This only affects internal planning
  quality for hypothetical future items, not the current decision's correctness.

## Reference

A working example exercising both usage modes end-to-end (with fixture data, no real robot
needed) is `scratch/verify_robot_interface.py` — useful as a template to copy from.
