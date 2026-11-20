# Risk-aware MCTS — grocery bin-packing simulations (minimal bundle)

Risk-averse planning for packing groceries into bags under perception noise.
The main algorithm is **ERM-MCTS** (`algos/erm_mcts.py`). It minimizes the entropic
risk measure ERM_beta(C) = (1/beta) * log E[exp(beta * C)] of the episode cost.
beta > 0 is risk-averse, and beta near 0 (e.g. 1e-6) is close to risk-neutral.

Environment: `envs/binpacking_env_v3.py` (32-item catalog, perception-confusion
model, costs: illegal move 100, crush 50, spill 40, contamination 10, bag opening 10).

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Scripts

Every script has a `CONFIG` dict at the top. Edit it, then run the script.
Results go to `data/<experiment_name>/exp_data.json`.

| Script | What it does |
|---|---|
| `simulate_mcts_multiple_betas.py` | Main experiment: random item streams, sweeping `erm_betas` for each algo in `algos` (`erm-mcts`, `acc-mcts` = MCTS on accrued-cost augmented state, `mcts` = risk-neutral). |
| `simulate_benchmark_scenes.py` | Runs planners on real benchmark scenes (`benchmark/`: perceived items plus ground truth). `algos` can include `erm-mcts`, `rule-based-planner`, `gemini-planner`. |
| `simulate_rule_based_planner.py` | Baseline: an OR-Tools CP-SAT solver with hard safety rules. |
| `simulate_gemini_planner.py` | Baseline: Gemini bags the whole episode from item names only. |
| `merge_exps.py` | Merges per-task `exp_data.json` files after a SLURM array run. |

Run them like this:

```bash
python3 simulate_mcts_multiple_betas.py                  # writes to ./data/
python3 simulate_mcts_multiple_betas.py <seed> <out_dir>  # e.g. SLURM: $SLURM_ARRAY_TASK_ID
python3 simulate_benchmark_scenes.py --dry-run            # gemini-planner uses a fake bagger (no API calls)
```

The Gemini planner needs `GEMINI_API_KEY`, either as an environment variable or in a `.env`
file next to the scripts. Use `--dry-run` to test without it.

Each saved `exp_data.json` contains the config, `f_vals` (the discounted cost per episode)
and safety-violation counts (`violation_metrics.py`).

The planning budget (`n_iter_per_timestep`) matters a lot. Budgets of 500 or less give
inconsistent allocations; 5000 or more is recommended.
