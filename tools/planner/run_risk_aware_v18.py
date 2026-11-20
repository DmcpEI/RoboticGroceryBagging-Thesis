#!/usr/bin/env python3
"""Run Jacopo Silvestrin's planner experiments (external/risk_aware_mcts, kept
unchanged) with the perception-noise table of the deployed v18 detector.

His environment hardcodes the noise table of the v15b detector. This module
loads tools/planner/binpacking_env_v3_v18.py -- his environment with only that
table replaced (tools/eval/build_planner_noise_env.py) -- under his module
name, then calls his script's main() with the settings of the paper's
captions. The swap sits at module level so that spawned worker processes,
which re-import this file, get it too.

  Dissertation Table 5.2 (simulation, 500 episodes, H=5, n0 ~ Poisson(5), 5000 iterations):
    .venv/bin/python3.12 tools/planner/run_risk_aware_v18.py sim-mcts
    .venv/bin/python3.12 tools/planner/run_risk_aware_v18.py sim-rules
    .venv/bin/python3.12 tools/planner/run_risk_aware_v18.py sim-gemini
  Dissertation Table 5.3 (the 150 benchmark scenes, perceived by v18):
    .venv/bin/python3.12 tools/planner/run_risk_aware_v18.py bench          # ERM-MCTS, rules
    .venv/bin/python3.12 tools/planner/run_risk_aware_v18.py bench-gemini
    .venv/bin/python3.12 tools/planner/run_risk_aware_v18.py bench-gt       # true inventory
  Add --dry-run to replace Gemini by his stub bagger; --n N for a smoke test;
  --resume EXP_NAME to continue an interrupted sim-gemini run.

Gemini's free tier allows 20 requests per day per project and model. Every
Gemini call goes through _rotating_call below, which uses the keys in .env in
turn, records in gemini_keys.json until when each is exhausted (shared by
concurrent runs), and sleeps until the first one resets when all are.
"""
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / "external/risk_aware_mcts"
OUT = ROOT / "runs/eval/risk_aware_v18"
sys.path.insert(0, str(EXT))

import envs.envs  # noqa: E402,F401  (the package must exist before the swap)

_spec = importlib.util.spec_from_file_location(
    "envs.binpacking_env_v3", ROOT / "tools/planner/binpacking_env_v3_v18.py")
_v18 = importlib.util.module_from_spec(_spec)
sys.modules["envs.binpacking_env_v3"] = _v18
_spec.loader.exec_module(_v18)
import envs  # noqa: E402
envs.binpacking_env_v3 = _v18

KEY_VARS = ["GEMINI_API_KEY", "GEMINI_API_KEY_ALT", "GEMINI_API_KEY_ALT2", "GEMINI_API_KEY_ALT3"]
KEY_STATE = OUT / "gemini_keys.json"
_current_key = None


def _exhausted() -> dict:
    try:
        return json.loads(KEY_STATE.read_text())
    except (OSError, ValueError):
        return {}


def _rotating_call(model, prompt: str, max_retries: int) -> str:
    """Drop-in for simulate_gemini_planner.call_gemini_with_backoff."""
    import google.generativeai as genai
    global _current_key
    other = 0
    while True:
        state, now = _exhausted(), time.time()
        free = [v for v in KEY_VARS if os.environ.get(v) and state.get(v, 0) <= now]
        if not free:
            wake = min(state.get(v, now) for v in KEY_VARS if os.environ.get(v))
            print(f"  all keys exhausted, sleeping {min(wake - now, 1800):.0f}s", flush=True)
            time.sleep(max(60, min(wake - now, 1800)))
            continue
        if free[0] != _current_key:
            _current_key = free[0]
            genai.configure(api_key=os.environ[_current_key])
            model._client = None  # the model caches its client; make it pick up the new key
            print(f"  using {_current_key}", flush=True)
        try:
            return model.generate_content(prompt).text
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "PerDay" in msg:
                m = re.search(r"seconds: (\d+)", msg)
                state = _exhausted()
                # the reported reset time is not reliable (keys come back hours early): retry hourly
                state[_current_key] = time.time() + min(int(m.group(1)) if m else 3600, 3600) + 60
                KEY_STATE.write_text(json.dumps(state, indent=1))
                print(f"  {_current_key} exhausted for the day", flush=True)
            elif "429" in msg or "quota" in msg.lower():
                m = re.search(r"retry in ([\d.]+)s", msg)
                time.sleep(float(m.group(1)) + 2 if m else 60)
            else:
                other += 1
                if other >= max_retries:
                    raise
                time.sleep(2 * other)


CAPTION = {"N": 500, "H": 5, "n_iter_per_timestep": 5000, "num_processors": 8}


def main():
    argv = sys.argv[1:]
    what = argv[0]
    dry = "--dry-run" in argv
    n = int(argv[argv.index("--n") + 1]) if "--n" in argv else None
    OUT.mkdir(parents=True, exist_ok=True)
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")  # GEMINI_API_KEY; his scripts also look next to themselves
    os.chdir(EXT)  # his scripts resolve Gemini-planning/ and .env relative to themselves
    import simulate_gemini_planner
    simulate_gemini_planner.call_gemini_with_backoff = _rotating_call
    resume = argv[argv.index("--resume") + 1] if "--resume" in argv else None

    if what == "sim-mcts":
        import simulate_mcts_multiple_betas as m
        cfg = {**m.CONFIG, **CAPTION, "algos": ["erm-mcts", "mcts"], "erm_betas": [1]}
        m.main({**cfg, "N": n or cfg["N"]}, 0, str(OUT / "sim") + "/")
    elif what == "sim-rules":
        import simulate_rule_based_planner as m
        cfg = {**m.CONFIG, "N": n or 500, "H": 5}
        m.main(cfg, 0, str(OUT / "sim") + "/")
    elif what == "sim-gemini":
        import simulate_gemini_planner as m
        cfg = {**m.CONFIG, "N": n or 500, "H": 5}
        m.main(cfg, 0, str(OUT / "sim") + "/", dry_run=dry, resume_exp_name=resume)
    elif what in ("bench", "bench-gemini", "bench-gt"):
        import simulate_benchmark_scenes as m
        cfg = {**m.CONFIG, "n_iter_per_timestep": 5000, "num_processors": 8,
               "algos": ["gemini-planner"] if what == "bench-gemini" else ["erm-mcts", "rule-based-planner"],
               "erm_betas": [1e-6, 1],
               # bench-gt: the planner sees the true inventory, isolating planning from perception
               "scenes_dir": str(ROOT / "runs/eval/planner_inputs_b50" /
                                 ("ground_truth" if what == "bench-gt" else "our_perception_v18")),
               "ground_truth_dir": str(ROOT / "runs/eval/planner_inputs_b50/ground_truth"),
               "N": n or 150}
        if what == "bench-gemini":
            # his benchmark script cannot resume; run only the scenes no earlier run has done
            done = set()
            for f in (OUT / "bench").glob("*gemini*/exp_data.json"):
                done |= set(json.loads(json.load(open(f)))["scene_ids"])
            ids = sorted(p.stem for p in Path(cfg["scenes_dir"]).glob("*.json") if p.stem not in done)
            print(f"{len(done)} scenes already done, {len(ids)} to go", flush=True)
            cfg = {**cfg, "scene_ids": ids}
        m.main(cfg, 0, str(OUT / ("bench_gt" if what == "bench-gt" else "bench")) + "/", dry_run=dry)
    else:
        sys.exit(f"unknown experiment {what!r}")


if __name__ == "__main__":
    main()
