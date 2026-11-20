import os
import sys
import json
import multiprocessing as mp
import pathlib
import random
from datetime import datetime

import numpy as np
from tqdm import tqdm

from algos.erm_mcts import ERMMCTS
from algos.mcts import MCTS
from algos.standard_mcts import StandardMCTS
from envs.envs import get_env, MDPs
#from envs.binpacking_env_sequential import BinPackingEnv
from envs.bandit_env import DeceptiveBanditEnv
from envs.binpacking_env_v3 import BinPackingEnv
from violation_metrics import count_violations, VIOLATION_TYPES, violations_affected_stats

DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent) + '/data/'
print(DATA_FOLDER_PATH)
DEBUG = False

CONFIG = {
    "N": 500,                    # Number of experiments to run per beta.
    "num_processors": 8,
    "env": "binpacking",
    "algo": "erm-mcts",          # "erm-mcts", "acc-mcts", or "mcts" (standard/risk-neutral) --
                                    # used as-is by main_single_beta()/run() (and by other
                                    # scripts that import them directly, e.g.
                                    # run_bandit_experiments.py, sweep_n_iter_erm_vs_acc.py).
                                    # main() instead loops over "algos" below if present.
    "algos": ["erm-mcts"],  # maican() sweeps every algo here (each with the
                                    # SAME arg_seed, for a consistent-seeding comparison across
                                    # algorithms -- see run()/main_single_beta() for how the
                                    # seed -> per-episode RNG mapping works). Falls back to
                                    # [cfg["algo"]] if this key is absent, so any caller that
                                    # only sets "algo" (or calls main_single_beta() directly)
                                    # is unaffected.
    "H": 5,                      # Truncation length.
    "n_iter_per_timestep":20_000,  # MCTS number of tree expansion steps per timestep.
    "erm_betas": [1e-6,1,5], #1e-6,1  # for "algo": "mcts", set a single-element placeholder
                                    # (e.g. [0]) -- StandardMCTS ignores it, kept only so
                                    # create_exp_name/main_single_beta (which always read
                                    # cfg['erm_beta']) keep working unmodified. main() only
                                    # runs "mcts" once regardless of how many betas are listed
                                    # here (see main()) since they'd otherwise be identical
                                    # wasted re-runs -- same seed, erm_beta has no effect.
    "best_action_criterion": "min_erm",  # "min_erm"/"min_cost" (per-algo "correct" default)
                                    # or "most_visited" (legacy for erm-mcts, alt. for mcts).
                                    # run() translates "min_erm" -> "min_cost" for algo="mcts".
    "gamma": 1,                   # BinPackingEnv discount factor (both algos read it via env.gamma)
}

def create_exp_name(args: dict) -> str:
    return args['env'] + '_' + args['algo'] + '_gamma_' + str(args['gamma']) + '_beta_' + str(args['erm_beta']) + '_' + \
        str(datetime.today().strftime('%Y-%m-%d-%H-%M-%S')) + str(args['seed'])


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


MAX_EXP_ARG = 700.0  # np.exp overflows to inf above ~709.78 (float64); 700 leaves
                      # headroom for cumulative_reward to sum many such terminal
                      # visits (n_iter_per_timestep up to ~1e4) before approaching
                      # float64's max (~1.8e308).

def _safe_exp(beta_times_cost):
    """np.exp with the exponent clipped so it can never overflow to inf."""
    return np.exp(np.minimum(beta_times_cost, MAX_EXP_ARG))


class AccruedCosts_MDP:

    def __init__(self, mdp, H, erm_beta):
        self.mdp = mdp
        self.H = H
        self.gamma = mdp["gamma"]
        self.erm_beta = erm_beta

    def available_actions(self, state):
        return self.mdp["actions"]

    def sample_initial_state(self):
        state = np.random.choice(self.mdp["states"], p=self.mdp["p_0"])
        return {"state": state, "accrued_costs": 0.0, "t": 0}

    def step(self, extended_state, a):
        state_t = extended_state["state"]
        accrued_costs_t = extended_state["accrued_costs"]
        timestep_t = extended_state["t"]
        next_accrued_cost = accrued_costs_t + self.gamma**timestep_t * self.mdp["C"][state_t, a]
        next_state = np.random.choice(self.mdp["states"], p=self.mdp["P"][a, state_t, :])
        next_timestep = timestep_t + 1
        next_extended_state = {"state": next_state, "accrued_costs": next_accrued_cost, "t": next_timestep}
        if next_timestep == self.H:
            cost = _safe_exp(self.erm_beta * next_accrued_cost)
            terminated = True
        else:
            cost = 0.0
            terminated = False
        return next_extended_state, cost, terminated


class AccruedCosts_BinPackingEnv:

    def __init__(self, env, erm_beta):
        self.env = env
        self.gamma = env.gamma
        self.erm_beta = erm_beta

    def available_actions(self, state):
        return self.env.available_actions(state)

    def sample_initial_state(self):
        s = self.env.sample_initial_state()
        s["accrued_costs"] = 0.0
        return s

    def step(self, extended_state, a):
        accrued_costs = extended_state["accrued_costs"]
        t = extended_state["t"]
        next_state, cost, terminated = self.env.step(extended_state, a)
        next_accrued_cost = accrued_costs + self.gamma**t * cost
        next_state["accrued_costs"] = next_accrued_cost
        if terminated:
            return next_state, _safe_exp(self.erm_beta * next_accrued_cost), True
        else:
            return next_state, 0.0, False


def get_accrued_env(env_name, H, erm_beta, gamma=1):
    if env_name == "binpacking":
        return AccruedCosts_BinPackingEnv(BinPackingEnv(num_items_to_pack=H, gamma=gamma), erm_beta)
    if env_name == "bandit":
        return AccruedCosts_BinPackingEnv(DeceptiveBanditEnv(gamma=gamma), erm_beta)
    return AccruedCosts_MDP(MDPs[env_name], H, erm_beta)


def _build_raw_env(cfg):
    """Construct the raw (unwrapped) env for a given cfg. Shared by the erm-mcts and
    mcts (standard) branches of run() - only acc-mcts uses get_accrued_env() instead."""
    if cfg["env"] == "binpacking":
        return BinPackingEnv(num_items_to_pack=cfg["H"], gamma=cfg.get("gamma", 1))
    elif cfg["env"] == "bandit":
        return DeceptiveBanditEnv(gamma=cfg.get("gamma", 1))
    else:
        return get_env(cfg["env"], cfg["H"])


def simulate_accrued_MCTS(env, H, n_iter_per_timestep=1_000):

    extended_state = env.sample_initial_state()
    mcts = MCTS(initial_state=extended_state, env=env, K_ucb=np.sqrt(2), erm_beta=env.erm_beta, rollout_policy=None)

    for t in tqdm(range(H)):
        random_state = random.getstate()
        np_state = np.random.get_state()
        mcts.learn(n_iters=n_iter_per_timestep)
        random.setstate(random_state)
        np.random.set_state(np_state)

        selected_action = mcts.best_action()
        extended_state, cost, terminated = env.step(extended_state, selected_action)

        if terminated:
            # Episode can end before t==H now (e.g. binpacking_env_v4: no items-
            # remaining budget, so a state with no available actions ends the
            # episode outright) -- don't plan/act on a state we won't use again.
            break

        updated_root = mcts.update_root_node(selected_action, extended_state)
        if not updated_root:
            mcts = MCTS(initial_state=extended_state, env=env, K_ucb=np.sqrt(2), erm_beta=env.erm_beta, rollout_policy=None)

    cumulative_cost = extended_state["accrued_costs"]
    violations = count_violations(extended_state["bags"]) if "bags" in extended_state else None
    print("final discounted cumulative cost:", cumulative_cost)
    print("violations:", violations)
    return cumulative_cost, violations


def simulate_ERM_MCTS(env, H, erm_beta, n_iter_per_timestep=1_000, best_action_criterion="min_erm",
                       risk_neutral_beta_threshold=1e-6):

    # Sample initial state.
    extended_state = env.sample_initial_state()
    K_ucb = np.sqrt(2)

    mcts = ERMMCTS(initial_state=extended_state, env=env, K_ucb=K_ucb,
                    erm_beta=erm_beta, rollout_policy=None, root_depth=0,
                    best_action_criterion=best_action_criterion,
                    risk_neutral_beta_threshold=risk_neutral_beta_threshold)

    # Simulate until termination.
    cumulative_discounted_cost = 0.0
    for t in tqdm(range(H)):

        if DEBUG:
            print("state:", extended_state)

        random_state = random.getstate()
        np_state = np.random.get_state()
        mcts.learn(n_iters=n_iter_per_timestep)
        random.setstate(random_state)
        np.random.set_state(np_state)

        selected_action = mcts.best_action()

        if DEBUG:
            print("action:", selected_action)

        extended_state, cost, terminated = env.step(extended_state, selected_action)
        cumulative_discounted_cost += cost * env.gamma**t

        if DEBUG:
            print("cost:", cost)

        if terminated:
            # Episode can end before t==H now (e.g. binpacking_env_v4: no items-
            # remaining budget, so a state with no available actions ends the
            # episode outright) -- don't plan/act on a state we won't use again.
            break

        updated_root = mcts.update_root_node(selected_action, extended_state)
        if updated_root:
            mcts.set_root_depth(t+1)
        else:
            mcts = ERMMCTS(initial_state=extended_state, env=env, K_ucb=K_ucb,
                    erm_beta=erm_beta, rollout_policy=None, root_depth=t+1,
                    best_action_criterion=best_action_criterion,
                    risk_neutral_beta_threshold=risk_neutral_beta_threshold)

    violations = count_violations(extended_state["bags"]) if "bags" in extended_state else None
    print("final discounted cumulative cost:", cumulative_discounted_cost)
    print("violations:", violations)

    return cumulative_discounted_cost, violations


def simulate_standard_MCTS(env, H, n_iter_per_timestep=1_000, best_action_criterion="min_cost"):

    extended_state = env.sample_initial_state()
    K_ucb = np.sqrt(2)

    mcts = StandardMCTS(initial_state=extended_state, env=env, K_ucb=K_ucb,
                         rollout_policy=None, best_action_criterion=best_action_criterion)

    cumulative_discounted_cost = 0.0
    for t in tqdm(range(H)):

        random_state = random.getstate()
        np_state = np.random.get_state()
        mcts.learn(n_iters=n_iter_per_timestep)
        random.setstate(random_state)
        np.random.set_state(np_state)

        selected_action = mcts.best_action()

        extended_state, cost, terminated = env.step(extended_state, selected_action)
        cumulative_discounted_cost += cost * env.gamma**t

        if terminated:
            # Episode can end before t==H now (e.g. binpacking_env_v4: no items-
            # remaining budget, so a state with no available actions ends the
            # episode outright) -- don't plan/act on a state we won't use again.
            break

        updated_root = mcts.update_root_node(selected_action, extended_state)
        if not updated_root:
            mcts = StandardMCTS(initial_state=extended_state, env=env, K_ucb=K_ucb,
                                 rollout_policy=None, best_action_criterion=best_action_criterion)

    violations = count_violations(extended_state["bags"]) if "bags" in extended_state else None
    print("final discounted cumulative cost:", cumulative_discounted_cost)
    print("violations:", violations)
    return cumulative_discounted_cost, violations


def run(cfg, seed):

    print('Running seed=', seed)

    np.random.seed(seed)
    random.seed(seed)

    if cfg["algo"] == "acc-mcts":
        env = get_accrued_env(cfg["env"], cfg["H"], cfg["erm_beta"], gamma=cfg.get("gamma", 1))
        mcts_f_val, violations = simulate_accrued_MCTS(env=env,
                                           H=cfg["H"],
                                           n_iter_per_timestep=cfg["n_iter_per_timestep"])
    elif cfg["algo"] == "mcts":
        env = _build_raw_env(cfg)
        # "min_erm" is ERM-MCTS's config vocabulary; translate its default to this
        # algo's equivalent ("min_cost"). "most_visited" passes through unchanged.
        criterion = cfg.get("best_action_criterion", "min_erm")
        criterion = "min_cost" if criterion == "min_erm" else criterion
        mcts_f_val, violations = simulate_standard_MCTS(env=env,
                                            H=cfg["H"],
                                            n_iter_per_timestep=cfg["n_iter_per_timestep"],
                                            best_action_criterion=criterion)
    else:
        env = _build_raw_env(cfg)
        mcts_f_val, violations = simulate_ERM_MCTS(env=env,
                                       H=cfg["H"],
                                       erm_beta=cfg["erm_beta"],
                                       n_iter_per_timestep=cfg["n_iter_per_timestep"],
                                       best_action_criterion=cfg.get("best_action_criterion", "min_erm"),
                                       risk_neutral_beta_threshold=cfg.get("risk_neutral_beta_threshold", 1e-6))

    return mcts_f_val, violations


def main_single_beta(cfg, arg_seed=0, data_folder_path=None):
    """Run N experiments for a single erm_beta. cfg must contain 'erm_beta'."""

    if not data_folder_path:
        data_folder_path = DATA_FOLDER_PATH

    if cfg['env'] not in MDPs:
        gamma_val = cfg.get('gamma', 1)
    else:
        gamma_val = MDPs.get(cfg['env'], {}).get('gamma', 0.99)
    exp_name = create_exp_name({'env': cfg['env'],
                                'algo': cfg['algo'],
                                'gamma': gamma_val,
                                'erm_beta': cfg['erm_beta'],
                                'seed': arg_seed,
                                })
    exp_path = data_folder_path + exp_name
    os.makedirs(exp_path, exist_ok=True)
    print('\nExperiment ID:', exp_name)
    print('Config:')
    print(cfg)

    print('\nSimulating...')

    with mp.Pool(processes=cfg["num_processors"]) as pool:
        results = pool.starmap(run, [(cfg, (1_000*arg_seed) + t) for t in range(cfg["N"])])
        pool.close()
        pool.join()

    f_vals = np.array([r[0] for r in results])
    violations_list = [r[1] for r in results]

    exp_data = {}
    exp_data["config"] = cfg
    exp_data["f_vals"] = f_vals
    exp_data["env"] = cfg['env']

    if violations_list and all(v is not None for v in violations_list):
        for k in VIOLATION_TYPES:
            exp_data[k] = [v[k] for v in violations_list]
        exp_data["violations_affected"] = violations_affected_stats(violations_list)

        stats = exp_data["violations_affected"]
        total_violations = {k: sum(v[k] for v in violations_list) for k in VIOLATION_TYPES}
        print("\n=== Violations summary ===")
        print(f"  total violations: {total_violations}  (grand total: {sum(total_violations.values())})")
        print(f"  episodes affected by >=1 violation: {stats['episodes_affected_any']}/{stats['n_episodes']}"
              f" ({stats['fraction_episodes_affected_any']:.1%})")
        print(f"  episodes affected by violation type: {stats['episodes_affected_by_type']}")

    f = open(exp_path + "/exp_data.json", "w")
    dumped = json.dumps(exp_data, cls=NumpyEncoder)
    json.dump(dumped, f)
    f.close()

    print(exp_name)

    return exp_name


def main(cfg, arg_seed=0, data_folder_path=None):
    """Run experiments for every (algo, erm_beta) combination in
    cfg['algos'] x cfg['erm_betas'] sequentially. cfg['algos'] defaults to
    [cfg['algo']] if absent, so existing single-algo CONFIGs/callers are
    unaffected.

    Seeding: `arg_seed` is passed through unchanged to every
    main_single_beta()/run() call below, regardless of algo or beta -- the
    per-episode seed (1_000*arg_seed + t, set inside run()) depends ONLY on
    arg_seed and the episode index t, never on algo/erm_beta. So every algo
    in this sweep sees the exact same set of N per-episode seeds -- this is
    what makes a cross-algo comparison from one main() call "consistently
    seeded". Do not thread algo/erm_beta into the seed calculation.
    """

    algos = cfg.get("algos", [cfg["algo"]])
    exp_names = {}
    for algo in algos:
        # StandardMCTS ("mcts") ignores erm_beta entirely (see algos/standard_mcts.py) --
        # looping it over multiple erm_betas would just re-run the IDENTICAL experiment
        # (same seeds, same algo, beta has no effect) once per beta for no reason.
        # Confirmed empirically 2026-08-18: three such re-runs produced bitwise-identical
        # f_vals. Run it exactly once, using the first erm_betas entry as a placeholder label.
        betas = cfg["erm_betas"] if algo != "mcts" else cfg["erm_betas"][:1]
        for erm_beta in betas:
            print(f"\n{'='*60}")
            print(f"Running algo = {algo}, beta = {erm_beta}")
            print(f"{'='*60}")

            cfg_run = {**cfg, "algo": algo, "erm_beta": erm_beta}
            exp_name = main_single_beta(cfg_run, arg_seed=arg_seed, data_folder_path=data_folder_path)
            exp_names[(algo, erm_beta)] = exp_name

    print("\nAll experiments completed:")
    for (algo, beta), name in exp_names.items():
        print(f"  algo={algo}, beta={beta}: {name}")

    return exp_names


if __name__ == "__main__":
    if len(sys.argv) > 2:
        main(CONFIG, int(sys.argv[1]), sys.argv[2])
    else:
        main(CONFIG)
