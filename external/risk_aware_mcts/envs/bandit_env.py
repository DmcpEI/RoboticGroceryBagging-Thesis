import random


class DeceptiveBanditEnv:
    """
    One-shot (H=1) two-armed bandit designed to make acc-mcts's UCB exploration
    lockout directly visible on a minimal example, mirroring the mechanism
    observed on the real binpacking env at high erm_beta.

    Arm 0 ("risky-but-better"): cost = 0 w.p. (1 - shock_prob), cost = shock_cost
        w.p. shock_prob.
    Arm 1 ("safe-but-worse"): cost = safe_cost, deterministic.

    Default parameters (shock_cost=12, shock_prob=0.02, safe_cost=10) were chosen
    so that arm 0 is the objectively better choice under ERM_beta at beta=1 by a
    substantial margin (E[arm0]=0.24 vs E[arm1]=10 in expectation; exact
    log-sum-exp: ERM(arm0)~=8.09 < ERM(arm1)=10, a ~19% gap -- vs an earlier
    shock_prob=0.1 version where the gap was only ~3% and got lost in sampling
    noise), while a single shock sample still produces exp(beta*12)=162755 vs
    exp(beta*10)=22026 -- ~7.4x larger, dwarfing a UCB exploration bonus of
    order 1-10 (unchanged from before -- this ratio depends only on
    shock_cost - safe_cost, not on shock_prob).
    """

    def __init__(self, gamma=1.0, shock_cost=12.0, shock_prob=0.02, safe_cost=10.0):
        self.gamma = gamma
        self.shock_cost = shock_cost
        self.shock_prob = shock_prob
        self.safe_cost = safe_cost

    def available_actions(self, state):
        return [0, 1]

    def sample_initial_state(self):
        return {"state": 0, "t": 0}

    def step(self, state, action):
        if action == 0:
            cost = self.shock_cost if random.random() < self.shock_prob else 0.0
        else:
            cost = self.safe_cost

        next_state = {"state": 0, "t": state["t"] + 1}
        terminated = True
        return next_state, cost, terminated
