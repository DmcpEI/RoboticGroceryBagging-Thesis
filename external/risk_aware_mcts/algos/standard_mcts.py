import numpy as np
from tqdm import tqdm


class DecisionNode:
    """
    The Decision Node class.

    A decision node stores a set of children nodes corresponding
    to the possible actions (of type RandomNode).

    :param state: (tuple) defining the state.
    :param available_actions: (tuple) defining the available actions at the state.
    :param father: (RandomNode) The father node; None if root.
    :param is_root: (bool)
    :param is_final: (bool)
    """

    def __init__(self, state=None, available_actions=None,
                    father=None, is_root=False, is_final=False):
        self.state = state
        self.available_actions = available_actions
        self.children = {}
        self.is_final = is_final
        self.visits = 0
        self.father = father
        self.is_root = is_root

        # Initialize children nodes.
        for action in self.available_actions:
            self.children[action] = RandomNode(action, father=self)

    def get_random_node(self, action: int):
        return self.children[action]


class RandomNode:
    """
    The RandomNode class defined by the state and the action.

    A random node stores a set of children nodes corresponding
    to the possible next states (of type DecisionNode).

    :param action: (int) taken in the decision node
    :param father: (DecisionNode)

    """
    def __init__(self, action: int, father: DecisionNode):
        self.action = action
        self.father = father
        self.children = {}
        self.cumulative_cost = 0.0  # scalar running sum -- a plain mean needs no full
        self.visits = 0             # history, unlike ERMMCTS's log-sum-exp costs_list.

    def add_child(self, child: DecisionNode, hash_preprocess):
        state_hash = hash_preprocess(child.state)
        if state_hash not in self.children.keys():
            self.children[state_hash] = child
        else:
            del child
            child = self.children[state_hash]

        return child


class StandardMCTS:
    """
    Genuine risk-neutral (expected discounted cost-minimizing) MCTS/UCT. Unlike
    ERMMCTS (algos/erm_mcts.py) and MCTS (algos/mcts.py), no erm_beta and no
    exp/log ERM transform is used anywhere -- node values are plain empirical
    mean discounted cost, and selection is standard UCB1.
    """

    def __init__(self, initial_state, env, K_ucb, rollout_policy=None,
                    best_action_criterion="min_cost"):
        """
        :param best_action_criterion: (str) how best_action() picks the root action
            once the tree is grown. "min_cost" (default): the action with the lowest
            empirical mean cost. "most_visited": the action visited most during tree
            search (classic UCT criterion). Unlike ERMMCTS's "min_erm"/"most_visited"
            split, both options are valid here -- visit counts DO track mean-cost
            ranking for an expected-value objective, so neither is a "legacy" fallback.
        """
        assert best_action_criterion in ("min_cost", "most_visited")

        self.K_ucb = K_ucb
        self.env = env
        self.rollout_policy = rollout_policy
        self.best_action_criterion = best_action_criterion

        # Create tree.
        self.root = self.init_tree(initial_state)

    def _hash_state(self, state):
        # Node identity must reflect the TRUE (physical) state, not just the
        # observed one -- envs like BinPackingEnv deliberately let observed
        # and true bag fill diverge (perception noise), and two branches that
        # look identical to the agent can still be in different true bag
        # states. Hashing only the observed "state" field would silently
        # merge those into one tree node, corrupting cost/legality estimates
        # with whichever branch's true bags got cached first. See
        # TRUE_STATE_HASH_COLLISION_BUG.md. state["bags"] (true bags) is
        # included when the env provides it; state["true_items"] is
        # deliberately NOT hashed -- for the current (non-scripted) env it is
        # always None until an item is actually placed, so it carries no
        # information and would only add a constant to every hash.
        bags = state.get("bags")
        true_bags_repr = tuple(bag.get_features() for bag in bags) if bags is not None else None
        node_repr = str(state["state"]) + "_" + \
                    str(state["t"]) + "_" + \
                    str(true_bags_repr)
        return hash(node_repr)

    def init_tree(self, initial_state):
        # Initialize tree.
        avail_actions = self.env.available_actions(initial_state)
        root = DecisionNode(state=initial_state,
                available_actions=avail_actions, is_root=True)
        return root

    def grow_tree(self):
        """
        Monte Carlo Tree Seach main loop: (i) Selection; (ii) Expansion;
            (iii) Simulation (rollout); and (iv) Backup.
        """
        decision_node = self.root
        decision_node.visits += 1 # Increment visits counter at root.

        # Selection and expansion.
        while (not decision_node.is_final) and decision_node.visits > 0:

            # Select action (and retrieve the random node associated with the action).
            a = self.select(decision_node)

            random_node = decision_node.get_random_node(a)

            # Generate the next decision node (next state).
            (new_decision_node, c) = self.select_outcome(random_node)

            # Store cost in random node.
            random_node.cost = c

            # Add next decision node to random node children. Internally checks whether
            # the created next_decision_node (next state) already exists in tree; if it
            # exists, the existing node is returned instead.
            new_decision_node = random_node.add_child(new_decision_node, self._hash_state)

            decision_node = new_decision_node

        # Simulation (rollout).
        cumulative_cost = self.rollout(decision_node)

        # Backup.
        while not decision_node.is_root:
            decision_node.visits += 1
            random_node = decision_node.father
            cumulative_cost = random_node.cost + self.env.gamma*cumulative_cost
            random_node.cumulative_cost += cumulative_cost
            random_node.visits += 1
            decision_node = random_node.father

    def select_outcome(self, random_node: RandomNode):
        """
        Given a RandomNode, returns a new DecisionNode corresponding to the next state.

        :param: random_node: (RandomNode) the random node from which selects the next state.
        :return: (DecisionNode) the selected Decision Node.
        """
        new_state, c, done = self.env.step(random_node.father.state, random_node.action)
        avail_actions = self.env.available_actions(new_state)
        new_decision_node = DecisionNode(state=new_state, father=random_node,
                            available_actions=avail_actions, is_final=done)
        return new_decision_node, c

    def select(self, x: DecisionNode):
        """
        Selects the action to play from the current decision node, via plain
        UCB1 scoring for cost minimization: argmin(mean_cost - bonus).

        :param x: (DecisionNode) current decision node.
        :return: (int) action.
        """
        def scoring(k):
            child = x.children[k]
            if child.visits > 0:
                mean_cost = child.cumulative_cost / child.visits
                return mean_cost - self.K_ucb * np.sqrt(np.sqrt(x.visits) / child.visits)
            else:
                return -np.inf

        return min(x.children, key=scoring)

    def rollout(self, initial_node: DecisionNode):
        """
        Evaluates a DecisionNode by performing a rollout with a given policy.

        :param initial_node: (DecisionNone) initial node.
        :return: (float) the cumulative cost observed during the tree traversing.
        """
        discounted_cumulative_costs = 0
        done = initial_node.is_final
        s = initial_node.state
        discount = 1.0
        while not done:
            if self.rollout_policy:
                a = self.rollout_policy(s)
            else:
                # Random policy.
                avail_actions = self.env.available_actions(s)
                a = np.random.choice(avail_actions)

            s, c, done = self.env.step(s,a)
            discounted_cumulative_costs += c * discount
            discount *= self.env.gamma

        return discounted_cumulative_costs

    def learn(self, n_iters, progress_bar=False):
        """
        Expand the tree.

        :param: n_iters: (int) number of tree traversals.
        :param: progress_bar: (bool) whether to show a progress bar (tqdm).
        """
        if progress_bar:
            iterations = tqdm(range(n_iters))
        else:
            iterations = range(n_iters)
        for _ in iterations:
            self.grow_tree()

    def best_action(self):
        """
        Returns the root action to commit to, according to self.best_action_criterion:
        - "min_cost": the action with the lowest empirical mean cost (no exploration
          bonus -- this is the final commit, not tree search).
        - "most_visited": the most-visited child (classic UCT commit criterion).

        :return: (int) the selected action.
        """
        children = list(self.root.children.values())

        if self.best_action_criterion == "most_visited":
            visits = [node.visits for node in children]
            return children[np.argmax(visits)].action

        # "min_cost"
        mean_costs = [
            node.cumulative_cost / node.visits if node.visits > 0 else np.inf
            for node in children
        ]
        return children[np.argmin(mean_costs)].action

    def update_root_node(self, selected_action : int, new_state : dict):
        """
            Updates the root node of the planning tree.

            :param: (int) selected_action: previously selected action.
            :param: (dict) new_state: the new state of the environment.
        """
        next_state_hash = self._hash_state(new_state)
        if next_state_hash in self.root.children[selected_action].children:
            self.root = self.root.children[selected_action].children[next_state_hash]
            self.root.is_root = True
            return True
        else:
            return False


if __name__ == "__main__":
    pass
