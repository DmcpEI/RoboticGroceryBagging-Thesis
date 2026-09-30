import numpy as np
import random
from envs.envs import Env

'''
    Here I want to make the agent see a variable number of objects at each time

'''


CATEGORY_MAP = {
    'Dairy': 0, 'Bakery': 1, 'Produce': 2, 'Snacks': 3,
    'Pantry': 4, 'Frozen': 5, 'Raw Meat': 6, 'Cleaning': 7, 'Household': 8, 'Hardware': 9
}
TEMPERATURE_MAP = {
    'Ambient': 0, 'Refrigerated': 1, 'Frozen': 2
}

# Cost constants — tune these to calibrate accident signal vs. packing noise.
# Illegal move reduced so it doesn't completely swamp accident variance during rollouts.
# Crush/spill raised so a single accident is visible on the scale of packing-reward differences.
ILLEGAL_MOVE_COST  = 100.0   # volume/weight constraint violation
CRUSH_COST         = 50.0   # per fragile item crushed
SPILL_COST         = 40.0   # base cost per spill event
CONTAMINATION_COST = 10.0   # added per spill-vulnerable item contaminated by a spill
CHEMICAL_CONTAMINATION_BONUS = 30.0  # additional cost (on top of CONTAMINATION_COST), per
                                     # contaminated spill-vulnerable item, when the spilling
                                     # item is a "Cleaning" category product rather than food
GENERAL_MESS_COST = CONTAMINATION_COST  # cost charged to any OTHER (non-spill-vulnerable)
                                     # item in a bag when a "Cleaning" product spills -- a
                                     # separate, independently-tunable constant (not the same
                                     # object as CONTAMINATION_COST), currently initialized to
                                     # the same value; food-on-food spills still cost
                                     # non-vulnerable items nothing
BAG_OPEN_COST      = 10.0    # charged once when the first item is placed in a previously empty bag

# Placement-time accident probabilities -- independent of, and in addition to,
# the squeeze-based crush/spill risk that accrues from later items (below).
# Model mishandling/impact damage at the moment an item is set down, not just
# being crushed/squeezed by something placed near it afterward.
PLACEMENT_CRUSH_PROB = 0.02  # chance a fragile item (crush_score >= 8) breaks the instant it's placed
PLACEMENT_SPILL_PROB = 0.02  # chance a spill_risk item spills the instant it's placed

class Item:
    def __init__(self, id, name, est_weight_g, est_volume_cc, crush_score, category, temperature, spill_risk, spill_vulnerable, orientation_sensitive, is_crushed=False, risk_level=0.0, is_spilled=False, spill_risk_level=0.0):
        self.id = id
        self.name = name
        self.weight = est_weight_g
        self.volume = est_volume_cc
        self.crush_score = crush_score
        self.category = category
        self.temperature = temperature
        self.spill_risk = spill_risk
        self.spill_vulnerable = spill_vulnerable
        self.orientation_sensitive = orientation_sensitive
        self.is_crushed = is_crushed
        self.risk_level = risk_level
        self.is_spilled = is_spilled
        self.spill_risk_level = spill_risk_level

    def copy(self):
        return Item(
            self.id, self.name, self.weight, self.volume, self.crush_score,
            self.category, self.temperature, self.spill_risk, self.spill_vulnerable,
            self.orientation_sensitive, self.is_crushed, self.risk_level,
            self.is_spilled, self.spill_risk_level
        )

    def get_features(self):
        category_id = CATEGORY_MAP.get(self.category, -1)
        temp_id = TEMPERATURE_MAP.get(self.temperature, -1)
        return (
            self.weight,
            self.volume,
            self.crush_score,
            category_id,
            temp_id,
            int(self.spill_risk),
            int(self.spill_vulnerable),
            int(self.orientation_sensitive),
            int(self.is_crushed),
            self.risk_level,
            int(self.is_spilled),
            self.spill_risk_level,
        )

    def __repr__(self):
        return f"Item({self.name}_{self.id})"

class Bag:
    MAX_VOLUME = 5000
    MAX_WEIGHT = 5000

    def __init__(self):
        self.volume_used = 0
        self.weight_used = 0
        self.items = []
        self.fragile_count = 0
        self.spill_risk_count = 0
        self.spill_vulnerable_count = 0
        self.orientation_sensitive_count = 0
        self.top_level_risk_score = 0
        self.total_items = 0
        self.spilled_count = 0
        
    def copy(self):
        new_bag = Bag()
        new_bag.volume_used = self.volume_used
        new_bag.weight_used = self.weight_used
        new_bag.items = [it.copy() for it in self.items]
        new_bag.fragile_count = self.fragile_count
        new_bag.spill_risk_count = self.spill_risk_count
        new_bag.spill_vulnerable_count = self.spill_vulnerable_count
        new_bag.orientation_sensitive_count = self.orientation_sensitive_count
        new_bag.top_level_risk_score = self.top_level_risk_score
        new_bag.total_items = self.total_items
        new_bag.spilled_count = self.spilled_count
        return new_bag

    def get_features(self):
        crushed_count = sum(1 for it in self.items if it.is_crushed)
        total_risk = sum(it.risk_level for it in self.items)
        total_spill_risk = sum(it.spill_risk_level for it in self.items)
        return (
            self.volume_used,
            self.weight_used,
            crushed_count,
            total_risk,
            self.fragile_count,
            self.spill_risk_count,
            self.spill_vulnerable_count,
            self.orientation_sensitive_count,
            self.spilled_count,
            total_spill_risk,
        )

    def get_observed_features(self):
        """Bag features as observed by the agent: no crushed/spilled counts."""
        total_risk = sum(it.risk_level for it in self.items)
        total_spill_risk = sum(it.spill_risk_level for it in self.items)
        return (
            self.volume_used,
            self.weight_used,
            total_risk,
            self.fragile_count,
            self.spill_risk_count,
            self.spill_vulnerable_count,
            self.orientation_sensitive_count,
            total_spill_risk,
        )

    def place_observed_item(self, item):
        """Deterministic accumulation (no crush/spill simulation, no cost) — used by
        step()'s observed_bags update and by the real-robot interface's internal
        bag-state tracking (replaying its own chosen actions)."""
        self.volume_used += item.volume
        self.weight_used += item.weight
        self.total_items += 1

        for packed_item in self.items:
            if packed_item.crush_score >= 8:
                if item.crush_score <= 2:
                    packed_item.risk_level += item.weight / 500.0
                else:
                    packed_item.risk_level += item.weight / 1000.0

        squeeze = self.volume_used / Bag.MAX_VOLUME
        for packed_item in self.items:
            if packed_item.spill_risk:
                if packed_item.orientation_sensitive:
                    packed_item.spill_risk_level += (item.weight / 500.0) * squeeze
                else:
                    packed_item.spill_risk_level += (item.weight / 1000.0) * squeeze

        self.items.append(item.copy())
        if item.crush_score >= 8:
            self.fragile_count += 1
        if item.spill_risk:
            self.spill_risk_count += 1
        if item.spill_vulnerable:
            self.spill_vulnerable_count += 1
        if item.orientation_sensitive:
            self.orientation_sensitive_count += 1

    def __repr__(self):
        return f"Bag(features={self.get_features()})"


def _contamination_cost(bag_items, spiller) -> float:
    """Cost every OTHER item in bag_items incurs given that `spiller` has just
    spilled (or already spilled) in this bag. Shared by every spill trigger
    point in step() (squeeze-based, retroactive-arrival, and placement-time)
    instead of duplicating this logic at each one:
      - spill_vulnerable items: CONTAMINATION_COST, +CHEMICAL_CONTAMINATION_BONUS
        if spiller is a "Cleaning" product.
      - every OTHER item (not spill_vulnerable): GENERAL_MESS_COST, but ONLY if
        spiller is "Cleaning" -- general mess/cross-contamination cost, cheaper
        than the chemical bonus which stays reserved for absorbent/vulnerable
        items. Food-on-food spills still cost non-vulnerable items nothing.
    Excludes spiller itself and any already-crushed item (an already-destroyed
    item doesn't separately track contamination)."""
    is_cleaning = spiller.category == "Cleaning"
    total = 0.0
    for other in bag_items:
        if other is spiller or other.is_crushed:
            continue
        if other.spill_vulnerable:
            total += CONTAMINATION_COST
            if is_cleaning:
                total += CHEMICAL_CONTAMINATION_BONUS
        elif is_cleaning:
            total += GENERAL_MESS_COST
    return total


# Items transcribed from the benchmark ground-truth catalogue (150 scenes under
# planner_input_ground_truth/*.json, collapsed by display_name into 32 conflict-free
# items; ids assigned alphabetically -- see scratch/build_v3_catalog.py). The source
# has no orientation_sensitive field, so it defaults to False for every item here.
ITEMS_DATA = [
    Item( 1, "bag of bananas", 477, 772, 10, "Produce", "Ambient", False, True, False),
    Item( 2, "bag of cherries", 33, 60, 10, "Produce", "Ambient", False, True, False),
    Item( 3, "bag of green apples", 733, 884, 8, "Produce", "Ambient", False, True, False),
    Item( 4, "bag of limes", 285, 624, 8, "Produce", "Ambient", False, True, False),
    Item( 5, "bag of mangoes", 1349, 1400, 8, "Produce", "Ambient", False, True, False),
    Item( 6, "bag of oranges", 609, 816, 8, "Produce", "Ambient", False, True, False),
    Item( 7, "bag of pears", 717, 1376, 8, "Produce", "Ambient", False, True, False),
    Item( 8, "bag of red apples", 733, 884, 8, "Produce", "Ambient", False, True, False),
    Item( 9, "bag of strawberries", 53, 332, 10, "Produce", "Ambient", False, True, False),
    Item(10, "bag of yellow lemons", 485, 624, 8, "Produce", "Ambient", False, True, False),
    Item(11, "bleach bottle", 1131, 1592, 2, "Cleaning", "Ambient", True, False, False),
    Item(12, "bowl", 147, 1052, 2, "Household", "Ambient", False, False, False),
    Item(13, "chocolate pudding box", 187, 342, 6, "Snacks", "Ambient", False, True, False),
    Item(14, "coffee can", 414, 1136, 2, "Pantry", "Ambient", False, False, False),
    Item(15, "cola bottle", 545, 550, 2, "Pantry", "Ambient", True, False, False),
    Item(16, "cracker box", 411, 1991, 6, "Snacks", "Ambient", False, True, False),
    Item(17, "cup", 118, 450, 8, "Household", "Ambient", False, False, False),
    Item(18, "egg carton", 385, 1300, 9, "Dairy", "Refrigerated", False, False, False),
    Item(19, "energy drink can", 270, 290, 2, "Pantry", "Ambient", False, False, False),
    Item(20, "gelatin dessert box", 97, 174, 6, "Snacks", "Ambient", False, True, False),
    Item(21, "glass cleaner spray bottle", 1022, 2268, 2, "Cleaning", "Ambient", True, False, False),
    Item(22, "kitchen sponge", 12, 160, 5, "Household", "Ambient", False, False, False),
    Item(23, "mustard bottle", 603, 744, 2, "Pantry", "Ambient", True, False, False),
    Item(24, "plate", 279, 1255, 2, "Household", "Ambient", False, False, False),
    Item(25, "potato chip can", 205, 1104, 2, "Snacks", "Ambient", False, True, False),
    Item(26, "spam can", 370, 398, 2, "Pantry", "Ambient", False, False, False),
    Item(27, "sugar box", 514, 592, 5, "Pantry", "Ambient", False, False, False),
    Item(28, "tomato soup can", 349, 346, 2, "Pantry", "Ambient", False, False, False),
    Item(29, "toothpaste box", 125, 300, 5, "Cleaning", "Ambient", False, False, False),
    Item(30, "tuna can", 171, 187, 2, "Pantry", "Ambient", False, False, False),
    Item(31, "water bottle", 520, 550, 2, "Pantry", "Ambient", True, False, False),
    Item(32, "wine cup", 133, 852, 8, "Household", "Ambient", False, False, False),
]

_ITEMS_BY_NAME = {item.name: item for item in ITEMS_DATA}
DEBUG_ITEMS_DATA = [
    _ITEMS_BY_NAME["egg carton"],
    _ITEMS_BY_NAME["bleach bottle"],
    _ITEMS_BY_NAME["glass cleaner spray bottle"],
    _ITEMS_BY_NAME["cracker box"],
]

# Sentinel for empty visible slots when all items for the episode have been revealed.
PLACEHOLDER_ITEM = Item(0, "placeholder", 0, 0, 0, "Household", "Ambient", False, False, False)

# Fast id → Item lookup used by the observation model.
_ID_TO_ITEM = {item.id: item for item in ITEMS_DATA}

# Per-item perception outcomes from perception_uncertainty_our_perception.json's
# per_object confusion stats (see scratch/build_v3_catalog.py). For each true item: a
# list of outcomes with the fraction of observations and the estimated planner features
# the agent sees for that classification. perceived_as=None means NOT_DETECTED (item
# present but invisible to the agent -> PLACEHOLDER_ITEM in the visible window).
# "observations" is the real per-item appearance count from the benchmark's 150 scenes
# (per_object[name]["observations"]) -- this is P(true item), used both as the sampling
# prior in NOISY_ITEMS/_NOISY_WEIGHTS (how often each true item shows up while
# generating episodes) and, multiplied by each outcome's fraction, as the unnormalized
# Bayes posterior weight in _PERCEIVED_CANDIDATES (P(true|observed) ~ P(observed|true) *
# P(true) -- see build_v3_catalog.py's docstring for the derivation and a worked
# example of why dropping this prior in favor of a uniform one gets the posterior wrong).
# Deliberate deviations from the raw source: (1) "unknown_object" classification
# outcomes are dropped entirely rather than kept or folded into the miss outcome --
# remaining fractions are left as-is (not renormalized), since random.choices()
# normalizes weights internally wherever these are sampled from, so the dropped mass is
# redistributed proportionally over what's left automatically; (2) "bag of red apples"
# has no entry in the source per_object (never observed as a true item there) -- it
# gets a synthetic single-outcome, perfect-detection entry, and its "observations"
# borrows "bag of green apples"'s real count (28) as a stand-in, since the two items
# are physically identical in the benchmark catalog.
FEATURE_DISTRIBUTION = {
    "bag of bananas": {
        "id": 1, "observations": 15,
        "outcomes": [
            {"fraction": 0.0, "perceived_as": None, "features": None},
            {"fraction": 1.0, "perceived_as": "bag of bananas", "features": {"weight": 477, "volume": 772, "crush_score": 10, "category": "Produce", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": True, "orientation_sensitive": False}},
        ],
    },
    "bag of cherries": {
        "id": 2, "observations": 11,
        "outcomes": [
            {"fraction": 0.0, "perceived_as": None, "features": None},
            {"fraction": 0.909, "perceived_as": "bag of cherries", "features": {"weight": 33, "volume": 60, "crush_score": 10, "category": "Produce", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": True, "orientation_sensitive": False}},
            {"fraction": 0.045, "perceived_as": "cracker box", "features": {"weight": 411, "volume": 1991, "crush_score": 6, "category": "Snacks", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": True, "orientation_sensitive": False}},
            {"fraction": 0.045, "perceived_as": "sugar box", "features": {"weight": 514, "volume": 592, "crush_score": 5, "category": "Pantry", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
        ],
    },
    "bag of green apples": {
        "id": 3, "observations": 28,
        "outcomes": [
            {"fraction": 0.023800000000000043, "perceived_as": None, "features": None},
            {"fraction": 0.6433158, "perceived_as": "bag of green apples", "features": {"weight": 733, "volume": 884, "crush_score": 8, "category": "Produce", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": True, "orientation_sensitive": False}},
            {"fraction": 0.04295279999999999, "perceived_as": "tuna can", "features": {"weight": 171, "volume": 187, "crush_score": 2, "category": "Pantry", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
            {"fraction": 0.0263574, "perceived_as": "bleach bottle", "features": {"weight": 1131, "volume": 1592, "crush_score": 2, "category": "Cleaning", "temperature": "Ambient", "spill_risk": True, "spill_vulnerable": False, "orientation_sensitive": False}},
            {"fraction": 0.0234288, "perceived_as": "coffee can", "features": {"weight": 414, "volume": 1136, "crush_score": 2, "category": "Pantry", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
            {"fraction": 0.0224526, "perceived_as": "cracker box", "features": {"weight": 411, "volume": 1991, "crush_score": 6, "category": "Snacks", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": True, "orientation_sensitive": False}},
            {"fraction": 0.017571599999999996, "perceived_as": "bag of pears", "features": {"weight": 717, "volume": 1376, "crush_score": 8, "category": "Produce", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": True, "orientation_sensitive": False}},
            {"fraction": 0.0136668, "perceived_as": "bag of cherries", "features": {"weight": 33, "volume": 60, "crush_score": 10, "category": "Produce", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": True, "orientation_sensitive": False}},
            {"fraction": 0.0117144, "perceived_as": "sugar box", "features": {"weight": 514, "volume": 592, "crush_score": 5, "category": "Pantry", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
            {"fraction": 0.008785799999999998, "perceived_as": "bag of limes", "features": {"weight": 285, "volume": 624, "crush_score": 8, "category": "Produce", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": True, "orientation_sensitive": False}},
            {"fraction": 0.008785799999999998, "perceived_as": "water bottle", "features": {"weight": 520, "volume": 550, "crush_score": 2, "category": "Pantry", "temperature": "Ambient", "spill_risk": True, "spill_vulnerable": False, "orientation_sensitive": False}},
            {"fraction": 0.008785799999999998, "perceived_as": "wine cup", "features": {"weight": 133, "volume": 852, "crush_score": 8, "category": "Household", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
            {"fraction": 0.0068334, "perceived_as": "bag of strawberries", "features": {"weight": 53, "volume": 332, "crush_score": 10, "category": "Produce", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": True, "orientation_sensitive": False}},
        ],
    },
    "bag of limes": {
        "id": 4, "observations": 12,
        "outcomes": [
            {"fraction": 0.25, "perceived_as": None, "features": None},
            {"fraction": 0.1665, "perceived_as": "bag of limes", "features": {"weight": 285, "volume": 624, "crush_score": 8, "category": "Produce", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": True, "orientation_sensitive": False}},
            {"fraction": 0.1665, "perceived_as": "bag of strawberries", "features": {"weight": 53, "volume": 332, "crush_score": 10, "category": "Produce", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": True, "orientation_sensitive": False}},
            {"fraction": 0.06975, "perceived_as": "tuna can", "features": {"weight": 171, "volume": 187, "crush_score": 2, "category": "Pantry", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
            {"fraction": 0.055499999999999994, "perceived_as": "bag of pears", "features": {"weight": 717, "volume": 1376, "crush_score": 8, "category": "Produce", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": True, "orientation_sensitive": False}},
            {"fraction": 0.027749999999999997, "perceived_as": "wine cup", "features": {"weight": 133, "volume": 852, "crush_score": 8, "category": "Household", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
        ],
    },
    "bag of mangoes": {
        "id": 5, "observations": 10,
        "outcomes": [
            {"fraction": 0.0, "perceived_as": None, "features": None},
            {"fraction": 0.8, "perceived_as": "bag of mangoes", "features": {"weight": 1349, "volume": 1400, "crush_score": 8, "category": "Produce", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": True, "orientation_sensitive": False}},
            {"fraction": 0.05, "perceived_as": "bag of green apples", "features": {"weight": 733, "volume": 884, "crush_score": 8, "category": "Produce", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": True, "orientation_sensitive": False}},
            {"fraction": 0.05, "perceived_as": "tuna can", "features": {"weight": 171, "volume": 187, "crush_score": 2, "category": "Pantry", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
        ],
    },
    "bag of oranges": {
        "id": 6, "observations": 18,
        "outcomes": [
            {"fraction": 0.0, "perceived_as": None, "features": None},
            {"fraction": 0.833, "perceived_as": "bag of oranges", "features": {"weight": 609, "volume": 816, "crush_score": 8, "category": "Produce", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": True, "orientation_sensitive": False}},
            {"fraction": 0.037, "perceived_as": "bag of green apples", "features": {"weight": 733, "volume": 884, "crush_score": 8, "category": "Produce", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": True, "orientation_sensitive": False}},
            {"fraction": 0.037, "perceived_as": "tuna can", "features": {"weight": 171, "volume": 187, "crush_score": 2, "category": "Pantry", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
        ],
    },
    "bag of pears": {
        "id": 7, "observations": 13,
        "outcomes": [
            {"fraction": 0.0, "perceived_as": None, "features": None},
            {"fraction": 1.0, "perceived_as": "bag of pears", "features": {"weight": 717, "volume": 1376, "crush_score": 8, "category": "Produce", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": True, "orientation_sensitive": False}},
        ],
    },
    "bag of red apples": {
        "id": 8, "observations": 28,
        "outcomes": [
            {"fraction": 1.0, "perceived_as": "bag of red apples", "features": {"weight": 733, "volume": 884, "crush_score": 8, "category": "Produce", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": True, "orientation_sensitive": False}},
        ],
    },
    "bag of strawberries": {
        "id": 9, "observations": 12,
        "outcomes": [
            {"fraction": 0.0, "perceived_as": None, "features": None},
            {"fraction": 0.75, "perceived_as": "bag of strawberries", "features": {"weight": 53, "volume": 332, "crush_score": 10, "category": "Produce", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": True, "orientation_sensitive": False}},
            {"fraction": 0.062, "perceived_as": "bag of limes", "features": {"weight": 285, "volume": 624, "crush_score": 8, "category": "Produce", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": True, "orientation_sensitive": False}},
            {"fraction": 0.042, "perceived_as": "coffee can", "features": {"weight": 414, "volume": 1136, "crush_score": 2, "category": "Pantry", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
            {"fraction": 0.021, "perceived_as": "tuna can", "features": {"weight": 171, "volume": 187, "crush_score": 2, "category": "Pantry", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
            {"fraction": 0.021, "perceived_as": "bag of cherries", "features": {"weight": 33, "volume": 60, "crush_score": 10, "category": "Produce", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": True, "orientation_sensitive": False}},
        ],
    },
    "bag of yellow lemons": {
        "id": 10, "observations": 11,
        "outcomes": [
            {"fraction": 0.0, "perceived_as": None, "features": None},
            {"fraction": 0.818, "perceived_as": "bag of yellow lemons", "features": {"weight": 485, "volume": 624, "crush_score": 8, "category": "Produce", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": True, "orientation_sensitive": False}},
            {"fraction": 0.091, "perceived_as": "bag of strawberries", "features": {"weight": 53, "volume": 332, "crush_score": 10, "category": "Produce", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": True, "orientation_sensitive": False}},
        ],
    },
    "bleach bottle": {
        "id": 11, "observations": 21,
        "outcomes": [
            {"fraction": 0.0, "perceived_as": None, "features": None},
            {"fraction": 1.0, "perceived_as": "bleach bottle", "features": {"weight": 1131, "volume": 1592, "crush_score": 2, "category": "Cleaning", "temperature": "Ambient", "spill_risk": True, "spill_vulnerable": False, "orientation_sensitive": False}},
        ],
    },
    "bowl": {
        "id": 12, "observations": 30,
        "outcomes": [
            {"fraction": 0.016700000000000048, "perceived_as": None, "features": None},
            {"fraction": 0.9331516999999999, "perceived_as": "bowl", "features": {"weight": 147, "volume": 1052, "crush_score": 2, "category": "Household", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
            {"fraction": 0.0137662, "perceived_as": "coffee can", "features": {"weight": 414, "volume": 1136, "crush_score": 2, "category": "Pantry", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
            {"fraction": 0.007866399999999999, "perceived_as": "cracker box", "features": {"weight": 411, "volume": 1991, "crush_score": 6, "category": "Snacks", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": True, "orientation_sensitive": False}},
            {"fraction": 0.007866399999999999, "perceived_as": "tuna can", "features": {"weight": 171, "volume": 187, "crush_score": 2, "category": "Pantry", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
            {"fraction": 0.0068831, "perceived_as": "bag of strawberries", "features": {"weight": 53, "volume": 332, "crush_score": 10, "category": "Produce", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": True, "orientation_sensitive": False}},
        ],
    },
    "chocolate pudding box": {
        "id": 13, "observations": 25,
        "outcomes": [
            {"fraction": 0.13329999999999997, "perceived_as": None, "features": None},
            {"fraction": 0.7999641000000001, "perceived_as": "chocolate pudding box", "features": {"weight": 187, "volume": 342, "crush_score": 6, "category": "Snacks", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": True, "orientation_sensitive": False}},
            {"fraction": 0.0234009, "perceived_as": "cracker box", "features": {"weight": 411, "volume": 1991, "crush_score": 6, "category": "Snacks", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": True, "orientation_sensitive": False}},
            {"fraction": 0.0130005, "perceived_as": "sugar box", "features": {"weight": 514, "volume": 592, "crush_score": 5, "category": "Pantry", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
            {"fraction": 0.0104004, "perceived_as": "tuna can", "features": {"weight": 171, "volume": 187, "crush_score": 2, "category": "Pantry", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
        ],
    },
    "coffee can": {
        "id": 14, "observations": 126,
        "outcomes": [
            {"fraction": 0.008700000000000041, "perceived_as": None, "features": None},
            {"fraction": 0.9367785, "perceived_as": "coffee can", "features": {"weight": 414, "volume": 1136, "crush_score": 2, "category": "Pantry", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
            {"fraction": 0.0317216, "perceived_as": "wine cup", "features": {"weight": 133, "volume": 852, "crush_score": 8, "category": "Household", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
            {"fraction": 0.0079304, "perceived_as": "tuna can", "features": {"weight": 171, "volume": 187, "crush_score": 2, "category": "Pantry", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
        ],
    },
    "cola bottle": {
        "id": 15, "observations": 36,
        "outcomes": [
            {"fraction": 0.0, "perceived_as": None, "features": None},
            {"fraction": 1.0, "perceived_as": "cola bottle", "features": {"weight": 545, "volume": 550, "crush_score": 2, "category": "Pantry", "temperature": "Ambient", "spill_risk": True, "spill_vulnerable": False, "orientation_sensitive": False}},
        ],
    },
    "cracker box": {
        "id": 16, "observations": 216,
        "outcomes": [
            {"fraction": 0.035599999999999965, "perceived_as": None, "features": None},
            {"fraction": 0.8978564000000001, "perceived_as": "cracker box", "features": {"weight": 411, "volume": 1991, "crush_score": 6, "category": "Snacks", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": True, "orientation_sensitive": False}},
            {"fraction": 0.0067508, "perceived_as": "spam can", "features": {"weight": 370, "volume": 398, "crush_score": 2, "category": "Pantry", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
        ],
    },
    "cup": {
        "id": 17, "observations": 24,
        "outcomes": [
            {"fraction": 0.07640000000000002, "perceived_as": None, "features": None},
            {"fraction": 0.8330872, "perceived_as": "cup", "features": {"weight": 118, "volume": 450, "crush_score": 8, "category": "Household", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
            {"fraction": 0.0350968, "perceived_as": "bleach bottle", "features": {"weight": 1131, "volume": 1592, "crush_score": 2, "category": "Cleaning", "temperature": "Ambient", "spill_risk": True, "spill_vulnerable": False, "orientation_sensitive": False}},
            {"fraction": 0.0101596, "perceived_as": "wine cup", "features": {"weight": 133, "volume": 852, "crush_score": 8, "category": "Household", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
        ],
    },
    "egg carton": {
        "id": 18, "observations": 45,
        "outcomes": [
            {"fraction": 0.0, "perceived_as": None, "features": None},
            {"fraction": 1.0, "perceived_as": "egg carton", "features": {"weight": 385, "volume": 1300, "crush_score": 9, "category": "Dairy", "temperature": "Refrigerated", "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
        ],
    },
    "energy drink can": {
        "id": 19, "observations": 35,
        "outcomes": [
            {"fraction": 0.0, "perceived_as": None, "features": None},
            {"fraction": 0.914, "perceived_as": "energy drink can", "features": {"weight": 270, "volume": 290, "crush_score": 2, "category": "Pantry", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
            {"fraction": 0.014, "perceived_as": "chocolate pudding box", "features": {"weight": 187, "volume": 342, "crush_score": 6, "category": "Snacks", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": True, "orientation_sensitive": False}},
            {"fraction": 0.007, "perceived_as": "cracker box", "features": {"weight": 411, "volume": 1991, "crush_score": 6, "category": "Snacks", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": True, "orientation_sensitive": False}},
        ],
    },
    "gelatin dessert box": {
        "id": 20, "observations": 25,
        "outcomes": [
            {"fraction": 0.2267, "perceived_as": None, "features": None},
            {"fraction": 0.44000769999999995, "perceived_as": "gelatin dessert box", "features": {"weight": 97, "volume": 174, "crush_score": 6, "category": "Snacks", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": True, "orientation_sensitive": False}},
            {"fraction": 0.0703703, "perceived_as": "cracker box", "features": {"weight": 411, "volume": 1991, "crush_score": 6, "category": "Snacks", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": True, "orientation_sensitive": False}},
            {"fraction": 0.0533577, "perceived_as": "bleach bottle", "features": {"weight": 1131, "volume": 1592, "crush_score": 2, "category": "Cleaning", "temperature": "Ambient", "spill_risk": True, "spill_vulnerable": False, "orientation_sensitive": False}},
            {"fraction": 0.046397999999999995, "perceived_as": "chocolate pudding box", "features": {"weight": 187, "volume": 342, "crush_score": 6, "category": "Snacks", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": True, "orientation_sensitive": False}},
            {"fraction": 0.0131461, "perceived_as": "glass cleaner spray bottle", "features": {"weight": 1022, "volume": 2268, "crush_score": 2, "category": "Cleaning", "temperature": "Ambient", "spill_risk": True, "spill_vulnerable": False, "orientation_sensitive": False}},
            {"fraction": 0.0131461, "perceived_as": "sugar box", "features": {"weight": 514, "volume": 592, "crush_score": 5, "category": "Pantry", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
            {"fraction": 0.0100529, "perceived_as": "tuna can", "features": {"weight": 171, "volume": 187, "crush_score": 2, "category": "Pantry", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
        ],
    },
    "glass cleaner spray bottle": {
        "id": 21, "observations": 21,
        "outcomes": [
            {"fraction": 0.0, "perceived_as": None, "features": None},
            {"fraction": 1.0, "perceived_as": "glass cleaner spray bottle", "features": {"weight": 1022, "volume": 2268, "crush_score": 2, "category": "Cleaning", "temperature": "Ambient", "spill_risk": True, "spill_vulnerable": False, "orientation_sensitive": False}},
        ],
    },
    "kitchen sponge": {
        "id": 22, "observations": 25,
        "outcomes": [
            {"fraction": 0.10329999999999995, "perceived_as": None, "features": None},
            {"fraction": 0.3201219, "perceived_as": "kitchen sponge", "features": {"weight": 12, "volume": 160, "crush_score": 5, "category": "Household", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
            {"fraction": 0.0564921, "perceived_as": "bleach bottle", "features": {"weight": 1131, "volume": 1592, "crush_score": 2, "category": "Cleaning", "temperature": "Ambient", "spill_risk": True, "spill_vulnerable": False, "orientation_sensitive": False}},
            {"fraction": 0.0457317, "perceived_as": "coffee can", "features": {"weight": 414, "volume": 1136, "crush_score": 2, "category": "Pantry", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
            {"fraction": 0.0403515, "perceived_as": "sugar box", "features": {"weight": 514, "volume": 592, "crush_score": 5, "category": "Pantry", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
            {"fraction": 0.0403515, "perceived_as": "bag of green apples", "features": {"weight": 733, "volume": 884, "crush_score": 8, "category": "Produce", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": True, "orientation_sensitive": False}},
            {"fraction": 0.0331779, "perceived_as": "gelatin dessert box", "features": {"weight": 97, "volume": 174, "crush_score": 6, "category": "Snacks", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": True, "orientation_sensitive": False}},
            {"fraction": 0.0197274, "perceived_as": "mustard bottle", "features": {"weight": 603, "volume": 744, "crush_score": 2, "category": "Pantry", "temperature": "Ambient", "spill_risk": True, "spill_vulnerable": False, "orientation_sensitive": False}},
            {"fraction": 0.0197274, "perceived_as": "glass cleaner spray bottle", "features": {"weight": 1022, "volume": 2268, "crush_score": 2, "category": "Cleaning", "temperature": "Ambient", "spill_risk": True, "spill_vulnerable": False, "orientation_sensitive": False}},
            {"fraction": 0.015243900000000001, "perceived_as": "cracker box", "features": {"weight": 411, "volume": 1991, "crush_score": 6, "category": "Snacks", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": True, "orientation_sensitive": False}},
            {"fraction": 0.015243900000000001, "perceived_as": "tuna can", "features": {"weight": 171, "volume": 187, "crush_score": 2, "category": "Pantry", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
            {"fraction": 0.015243900000000001, "perceived_as": "bag of cherries", "features": {"weight": 33, "volume": 60, "crush_score": 10, "category": "Produce", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": True, "orientation_sensitive": False}},
            {"fraction": 0.0098637, "perceived_as": "tomato soup can", "features": {"weight": 349, "volume": 346, "crush_score": 2, "category": "Pantry", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
            {"fraction": 0.0098637, "perceived_as": "bag of limes", "features": {"weight": 285, "volume": 624, "crush_score": 8, "category": "Produce", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": True, "orientation_sensitive": False}},
            {"fraction": 0.0098637, "perceived_as": "wine cup", "features": {"weight": 133, "volume": 852, "crush_score": 8, "category": "Household", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
            {"fraction": 0.0098637, "perceived_as": "bag of pears", "features": {"weight": 717, "volume": 1376, "crush_score": 8, "category": "Produce", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": True, "orientation_sensitive": False}},
            {"fraction": 0.008070299999999999, "perceived_as": "bag of strawberries", "features": {"weight": 53, "volume": 332, "crush_score": 10, "category": "Produce", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": True, "orientation_sensitive": False}},
        ],
    },
    "mustard bottle": {
        "id": 23, "observations": 21,
        "outcomes": [
            {"fraction": 0.0, "perceived_as": None, "features": None},
            {"fraction": 1.0, "perceived_as": "mustard bottle", "features": {"weight": 603, "volume": 744, "crush_score": 2, "category": "Pantry", "temperature": "Ambient", "spill_risk": True, "spill_vulnerable": False, "orientation_sensitive": False}},
        ],
    },
    "plate": {
        "id": 24, "observations": 25,
        "outcomes": [
            {"fraction": 0.0, "perceived_as": None, "features": None},
            {"fraction": 1.0, "perceived_as": "plate", "features": {"weight": 279, "volume": 1255, "crush_score": 2, "category": "Household", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
        ],
    },
    "potato chip can": {
        "id": 25, "observations": 38,
        "outcomes": [
            {"fraction": 0.0, "perceived_as": None, "features": None},
            {"fraction": 0.974, "perceived_as": "potato chip can", "features": {"weight": 205, "volume": 1104, "crush_score": 2, "category": "Snacks", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": True, "orientation_sensitive": False}},
            {"fraction": 0.009, "perceived_as": "tuna can", "features": {"weight": 171, "volume": 187, "crush_score": 2, "category": "Pantry", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
            {"fraction": 0.009, "perceived_as": "coffee can", "features": {"weight": 414, "volume": 1136, "crush_score": 2, "category": "Pantry", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
        ],
    },
    "spam can": {
        "id": 26, "observations": 95,
        "outcomes": [
            {"fraction": 0.006299999999999972, "perceived_as": None, "features": None},
            {"fraction": 0.9579268, "perceived_as": "spam can", "features": {"weight": 370, "volume": 398, "crush_score": 2, "category": "Pantry", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
            {"fraction": 0.0049685, "perceived_as": "tomato soup can", "features": {"weight": 349, "volume": 346, "crush_score": 2, "category": "Pantry", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
            {"fraction": 0.0049685, "perceived_as": "bag of green apples", "features": {"weight": 733, "volume": 884, "crush_score": 8, "category": "Produce", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": True, "orientation_sensitive": False}},
            {"fraction": 0.0049685, "perceived_as": "wine cup", "features": {"weight": 133, "volume": 852, "crush_score": 8, "category": "Household", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
        ],
    },
    "sugar box": {
        "id": 27, "observations": 190,
        "outcomes": [
            {"fraction": 0.062000000000000055, "perceived_as": None, "features": None},
            {"fraction": 0.85827, "perceived_as": "sugar box", "features": {"weight": 514, "volume": 592, "crush_score": 5, "category": "Pantry", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
            {"fraction": 0.011256, "perceived_as": "cracker box", "features": {"weight": 411, "volume": 1991, "crush_score": 6, "category": "Snacks", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": True, "orientation_sensitive": False}},
            {"fraction": 0.007503999999999999, "perceived_as": "tuna can", "features": {"weight": 171, "volume": 187, "crush_score": 2, "category": "Pantry", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
        ],
    },
    "tomato soup can": {
        "id": 28, "observations": 84,
        "outcomes": [
            {"fraction": 0.0, "perceived_as": None, "features": None},
            {"fraction": 0.893, "perceived_as": "tomato soup can", "features": {"weight": 349, "volume": 346, "crush_score": 2, "category": "Pantry", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
            {"fraction": 0.012, "perceived_as": "spam can", "features": {"weight": 370, "volume": 398, "crush_score": 2, "category": "Pantry", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
            {"fraction": 0.008, "perceived_as": "wine cup", "features": {"weight": 133, "volume": 852, "crush_score": 8, "category": "Household", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
            {"fraction": 0.006, "perceived_as": "bag of strawberries", "features": {"weight": 53, "volume": 332, "crush_score": 10, "category": "Produce", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": True, "orientation_sensitive": False}},
            {"fraction": 0.006, "perceived_as": "cracker box", "features": {"weight": 411, "volume": 1991, "crush_score": 6, "category": "Snacks", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": True, "orientation_sensitive": False}},
        ],
    },
    "toothpaste box": {
        "id": 29, "observations": 25,
        "outcomes": [
            {"fraction": 0.0, "perceived_as": None, "features": None},
            {"fraction": 0.88, "perceived_as": "toothpaste box", "features": {"weight": 125, "volume": 300, "crush_score": 5, "category": "Cleaning", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
            {"fraction": 0.068, "perceived_as": "cracker box", "features": {"weight": 411, "volume": 1991, "crush_score": 6, "category": "Snacks", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": True, "orientation_sensitive": False}},
            {"fraction": 0.008, "perceived_as": "tuna can", "features": {"weight": 171, "volume": 187, "crush_score": 2, "category": "Pantry", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
            {"fraction": 0.008, "perceived_as": "gelatin dessert box", "features": {"weight": 97, "volume": 174, "crush_score": 6, "category": "Snacks", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": True, "orientation_sensitive": False}},
        ],
    },
    "tuna can": {
        "id": 30, "observations": 62,
        "outcomes": [
            {"fraction": 0.028200000000000003, "perceived_as": None, "features": None},
            {"fraction": 0.7900733999999999, "perceived_as": "tuna can", "features": {"weight": 171, "volume": 187, "crush_score": 2, "category": "Pantry", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
            {"fraction": 0.017492399999999998, "perceived_as": "coffee can", "features": {"weight": 414, "volume": 1136, "crush_score": 2, "category": "Pantry", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
            {"fraction": 0.0165206, "perceived_as": "spam can", "features": {"weight": 370, "volume": 398, "crush_score": 2, "category": "Pantry", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
            {"fraction": 0.0116616, "perceived_as": "tomato soup can", "features": {"weight": 349, "volume": 346, "crush_score": 2, "category": "Pantry", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
            {"fraction": 0.0077744, "perceived_as": "glass cleaner spray bottle", "features": {"weight": 1022, "volume": 2268, "crush_score": 2, "category": "Cleaning", "temperature": "Ambient", "spill_risk": True, "spill_vulnerable": False, "orientation_sensitive": False}},
            {"fraction": 0.0077744, "perceived_as": "chocolate pudding box", "features": {"weight": 187, "volume": 342, "crush_score": 6, "category": "Snacks", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": True, "orientation_sensitive": False}},
            {"fraction": 0.0058308, "perceived_as": "gelatin dessert box", "features": {"weight": 97, "volume": 174, "crush_score": 6, "category": "Snacks", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": True, "orientation_sensitive": False}},
        ],
    },
    "water bottle": {
        "id": 31, "observations": 30,
        "outcomes": [
            {"fraction": 0.0, "perceived_as": None, "features": None},
            {"fraction": 1.0, "perceived_as": "water bottle", "features": {"weight": 520, "volume": 550, "crush_score": 2, "category": "Pantry", "temperature": "Ambient", "spill_risk": True, "spill_vulnerable": False, "orientation_sensitive": False}},
        ],
    },
    "wine cup": {
        "id": 32, "observations": 16,
        "outcomes": [
            {"fraction": 0.031200000000000006, "perceived_as": None, "features": None},
            {"fraction": 0.8128232, "perceived_as": "wine cup", "features": {"weight": 133, "volume": 852, "crush_score": 8, "category": "Household", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
            {"fraction": 0.0939736, "perceived_as": "tomato soup can", "features": {"weight": 349, "volume": 346, "crush_score": 2, "category": "Pantry", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
            {"fraction": 0.0310016, "perceived_as": "gelatin dessert box", "features": {"weight": 97, "volume": 174, "crush_score": 6, "category": "Snacks", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": True, "orientation_sensitive": False}},
            {"fraction": 0.0155008, "perceived_as": "cracker box", "features": {"weight": 411, "volume": 1991, "crush_score": 6, "category": "Snacks", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": True, "orientation_sensitive": False}},
            {"fraction": 0.0155008, "perceived_as": "tuna can", "features": {"weight": 171, "volume": 187, "crush_score": 2, "category": "Pantry", "temperature": "Ambient", "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
        ],
    },
}

NOISY_ITEMS = [
    {"name": name, "id": data["id"], "weight": data["observations"], "outcomes": data["outcomes"]}
    for name, data in FEATURE_DISTRIBUTION.items()
]
_NOISY_WEIGHTS = [e["weight"] for e in NOISY_ITEMS]

# Bayes-inverted posterior P(true_item | perceived_as), used to resolve the
# hidden true item lazily (in step(), at placement time) instead of pairing
# it with the observation at draw time. P(perceived|true) is the outcome's
# fraction; P(true) is the same "observations"-count prior NOISY_ITEMS uses.
# perceived label -> list of (true_id, unnormalized weight = fraction * observations)
_PERCEIVED_CANDIDATES: dict = {}
for _entry in NOISY_ITEMS:
    for _outcome in _entry["outcomes"]:
        _label = _outcome["perceived_as"]
        if _label is None:
            continue
        _PERCEIVED_CANDIDATES.setdefault(_label, []).append(
            (_entry["id"], _outcome["fraction"] * _entry["weight"])
        )


def _sample_true_item(obs_item):
    """Sample a true Item from P(true | observed=obs_item.name) via Bayes' rule.
    Falls back to true == observed when obs_item.name isn't in the noise-model
    catalog (e.g. real VLM display names not among the ~17 hardcoded labels)."""
    candidates = _PERCEIVED_CANDIDATES.get(obs_item.name)
    if candidates is None:
        return obs_item
    ids = [c[0] for c in candidates]
    weights = [c[1] for c in candidates]
    true_id = random.choices(ids, weights=weights, k=1)[0]
    return _ID_TO_ITEM[true_id]


class BinPackingEnv(Env):
    def __init__(self, num_bags=5, num_items_to_pack=5, gamma=0.99, n_visible_items=5, p_double_draw=0.2, p_zero_draw=0.1, debug=False):
        # We don't call super().__init__(mdp, H) because we don't use an explicit MDP dict.
        self.num_bags = num_bags
        self.num_items_to_pack = num_items_to_pack
        self.gamma = gamma
        self.H = num_items_to_pack  # Horizon is exactly the number of items to pack
        self.n_visible_items = n_visible_items
        self.p_double_draw = p_double_draw
        self.p_zero_draw = p_zero_draw
        self.items_data = ITEMS_DATA
        self.debug = debug
        self.debug_items = DEBUG_ITEMS_DATA

    def available_actions(self, state):
        # Action a encodes: item_idx = a // num_bags, bag_idx = a % num_bags
        # Exclude slots occupied by the placeholder (no item left to reveal there).
        return [
            item_idx * self.num_bags + bag_idx
            for item_idx, item in enumerate(state["visible_items"])
            if item.id != 0
            for bag_idx in range(self.num_bags)
        ]

    def _get_rl_vector(self, observed_bags, visible_items):
        vec = []
        for item in visible_items:
            vec.extend(item.get_features()[:8])
        for bag in observed_bags:
            vec.extend(bag.get_observed_features())
        return tuple(vec)

    def _draw_item(self, slot_idx, t):
        """Return (true_item_or_None, observed_item) for a new visible slot.

        Observed item is built from the per-outcome features in FEATURE_DISTRIBUTION:
        different perceived classifications yield different estimated feature vectors.
        NOT_DETECTED outcomes (features=None) are ignored — resampled until a detected
        classification comes back, so the item is always revealed to the agent.

        The true item is left unresolved (None) here: the true-first ancestral draw
        below (entry ~ NOISY_ITEMS prior, outcome ~ fraction likelihood) is only used
        to get the correct marginal distribution over observed labels — the specific
        true item it happened to pass through is discarded rather than paired with
        this observation. Identity is instead resampled fresh from the Bayes-inverted
        posterior P(true|observed) every time step() actually places this slot (see
        _sample_true_item), so independent MCTS rollouts that reach the same node see
        independent, plausible real states instead of one true item shared by all of
        them. Debug mode is unaffected: true == observed, deterministic, no ambiguity.
        """
        if self.debug:
            idx = (t + slot_idx) % len(self.debug_items)
            item = self.debug_items[idx]
            return item, item
        entry = random.choices(NOISY_ITEMS, weights=_NOISY_WEIGHTS, k=1)[0]
        fracs = [o["fraction"] for o in entry["outcomes"]]
        outcome = random.choices(entry["outcomes"], weights=fracs, k=1)[0]
        while outcome["features"] is None:
            outcome = random.choices(entry["outcomes"], weights=fracs, k=1)[0]
        f = outcome["features"]
        # Note: id here is just the id of whichever true item generated this marginal
        # draw — it is NOT necessarily the true id step() will later resolve to.
        # Harmless: id/name aren't part of Item.get_features() (the RL vector), only
        # used for placeholder detection (id != 0) and debug printing.
        obs_item = Item(
            entry["id"], outcome["perceived_as"],
            f["weight"], f["volume"], f["crush_score"],
            f["category"], f["temperature"],
            f["spill_risk"], f["spill_vulnerable"], f["orientation_sensitive"],
        )
        return None, obs_item

    def sample_initial_state(self):
        bags = tuple(Bag() for _ in range(self.num_bags))
        observed_bags = tuple(Bag() for _ in range(self.num_bags))
        if self.debug:
            n_initial = min(self.n_visible_items, len(self.debug_items), self.num_items_to_pack)
        else:
            # Poisson(5) gives a mean of ~5 visible items regardless of window size.
            n_initial = int(np.clip(np.random.poisson(5), 1, min(self.n_visible_items, self.num_items_to_pack)))
        # Retry until at least one slot is visible — NOT_DETECTED on all draws would
        # leave the agent with no available actions at t=0.
        while True:
            true_items = []
            visible_items = []
            for i in range(self.n_visible_items):
                if i < n_initial:
                    true_item, observed_item = self._draw_item(i, 0)
                else:
                    true_item, observed_item = PLACEHOLDER_ITEM, PLACEHOLDER_ITEM
                true_items.append(true_item)
                visible_items.append(observed_item)
            if any(it.id != 0 for it in visible_items):
                break
        true_items = tuple(true_items)
        visible_items = tuple(visible_items)
        n_detected = sum(1 for it in visible_items if it.id != 0)
        extended_state = {
            "state": self._get_rl_vector(observed_bags, visible_items),
            "bags": bags,
            "observed_bags": observed_bags,
            "visible_items": visible_items,
            "true_items": true_items,
            "items_remaining": self.num_items_to_pack - n_detected,
            "t": 0
        }
        return extended_state

    def step(self, extended_state, a):
        t = extended_state["t"]
        bags = list(bag.copy() for bag in extended_state["bags"])
        observed_bags = list(bag.copy() for bag in extended_state["observed_bags"])
        visible_items = list(extended_state["visible_items"])
        true_items = list(extended_state["true_items"])

        # Decode action: which visible item to place, and in which bag
        item_idx = a // self.num_bags
        bag_idx  = a % self.num_bags

        obs_item = visible_items[item_idx]  # perceived item drives observed_bags
        pinned_true = true_items[item_idx]
        # Physics driven by the hidden true item: use it if already pinned (debug
        # mode, or a scripted draw from a subclass like FixedItemsBinPackingEnv);
        # otherwise resample fresh from the posterior P(true|observed) right now,
        # so different rollouts through this same node get independent real states.
        item = pinned_true if pinned_true is not None else _sample_true_item(obs_item)
        # Copy immediately: both `pinned_true` (e.g. a scripted true-item queue
        # entry) and _sample_true_item's return value (always _ID_TO_ITEM[id],
        # a SHARED catalog object) can be direct references, not private
        # copies. Every existing block below only ever mutates `packed_item`
        # (already a safe .copy() sitting in bag.items) -- but the placement-
        # time accident blocks (1b, 2c) mutate `item` itself, before block 3's
        # own .copy(). Without this, that mutation would land on the shared
        # global object (e.g. _ID_TO_ITEM's "egg carton" entry), permanently
        # corrupting every future draw of that item for the rest of the
        # process -- confirmed as a real, reproducible bug before this line
        # was added, not just a theoretical risk.
        item = item.copy()
        bag  = bags[bag_idx]
        obs_bag = observed_bags[bag_idx]

        cost_t = 0

        illegal_move = (
            bag.volume_used + item.volume > bag.MAX_VOLUME or
            bag.weight_used + item.weight > bag.MAX_WEIGHT
        )

        if illegal_move:
            cost_t += ILLEGAL_MOVE_COST
        else:
            # ── True bag update (drives physics and cost) ──────────────────
            bag.volume_used += item.volume
            bag.weight_used += item.weight
            bag.total_items += 1
            if bag.total_items == 1:
                cost_t += BAG_OPEN_COST

            # 1. Update crush risk for each fragile item ALREADY in the bag
            for packed_item in bag.items:
                if packed_item.crush_score >= 8 and not packed_item.is_crushed:
                    if item.crush_score <= 2:
                        packed_item.risk_level += item.weight / 500.0
                    else:
                        packed_item.risk_level += item.weight / 1000.0

                    fail_prob = min(packed_item.risk_level * 0.05, 1.0)
                    if np.random.rand() < fail_prob:
                        packed_item.is_crushed = True
                        cost_t += CRUSH_COST

            # 1b. The item being placed, if fragile, can break immediately upon
            # placement -- models mishandling/impact damage at the moment of
            # placement itself, independent of being crushed later by a squeeze
            # from a subsequent item (block 1 above). Setting is_crushed=True
            # here correctly prevents block 1 from crushing it again later
            # (same "not packed_item.is_crushed" guard already in place).
            if item.crush_score >= 8 and np.random.rand() < PLACEMENT_CRUSH_PROB:
                item.is_crushed = True
                cost_t += CRUSH_COST

            # 2. Update spill risk for each spill-risk item ALREADY in the bag
            squeeze_factor = bag.volume_used / Bag.MAX_VOLUME
            for packed_item in bag.items:
                if packed_item.spill_risk and not packed_item.is_spilled:
                    if packed_item.orientation_sensitive:
                        packed_item.spill_risk_level += (item.weight / 500.0) * squeeze_factor
                    else:
                        packed_item.spill_risk_level += (item.weight / 1000.0) * squeeze_factor

                    fail_prob = min(packed_item.spill_risk_level * 0.05, 1.0)
                    if np.random.rand() < fail_prob:
                        packed_item.is_spilled = True
                        bag.spilled_count += 1
                        cost_t += SPILL_COST
                        cost_t += _contamination_cost(bag.items, packed_item)

            # 2b. Retroactive: charge the arriving item for any item already
            # spilled in this bag -- whether that spill happened in an earlier
            # step, or just now in step 2 above (bag.items/is_spilled already
            # reflect this step's spills). Mirrors step 2's contamination from
            # the other direction: there, an existing item gets charged when a
            # NEW spill happens; here, a NEWLY-arriving item gets charged for
            # spills that already happened, so cost doesn't depend on which of
            # the two items happened to arrive first. Spill-vulnerable arrivals
            # get the full (possibly chemical-bonused) treatment; any OTHER
            # arriving item still picks up GENERAL_MESS_COST if the earlier
            # spill was a cleaning product.
            for spilled_item in bag.items:
                if spilled_item.is_spilled:
                    cost_t += _contamination_cost([item], spilled_item)

            # 2c. The item being placed can itself spill immediately, independent
            # of being squeezed by a later item (block 2 above) -- models
            # mishandling at the moment of placement. Runs before block 3 (item
            # not yet appended to bag.items), so the contamination charge below
            # correctly reflects "everything already in the bag" as the
            # affected set, same convention as block 2. Setting is_spilled=True
            # here prevents block 2 from spilling it again in a future step.
            if item.spill_risk and np.random.rand() < PLACEMENT_SPILL_PROB:
                item.is_spilled = True
                bag.spilled_count += 1
                cost_t += SPILL_COST
                cost_t += _contamination_cost(bag.items, item)

            # 3. Add placed item to true bag
            bag.items.append(item.copy())
            if item.crush_score >= 8:
                bag.fragile_count += 1
            if item.spill_risk:
                bag.spill_risk_count += 1
            if item.spill_vulnerable:
                bag.spill_vulnerable_count += 1
            if item.orientation_sensitive:
                bag.orientation_sensitive_count += 1

            # ── Observed bag update (drives state representation) ──────────
            # Mirrors the true update using obs_item properties; no crash/spill events.
            obs_bag.place_observed_item(obs_item)

        # Replace the placed slot, then optionally fill one extra placeholder slot.
        # Three outcomes (when items_remaining > 0):
        #   p_zero_draw              → 0 new items (placed slot becomes placeholder, budget unchanged)
        #   (1-p_zero_draw)*(1-p_double_draw) → 1 new item  (normal draw into placed slot)
        #   (1-p_zero_draw)*p_double_draw      → 2 new items (normal draw + extra placeholder slot)
        #
        # Zero-draw is suppressed when the placed slot is the last visible real item in the window;
        # otherwise the next state would have no available actions despite items remaining.
        next_t = t + 1
        items_remaining = extended_state["items_remaining"]
        next_visible = list(visible_items)
        next_true = list(true_items)
        other_real = sum(1 for i, it in enumerate(next_visible) if i != item_idx and it.id != 0)
        can_zero_draw = other_real > 0
        if items_remaining > 0 and can_zero_draw and random.random() < self.p_zero_draw:
            next_visible[item_idx] = PLACEHOLDER_ITEM
            next_true[item_idx] = PLACEHOLDER_ITEM
        elif items_remaining > 0:
            t_item, o_item = self._draw_item(item_idx, next_t)
            # NOT_DETECTED (placeholder observed): slot goes empty but item is not yet
            # revealed to the agent, so don't consume the budget. Exception: if this
            # is the last visible slot a deadlock would result — force re-draw until detected.
            if o_item.id == 0 and other_real == 0:
                while o_item.id == 0:
                    t_item, o_item = self._draw_item(item_idx, next_t)
            next_true[item_idx] = t_item
            next_visible[item_idx] = o_item
            if o_item.id != 0:
                items_remaining -= 1

            if items_remaining > 0 and random.random() < self.p_double_draw:
                placeholder_slots = [i for i, it in enumerate(next_visible) if it.id == 0]
                if placeholder_slots:
                    extra_slot = random.choice(placeholder_slots)
                    t_item, o_item = self._draw_item(extra_slot, next_t)
                    next_true[extra_slot] = t_item
                    next_visible[extra_slot] = o_item
                    items_remaining -= 1
        else:
            next_visible[item_idx] = PLACEHOLDER_ITEM
            next_true[item_idx] = PLACEHOLDER_ITEM

        next_visible = tuple(next_visible)
        next_true = tuple(next_true)

        next_state = {
            "state": self._get_rl_vector(observed_bags, next_visible),
            "bags": tuple(bags),
            "observed_bags": tuple(observed_bags),
            "visible_items": next_visible,
            "true_items": next_true,
            "items_remaining": items_remaining,
            "t": next_t
        }

        terminated = next_t >= self.H
        return next_state, cost_t, terminated
