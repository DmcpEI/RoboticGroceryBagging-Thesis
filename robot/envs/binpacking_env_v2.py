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
BAG_OPEN_COST      = 10.0    # charged once when the first item is placed in a previously empty bag

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


# Items from robot_item_catalog_planner_products.json (YCB-calibrated weights/volumes).
# orientation_sensitive = spill_risk, except egg carton which is True despite no spill risk.
ITEMS_DATA = [
    # Item(id, name, est_weight_g, est_volume_cc, crush_score, category, temperature, spill_risk, spill_vulnerable, orientation_sensitive)
    Item(1,  "cracker box",                  411,  1991, 5, "Snacks",   "Ambient",      False, True,  False),
    Item(2,  "sugar box",                    514,   592, 5, "Pantry",   "Ambient",      False, False, False),
    Item(3,  "mustard bottle",               603,   744, 5, "Pantry",   "Ambient",      True,  False, True),
    Item(4,  "water bottle",                 520,   550, 5, "Pantry",   "Ambient",      True,  False, True),
    Item(5,  "cola bottle",                  545,   550, 5, "Pantry",   "Ambient",      True,  False, True),
    Item(6,  "energy drink can",             270,   290, 3, "Pantry",   "Ambient",      False, False, False),
    Item(7,  "gelatin dessert box",           97,   174, 6, "Snacks",   "Ambient",      False, True,  False),
    Item(8,  "tuna can",                     171,   187, 5, "Pantry",   "Ambient",      False, False, False),
    Item(9,  "toothpaste box",               125,   300, 6, "Cleaning", "Ambient",      False, False, False),
    Item(10, "bleach bottle",               1131,  1592, 3, "Cleaning", "Ambient",      True,  False, True),
    Item(11, "glass cleaner spray bottle",  1022,  2268, 3, "Cleaning", "Ambient",      True,  False, True),
    Item(12, "spam can",                     370,   398, 3, "Pantry",   "Ambient",      False, False, False),
    Item(13, "coffee can",                   414,  1136, 3, "Pantry",   "Ambient",      False, False, False),
    Item(14, "tomato soup can",              349,   346, 3, "Pantry",   "Ambient",      False, False, False),
    Item(15, "potato chip can",              205,  1104, 5, "Snacks",   "Ambient",      False, True,  False),
    Item(16, "egg carton",                   385,  1300, 9, "Dairy",    "Refrigerated", False, False, True),
    Item(17, "chocolate pudding box",        187,   342, 6, "Snacks",   "Ambient",      False, True,  False),
]

# --- OLD ITEMS_DATA (robot_item_catalog_planner.json, rough estimates) ---
# # Items from robot_item_catalog_planner.json. orientation_sensitive derived as = spill_risk,
# # except egg carton which is orientation-sensitive despite no spill risk.
# ITEMS_DATA = [
#     # Item(id, name, est_weight_g, est_volume_cc, crush_score, category, temperature, spill_risk, spill_vulnerable, orientation_sensitive)
#     Item(1,  "cracker box",                 800, 1000, 5, "Snacks",    "Ambient",      False, True,  False),
#     Item(2,  "cracker box",                 800, 1000, 5, "Snacks",    "Ambient",      False, True,  False),
#     Item(3,  "cracker box",                 800, 1000, 5, "Snacks",    "Ambient",      False, True,  False),
#     Item(4,  "sugar box",                   800, 1000, 5, "Pantry",    "Ambient",      False, False, False),
#     Item(5,  "sugar box",                   800, 1000, 5, "Pantry",    "Ambient",      False, False, False),
#     Item(6,  "sugar box",                   800, 1000, 5, "Pantry",    "Ambient",      False, False, False),
#     Item(7,  "mustard bottle",              800,  750, 5, "Pantry",    "Ambient",      True,  False, True),
#     Item(8,  "hammer",                     2500, 4000, 1, "Hardware",  "Ambient",      False, False, False),
#     Item(9,  "toothbrush",                  200,  350, 6, "Cleaning",  "Ambient",      False, False, False),
#     Item(10, "water bottle",                800,  750, 5, "Pantry",    "Ambient",      True,  False, True),
#     Item(11, "cola bottle",                 800,  750, 5, "Pantry",    "Ambient",      True,  False, True),
#     Item(12, "energy drink can",            800,  400, 3, "Pantry",    "Ambient",      False, False, False),
#     Item(13, "fork",                        200,  350, 5, "Household", "Ambient",      False, False, False),
#     Item(14, "spoon",                       200,  350, 5, "Household", "Ambient",      False, False, False),
#     Item(15, "knife",                       200,  350, 5, "Household", "Ambient",      False, False, False),
#     Item(16, "gelatin dessert box",         200,  400, 6, "Snacks",    "Ambient",      False, True,  False),
#     Item(17, "gelatin dessert box",         200,  400, 6, "Snacks",    "Ambient",      False, True,  False),
#     Item(18, "kitchen sponge",              200,  350, 7, "Household", "Ambient",      False, False, False),
#     Item(19, "mango",                       800, 1200, 9, "Produce",   "Ambient",      False, True,  False),
#     Item(20, "tuna can",                    200,  200, 5, "Pantry",    "Ambient",      False, False, False),
#     Item(21, "tuna can",                    200,  200, 5, "Pantry",    "Ambient",      False, False, False),
#     Item(22, "cherries",                    200,  350, 9, "Produce",   "Ambient",      False, True,  False),
#     Item(23, "banana",                      200,  350, 9, "Produce",   "Ambient",      False, True,  False),
#     Item(24, "banana",                      200,  350, 9, "Produce",   "Ambient",      False, True,  False),
#     Item(25, "banana",                      200,  350, 9, "Produce",   "Ambient",      False, True,  False),
#     Item(26, "orange",                      200,  350, 9, "Produce",   "Ambient",      False, True,  False),
#     Item(27, "orange",                      200,  350, 9, "Produce",   "Ambient",      False, True,  False),
#     Item(28, "red apple",                   200,  350, 9, "Produce",   "Ambient",      False, True,  False),
#     Item(29, "red apple",                   200,  350, 9, "Produce",   "Ambient",      False, True,  False),
#     Item(30, "green apple",                 200,  350, 9, "Produce",   "Ambient",      False, True,  False),
#     Item(31, "yellow lemon",                200,  350, 9, "Produce",   "Ambient",      False, True,  False),
#     Item(32, "yellow lemon",                200,  350, 9, "Produce",   "Ambient",      False, True,  False),
#     Item(33, "green lemon",                 200,  350, 9, "Produce",   "Ambient",      False, True,  False),
#     Item(34, "strawberry",                  200,  350, 9, "Produce",   "Ambient",      False, True,  False),
#     Item(35, "strawberry",                  200,  350, 9, "Produce",   "Ambient",      False, True,  False),
#     Item(36, "pear",                        200,  350, 9, "Produce",   "Ambient",      False, True,  False),
#     Item(37, "pear",                        200,  350, 9, "Produce",   "Ambient",      False, True,  False),
#     Item(38, "baseball",                    200,  350, 5, "Household", "Ambient",      False, False, False),
#     Item(39, "sponge ball",                 200,  350, 7, "Household", "Ambient",      False, False, False),
#     Item(40, "rubber ball",                 200,  350, 7, "Household", "Ambient",      False, False, False),
#     Item(41, "toothpaste box",              200,  400, 6, "Cleaning",  "Ambient",      False, False, False),
#     Item(42, "bowl",                        800, 1200, 9, "Household", "Ambient",      False, False, False),
#     Item(43, "cup",                         200,  350, 9, "Household", "Ambient",      False, False, False),
#     Item(44, "bleach bottle",              2500, 2000, 3, "Cleaning",  "Ambient",      True,  False, True),
#     Item(45, "glass cleaner spray bottle",  800,  750, 5, "Cleaning",  "Ambient",      True,  False, True),
#     Item(46, "electric screwdriver",       2500, 4000, 1, "Hardware",  "Ambient",      False, False, False),
#     Item(47, "spam can",                    200,  200, 5, "Pantry",    "Ambient",      False, False, False),
#     Item(48, "spam can",                    200,  200, 5, "Pantry",    "Ambient",      False, False, False),
#     Item(49, "coffee can",                  800,  400, 3, "Pantry",    "Ambient",      False, False, False),
#     Item(50, "coffee can",                  800,  400, 3, "Pantry",    "Ambient",      False, False, False),
#     Item(51, "tomato soup can",             800,  400, 3, "Pantry",    "Ambient",      False, False, False),
#     Item(52, "tomato soup can",             800,  400, 3, "Pantry",    "Ambient",      False, False, False),
#     Item(53, "dog chew toy",                200,  350, 7, "Household", "Ambient",      False, False, False),
#     Item(54, "wine cup",                    200,  350, 9, "Household", "Ambient",      False, False, False),
#     Item(55, "golf ball",                   200,  350, 5, "Household", "Ambient",      False, False, False),
#     Item(56, "potato chip can",             200,  200, 5, "Snacks",    "Ambient",      False, True,  False),
#     Item(57, "egg carton",                  200,  300, 9, "Dairy",     "Refrigerated", False, False, True),
#     Item(58, "plate",                       200,  300, 9, "Household", "Ambient",      False, False, False),
#     Item(59, "chocolate pudding box",       200,  400, 6, "Snacks",    "Ambient",      False, True,  False),
# ]

DEBUG_ITEMS_DATA = [
    ITEMS_DATA[15],  # egg carton (id=16)
    ITEMS_DATA[9],   # bleach bottle (id=10)
    ITEMS_DATA[10],  # glass cleaner spray bottle (id=11)
    ITEMS_DATA[0],   # cracker box (id=1)
]

# Sentinel for empty visible slots when all items for the episode have been revealed.
PLACEHOLDER_ITEM = Item(0, "placeholder", 0, 0, 0, "Household", "Ambient", False, False, False)

# Fast id → Item lookup used by the observation model.
_ID_TO_ITEM = {item.id: item for item in ITEMS_DATA}

# Per-item perception outcomes from scratch/feature_distribution.json.
# For each true item: a list of outcomes with the fraction of observations and the estimated
# planner features the agent sees for that classification. perceived_as=None means NOT_DETECTED
# (item present but invisible to the agent → PLACEHOLDER_ITEM in the visible window).
# orientation_sensitive follows the catalog convention (= spill_risk), except egg carton (True).
FEATURE_DISTRIBUTION = {
    "bleach bottle": {
        "id": 10, "observations": 10,
        "outcomes": [
            {"fraction": 0.5, "perceived_as": None,                 "features": None},
            {"fraction": 0.3, "perceived_as": "bleach bottle",      "features": {"weight": 1131, "volume": 1592, "crush_score": 3, "category": "Cleaning", "temperature": "Ambient",      "spill_risk": True,  "spill_vulnerable": False, "orientation_sensitive": True}},
            {"fraction": 0.1, "perceived_as": "mustard bottle",     "features": {"weight": 603,  "volume": 744,  "crush_score": 5, "category": "Pantry",   "temperature": "Ambient",      "spill_risk": True,  "spill_vulnerable": False, "orientation_sensitive": True}},
            {"fraction": 0.1, "perceived_as": "hand soap bottle",   "features": {"weight": 800,  "volume": 750,  "crush_score": 5, "category": "Cleaning", "temperature": "Ambient",      "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
        ],
    },
    "chocolate pudding box": {
        "id": 17, "observations": 10,
        "outcomes": [
            {"fraction": 1.0, "perceived_as": "chocolate pudding box", "features": {"weight": 187, "volume": 342,  "crush_score": 6, "category": "Snacks",   "temperature": "Ambient",      "spill_risk": False, "spill_vulnerable": True,  "orientation_sensitive": False}},
        ],
    },
    "coffee can": {
        "id": 13, "observations": 10,
        "outcomes": [
            {"fraction": 0.9, "perceived_as": "coffee can",         "features": {"weight": 414, "volume": 1136, "crush_score": 3, "category": "Pantry",   "temperature": "Ambient",      "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
            {"fraction": 0.1, "perceived_as": None,                 "features": None},
        ],
    },
    "cola bottle": {
        "id": 5, "observations": 5,
        "outcomes": [
            {"fraction": 1.0, "perceived_as": "cola bottle",        "features": {"weight": 545, "volume": 550,  "crush_score": 5, "category": "Pantry",   "temperature": "Ambient",      "spill_risk": True,  "spill_vulnerable": False, "orientation_sensitive": True}},
        ],
    },
    "cracker box": {
        "id": 1, "observations": 10,
        "outcomes": [
            {"fraction": 0.8, "perceived_as": "cracker box",            "features": {"weight": 411, "volume": 1991, "crush_score": 5, "category": "Snacks",   "temperature": "Ambient",      "spill_risk": False, "spill_vulnerable": True,  "orientation_sensitive": False}},
            {"fraction": 0.2, "perceived_as": "chocolate pudding box",  "features": {"weight": 187, "volume": 342,  "crush_score": 6, "category": "Snacks",   "temperature": "Ambient",      "spill_risk": False, "spill_vulnerable": True,  "orientation_sensitive": False}},
        ],
    },
    "egg carton": {
        "id": 16, "observations": 5,
        "outcomes": [
            {"fraction": 1.0, "perceived_as": "egg carton",         "features": {"weight": 385, "volume": 1300, "crush_score": 9, "category": "Dairy",    "temperature": "Refrigerated", "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": True}},
        ],
    },
    "energy drink can": {
        "id": 6, "observations": 5,
        "outcomes": [
            {"fraction": 0.8, "perceived_as": "energy drink can",   "features": {"weight": 270, "volume": 290,  "crush_score": 3, "category": "Pantry",   "temperature": "Ambient",      "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
            {"fraction": 0.2, "perceived_as": None,                 "features": None},
        ],
    },
    "gelatin dessert box": {
        "id": 7, "observations": 5,
        "outcomes": [
            {"fraction": 0.8, "perceived_as": None,                 "features": None},
            {"fraction": 0.2, "perceived_as": "cracker box",        "features": {"weight": 411, "volume": 1991, "crush_score": 5, "category": "Snacks",   "temperature": "Ambient",      "spill_risk": False, "spill_vulnerable": True,  "orientation_sensitive": False}},
        ],
    },
    "glass cleaner spray bottle": {
        "id": 11, "observations": 10,
        "outcomes": [
            {"fraction": 0.5, "perceived_as": None,                          "features": None},
            {"fraction": 0.5, "perceived_as": "glass cleaner spray bottle",  "features": {"weight": 1022, "volume": 2268, "crush_score": 3, "category": "Cleaning", "temperature": "Ambient", "spill_risk": True, "spill_vulnerable": False, "orientation_sensitive": True}},
        ],
    },
    "mustard bottle": {
        "id": 3, "observations": 5,
        "outcomes": [
            {"fraction": 1.0, "perceived_as": "mustard bottle",     "features": {"weight": 603, "volume": 744,  "crush_score": 5, "category": "Pantry",   "temperature": "Ambient",      "spill_risk": True,  "spill_vulnerable": False, "orientation_sensitive": True}},
        ],
    },
    "potato chip can": {
        "id": 15, "observations": 5,
        "outcomes": [
            {"fraction": 0.4, "perceived_as": "coffee can",         "features": {"weight": 414, "volume": 1136, "crush_score": 3, "category": "Pantry",   "temperature": "Ambient",      "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
            {"fraction": 0.4, "perceived_as": None,                 "features": None},
            {"fraction": 0.2, "perceived_as": "tuna can",           "features": {"weight": 171, "volume": 187,  "crush_score": 5, "category": "Pantry",   "temperature": "Ambient",      "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
        ],
    },
    "spam can": {
        "id": 12, "observations": 10,
        "outcomes": [
            {"fraction": 0.6, "perceived_as": "spam can",           "features": {"weight": 370, "volume": 398,  "crush_score": 3, "category": "Pantry",   "temperature": "Ambient",      "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
            {"fraction": 0.3, "perceived_as": None,                 "features": None},
            {"fraction": 0.1, "perceived_as": "tuna can",           "features": {"weight": 171, "volume": 187,  "crush_score": 5, "category": "Pantry",   "temperature": "Ambient",      "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
        ],
    },
    "sugar box": {
        "id": 2, "observations": 10,
        "outcomes": [
            {"fraction": 0.9, "perceived_as": "sugar box",          "features": {"weight": 514, "volume": 592,  "crush_score": 5, "category": "Pantry",   "temperature": "Ambient",      "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
            {"fraction": 0.1, "perceived_as": None,                 "features": None},
        ],
    },
    "tomato soup can": {
        "id": 14, "observations": 5,
        "outcomes": [
            {"fraction": 0.6, "perceived_as": None,                 "features": None},
            {"fraction": 0.2, "perceived_as": "coffee can",         "features": {"weight": 414, "volume": 1136, "crush_score": 3, "category": "Pantry",   "temperature": "Ambient",      "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
            {"fraction": 0.2, "perceived_as": "potato chip can",    "features": {"weight": 205, "volume": 1104, "crush_score": 5, "category": "Snacks",   "temperature": "Ambient",      "spill_risk": False, "spill_vulnerable": True,  "orientation_sensitive": False}},
        ],
    },
    "toothpaste box": {
        "id": 9, "observations": 5,
        "outcomes": [
            {"fraction": 0.8, "perceived_as": "toothpaste box",     "features": {"weight": 125, "volume": 300,  "crush_score": 6, "category": "Cleaning", "temperature": "Ambient",      "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
            {"fraction": 0.2, "perceived_as": None,                 "features": None},
        ],
    },
    "tuna can": {
        "id": 8, "observations": 10,
        "outcomes": [
            {"fraction": 0.6, "perceived_as": None,                 "features": None},
            {"fraction": 0.3, "perceived_as": "energy drink can",   "features": {"weight": 270, "volume": 290,  "crush_score": 3, "category": "Pantry",   "temperature": "Ambient",      "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
            {"fraction": 0.1, "perceived_as": "tuna can",           "features": {"weight": 171, "volume": 187,  "crush_score": 5, "category": "Pantry",   "temperature": "Ambient",      "spill_risk": False, "spill_vulnerable": False, "orientation_sensitive": False}},
        ],
    },
    "water bottle": {
        "id": 4, "observations": 5,
        "outcomes": [
            {"fraction": 1.0, "perceived_as": "water bottle",       "features": {"weight": 520, "volume": 550,  "crush_score": 5, "category": "Pantry",   "temperature": "Ambient",      "spill_risk": True,  "spill_vulnerable": False, "orientation_sensitive": True}},
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
                        contaminated = sum(
                            1 for it in bag.items
                            if it.spill_vulnerable and not it.is_crushed and it is not packed_item
                        )
                        cost_t += CONTAMINATION_COST * contaminated
                        if packed_item.category == "Cleaning":
                            cost_t += CHEMICAL_CONTAMINATION_BONUS * contaminated

            # 2b. If the item being placed is itself spill-vulnerable, charge
            # contamination for any item already spilled in this bag -- whether
            # that spill happened in an earlier step, or just now in step 2
            # above (bag.items/is_spilled already reflect this step's spills).
            # Mirrors step 2's contamination from the other direction: there,
            # an existing vulnerable item gets charged when a NEW spill
            # happens; here, a NEWLY-arriving vulnerable item gets charged for
            # spills that already happened, so the cost doesn't depend on
            # which of the two items happened to arrive first.
            if item.spill_vulnerable:
                for spilled_item in bag.items:
                    if spilled_item.is_spilled:
                        cost_t += CONTAMINATION_COST
                        if spilled_item.category == "Cleaning":
                            cost_t += CHEMICAL_CONTAMINATION_BONUS

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
