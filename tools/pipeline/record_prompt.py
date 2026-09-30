"""Prompt for the vision-language model: every object on the table as a record
of the characteristics the bagging planner reads.

One prompt serves training and inference. The definitions of rigidity and spill
risk are the ones used to annotate the catalog, so the model and the ground
truth answer the same question. CROP_SUFFIX is appended when the image is a
single detector region (the hybrid configuration).
"""

GROUPS = ("baby, bakery, batteries, cleaning, dairy, drink, frozen, hardware, household, "
          "hygiene, other, pantry, pet, pharmacy, produce, raw_meat, seafood, snack")
PACKAGING = ("aerosol, bag, blister, bottle, box, can, carton, cup, jar, loose, other, pouch, "
             "tray, tub, tube, wrap")

PROMPT = f"""This is a top-down photograph of a robot work table with grocery and household products on it. List every product resting on the table, so that a robot can pack them into bags.

Ignore the robot, the table, the markings on the table, the floor and anything behind the table.

Read the printed brand or product text on each item and use it to identify the product. Fruit sealed in a mesh bag is one item, named "bag of <fruit>".

For each product give:
- name: a short name of the product, such as "tuna can" or "bag of oranges".
- group: one of {GROUPS}.
- packaging: the container, one of {PACKAGING}. An unpackaged item is loose.
- quantity: how many identical units of this product are visible.
- weight_class: light (under 200 g), medium (200 g to 1 kg) or heavy (over 1 kg).
- rigidity: how much load the item carries without deforming. soft: a bag or a wrapped pack. semi: a thin card box or a plastic bottle. rigid: a can, a jar, a glass or a thick box.
- cold_chain: true if the product must be kept refrigerated or frozen.
- spill_risk: true if the contents could escape into the bag when the bag is tipped over or the item is laid on its side, which needs liquid or pourable contents and a closure that opens, such as a screw cap, a spray head or a snap lid. A sealed can, a sealed carton, a box, a bag, loose produce and crockery are false.

Answer with JSON only, in this form:
{{"items": [{{"name": "", "group": "", "packaging": "", "quantity": 1, "weight_class": "", "rigidity": "", "cold_chain": false, "spill_risk": false}}]}}
If there is no product on the table, answer {{"items": []}}."""

CROP_SUFFIX = """

This image is one region of the table. Report only the product at its centre, as one item with quantity 1, and ignore objects cut off at the edges. If there is no product at the centre, answer {"items": []}."""

FIELDS = ("name", "group", "packaging", "quantity", "weight_class", "rigidity",
          "cold_chain", "spill_risk")
