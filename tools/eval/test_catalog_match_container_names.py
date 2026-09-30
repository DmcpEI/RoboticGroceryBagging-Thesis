"""One check: a product whose whole name is a container word must resolve,
and a container word that is not a product must not."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
sys.path.insert(0, "tools/eval")
from run_gemini_robotics import full_catalog_attrs

def demo():
    assert full_catalog_attrs("cup")["packaging"] == "cup", "the catalog holds a cup"
    assert full_catalog_attrs("Cup")["packaging"] == "cup", "case must not matter"
    assert full_catalog_attrs("can")["group"] is None, "a bare can is not a product"
    assert full_catalog_attrs("bottle")["group"] is None, "nor a bare bottle"
    assert full_catalog_attrs("tuna can")["group"] == "pantry", "named cans still resolve"
    assert full_catalog_attrs("wine cup")["packaging"] == "cup", "two-word names unaffected"
    print("ok")

if __name__ == "__main__":
    demo()
