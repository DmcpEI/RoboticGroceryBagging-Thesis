"""One check: the crop must filter boxes, not shrink the image the model sees."""
import sys
from pathlib import Path
sys.path.insert(0, "robot/robot_pc_package")
import perceive_local as pl


class FakeBoxes:
    def __init__(self, rows):
        self.cls = type("T", (), {"tolist": lambda s: [r[0] for r in rows]})()
        self.conf = type("T", (), {"tolist": lambda s: [r[1] for r in rows]})()
        self.xyxy = type("T", (), {"tolist": lambda s: [r[2] for r in rows]})()


class FakeModel:
    """Records what it was handed, and answers with one on-table and one off-table box."""
    def __init__(self):
        self.saw = None
    def __call__(self, src, **kw):
        self.saw = src
        return [type("R", (), {"boxes": FakeBoxes([
            (0, 0.9, [200.0, 200.0, 260.0, 260.0]),     # centre inside the crop
            (1, 0.8, [10.0, 10.0, 40.0, 40.0]),         # centre outside it
        ])})()]


def demo():
    crop = (110, 125, 565, 412)
    m = FakeModel()
    out = pl.detect(m, Path("unused.png"), ["on_table", "on_floor"], 640, 0.15, crop)
    assert m.saw == "unused.png", "the model must be handed the whole frame, not a crop"
    names = [o[0] for o in out]
    assert names == ["on_table"], names
    assert out[0][2] == [200.0, 200.0, 260.0, 260.0], "boxes stay in full-frame coordinates"
    print("ok")


if __name__ == "__main__":
    demo()
