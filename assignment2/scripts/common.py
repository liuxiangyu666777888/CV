from __future__ import annotations

from pathlib import Path


VISDRONE_CLASS_MAP = {
    1: 0,   # pedestrian
    2: 1,   # people
    3: 2,   # bicycle
    4: 3,   # car
    5: 4,   # van
    6: 5,   # truck
    7: 6,   # tricycle
    8: 7,   # awning-tricycle
    9: 8,   # bus
    10: 9,  # motor
}

VISDRONE_CLASS_NAMES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_nested_dir(root: Path, split_name: str) -> Path:
    direct = root / split_name
    nested = root / split_name / split_name
    if nested.exists():
        return nested
    if direct.exists():
        return direct
    raise FileNotFoundError(f"Could not find split directory for {split_name} under {root}")
