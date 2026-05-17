from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from common import VISDRONE_CLASS_MAP, VISDRONE_CLASS_NAMES, ensure_dir, find_nested_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=str, help="Path to data/archive")
    parser.add_argument("--target", required=True, type=str, help="Output YOLO dataset directory")
    return parser.parse_args()


def convert_annotation(annotation_path: Path, image_path: Path) -> list[str]:
    with open(image_path, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    height, width = image.shape[:2]

    yolo_lines: list[str] = []
    with annotation_path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip().rstrip(",")
            if not raw:
                continue
            x, y, w, h, score, category, truncation, occlusion = map(int, raw.split(","))
            if score == 0:
                continue
            if category not in VISDRONE_CLASS_MAP:
                continue
            cls = VISDRONE_CLASS_MAP[category]
            xc = (x + w / 2.0) / width
            yc = (y + h / 2.0) / height
            wn = w / width
            hn = h / height
            yolo_lines.append(f"{cls} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
    return yolo_lines


def process_split(source_root: Path, target_root: Path, split_dir_name: str, split_name: str) -> None:
    split_root = find_nested_dir(source_root, split_dir_name)
    images_src = split_root / "images"
    ann_src = split_root / "annotations"

    images_dst = ensure_dir(target_root / "images" / split_name)
    labels_dst = ensure_dir(target_root / "labels" / split_name)

    image_paths = sorted(images_src.glob("*.jpg"))
    for image_path in tqdm(image_paths, desc=f"Preparing {split_name}"):
        ann_path = ann_src / f"{image_path.stem}.txt"
        label_path = labels_dst / f"{image_path.stem}.txt"
        shutil.copy2(image_path, images_dst / image_path.name)
        if ann_path.exists():
            lines = convert_annotation(ann_path, image_path)
            label_path.write_text("\n".join(lines), encoding="utf-8")
        else:
            label_path.write_text("", encoding="utf-8")


def write_data_yaml(target_root: Path) -> None:
    data = {
        "path": str(target_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(VISDRONE_CLASS_NAMES),
        "names": VISDRONE_CLASS_NAMES,
    }
    with (target_root / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def main() -> None:
    args = parse_args()
    source_root = Path(args.source).resolve()
    target_root = ensure_dir(args.target)

    process_split(source_root, target_root, "VisDrone2019-DET-train", "train")
    process_split(source_root, target_root, "VisDrone2019-DET-val", "val")
    process_split(source_root, target_root, "VisDrone2019-DET-test-dev", "test")
    write_data_yaml(target_root)
    print(f"Prepared YOLO dataset at: {target_root.resolve()}")


if __name__ == "__main__":
    main()
