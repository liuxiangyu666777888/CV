from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-root", required=True, type=str)
    parser.add_argument("--topk", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.images_root)
    counter: Counter[str] = Counter()
    for path in root.glob("*.jpg"):
        prefix = path.stem.split("_d_")[0]
        counter[prefix] += 1

    for prefix, count in counter.most_common(args.topk):
        print(f"{prefix}\t{count}")


if __name__ == "__main__":
    main()
