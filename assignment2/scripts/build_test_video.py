from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from common import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-root", required=True, type=str)
    parser.add_argument("--sequence", required=True, type=str, help="Prefix before _d_, e.g. 9999979_00000")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--output", required=True, type=str)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images_root = Path(args.images_root)
    output = Path(args.output)
    ensure_dir(output.parent)

    pattern = f"{args.sequence}_d_*.jpg"
    frames = sorted(images_root.glob(pattern))
    if not frames:
        raise FileNotFoundError(f"No frames found for pattern: {pattern}")

    first = cv2.imread(str(frames[0]))
    if first is None:
        raise RuntimeError(f"Failed to read: {frames[0]}")
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (width, height),
    )
    for frame_path in frames:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        writer.write(frame)
    writer.release()
    print(f"Saved video to: {output.resolve()}")
    print(f"Frame count: {len(frames)}")


if __name__ == "__main__":
    main()
