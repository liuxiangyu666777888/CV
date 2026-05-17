from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from common import VISDRONE_CLASS_NAMES, ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--source", required=True, type=str)
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--line", nargs=4, type=int, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--output-dir", required=True, type=str)
    return parser.parse_args()


def side_of_line(point: tuple[float, float], line: tuple[int, int, int, int]) -> float:
    x, y = point
    x1, y1, x2, y2 = line
    return (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)


def main() -> None:
    from ultralytics import YOLO

    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    video_out = output_dir / "tracked.mp4"
    csv_out = output_dir / "tracks.csv"
    summary_out = output_dir / "summary.json"

    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source video: {args.source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    writer = cv2.VideoWriter(
        str(video_out),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    line = tuple(args.line) if args.line else None
    line_cross_count = 0
    last_side: dict[int, float] = {}
    track_lengths: defaultdict[int, int] = defaultdict(int)
    seen_ids: set[int] = set()
    rows: list[list[object]] = []

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.track(
            source=frame,
            persist=True,
            tracker=args.tracker,
            conf=args.conf,
            iou=args.iou,
            verbose=False,
        )
        result = results[0]
        boxes = result.boxes

        if line is not None:
            cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (0, 255, 255), 2)

        if boxes is not None and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            track_ids = boxes.id.cpu().numpy().astype(int)

            for box, cls_id, conf, track_id in zip(xyxy, cls_ids, confs, track_ids):
                x1, y1, x2, y2 = map(int, box.tolist())
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                label = VISDRONE_CLASS_NAMES[cls_id] if 0 <= cls_id < len(VISDRONE_CLASS_NAMES) else str(cls_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"ID {track_id} {label} {conf:.2f}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                cv2.circle(frame, (int(cx), int(cy)), 3, (0, 0, 255), -1)

                if line is not None:
                    current_side = side_of_line((cx, cy), line)
                    if track_id in last_side and last_side[track_id] * current_side < 0:
                        line_cross_count += 1
                    last_side[track_id] = current_side

                track_lengths[track_id] += 1
                seen_ids.add(track_id)
                rows.append([frame_index, track_id, cls_id, float(conf), x1, y1, x2, y2, cx, cy])

        cv2.putText(
            frame,
            f"Cross Count: {line_cross_count}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)
        frame_index += 1

    cap.release()
    writer.release()

    with csv_out.open("w", newline="", encoding="utf-8") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["frame", "track_id", "class_id", "confidence", "x1", "y1", "x2", "y2", "cx", "cy"])
        writer_csv.writerows(rows)

    summary = {
        "source": str(Path(args.source).resolve()),
        "model": args.model,
        "frame_count": frame_index,
        "unique_track_ids": len(seen_ids),
        "line_cross_count": line_cross_count,
        "mean_track_length": float(np.mean(list(track_lengths.values()))) if track_lengths else 0.0,
        "output_video": str(video_out.resolve()),
        "tracks_csv": str(csv_out.resolve()),
    }
    with summary_out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
