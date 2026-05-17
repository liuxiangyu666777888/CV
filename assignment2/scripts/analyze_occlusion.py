from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd

from common import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=str)
    parser.add_argument("--tracks", required=True, type=str)
    parser.add_argument("--start-frame", required=True, type=int)
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--output-dir", required=True, type=str)
    return parser.parse_args()


def draw_frame(frame, frame_tracks: pd.DataFrame):
    for _, row in frame_tracks.iterrows():
        x1, y1, x2, y2 = int(row.x1), int(row.y1), int(row.x2), int(row.y2)
        track_id = int(row.track_id)
        cls_id = int(row.class_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"ID {track_id} C{cls_id}",
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return frame


def save_contact_sheet(image_paths: list[Path], output_path: Path) -> None:
    images = [cv2.imread(str(path)) for path in image_paths]
    images = [img for img in images if img is not None]
    if not images:
        return
    height = min(img.shape[0] for img in images)
    width = min(img.shape[1] for img in images)
    resized = [cv2.resize(img, (width, height)) for img in images]
    sheet = cv2.hconcat(resized)
    cv2.imwrite(str(output_path), sheet)


def build_markdown_report(df: pd.DataFrame, start_frame: int, num_frames: int) -> str:
    subset = df[(df["frame"] >= start_frame) & (df["frame"] < start_frame + num_frames)]
    unique_ids = sorted(subset["track_id"].unique().tolist())
    frame_counts = subset.groupby("frame")["track_id"].nunique().to_dict()
    return (
        "# Occlusion / ID Switch Analysis\n\n"
        f"- Frame window: {start_frame} ~ {start_frame + num_frames - 1}\n"
        f"- Visible track IDs: {unique_ids}\n"
        f"- Objects per frame: {frame_counts}\n\n"
        "## Manual Analysis Notes\n\n"
        "1. Observe whether the same object keeps the same track ID across consecutive frames.\n"
        "2. Check whether dense overlap or partial occlusion causes missed detections.\n"
        "3. If IDs change unexpectedly, describe the likely reason: detector miss, overlap, or tracker association failure.\n"
    )


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    df = pd.read_csv(args.tracks)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")

    saved_frames: list[Path] = []
    for frame_id in range(args.start_frame, args.start_frame + args.num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = cap.read()
        if not ok:
            continue
        frame_tracks = df[df["frame"] == frame_id]
        frame = draw_frame(frame, frame_tracks)
        out_path = output_dir / f"frame_{frame_id:05d}.jpg"
        cv2.imwrite(str(out_path), frame)
        saved_frames.append(out_path)

    cap.release()

    contact_sheet = output_dir / "contact_sheet.jpg"
    save_contact_sheet(saved_frames, contact_sheet)

    report = build_markdown_report(df, args.start_frame, args.num_frames)
    (output_dir / "analysis.md").write_text(report, encoding="utf-8")
    print(f"Saved analysis to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
