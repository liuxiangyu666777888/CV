from __future__ import annotations

import argparse
import os
from pathlib import Path

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=str)
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--project", type=str, default="outputs/train")
    parser.add_argument("--name", type=str, default="visdrone_yolo")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--resume-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    from ultralytics import YOLO, settings

    args = parse_args()
    project = Path(args.project)
    project.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    if args.resume_only:
        print(f"Loaded model only: {args.model}")
        return

    # Enable wandb in Ultralytics settings
    if HAS_WANDB:
        os.environ["WANDB_MODE"] = "online"
        settings.update({"wandb": True})
        wandb.init(
            project="assignment2-visdrone-detection",
            name=args.name,
            config={
                "model": args.model,
                "epochs": args.epochs,
                "imgsz": args.imgsz,
                "batch": args.batch,
            },
        )

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(project),
        name=args.name,
        device=args.device,
        amp=False,
        workers=0,
        cache="ram",
    )

    if HAS_WANDB:
        wandb.finish()


if __name__ == "__main__":
    main()
