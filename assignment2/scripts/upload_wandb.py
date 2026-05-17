"""Upload existing YOLO training results.csv to wandb — no re-training needed."""
import csv
from pathlib import Path

import wandb

RESULTS_DIR = Path("runs/detect/outputs/train/visdrone_yolo_wandb-4")

wandb.init(
    project="assignment2-visdrone-detection",
    name="visdrone_yolo_wandb-4",
    config={
        "model": "yolo11n.pt",
        "epochs": 30,
        "imgsz": 640,
        "batch": 8,
    },
)

with open(RESULTS_DIR / "results.csv") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Remove trailing empty column
for row in rows:
    row.pop("", None)
    epoch = int(row["epoch"])
    wandb.log(
        {
            "epoch": epoch,
            "train/box_loss": float(row["train/box_loss"]),
            "train/cls_loss": float(row["train/cls_loss"]),
            "train/dfl_loss": float(row["train/dfl_loss"]),
            "val/box_loss": float(row["val/box_loss"]),
            "val/cls_loss": float(row["val/cls_loss"]),
            "val/dfl_loss": float(row["val/dfl_loss"]),
            "metrics/precision(B)": float(row["metrics/precision(B)"]),
            "metrics/recall(B)": float(row["metrics/recall(B)"]),
            "metrics/mAP50(B)": float(row["metrics/mAP50(B)"]),
            "metrics/mAP50-95(B)": float(row["metrics/mAP50-95(B)"]),
        }
    )

print(f"Uploaded {len(rows)} epochs to wandb")
wandb.finish()
