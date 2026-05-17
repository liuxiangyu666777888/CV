"""Train U-Net on Oxford Pet with configurable loss function."""

from __future__ import annotations

import argparse
import csv
import random

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from config import (
    ANNOTATIONS_DIR,
    BATCH_SIZE,
    IMAGE_SIZE,
    IMAGES_DIR,
    LEARNING_RATE,
    NUM_CLASSES,
    NUM_EPOCHS,
    NUM_WORKERS,
    OUTPUT_DIR,
    SEED,
    TRAINVAL_LIST,
    TRIMAPS_DIR,
    VAL_SPLIT,
    WEIGHT_DECAY,
)
from data.dataset import OxfordPetDataset, build_dataloaders
from losses.dice_loss import CombinedLoss, DiceLoss
from models.unet import UNet

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def validate(model: UNet, loader, criterion, device: torch.device, num_classes: int):
    """Validation with incremental mIoU to avoid memory blow-up."""
    model.eval()
    total_loss = 0.0
    # Accumulate intersection & union per class (no intermediate storage)
    intersections = torch.zeros(num_classes, device=device)
    unions = torch.zeros(num_classes, device=device)
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            loss = criterion(logits, masks)
            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(1)
            for c in range(num_classes):
                pred_c = (preds == c)
                target_c = (masks == c)
                intersections[c] += (pred_c & target_c).sum()
                unions[c] += (pred_c | target_c).sum()
    avg_loss = total_loss / len(loader.dataset)
    # Compute mIoU from accumulated values
    ious = []
    for c in range(num_classes):
        if unions[c] > 0:
            ious.append((intersections[c] / unions[c]).item())
    miou = float(np.mean(ious)) if ious else 0.0
    return avg_loss, miou


def run_training(args: argparse.Namespace) -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    run_name = f"unet_{args.loss}"
    output_dir = OUTPUT_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    metrics_csv = output_dir / "metrics.csv"
    if HAS_WANDB:
        wandb.init(
            project="assignment3-unet",
            name=run_name,
            config={
                "model": "U-Net",
                "loss": args.loss,
                "batch_size": BATCH_SIZE,
                "epochs": NUM_EPOCHS,
                "lr": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "image_size": IMAGE_SIZE,
                "scheduler": "CosineAnnealingLR",
            },
        )

    # Build model
    model = UNet(in_channels=3, num_classes=NUM_CLASSES).to(device)
    print(f"U-Net params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Build loss
    if args.loss == "ce":
        criterion = nn.CrossEntropyLoss()
    elif args.loss == "dice":
        criterion = DiceLoss()
    elif args.loss == "combined":
        criterion = CombinedLoss()
    else:
        raise ValueError(f"Unknown loss: {args.loss}")
    print(f"Loss: {args.loss}")

    # Build dataloaders
    train_loader, val_loader = build_dataloaders(
        images_dir=IMAGES_DIR,
        trimaps_dir=TRIMAPS_DIR,
        trainval_file=TRAINVAL_LIST,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        val_split=VAL_SPLIT,
        num_workers=NUM_WORKERS,
        seed=SEED,
    )
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_miou = 0.0
    csv_rows = []
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{NUM_EPOCHS}")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()

        train_loss /= len(train_loader.dataset)
        val_loss, val_miou = validate(model, val_loader, criterion, device, NUM_CLASSES)

        lr = scheduler.get_last_lr()[0]
        csv_rows.append([epoch, train_loss, val_loss, val_miou, lr])

        if HAS_WANDB:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "val/mIoU": val_miou,
                "lr": lr,
            })

        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {val_miou:.4f}")

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), output_dir / "best.pt")
            print(f"  Saved best model (mIoU={best_miou:.4f})")

    # Save metrics CSV
    with metrics_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_mIoU", "lr"])
        writer.writerows(csv_rows)

    # Save final model
    torch.save(model.state_dict(), output_dir / "last.pt")
    print(f"\nFinished. Best mIoU: {best_miou:.4f}")
    print(f"Output: {output_dir}")
    print(f"Metrics: {metrics_csv}")

    if HAS_WANDB:
        wandb.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss", type=str, default="ce", choices=["ce", "dice", "combined"])
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
