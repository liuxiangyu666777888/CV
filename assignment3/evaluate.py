"""Evaluate trained U-Net on the VisDrone test set and generate visualizations."""

from __future__ import annotations

import argparse
import csv

import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    CLASS_NAMES,
    IMAGE_SIZE,
    IMAGES_DIR,
    NUM_CLASSES,
    NUM_WORKERS,
    OUTPUT_DIR,
    TEST_LIST,
    TRIMAPS_DIR,
)
from data.dataset import OxfordPetDataset
from models.unet import UNet


def compute_per_class_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> list[float]:
    ious = []
    for c in range(num_classes):
        pred_c = pred == c
        target_c = target == c
        intersection = (pred_c & target_c).sum().float()
        union = (pred_c | target_c).sum().float()
        ious.append((intersection / union).item() if union > 0 else 0.0)
    return ious


def generate_visualization(image_np, gt_mask, pred_mask, save_path: Path):
    """Save a side-by-side comparison: original | GT mask | predicted mask."""
    try:
        h, w = image_np.shape[:2]
        # Pad to match original aspect if needed — just use resized version
        fig = np.zeros((IMAGE_SIZE, IMAGE_SIZE * 3, 3), dtype=np.uint8)
        fig[:, :IMAGE_SIZE] = (image_np * 255).astype(np.uint8)

        # Colorize masks
        colors = {0: (0, 255, 0), 1: (0, 0, 255), 2: (255, 255, 0)}  # pet=green, bg=blue, boundary=yellow
        for c in range(NUM_CLASSES):
            r, g, b = colors.get(c, (128, 128, 128))
            mask_c = (gt_mask == c)
            fig[:, IMAGE_SIZE:IMAGE_SIZE * 2, 0][mask_c] = r
            fig[:, IMAGE_SIZE:IMAGE_SIZE * 2, 1][mask_c] = g
            fig[:, IMAGE_SIZE:IMAGE_SIZE * 2, 2][mask_c] = b

            mask_c = (pred_mask == c)
            fig[:, IMAGE_SIZE * 2:, 0][mask_c] = r
            fig[:, IMAGE_SIZE * 2:, 1][mask_c] = g
            fig[:, IMAGE_SIZE * 2:, 2][mask_c] = b

        pil_img = Image.fromarray(fig)
        pil_img.save(str(save_path))
    except Exception:
        import cv2
        cv2.imwrite(str(save_path), cv2.cvtColor((image_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Auto-detect model path
    model_path = args.model or str(OUTPUT_DIR / "unet_combined" / "best.pt")
    out_dir = args.output_dir or str(OUTPUT_DIR / "evaluation")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = UNet(in_channels=3, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded: {model_path}")

    test_ds = OxfordPetDataset(
        IMAGES_DIR, TRIMAPS_DIR, TEST_LIST, image_size=IMAGE_SIZE, augment=False
    )
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=NUM_WORKERS)

    # Use incremental accumulation to avoid memory blow-up
    intersections = torch.zeros(NUM_CLASSES, device=device)
    unions = torch.zeros(NUM_CLASSES, device=device)
    vis_dir = out_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    vis_count = 0

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(test_loader, desc="Testing")):
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            preds = logits.argmax(1)

            # Incremental mIoU accumulation
            for c in range(NUM_CLASSES):
                pred_c = (preds == c)
                target_c = (masks == c)
                intersections[c] += (pred_c & target_c).sum()
                unions[c] += (pred_c | target_c).sum()

            # Save a few visualizations
            if vis_count < 10:
                for j in range(min(images.size(0), 10 - vis_count)):
                    img_np = images[j].cpu().permute(1, 2, 0).numpy()
                    gt = masks[j].cpu()
                    pd = preds[j].cpu()
                    generate_visualization(img_np, gt, pd, vis_dir / f"sample_{vis_count:03d}.png")
                    vis_count += 1

    # Per-class IoU from accumulated intersections/unions
    per_class_iou = []
    for c in range(NUM_CLASSES):
        iou = (intersections[c] / unions[c]).item() if unions[c] > 0 else 0.0
        per_class_iou.append(iou)
    miou = float(np.mean(per_class_iou))

    print("\n=== Test Results ===")
    print(f"mIoU: {miou:.4f}")
    for c in range(NUM_CLASSES):
        print(f"  {CLASS_NAMES[c]:>12s}: IoU={per_class_iou[c]:.4f}")

    # Save results
    with (out_dir / "results.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "iou"])
        for c in range(NUM_CLASSES):
            writer.writerow([CLASS_NAMES[c], f"{per_class_iou[c]:.4f}"])
        writer.writerow(["mIoU", f"{miou:.4f}"])

    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
