from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from config import load_config
from dataset import OxfordPetClassificationDataset
from metrics import (
    accuracy_from_logits,
    confusion_matrix_numpy,
    save_confusion_matrix_plot,
    save_curves,
    save_history_csv,
    save_predictions_csv,
)
from models import build_model, set_backbone_trainable, trainable_parameters

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        help="Override config values with key=value, e.g. train.lr=0.001",
    )
    return parser.parse_args()


def parse_override_value(value: str):
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "none":
        return None
    try:
        if any(ch in value for ch in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override: {item}")
        key, value = item.split("=", 1)
        target = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                target[part] = {}
            target = target[part]
        target[parts[-1]] = parse_override_value(value)
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(image_size: int):
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return train_tf, test_tf


def create_dataloaders(cfg: dict):
    root = Path(cfg["data"]["root"])
    train_tf, test_tf = build_transforms(int(cfg["data"]["image_size"]))
    train_ds = OxfordPetClassificationDataset(
        root=root,
        split="train",
        images_dir=cfg["data"]["images_dir"],
        annotations_dir=cfg["data"]["annotations_dir"],
        transform=train_tf,
        limit=cfg["data"].get("train_subset"),
    )
    test_ds = OxfordPetClassificationDataset(
        root=root,
        split="test",
        images_dir=cfg["data"]["images_dir"],
        annotations_dir=cfg["data"]["annotations_dir"],
        transform=test_tf,
        limit=cfg["data"].get("test_subset"),
    )
    loader_kwargs = {
        "batch_size": int(cfg["train"]["batch_size"]),
        "num_workers": int(cfg["data"]["num_workers"]),
        "pin_memory": True,
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    return train_ds, test_ds, train_loader, test_loader


def build_optimizer(cfg: dict, model: nn.Module):
    name = cfg["train"]["optimizer"].lower()
    params = trainable_parameters(model)
    if name == "adamw":
        return AdamW(
            params,
            lr=float(cfg["train"]["lr"]),
            weight_decay=float(cfg["train"]["weight_decay"]),
        )
    if name == "sgd":
        return SGD(
            params,
            lr=float(cfg["train"]["lr"]),
            momentum=0.9,
            weight_decay=float(cfg["train"]["weight_decay"]),
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(cfg: dict, optimizer, epochs: int):
    scheduler_cfg = cfg["scheduler"]
    warmup_epochs = int(scheduler_cfg.get("warmup_epochs", 0))
    cosine_epochs = max(epochs - warmup_epochs, 1)
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
        eta_min=float(scheduler_cfg.get("min_lr", 1e-6)),
    )
    if warmup_epochs <= 0:
        return cosine
    warmup = LinearLR(optimizer, start_factor=0.2, end_factor=1.0, total_iters=warmup_epochs)
    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    optimizer,
    scaler,
    device: torch.device,
    train: bool,
    amp_enabled: bool,
    grad_clip_norm: float | None,
):
    model.train(train)
    losses: list[float] = []
    accuracies: list[float] = []
    all_targets: list[int] = []
    all_preds: list[int] = []
    prediction_rows: list[dict[str, object]] = []

    context = torch.enable_grad if train else torch.no_grad
    with context():
        for batch in tqdm(loader, leave=False):
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["label"].to(device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, targets)

            if train:
                scaler.scale(loss).backward()
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()

            acc = accuracy_from_logits(logits, targets)
            preds = logits.argmax(dim=1)

            losses.append(loss.item())
            accuracies.append(acc)
            all_targets.extend(targets.detach().cpu().tolist())
            all_preds.extend(preds.detach().cpu().tolist())

            for path, target, pred in zip(batch["path"], targets.detach().cpu().tolist(), preds.detach().cpu().tolist()):
                prediction_rows.append(
                    {"path": path, "target": target, "prediction": pred}
                )

    return {
        "loss": float(np.mean(losses)),
        "acc": float(np.mean(accuracies)),
        "targets": all_targets,
        "preds": all_preds,
        "prediction_rows": prediction_rows,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.set)
    set_seed(int(cfg["seed"]))

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if HAS_WANDB:
        wandb.init(
            project="assignment1-pet-classification",
            name=cfg["experiment_name"],
            config={
                "model": cfg["model"]["name"],
                "pretrained": cfg["model"].get("pretrained", True),
                "image_size": cfg["data"]["image_size"],
                "batch_size": cfg["train"]["batch_size"],
                "epochs": cfg["train"]["epochs"],
                "lr": cfg["train"]["lr"],
                "optimizer": cfg["train"]["optimizer"],
                "weight_decay": cfg["train"]["weight_decay"],
                "scheduler": "CosineAnnealingLR",
            },
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, test_ds, train_loader, test_loader = create_dataloaders(cfg)

    model = build_model(cfg).to(device)
    freeze_epochs = int(cfg["train"].get("freeze_backbone_epochs", 0))
    set_backbone_trainable(model, trainable=freeze_epochs == 0)

    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(cfg["train"].get("label_smoothing", 0.0))
    )
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer, int(cfg["train"]["epochs"]))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg["train"].get("amp", True) and device.type == "cuda"))

    best_acc = -1.0
    best_state = None
    history: list[dict[str, float]] = []

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        if epoch == freeze_epochs + 1 and freeze_epochs > 0:
            set_backbone_trainable(model, trainable=True)
            optimizer = build_optimizer(cfg, model)
            scheduler = build_scheduler(cfg, optimizer, int(cfg["train"]["epochs"]) - epoch + 1)

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            train=True,
            amp_enabled=bool(cfg["train"].get("amp", True) and device.type == "cuda"),
            grad_clip_norm=float(cfg["train"].get("grad_clip_norm", 0.0)),
        )
        val_metrics = run_epoch(
            model=model,
            loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            train=False,
            amp_enabled=bool(cfg["train"].get("amp", True) and device.type == "cuda"),
            grad_clip_norm=None,
        )
        scheduler.step()

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["acc"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"],
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        if HAS_WANDB:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_metrics["loss"],
                "train/acc": train_metrics["acc"],
                "val/loss": val_metrics["loss"],
                "val/acc": val_metrics["acc"],
                "lr": optimizer.param_groups[0]["lr"],
            })

        if val_metrics["acc"] > best_acc:
            best_acc = val_metrics["acc"]
            best_state = {
                "model": model.state_dict(),
                "config": cfg,
                "epoch": epoch,
                "val_acc": best_acc,
            }

        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f}"
        )

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint.")

    checkpoint_path = out_dir / "best.pt"
    torch.save(best_state, checkpoint_path)
    save_history_csv(out_dir / "history.csv", history)
    save_curves(out_dir / "curves.png", history)

    model.load_state_dict(best_state["model"])
    final_metrics = run_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scaler=scaler,
        device=device,
        train=False,
        amp_enabled=bool(cfg["train"].get("amp", True) and device.type == "cuda"),
        grad_clip_norm=None,
    )
    cm = confusion_matrix_numpy(
        final_metrics["targets"],
        final_metrics["preds"],
        num_classes=int(cfg["data"]["num_classes"]),
    )

    if bool(cfg["eval"].get("save_predictions", True)):
        save_predictions_csv(out_dir / "predictions.csv", final_metrics["prediction_rows"])
    if bool(cfg["eval"].get("save_confusion_matrix", True)):
        save_confusion_matrix_plot(out_dir / "confusion_matrix.png", cm, test_ds.idx_to_class)

    summary = {
        "experiment_name": cfg["experiment_name"],
        "best_val_acc": best_acc,
        "num_train": len(train_ds),
        "num_test": len(test_ds),
        "checkpoint": str(checkpoint_path),
        "device": str(device),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if HAS_WANDB:
        wandb.finish()


if __name__ == "__main__":
    main()
