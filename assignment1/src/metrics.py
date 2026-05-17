from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def accuracy_from_logits(logits, targets) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def confusion_matrix_numpy(
    y_true: list[int],
    y_pred: list[int],
    num_classes: int,
) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def save_predictions_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_history_csv(path: Path, history: list[dict[str, float]]) -> None:
    pd.DataFrame(history).to_csv(path, index=False)


def save_confusion_matrix_plot(
    path: Path,
    cm: np.ndarray,
    class_names: list[str],
) -> None:
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, cmap="Blues", square=True)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_curves(path: Path, history: list[dict[str, float]]) -> None:
    df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(df["epoch"], df["train_loss"], label="train_loss")
    axes[0].plot(df["epoch"], df["val_loss"], label="val_loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(df["epoch"], df["train_acc"], label="train_acc")
    axes[1].plot(df["epoch"], df["val_acc"], label="val_acc")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close(fig)
