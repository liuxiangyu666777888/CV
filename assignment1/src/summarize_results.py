from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def main() -> None:
    outputs = Path("outputs")
    rows: list[dict] = []
    for summary_path in outputs.glob("*/summary.json"):
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        history_path = summary_path.parent / "history.csv"
        if history_path.exists():
            history = pd.read_csv(history_path)
            if not history.empty:
                summary["final_train_loss"] = float(history.iloc[-1]["train_loss"])
                summary["final_val_loss"] = float(history.iloc[-1]["val_loss"])
                summary["final_train_acc"] = float(history.iloc[-1]["train_acc"])
                summary["final_val_acc"] = float(history.iloc[-1]["val_acc"])
                summary["epochs_ran"] = int(history.iloc[-1]["epoch"])
        rows.append(summary)

    if not rows:
        print("No results found.")
        return

    df = pd.DataFrame(rows).sort_values("experiment_name")
    out_path = outputs / "experiment_summary.csv"
    df.to_csv(out_path, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved summary to {out_path}")


if __name__ == "__main__":
    main()
