from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Base config file.")
    parser.add_argument(
        "--overrides",
        nargs="+",
        required=True,
        help=(
            "Experiment overrides. Example: "
            "\"experiment_name=lr_1e3 output_dir=outputs/lr_1e3 train.lr=0.001\""
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    train_script = root / "src" / "train.py"
    experiments = args.overrides
    for experiment in experiments:
        pieces = experiment.split()
        cmd = [sys.executable, str(train_script), "--config", args.config, "--set", *pieces]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=root)


if __name__ == "__main__":
    main()
