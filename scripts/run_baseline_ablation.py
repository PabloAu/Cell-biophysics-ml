from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cell_biophysics_benchmark.data import read_samples  # noqa: E402
from cell_biophysics_benchmark.evaluation import classification_report  # noqa: E402
from cell_biophysics_benchmark.models import (  # noqa: E402
    predict_classical,
    save_classical_model,
    train_classical,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare measurement-blind and measurement-aware baselines"
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", default="artifacts/classical_ablation")
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()
    train = read_samples(args.data, split="train")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    reports: dict[str, object] = {}
    for aware in (False, True):
        name = "measurement_aware" if aware else "trajectory_only"
        artifact = train_classical(train, measurement_aware=aware, seed=args.seed)
        save_classical_model(artifact, output / f"{name}.joblib")
        split_reports: dict[str, object] = {}
        for split in ("validation", "test_matched", "test_stress"):
            samples = read_samples(args.data, split=split)
            probabilities, classes = predict_classical(artifact, samples)
            split_reports[split] = classification_report(
                [sample.state_label for sample in samples], probabilities, classes
            )
        reports[name] = split_reports
    (output / "comparison.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
