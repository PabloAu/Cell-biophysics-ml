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
    parser = argparse.ArgumentParser(description="Train the physics-feature baseline")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="artifacts/classical/model.joblib")
    parser.add_argument(
        "--measurement-aware",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include declared acquisition parameters",
    )
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()

    train = read_samples(args.data, split="train")
    validation = read_samples(args.data, split="validation")
    artifact = train_classical(
        train, measurement_aware=args.measurement_aware, seed=args.seed
    )
    save_classical_model(artifact, args.output)
    probabilities, classes = predict_classical(artifact, validation)
    report = classification_report(
        [sample.state_label for sample in validation], probabilities, classes
    )
    report_path = Path(args.output).with_suffix(".validation.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"model": args.output, "validation": report}, indent=2))


if __name__ == "__main__":
    main()
