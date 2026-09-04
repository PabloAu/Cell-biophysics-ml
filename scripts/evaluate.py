from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cell_biophysics_benchmark.data import read_samples  # noqa: E402
from cell_biophysics_benchmark.evaluation import (  # noqa: E402
    classification_report,
    identifiability_table,
)
from cell_biophysics_benchmark.models import (  # noqa: E402
    load_classical_model,
    predict_classical,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a classical model")
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", default="artifacts/classical/evaluation")
    parser.add_argument("--confidence-threshold", type=float, default=0.70)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    artifact = load_classical_model(args.model)
    combined: dict[str, object] = {}
    for split in ("test_matched", "test_stress"):
        samples = read_samples(args.data, split=split)
        probabilities, classes = predict_classical(artifact, samples)
        combined[split] = classification_report(
            [sample.state_label for sample in samples],
            probabilities,
            classes,
            confidence_threshold=args.confidence_threshold,
        )
        table = identifiability_table(
            samples,
            probabilities,
            classes,
            confidence_threshold=args.confidence_threshold,
        )
        table.to_csv(output / f"identifiability_{split}.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
