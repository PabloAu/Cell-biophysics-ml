from __future__ import annotations

import gzip
import json
from pathlib import Path

from cell_biophysics_benchmark.data.generation import generate_dataset


def _tiny_config(output: Path) -> dict[str, object]:
    return {
        "dataset": {
            "name": "tiny-determinism-check",
            "seed": 7,
            "states": ["brownian", "directed"],
            "samples_per_state": {
                "train": 3,
                "validation": 1,
                "test_matched": 1,
                "test_stress": 1,
            },
            "output": str(output),
        },
        "physical": {
            "diffusion_um2_s": [0.1, 0.2],
            "velocity_um_s": [0.5, 1.0],
            "confinement_radius_um": [0.5, 1.0],
            "anomalous_alpha": [0.5, 0.8],
            "switch_probability_per_frame": [0.05, 0.1],
        },
        "acquisition": {
            "matched": {
                "frame_interval_s": [0.01, 0.02],
                "exposure_fraction": [0.1, 0.5],
                "localization_sigma_um": [0.005, 0.01],
                "missing_probability": [0.0, 0.1],
                "trajectory_length": [8, 12],
                "oversample": 2,
            },
            "stress": {
                "frame_interval_s": [0.02, 0.03],
                "exposure_fraction": [0.8, 1.0],
                "localization_sigma_um": [0.05, 0.1],
                "missing_probability": [0.2, 0.3],
                "trajectory_length": [5, 7],
                "oversample": 2,
            },
        },
    }


def test_generation_is_deterministic_and_balanced(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl.gz"
    second = tmp_path / "second.jsonl.gz"

    first_manifest = generate_dataset(_tiny_config(first))
    generate_dataset(_tiny_config(second))

    assert first.read_bytes() == second.read_bytes()
    assert first_manifest["samples"] == 12
    assert first_manifest["split_counts"] == {
        "test_matched": 2,
        "test_stress": 2,
        "train": 6,
        "validation": 2,
    }
    assert first_manifest["state_counts"] == {"brownian": 6, "directed": 6}

    with gzip.open(first, mode="rt", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle]
    assert len(records) == 12
    assert {record["state_label"] for record in records} == {"brownian", "directed"}
