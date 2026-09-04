from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cell_biophysics_benchmark.constants import SIMULATOR_VERSION  # noqa: E402
from cell_biophysics_benchmark.data import load_config  # noqa: E402
from cell_biophysics_benchmark.simulation import simulate_latent  # noqa: E402
from cell_biophysics_benchmark.types import (  # noqa: E402
    AcquisitionParams,
    PhysicalParams,
)
from cell_biophysics_benchmark.validation import compare_ensembles  # noqa: E402


def _git_provenance() -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"commit": commit, "dirty": dirty}


def _benchmark_paths(
    *,
    trajectories: int,
    steps: int,
    diffusion: float,
    frame_interval: float,
    alpha: float,
    seed_sequence: np.random.SeedSequence,
) -> np.ndarray:
    state = "brownian" if alpha == 1.0 else "subdiffusive"
    acquisition = AcquisitionParams(
        n_frames=steps,
        frame_interval_s=frame_interval,
        exposure_time_s=0.0,
        localization_sigma_um=0.0,
        missing_probability=0.0,
        oversample=1,
    )
    physical = PhysicalParams(
        state=state,
        diffusion_um2_s=diffusion,
        anomalous_alpha=alpha,
    )
    paths = np.empty((trajectories, steps + 1, 2), dtype=float)
    for index, child_seed in enumerate(seed_sequence.spawn(trajectories)):
        paths[index] = simulate_latent(
            physical, acquisition, np.random.default_rng(child_seed)
        ).dense_xy_um
    return paths


def _andi_paths(
    *,
    trajectories: int,
    steps: int,
    diffusion: float,
    frame_interval: float,
    alpha: float,
    seed_sequence: np.random.SeedSequence,
) -> np.ndarray:
    try:
        from andi_datasets.models_phenom import models_phenom
    except ImportError as error:
        raise SystemExit(
            "Install optional cross-check dependencies with "
            "'python -m pip install andi_datasets==2.1.13 stochastic'."
        ) from error

    # AnDi's phenomenological generator uses one arbitrary time unit. Matching
    # D * dt**alpha gives the same per-coordinate variance, 2 D dt**alpha.
    andi_diffusion = diffusion * frame_interval**alpha
    paths = np.zeros((trajectories, steps + 1, 2), dtype=float)
    for index, child_seed in enumerate(seed_sequence.spawn(trajectories)):
        legacy_seed = int(child_seed.generate_state(1, dtype=np.uint32)[0])
        np.random.seed(legacy_seed)
        increments = np.column_stack(
            [
                models_phenom.disp_fbm(alpha, andi_diffusion, steps),
                models_phenom.disp_fbm(alpha, andi_diffusion, steps),
            ]
        )
        paths[index, 1:] = np.cumsum(increments, axis=0)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-check latent diffusion ensembles against AnDi"
    )
    parser.add_argument("--config", default="configs/andi_crosscheck.yaml")
    parser.add_argument(
        "--output", default="artifacts/validation/andi_crosscheck.json"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(config["seed"])
    trajectories = int(config["trajectories"])
    steps = int(config["steps"])
    frame_interval = float(config["frame_interval_s"])
    diffusion = float(config["diffusion_um2_s"])
    lags = [int(lag) for lag in config["lags_frames"]]
    thresholds = config["thresholds"]
    root_seed = np.random.SeedSequence(seed)
    model_seeds = root_seed.spawn(2 * len(config["models"]))

    results: dict[str, object] = {}
    for index, model in enumerate(config["models"]):
        alpha = float(model["alpha"])
        benchmark = _benchmark_paths(
            trajectories=trajectories,
            steps=steps,
            diffusion=diffusion,
            frame_interval=frame_interval,
            alpha=alpha,
            seed_sequence=model_seeds[2 * index],
        )
        andi = _andi_paths(
            trajectories=trajectories,
            steps=steps,
            diffusion=diffusion,
            frame_interval=frame_interval,
            alpha=alpha,
            seed_sequence=model_seeds[2 * index + 1],
        )
        results[str(model["name"])] = compare_ensembles(
            benchmark,
            andi,
            diffusion=diffusion,
            frame_interval=frame_interval,
            alpha=alpha,
            lags=lags,
            theory_relative_tolerance=float(thresholds["theory_relative"]),
            between_relative_tolerance=float(thresholds["between_relative"]),
            correlation_absolute_tolerance=float(thresholds["correlation_absolute"]),
        )

    report = {
        "status": "passed" if all(result["passed"] for result in results.values()) else "failed",
        "scope": (
            "Latent, unobserved Brownian and fractional-Brownian trajectories only. "
            "Directed drift, circular confinement, switching, and camera acquisition "
            "are not claimed to be cross-validated by this run."
        ),
        "config": config,
        "provenance": {
            "simulator_version": SIMULATOR_VERSION,
            "andi_datasets_version": importlib.metadata.version("andi_datasets"),
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "git": _git_provenance(),
        },
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
