"""Reproducible balanced benchmark generation."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..constants import SIMULATOR_VERSION, STATE_LABELS
from ..measurement import observe_trajectory
from ..simulation import simulate_latent
from ..types import AcquisitionParams, PhysicalParams, TrajectorySample
from .io import write_samples


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping")
    return config


def _uniform(bounds: list[float], rng: np.random.Generator) -> float:
    return float(rng.uniform(float(bounds[0]), float(bounds[1])))


def _log_uniform(bounds: list[float], rng: np.random.Generator) -> float:
    low, high = map(float, bounds)
    if low <= 0:
        raise ValueError("log-uniform bounds must be positive")
    return float(math.exp(rng.uniform(math.log(low), math.log(high))))


def _sample_parameters(
    state: str,
    physical_config: dict[str, Any],
    acquisition_config: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[PhysicalParams, AcquisitionParams]:
    diffusion = _log_uniform(physical_config["diffusion_um2_s"], rng)
    velocity = _log_uniform(physical_config["velocity_um_s"], rng)
    direction = float(rng.uniform(0.0, 2.0 * np.pi))
    radius = _log_uniform(physical_config["confinement_radius_um"], rng)
    alpha = _uniform(physical_config["anomalous_alpha"], rng)
    switch_probability = _log_uniform(
        physical_config["switch_probability_per_frame"], rng
    )
    physical = PhysicalParams(
        state=state,
        diffusion_um2_s=diffusion,
        velocity_um_s=velocity if state in {"directed", "switching"} else 0.0,
        direction_rad=direction,
        confinement_radius_um=radius if state == "confined" else None,
        anomalous_alpha=alpha if state == "subdiffusive" else 1.0,
        switch_probability_per_frame=switch_probability if state == "switching" else 0.0,
        second_state_diffusion_um2_s=(
            diffusion * float(rng.uniform(0.1, 0.8)) if state == "switching" else None
        ),
    )

    frame_interval = _log_uniform(acquisition_config["frame_interval_s"], rng)
    exposure_fraction = _uniform(acquisition_config["exposure_fraction"], rng)
    length_bounds = acquisition_config["trajectory_length"]
    n_frames = int(rng.integers(int(length_bounds[0]), int(length_bounds[1]) + 1))
    acquisition = AcquisitionParams(
        n_frames=n_frames,
        frame_interval_s=frame_interval,
        exposure_time_s=frame_interval * exposure_fraction,
        localization_sigma_um=_log_uniform(
            acquisition_config["localization_sigma_um"], rng
        ),
        missing_probability=_uniform(acquisition_config["missing_probability"], rng),
        oversample=int(acquisition_config.get("oversample", 8)),
    )
    return physical, acquisition


def _make_sample(
    *,
    state: str,
    split: str,
    regime: str,
    seed: int,
    physical_config: dict[str, Any],
    acquisition_config: dict[str, Any],
) -> TrajectorySample:
    root_seed = np.random.SeedSequence(seed)
    parameter_seed, latent_seed, observation_seed = root_seed.spawn(3)
    parameter_rng = np.random.default_rng(parameter_seed)
    physical, acquisition = _sample_parameters(
        state, physical_config, acquisition_config, parameter_rng
    )
    latent = simulate_latent(physical, acquisition, np.random.default_rng(latent_seed))
    t_s, latent_xy, observed_xy, mask = observe_trajectory(
        latent, acquisition, np.random.default_rng(observation_seed)
    )
    token = f"{SIMULATOR_VERSION}|{state}|{split}|{seed}".encode()
    sample = TrajectorySample(
        sample_id=hashlib.sha256(token).hexdigest()[:20],
        split=split,
        regime=regime,
        seed=seed,
        state_label=state,
        t_s=t_s,
        latent_xy_um=latent_xy,
        observed_xy_um=observed_xy,
        observed_mask=mask,
        latent_state=latent.frame_state,
        physical_params=physical,
        acquisition_params=acquisition,
    )
    sample.validate()
    return sample


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def generate_dataset(config: dict[str, Any]) -> dict[str, Any]:
    dataset = config["dataset"]
    states = list(dataset["states"])
    unknown = sorted(set(states) - set(STATE_LABELS))
    if unknown:
        raise ValueError(f"Unknown state labels: {unknown}")
    counts = dataset["samples_per_state"]
    root_seed = np.random.SeedSequence(int(dataset["seed"]))
    total = len(states) * sum(int(value) for value in counts.values())
    child_seeds = root_seed.spawn(total)

    samples: list[TrajectorySample] = []
    seed_index = 0
    for split, per_state in counts.items():
        regime = "stress" if split == "test_stress" else "matched"
        acquisition_config = config["acquisition"][regime]
        for state in states:
            for _ in range(int(per_state)):
                seed = int(child_seeds[seed_index].generate_state(1, dtype=np.uint64)[0])
                seed_index += 1
                samples.append(
                    _make_sample(
                        state=state,
                        split=split,
                        regime=regime,
                        seed=seed,
                        physical_config=config["physical"],
                        acquisition_config=acquisition_config,
                    )
                )

    # Deterministic shuffle avoids class-block ordering while preserving exact
    # regeneration from the root seed.
    order_rng = np.random.default_rng(root_seed.spawn(1)[0])
    order_rng.shuffle(samples)
    output = Path(dataset["output"])
    summary = write_samples(samples, output)
    split_counts = Counter(sample.split for sample in samples)
    state_counts = Counter(sample.state_label for sample in samples)
    manifest = {
        **summary,
        "dataset_name": dataset["name"],
        "simulator_version": SIMULATOR_VERSION,
        "root_seed": int(dataset["seed"]),
        "git_commit": _git_commit(),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "split_counts": dict(sorted(split_counts.items())),
        "state_counts": dict(sorted(state_counts.items())),
        "config_sha256": hashlib.sha256(
            yaml.safe_dump(config, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
