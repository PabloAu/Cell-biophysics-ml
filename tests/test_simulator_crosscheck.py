from __future__ import annotations

import numpy as np
import pytest

from cell_biophysics_benchmark.validation.simulator_crosscheck import (
    compare_ensembles,
    increment_correlation,
    summarize_ensemble,
    time_averaged_msd,
)


def brownian_paths(seed: int, trajectories: int = 400, steps: int = 128) -> np.ndarray:
    rng = np.random.default_rng(seed)
    increments = rng.normal(scale=np.sqrt(0.04), size=(trajectories, steps, 2))
    return np.concatenate(
        [np.zeros((trajectories, 1, 2)), np.cumsum(increments, axis=1)], axis=1
    )


def test_time_averaged_msd_rejects_invalid_lag() -> None:
    paths = brownian_paths(1, trajectories=2, steps=3)
    with pytest.raises(ValueError, match="lags"):
        time_averaged_msd(paths, [4])


def test_brownian_summary_recovers_scale_and_zero_correlation() -> None:
    paths = brownian_paths(2)
    summary = summarize_ensemble(
        paths,
        diffusion=1.0,
        frame_interval=0.02,
        alpha=1.0,
        lags=[1, 4, 16],
    )
    assert summary["max_msd_relative_error"] < 0.04
    assert abs(increment_correlation(paths)) < 0.02


def test_compare_ensembles_reports_passing_checks() -> None:
    comparison = compare_ensembles(
        brownian_paths(3),
        brownian_paths(4),
        diffusion=1.0,
        frame_interval=0.02,
        alpha=1.0,
        lags=[1, 2, 4, 8],
        theory_relative_tolerance=0.08,
        between_relative_tolerance=0.08,
        correlation_absolute_tolerance=0.02,
    )
    assert comparison["passed"] is True
    assert all(comparison["checks"].values())
