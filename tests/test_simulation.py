from __future__ import annotations

import numpy as np
import pytest

from cell_biophysics_benchmark.measurement import observe_trajectory
from cell_biophysics_benchmark.simulation import LatentTrajectory, simulate_latent
from cell_biophysics_benchmark.types import AcquisitionParams, PhysicalParams


def acquisition(**overrides: object) -> AcquisitionParams:
    values: dict[str, object] = {
        "n_frames": 200,
        "frame_interval_s": 0.02,
        "exposure_time_s": 0.0,
        "localization_sigma_um": 0.0,
        "missing_probability": 0.0,
        "oversample": 4,
    }
    values.update(overrides)
    return AcquisitionParams(**values)


def test_brownian_ensemble_recovers_expected_end_point_msd() -> None:
    observed = []
    params = PhysicalParams(state="brownian", diffusion_um2_s=0.4)
    camera = acquisition(n_frames=100, oversample=2)
    for seed in range(300):
        path = simulate_latent(params, camera, np.random.default_rng(seed)).dense_xy_um
        observed.append(np.square(path[-1]).sum())
    expected = 4.0 * params.diffusion_um2_s * camera.n_frames * camera.frame_interval_s
    assert np.mean(observed) == pytest.approx(expected, rel=0.15)


def test_confined_process_never_leaves_boundary() -> None:
    # Exercise the most difficult scale combination in benchmark_v1: a small
    # radius with high diffusion and a long frame interval requires many
    # specular reflections within a single dense integration step.
    radius = 0.05
    params = PhysicalParams(
        state="confined", diffusion_um2_s=5.0, confinement_radius_um=radius
    )
    latent = simulate_latent(
        params,
        acquisition(frame_interval_s=0.2, exposure_time_s=0.1, oversample=8),
        np.random.default_rng(11),
    )
    assert np.linalg.norm(latent.dense_xy_um, axis=1).max() <= radius * (1 + 1e-9)


def test_subdiffusive_increment_variance_has_requested_scale() -> None:
    params = PhysicalParams(
        state="subdiffusive", diffusion_um2_s=0.5, anomalous_alpha=0.7
    )
    camera = acquisition(n_frames=400, frame_interval_s=0.01, oversample=1)
    variances = []
    for seed in range(40):
        latent = simulate_latent(params, camera, np.random.default_rng(seed))
        increments = np.diff(latent.dense_xy_um[:, 0])
        variances.append(np.var(increments))
    expected = 2.0 * params.diffusion_um2_s * camera.frame_interval_s**params.anomalous_alpha
    assert np.mean(variances) == pytest.approx(expected, rel=0.20)


def test_full_exposure_averages_dense_positions() -> None:
    camera = acquisition(
        n_frames=3,
        frame_interval_s=1.0,
        exposure_time_s=1.0,
        oversample=4,
    )
    dense = np.column_stack([np.arange(13, dtype=float), np.zeros(13)])
    latent = LatentTrajectory(
        dense_xy_um=dense, frame_state=["directed", "directed", "directed"]
    )
    _, endpoints, observed, mask = observe_trajectory(
        latent, camera, np.random.default_rng(0)
    )
    np.testing.assert_allclose(endpoints[:, 0], [4.0, 8.0, 12.0])
    np.testing.assert_allclose(observed[:, 0], [2.5, 6.5, 10.5])
    assert mask.all()


def test_switching_sample_contains_both_states() -> None:
    params = PhysicalParams(
        state="switching",
        diffusion_um2_s=0.3,
        velocity_um_s=1.0,
        switch_probability_per_frame=1e-12,
        second_state_diffusion_um2_s=0.1,
    )
    latent = simulate_latent(params, acquisition(n_frames=30), np.random.default_rng(3))
    assert set(latent.frame_state) == {"brownian", "directed"}
