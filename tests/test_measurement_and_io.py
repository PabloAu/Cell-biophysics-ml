from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np

from cell_biophysics_benchmark.data.io import iter_samples, write_samples
from cell_biophysics_benchmark.measurement import observe_trajectory
from cell_biophysics_benchmark.simulation import simulate_latent
from cell_biophysics_benchmark.types import AcquisitionParams, PhysicalParams, TrajectorySample


def make_sample(seed: int = 7) -> TrajectorySample:
    acquisition = AcquisitionParams(
        n_frames=40,
        frame_interval_s=0.05,
        exposure_time_s=0.04,
        localization_sigma_um=0.03,
        missing_probability=0.25,
        oversample=4,
    )
    physical = PhysicalParams(state="brownian", diffusion_um2_s=0.4)
    root = np.random.SeedSequence(seed).spawn(2)
    latent = simulate_latent(physical, acquisition, np.random.default_rng(root[0]))
    t_s, latent_xy, observed, mask = observe_trajectory(
        latent, acquisition, np.random.default_rng(root[1])
    )
    return TrajectorySample(
        sample_id="test-sample",
        split="train",
        regime="matched",
        seed=seed,
        state_label="brownian",
        t_s=t_s,
        latent_xy_um=latent_xy,
        observed_xy_um=observed,
        observed_mask=mask,
        latent_state=latent.frame_state,
        physical_params=physical,
        acquisition_params=acquisition,
    )


def test_missing_mask_matches_nan_rows() -> None:
    sample = make_sample()
    sample.validate()
    assert np.array_equal(~np.isfinite(sample.observed_xy_um).all(axis=1), ~sample.observed_mask)
    assert sample.observed_mask.sum() >= 3


def test_jsonl_round_trip_and_deterministic_gzip(tmp_path: Path) -> None:
    sample = make_sample()
    first = tmp_path / "first.jsonl.gz"
    second = tmp_path / "second.jsonl.gz"
    first_manifest = write_samples([sample], first)
    second_manifest = write_samples([sample], second)
    assert first_manifest["sha256"] == second_manifest["sha256"]
    restored = list(iter_samples(first))[0]
    restored.validate()
    np.testing.assert_allclose(restored.latent_xy_um, sample.latent_xy_um)
    np.testing.assert_allclose(
        restored.observed_xy_um, sample.observed_xy_um, equal_nan=True
    )
    with gzip.open(first, "rt", encoding="utf-8") as handle:
        assert '"observed_xy_um"' in handle.readline()
