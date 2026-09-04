from __future__ import annotations

import numpy as np
import pytest

from cell_biophysics_benchmark.features import extract_features, feature_matrix, time_averaged_msd
from cell_biophysics_benchmark.models.temporal import temporal_inputs
from cell_biophysics_benchmark.types import AcquisitionParams, PhysicalParams, TrajectorySample


def linear_sample() -> TrajectorySample:
    n = 20
    xy = np.column_stack([np.arange(n, dtype=float), np.zeros(n)])
    acquisition = AcquisitionParams(
        n_frames=n,
        frame_interval_s=0.1,
        exposure_time_s=0.05,
        localization_sigma_um=0.02,
        missing_probability=0.0,
        oversample=4,
    )
    return TrajectorySample(
        sample_id="linear",
        split="train",
        regime="matched",
        seed=0,
        state_label="directed",
        t_s=(np.arange(n) + 1) * 0.1,
        latent_xy_um=xy.copy(),
        observed_xy_um=xy.copy(),
        observed_mask=np.ones(n, dtype=bool),
        latent_state=["directed"] * n,
        physical_params=PhysicalParams(
            state="directed", diffusion_um2_s=0.01, velocity_um_s=10.0
        ),
        acquisition_params=acquisition,
    )


def test_linear_path_features_have_expected_geometry() -> None:
    features = extract_features(linear_sample())
    assert features["straightness"] == pytest.approx(1.0)
    assert features["mean_turn_cosine"] == pytest.approx(1.0)
    assert features["asymmetry"] == pytest.approx(1.0)
    assert features["msd_alpha"] == pytest.approx(2.0, rel=0.01)


def test_msd_ignores_pairs_touching_missing_frames() -> None:
    sample = linear_sample()
    sample.observed_mask[2] = False
    sample.observed_xy_um[2] = np.nan
    lag, msd, counts = time_averaged_msd(
        sample.observed_xy_um, sample.observed_mask, 0.1, max_lag=1
    )
    np.testing.assert_allclose(lag, [0.1])
    np.testing.assert_allclose(msd, [1.0])
    assert counts.tolist() == [17]


def test_measurement_aware_schema_adds_only_acquisition_fields() -> None:
    blind, labels, blind_names, _ = feature_matrix([linear_sample()], measurement_aware=False)
    aware, _, aware_names, _ = feature_matrix([linear_sample()], measurement_aware=True)
    assert labels.tolist() == ["directed"]
    assert aware.shape[1] == blind.shape[1] + 4
    assert aware_names[: len(blind_names)] == blind_names


def test_temporal_input_marks_missing_displacement_pairs() -> None:
    sample = linear_sample()
    sample.observed_mask[5] = False
    sample.observed_xy_um[5] = np.nan
    sequence, metadata = temporal_inputs(sample)
    assert sequence.shape == (3, 19)
    assert metadata.shape == (5,)
    assert sequence[2, 4] == 0
    assert sequence[2, 5] == 0
    assert np.isfinite(sequence).all()
