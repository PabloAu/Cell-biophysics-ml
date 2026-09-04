"""Camera-like coordinate observation with exposure averaging and missingness."""

from __future__ import annotations

import numpy as np

from ..simulation import LatentTrajectory
from ..types import AcquisitionParams


def observe_trajectory(
    latent: LatentTrajectory,
    acquisition: AcquisitionParams,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return frame times, latent endpoints, observations, and detection mask.

    Exposure is modeled by uniformly averaging dense positions over the final
    exposed fraction of each frame. A zero exposure samples the frame endpoint.
    Gaussian localization uncertainty is added after integration.
    """

    acquisition.validate()
    expected = acquisition.n_frames * acquisition.oversample + 1
    if latent.dense_xy_um.shape != (expected, 2):
        raise ValueError(f"dense latent path must have shape ({expected}, 2)")

    blocks = latent.dense_xy_um[1:].reshape(
        acquisition.n_frames, acquisition.oversample, 2
    )
    latent_endpoints = blocks[:, -1, :].copy()
    if acquisition.exposure_time_s == 0:
        integrated = latent_endpoints.copy()
    else:
        fraction = acquisition.exposure_time_s / acquisition.frame_interval_s
        exposed_steps = max(1, int(np.ceil(fraction * acquisition.oversample)))
        integrated = blocks[:, -exposed_steps:, :].mean(axis=1)

    observed = integrated + acquisition.localization_sigma_um * rng.normal(
        size=integrated.shape
    )
    mask = rng.random(acquisition.n_frames) >= acquisition.missing_probability
    if int(mask.sum()) < 3:
        required = np.linspace(0, acquisition.n_frames - 1, 3, dtype=int)
        mask[required] = True
    observed[~mask] = np.nan
    t_s = (np.arange(acquisition.n_frames, dtype=float) + 1.0) * acquisition.frame_interval_s
    return t_s, latent_endpoints, observed, mask
