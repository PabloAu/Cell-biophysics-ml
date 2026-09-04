"""Typed benchmark records with explicit physical and acquisition parameters."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .constants import SIMULATOR_VERSION, STATE_LABELS


@dataclass(frozen=True)
class PhysicalParams:
    """Parameters of the latent two-dimensional process.

    ``diffusion_um2_s`` is the ordinary diffusion coefficient for Brownian,
    directed, confined, and switching trajectories. For fractional Brownian
    motion it acts as a generalized scale coefficient in um^2 / s^alpha.
    """

    state: str
    diffusion_um2_s: float
    velocity_um_s: float = 0.0
    direction_rad: float = 0.0
    confinement_radius_um: float | None = None
    anomalous_alpha: float = 1.0
    switch_probability_per_frame: float = 0.0
    second_state_diffusion_um2_s: float | None = None

    def validate(self) -> None:
        if self.state not in STATE_LABELS:
            raise ValueError(f"Unknown state {self.state!r}; expected one of {STATE_LABELS}")
        if self.diffusion_um2_s <= 0:
            raise ValueError("diffusion_um2_s must be positive")
        if self.velocity_um_s < 0:
            raise ValueError("velocity_um_s must be non-negative")
        if self.state == "confined" and (
            self.confinement_radius_um is None or self.confinement_radius_um <= 0
        ):
            raise ValueError("confined trajectories require a positive confinement radius")
        if self.state == "subdiffusive" and not 0 < self.anomalous_alpha < 1:
            raise ValueError("subdiffusive trajectories require 0 < anomalous_alpha < 1")
        if not 0 <= self.switch_probability_per_frame <= 1:
            raise ValueError("switch_probability_per_frame must be in [0, 1]")


@dataclass(frozen=True)
class AcquisitionParams:
    """Parameters of the camera-like observation process."""

    n_frames: int
    frame_interval_s: float
    exposure_time_s: float
    localization_sigma_um: float
    missing_probability: float = 0.0
    oversample: int = 8

    def validate(self) -> None:
        if self.n_frames < 3:
            raise ValueError("n_frames must be at least 3")
        if self.frame_interval_s <= 0:
            raise ValueError("frame_interval_s must be positive")
        if not 0 <= self.exposure_time_s <= self.frame_interval_s:
            raise ValueError("exposure_time_s must lie in [0, frame_interval_s]")
        if self.localization_sigma_um < 0:
            raise ValueError("localization_sigma_um must be non-negative")
        if not 0 <= self.missing_probability < 1:
            raise ValueError("missing_probability must lie in [0, 1)")
        if self.oversample < 1:
            raise ValueError("oversample must be at least 1")


@dataclass
class TrajectorySample:
    """A paired latent/observed trajectory and its full provenance."""

    sample_id: str
    split: str
    regime: str
    seed: int
    state_label: str
    t_s: np.ndarray
    latent_xy_um: np.ndarray
    observed_xy_um: np.ndarray
    observed_mask: np.ndarray
    latent_state: list[str]
    physical_params: PhysicalParams
    acquisition_params: AcquisitionParams
    simulator_version: str = SIMULATOR_VERSION

    def validate(self) -> None:
        self.physical_params.validate()
        self.acquisition_params.validate()
        n = self.acquisition_params.n_frames
        if self.state_label != self.physical_params.state:
            raise ValueError("state_label and physical_params.state disagree")
        if self.t_s.shape != (n,):
            raise ValueError(f"t_s must have shape ({n},)")
        if self.latent_xy_um.shape != (n, 2):
            raise ValueError(f"latent_xy_um must have shape ({n}, 2)")
        if self.observed_xy_um.shape != (n, 2):
            raise ValueError(f"observed_xy_um must have shape ({n}, 2)")
        if self.observed_mask.shape != (n,):
            raise ValueError(f"observed_mask must have shape ({n},)")
        if len(self.latent_state) != n:
            raise ValueError("latent_state must align with frames")
        if not np.all(np.diff(self.t_s) > 0):
            raise ValueError("t_s must be strictly increasing")
        missing_rows = ~np.isfinite(self.observed_xy_um).all(axis=1)
        if not np.array_equal(missing_rows, ~self.observed_mask):
            raise ValueError("observed_mask must match finite observed coordinates")

    def to_record(self) -> dict[str, Any]:
        """Convert to a JSON-compatible record using nulls for missing positions."""

        self.validate()
        observed: list[list[float | None]] = []
        for row in self.observed_xy_um:
            if np.isfinite(row).all():
                observed.append([float(row[0]), float(row[1])])
            else:
                observed.append([None, None])
        return {
            "sample_id": self.sample_id,
            "split": self.split,
            "regime": self.regime,
            "seed": int(self.seed),
            "simulator_version": self.simulator_version,
            "state_label": self.state_label,
            "t_s": self.t_s.astype(float).tolist(),
            "latent_xy_um": self.latent_xy_um.astype(float).tolist(),
            "observed_xy_um": observed,
            "observed_mask": self.observed_mask.astype(bool).tolist(),
            "latent_state": self.latent_state,
            "physical_params": asdict(self.physical_params),
            "acquisition_params": asdict(self.acquisition_params),
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> TrajectorySample:
        observed = np.asarray(
            [
                [np.nan, np.nan] if row[0] is None else [float(row[0]), float(row[1])]
                for row in record["observed_xy_um"]
            ],
            dtype=float,
        )
        sample = cls(
            sample_id=record["sample_id"],
            split=record["split"],
            regime=record["regime"],
            seed=int(record["seed"]),
            simulator_version=record.get("simulator_version", SIMULATOR_VERSION),
            state_label=record["state_label"],
            t_s=np.asarray(record["t_s"], dtype=float),
            latent_xy_um=np.asarray(record["latent_xy_um"], dtype=float),
            observed_xy_um=observed,
            observed_mask=np.asarray(record["observed_mask"], dtype=bool),
            latent_state=list(record["latent_state"]),
            physical_params=PhysicalParams(**record["physical_params"]),
            acquisition_params=AcquisitionParams(**record["acquisition_params"]),
        )
        sample.validate()
        return sample
