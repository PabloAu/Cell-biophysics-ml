"""Two-dimensional latent process simulators.

These simulators are deliberately explicit and small. They are benchmark
generators, not claims that each label maps to a unique cellular mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..types import AcquisitionParams, PhysicalParams


@dataclass(frozen=True)
class LatentTrajectory:
    dense_xy_um: np.ndarray
    frame_state: list[str]


def _fractional_gaussian_noise(
    n: int, hurst: float, rng: np.random.Generator
) -> np.ndarray:
    """Generate unit-variance fractional Gaussian noise with Davies-Harte.

    A small negative circulant eigenvalue can arise from floating-point error;
    materially negative eigenvalues raise instead of silently changing the
    process.
    """

    if not 0 < hurst < 1:
        raise ValueError("hurst must lie in (0, 1)")
    k = np.arange(n, dtype=float)
    gamma = 0.5 * (
        np.abs(k - 1.0) ** (2.0 * hurst)
        - 2.0 * k ** (2.0 * hurst)
        + (k + 1.0) ** (2.0 * hurst)
    )
    circulant = np.concatenate([gamma, [0.0], gamma[1:][::-1]])
    eigenvalues = np.fft.fft(circulant).real
    tolerance = 1e-10 * max(1.0, float(eigenvalues.max()))
    if eigenvalues.min() < -tolerance:
        raise RuntimeError("Davies-Harte embedding is not non-negative")
    eigenvalues = np.maximum(eigenvalues, 0.0)

    spectrum = np.zeros(2 * n, dtype=complex)
    spectrum[0] = np.sqrt(eigenvalues[0] / (2 * n)) * rng.normal()
    spectrum[n] = np.sqrt(eigenvalues[n] / (2 * n)) * rng.normal()
    normals = rng.normal(size=(n - 1, 2))
    spectrum[1:n] = np.sqrt(eigenvalues[1:n] / (4 * n)) * (
        normals[:, 0] + 1j * normals[:, 1]
    )
    spectrum[n + 1 :] = np.conjugate(spectrum[1:n][::-1])
    return np.fft.fft(spectrum).real[:n]


def _reflect_step(start: np.ndarray, delta: np.ndarray, radius: float) -> np.ndarray:
    """Specularly reflect a line segment at a circular boundary."""

    point = start.copy()
    remaining = delta.copy()
    reflections = 0
    while True:
        candidate = point + remaining
        if float(candidate @ candidate) <= radius * radius * (1.0 + 1e-12):
            return candidate
        reflections += 1
        if reflections > 100_000:
            raise RuntimeError("Reflection solver did not converge")
        a = float(remaining @ remaining)
        b = 2.0 * float(point @ remaining)
        c = float(point @ point) - radius * radius
        discriminant = max(0.0, b * b - 4.0 * a * c)
        candidate_roots = (
            (-b - np.sqrt(discriminant)) / (2 * a),
            (-b + np.sqrt(discriminant)) / (2 * a),
        )
        roots = [
            root
            for root in candidate_roots
            if 1e-12 <= root <= 1.0 + 1e-12
        ]
        if not roots:
            # Extremely large numerical overshoots are folded radially as a
            # safe, bounded fallback rather than returning an invalid point.
            norm = float(np.linalg.norm(candidate))
            folded = norm % (2.0 * radius)
            folded = folded if folded <= radius else 2.0 * radius - folded
            return candidate / norm * folded
        fraction = min(roots)
        hit = point + fraction * remaining
        normal = hit / radius
        tail = (1.0 - fraction) * remaining
        remaining = tail - 2.0 * float(tail @ normal) * normal
        point = hit - normal * (radius * 1e-12)


def _switching_states(
    n_frames: int, probability: float, rng: np.random.Generator
) -> np.ndarray:
    # The switching class is conditioned on at least one realized transition;
    # otherwise its label would be observationally identical to a single-state
    # trajectory by construction. Rejection sampling preserves the geometric
    # segment-length distribution under that condition.
    for _ in range(128):
        states = np.zeros(n_frames, dtype=np.int8)
        states[0] = int(rng.integers(0, 2))
        for index in range(1, n_frames):
            switch = rng.random() < probability
            states[index] = 1 - states[index - 1] if switch else states[index - 1]
        if np.any(states != states[0]):
            return states

    # Degenerate probabilities used in unit tests or interactive exploration
    # should still return a valid switching-class sample without hanging.
    change = int(rng.integers(1, n_frames - 1))
    states[change:] = 1 - states[0]
    return states


def simulate_latent(
    physical: PhysicalParams,
    acquisition: AcquisitionParams,
    rng: np.random.Generator,
) -> LatentTrajectory:
    """Simulate a dense latent path for later camera integration."""

    physical.validate()
    acquisition.validate()
    n_substeps = acquisition.n_frames * acquisition.oversample
    dt = acquisition.frame_interval_s / acquisition.oversample
    path = np.zeros((n_substeps + 1, 2), dtype=float)
    direction = np.array(
        [np.cos(physical.direction_rad), np.sin(physical.direction_rad)], dtype=float
    )

    if physical.state == "subdiffusive":
        hurst = physical.anomalous_alpha / 2.0
        scale = np.sqrt(2.0 * physical.diffusion_um2_s * dt**physical.anomalous_alpha)
        increments = np.column_stack(
            [
                _fractional_gaussian_noise(n_substeps, hurst, rng),
                _fractional_gaussian_noise(n_substeps, hurst, rng),
            ]
        )
        path[1:] = np.cumsum(scale * increments, axis=0)
        frame_state = ["subdiffusive"] * acquisition.n_frames
        return LatentTrajectory(path, frame_state)

    if physical.state == "switching":
        frame_codes = _switching_states(
            acquisition.n_frames, physical.switch_probability_per_frame, rng
        )
        frame_state = ["directed" if code else "brownian" for code in frame_codes]
    else:
        frame_codes = np.zeros(acquisition.n_frames, dtype=np.int8)
        frame_state = [physical.state] * acquisition.n_frames

    for step in range(n_substeps):
        frame = step // acquisition.oversample
        diffusion = physical.diffusion_um2_s
        drift = np.zeros(2, dtype=float)
        if physical.state == "directed" or (
            physical.state == "switching" and frame_codes[frame] == 1
        ):
            if physical.state == "switching" and physical.second_state_diffusion_um2_s:
                diffusion = physical.second_state_diffusion_um2_s
            drift = physical.velocity_um_s * direction * dt
        delta = drift + np.sqrt(2.0 * diffusion * dt) * rng.normal(size=2)
        if physical.state == "confined":
            path[step + 1] = _reflect_step(
                path[step], delta, float(physical.confinement_radius_um)
            )
        else:
            path[step + 1] = path[step] + delta

    return LatentTrajectory(path, frame_state)
