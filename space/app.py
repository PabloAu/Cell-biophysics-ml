from __future__ import annotations

import os
import sys
from pathlib import Path

import gradio as gr
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for candidate in (HERE / "src", ROOT / "src"):
    if candidate.exists():
        sys.path.insert(0, str(candidate))

from cell_biophysics_benchmark.constants import STATE_LABELS  # noqa: E402
from cell_biophysics_benchmark.features import extract_features, time_averaged_msd  # noqa: E402
from cell_biophysics_benchmark.measurement import observe_trajectory  # noqa: E402
from cell_biophysics_benchmark.models import (  # noqa: E402
    load_classical_model,
    predict_classical,
)
from cell_biophysics_benchmark.simulation import simulate_latent  # noqa: E402
from cell_biophysics_benchmark.types import (  # noqa: E402
    AcquisitionParams,
    PhysicalParams,
    TrajectorySample,
)

MODEL_PATH = Path(os.getenv("CELL_BIOPHYSICS_MODEL", HERE / "model.joblib"))
MODEL = load_classical_model(MODEL_PATH) if MODEL_PATH.exists() else None


def _heuristic_probabilities(features: dict[str, float]) -> np.ndarray:
    """Fallback for an unbundled development checkout; not a benchmark model."""

    alpha = float(np.nan_to_num(features["msd_alpha"], nan=1.0))
    straight = float(features["straightness"])
    asymmetry = float(features["asymmetry"])
    turn = float(np.nan_to_num(features["mean_turn_cosine"], nan=0.0))
    step_cv = float(features["step_std_um"] / (features["mean_step_um"] + 1e-9))
    logits = np.asarray(
        [
            -3.0 * abs(alpha - 1.0) - straight - asymmetry,
            3.5 * straight + 1.5 * asymmetry + max(0.0, alpha - 1.0) + turn,
            2.0 * max(0.0, 0.9 - alpha) - 1.5 * straight,
            2.5 * max(0.0, 0.95 - alpha) - 0.5 * asymmetry,
            1.3 * step_cv + 0.5 * asymmetry - abs(alpha - 1.0),
        ]
    )
    logits -= logits.max()
    return np.exp(logits) / np.exp(logits).sum()


def _failure_note(sample: TrajectorySample, confidence: float, threshold: float) -> str:
    acquisition = sample.acquisition_params
    physical = sample.physical_params
    diffusive_step = np.sqrt(4.0 * physical.diffusion_um2_s * acquisition.frame_interval_s)
    localization_ratio = acquisition.localization_sigma_um / max(diffusive_step, 1e-12)
    warnings: list[str] = []
    if localization_ratio > 0.5:
        warnings.append(
            "localization uncertainty is large relative to a diffusive frame-to-frame step"
        )
    if acquisition.exposure_time_s / acquisition.frame_interval_s > 0.8:
        warnings.append("long exposure strongly averages motion within each frame")
    if sample.observed_mask.sum() < 50:
        warnings.append("fewer than 50 positions were observed")
    if acquisition.missing_probability >= 0.2:
        warnings.append("missing detections substantially fragment the path")
    decision = (
        f"**Prediction accepted** at confidence `{confidence:.2f}`."
        if confidence >= threshold
        else f"**Abstain**: maximum probability `{confidence:.2f}` is below `{threshold:.2f}`."
    )
    if warnings:
        decision += "\n\nIdentifiability warning: " + "; ".join(warnings) + "."
    return decision


def simulate_and_explain(
    state: str,
    diffusion: float,
    velocity: float,
    confinement_radius: float,
    anomalous_alpha: float,
    switch_probability: float,
    localization_nm: float,
    frame_interval_ms: float,
    exposure_fraction: float,
    trajectory_length: int,
    missing_probability: float,
    confidence_threshold: float,
    seed: int,
):
    physical = PhysicalParams(
        state=state,
        diffusion_um2_s=float(diffusion),
        velocity_um_s=float(velocity) if state in {"directed", "switching"} else 0.0,
        direction_rad=np.pi / 6,
        confinement_radius_um=float(confinement_radius) if state == "confined" else None,
        anomalous_alpha=float(anomalous_alpha) if state == "subdiffusive" else 1.0,
        switch_probability_per_frame=(float(switch_probability) if state == "switching" else 0.0),
        second_state_diffusion_um2_s=(0.25 * float(diffusion) if state == "switching" else None),
    )
    frame_interval = float(frame_interval_ms) / 1000.0
    acquisition = AcquisitionParams(
        n_frames=int(trajectory_length),
        frame_interval_s=frame_interval,
        exposure_time_s=frame_interval * float(exposure_fraction),
        localization_sigma_um=float(localization_nm) / 1000.0,
        missing_probability=float(missing_probability),
        oversample=8,
    )
    seeds = np.random.SeedSequence(int(seed)).spawn(2)
    latent = simulate_latent(physical, acquisition, np.random.default_rng(seeds[0]))
    t_s, latent_xy, observed_xy, mask = observe_trajectory(
        latent, acquisition, np.random.default_rng(seeds[1])
    )
    sample = TrajectorySample(
        sample_id=f"interactive-{seed}",
        split="interactive",
        regime="user_selected",
        seed=int(seed),
        state_label=state,
        t_s=t_s,
        latent_xy_um=latent_xy,
        observed_xy_um=observed_xy,
        observed_mask=mask,
        latent_state=latent.frame_state,
        physical_params=physical,
        acquisition_params=acquisition,
    )
    features = extract_features(sample, measurement_aware=True)
    if MODEL is not None:
        probability_matrix, model_classes = predict_classical(MODEL, [sample])
        probability = probability_matrix[0]
        classes = list(model_classes)
        model_status = "Loaded trained physics-feature baseline."
    else:
        probability = _heuristic_probabilities(features)
        classes = list(STATE_LABELS)
        model_status = "Development fallback heuristic (not a trained benchmark result)."
    prediction_index = int(np.argmax(probability))
    prediction = classes[prediction_index]
    confidence = float(probability[prediction_index])

    figure, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    axes[0, 0].plot(latent_xy[:, 0], latent_xy[:, 1], color="#20639b", lw=1.5)
    axes[0, 0].scatter(latent_xy[0, 0], latent_xy[0, 1], color="#173f5f", s=30, label="start")
    axes[0, 0].set(title=f"Latent: {state}", xlabel="x (µm)", ylabel="y (µm)")
    axes[0, 0].axis("equal")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(observed_xy[:, 0], observed_xy[:, 1], color="#ed553b", lw=1.0)
    axes[0, 1].scatter(observed_xy[mask, 0], observed_xy[mask, 1], color="#ed553b", s=7)
    axes[0, 1].set(title="Observed trajectory", xlabel="x (µm)", ylabel="y (µm)")
    axes[0, 1].axis("equal")

    latent_mask = np.ones(len(latent_xy), dtype=bool)
    latent_lag, latent_msd, _ = time_averaged_msd(
        latent_xy, latent_mask, acquisition.frame_interval_s
    )
    observed_lag, observed_msd, _ = time_averaged_msd(
        observed_xy, mask, acquisition.frame_interval_s
    )
    axes[1, 0].loglog(latent_lag, latent_msd, "o-", label="latent", color="#20639b")
    axes[1, 0].loglog(observed_lag, observed_msd, "o-", label="observed", color="#ed553b")
    axes[1, 0].set(title="Time-averaged MSD", xlabel="lag (s)", ylabel="MSD (µm²)")
    axes[1, 0].legend(frameon=False)

    axes[1, 1].barh(classes, probability, color="#3caea3")
    axes[1, 1].axvline(confidence_threshold, color="#7a1f5c", ls="--", lw=1, label="threshold")
    axes[1, 1].set(title="Classifier probabilities", xlabel="probability", xlim=(0, 1))
    axes[1, 1].legend(frameon=False)

    selected_features = [
        "observed_fraction",
        "mean_step_um",
        "straightness",
        "radius_gyration_um",
        "asymmetry",
        "mean_turn_cosine",
        "msd_alpha",
        "apparent_diffusion_um2_s",
        "msd_plateau_ratio",
    ]
    feature_table = pd.DataFrame(
        {"feature": selected_features, "value": [features[name] for name in selected_features]}
    )
    probability_table = pd.DataFrame({"state": classes, "probability": probability})
    decision = (
        f"### `{prediction}`\n\n{_failure_note(sample, confidence, confidence_threshold)}"
        f"\n\n_Model status: {model_status}_"
    )
    return figure, feature_table, probability_table, decision


CSS = """
.gradio-container {max-width: 1320px !important;}
.intro {max-width: 900px;}
"""

with gr.Blocks(title="Cell Biophysics Trajectory Explorer") as demo:
    gr.Markdown(
        """
        # Cell Biophysics Trajectory Explorer
        Move from **latent physics** to **what the microscope reports**. The key
        output is not only a class: it is whether the observation supports an
        identifiable claim under the selected acquisition conditions.
        """,
        elem_classes=["intro"],
    )
    with gr.Row():
        with gr.Column(scale=1):
            state = gr.Dropdown(STATE_LABELS, value="brownian", label="Latent physical state")
            diffusion = gr.Slider(0.005, 5.0, value=0.5, step=0.005, label="D (µm²/s)")
            velocity = gr.Slider(0.0, 5.0, value=0.8, step=0.05, label="Velocity (µm/s)")
            confinement_radius = gr.Slider(
                0.05, 2.0, value=0.5, step=0.01, label="Confinement radius (µm)"
            )
            anomalous_alpha = gr.Slider(
                0.25, 0.95, value=0.65, step=0.01, label="Anomalous exponent α"
            )
            switch_probability = gr.Slider(
                0.001, 0.15, value=0.02, step=0.001, label="Switch probability / frame"
            )
        with gr.Column(scale=1):
            localization_nm = gr.Slider(
                0, 200, value=40, step=1, label="Localization uncertainty (nm)"
            )
            frame_interval_ms = gr.Slider(
                5, 250, value=50, step=1, label="Frame interval (ms)"
            )
            exposure_fraction = gr.Slider(
                0, 1, value=0.8, step=0.01, label="Exposure / frame interval"
            )
            trajectory_length = gr.Slider(
                30, 500, value=200, step=1, label="Trajectory length (frames)"
            )
            missing_probability = gr.Slider(
                0, 0.5, value=0.05, step=0.01, label="Missing-detection probability"
            )
            confidence_threshold = gr.Slider(
                0.4, 0.95, value=0.7, step=0.01, label="Abstention threshold"
            )
            seed = gr.Number(value=7, precision=0, label="Simulation seed")
            run = gr.Button("Simulate and infer", variant="primary")
    plot = gr.Plot(label="Latent process, observation, MSD, and inference")
    decision = gr.Markdown()
    with gr.Row():
        features = gr.Dataframe(label="Physical features", interactive=False)
        probabilities = gr.Dataframe(label="Class probabilities", interactive=False)

    inputs = [
        state,
        diffusion,
        velocity,
        confinement_radius,
        anomalous_alpha,
        switch_probability,
        localization_nm,
        frame_interval_ms,
        exposure_fraction,
        trajectory_length,
        missing_probability,
        confidence_threshold,
        seed,
    ]
    outputs = [plot, features, probabilities, decision]
    run.click(simulate_and_explain, inputs=inputs, outputs=outputs, api_name="simulate")
    demo.load(simulate_and_explain, inputs=inputs, outputs=outputs)


if __name__ == "__main__":
    demo.launch(css=CSS)
