"""Regenerate public documentation figures from synthetic examples and frozen summaries.

Run from any directory: python scripts/render_documentation.py
No experimental input, model fitting, or test-set selection occurs here.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cell_biophysics_benchmark.constants import SIMULATOR_VERSION  # noqa: E402
from cell_biophysics_benchmark.measurement import observe_trajectory  # noqa: E402
from cell_biophysics_benchmark.simulation import simulate_latent  # noqa: E402
from cell_biophysics_benchmark.types import AcquisitionParams, PhysicalParams  # noqa: E402

OUT = ROOT / "docs/assets"
RESULTS = ROOT / "docs/results/20260905"
INK, TEAL, ORANGE = "#182c42", "#007f86", "#cf633c"
STATES = ["brownian", "directed", "confined", "subdiffusive", "switching"]
COLORS = [TEAL, "#327ab7", "#8f5baf", ORANGE, "#568046"]
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": "#bcc8cf",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.titleweight": "bold",
        "savefig.facecolor": "white",
        "svg.hashsalt": "cell-biophysics-docs-v1",
    }
)
EXAMPLES: list[dict] = []


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=160, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", metadata={"Date": None})
    svg = OUT / f"{name}.svg"
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    plt.close(fig)


def card(ax, x, y, w, h, title, body, color=TEAL):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            facecolor="#f2f6f8",
            edgecolor=color,
            linewidth=1.5,
        )
    )
    ax.text(
        x + w / 2,
        y + h * 0.73,
        title,
        ha="center",
        va="center",
        fontsize=12,
        weight="bold",
        color=color,
    )
    ax.text(x + w / 2, y + h * 0.33, body, ha="center", va="center", fontsize=10)


def arrow(ax, start, end):
    ax.annotate("", end, start, arrowprops={"arrowstyle": "->", "color": INK, "lw": 1.6})


def overview():
    fig, ax = plt.subplots(figsize=(14, 5.8))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.text(0.01, 0.95, "From latent motion to an identifiable claim", fontsize=23, weight="bold")
    ax.text(0.01, 0.87, "IMPLEMENTED  /  synthetic trajectory benchmark", color=TEAL, fontsize=12)
    card(
        ax,
        0.02,
        0.48,
        0.20,
        0.30,
        "1  Latent dynamics",
        "Five physical families\nKnown state + parameters",
    )
    card(
        ax,
        0.27,
        0.48,
        0.20,
        0.30,
        "2  Acquisition",
        "Exposure + localization\nFrame rate + missingness",
    )
    card(
        ax,
        0.52,
        0.48,
        0.20,
        0.30,
        "3  Inference",
        "Physical features + logistic model\nDisplacements + temporal CNN",
    )
    card(
        ax,
        0.77,
        0.48,
        0.20,
        0.30,
        "4  Evaluation",
        "Matched and stress separately\nAccuracy, uncertainty, abstention",
    )
    for x in [0.22, 0.47, 0.72]:
        arrow(ax, (x + 0.005, 0.63), (x + 0.043, 0.63))
    ax.text(
        0.01, 0.37, "PLANNED  /  experimental Cell-iSCAT model sequence", color=ORANGE, fontsize=12
    )
    card(
        ax,
        0.02,
        0.04,
        0.28,
        0.24,
        "Optical event model",
        "Fringe-moving / stable / unassessable\nIndependent timestamp labels",
        ORANGE,
    )
    card(
        ax,
        0.36,
        0.04,
        0.28,
        0.24,
        "Qualified observables",
        "Support-matched transport\nVolume only after calibration",
        ORANGE,
    )
    card(
        ax,
        0.70,
        0.04,
        0.28,
        0.24,
        "Response model comparison",
        "Current measured state vs. slow state\nHeld-out cells, days and future pulses",
        ORANGE,
    )
    arrow(ax, (0.315, 0.16), (0.343, 0.16))
    arrow(ax, (0.655, 0.16), (0.683, 0.16))
    save(fig, "research-workflow")


def trajectories():
    acq = AcquisitionParams(180, 0.03, 0.015, 0.025, 0.08, 8)
    fig, axes = plt.subplots(1, 5, figsize=(16, 4.3))
    for i, (state, ax, color) in enumerate(zip(STATES, axes, COLORS, strict=True)):
        physical = PhysicalParams(
            state,
            0.12,
            velocity_um_s=1.2 if state in {"directed", "switching"} else 0,
            direction_rad=0.5,
            confinement_radius_um=0.3 if state == "confined" else None,
            anomalous_alpha=0.55 if state == "subdiffusive" else 1,
            switch_probability_per_frame=0.025 if state == "switching" else 0,
            second_state_diffusion_um2_s=0.04 if state == "switching" else None,
        )
        latent = simulate_latent(physical, acq, np.random.default_rng(71 + i))
        _, endpoints, observed, _ = observe_trajectory(latent, acq, np.random.default_rng(171 + i))
        ax.plot(*endpoints.T, color="#a6b0b8", lw=1, label="Latent endpoints")
        ax.plot(*observed.T, ".-", color=color, lw=0.7, ms=2, label="Observed")
        ax.scatter(*endpoints[0], marker="s", s=25, c=INK, zorder=5, label="Latent start")
        ax.set(title=state.capitalize(), xlabel="x (µm)", ylabel="y (µm)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.15)
        EXAMPLES.append(
            {
                "name": state,
                "physical": asdict(physical),
                "acquisition": asdict(acq),
                "latent_seed": 71 + i,
                "observation_seed": 171 + i,
            }
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Five synthetic motion families", x=0.04, ha="left", fontsize=22, weight="bold")
    fig.text(
        0.04,
        0.88,
        "Illustrative draws • independent axis scales • lines break at missing detections",
    )
    fig.subplots_adjust(left=0.05, right=0.99, top=0.73, bottom=0.22, wspace=0.48)
    save(fig, "synthetic-trajectories")


def observation():
    physical = PhysicalParams("brownian", 0.04)
    base = AcquisitionParams(120, 0.03, 0, 0.003, 0, 8)
    latent = simulate_latent(physical, base, np.random.default_rng(601))
    settings = [
        base,
        replace(base, exposure_time_s=0.03, localization_sigma_um=0.003),
        replace(base, exposure_time_s=0.03, localization_sigma_um=0.12, missing_probability=0.35),
    ]
    titles = [
        "Near-instantaneous, low noise",
        "Full-frame exposure, low noise",
        "Full exposure + noise + missingness",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True)
    for i, (acq, ax, title) in enumerate(zip(settings, axes, titles, strict=True)):
        t, endpoints, observed, mask = observe_trajectory(latent, acq, np.random.default_rng(602))
        ax.plot(t, endpoints[:, 0], color="#a6b0b8", lw=1.8, label="Latent endpoint x")
        ax.plot(t, observed[:, 0], ".-", color=TEAL, lw=1, ms=3, label="Observed x")
        ax.set(title=title, xlabel="Time (s)")
        ax.grid(alpha=0.15)
        EXAMPLES.append(
            {
                "name": f"observation_{i}",
                "physical": asdict(physical),
                "acquisition": asdict(acq),
                "latent_seed": 601,
                "observation_seed": 602,
                "observed_count": int(mask.sum()),
            }
        )
    axes[0].set_ylabel("x (µm)")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(
        "One latent path, three observations", x=0.05, ha="left", fontsize=22, weight="bold"
    )
    fig.text(
        0.05,
        0.87,
        "Controlled illustration; the underlying motion and random seeds are held fixed.",
    )
    fig.subplots_adjust(left=0.06, right=0.99, top=0.73, bottom=0.16, wspace=0.20)
    save(fig, "observation-effects")


def results():
    table = pd.read_csv(RESULTS / "comparison.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    models = ["trajectory_only", "measurement_aware", "temporal_measurement_aware"]
    for split, color, offset, label in [
        ("test_matched", TEAL, -0.12, "Matched"),
        ("test_stress", ORANGE, 0.12, "Stress"),
    ]:
        rows = table[table.split == split].set_index("model").loc[models]
        values = rows.accuracy.to_numpy()
        axes[0].errorbar(
            values,
            np.arange(3) + offset,
            xerr=[values - rows.accuracy_lower_95, rows.accuracy_upper_95 - values],
            fmt="o",
            capsize=4,
            color=color,
            label=label,
        )
        curve = json.loads((RESULTS / f"coverage-risk_{split}.json").read_text())
        points = [
            (r["coverage"], r["risk"]) for r in curve if r["risk"] is not None and r["coverage"] > 0
        ]
        axes[1].plot(*np.asarray(points).T, color=color, label=label, lw=2)
        row = rows.loc["measurement_aware"]
        axes[1].scatter(row.coverage, 1 - row.selective_accuracy, color=color, s=50)
    axes[0].set(
        yticks=range(3),
        yticklabels=["Classical: trajectory only", "Classical: metadata", "Temporal CNN: metadata"],
        xlabel="Accuracy (95% trajectory-bootstrap CI)",
        xlim=(0.25, 0.65),
        title="Same synthetic dataset, frozen models",
    )
    axes[0].invert_yaxis()
    axes[1].set(
        xlabel="Coverage: fraction accepted",
        ylabel="Risk: error among accepted",
        title="Classical measurement-aware model",
        xlim=(0, 1.02),
        ylim=(0, 1),
    )
    axes[1].text(
        0.04,
        0.94,
        "Dots: fixed confidence threshold 0.70",
        transform=axes[1].transAxes,
        fontsize=10,
        va="top",
    )
    for ax in axes:
        ax.legend(frameon=False, loc="best")
        ax.grid(alpha=0.15)
    fig.suptitle(
        "Acquisition shift limits reliability", x=0.025, ha="left", fontsize=22, weight="bold"
    )
    fig.subplots_adjust(left=0.20, right=0.99, top=0.79, bottom=0.15, wspace=0.35)
    save(fig, "benchmark-results")


def identifiability():
    table = pd.read_csv(RESULTS / "identifiability_test_stress.csv")
    missing_bins = ["[0.2, 0.4)", "[0.4, 1.0)"]
    columns = [
        (sigma, length)
        for sigma in ["[80.0, 150.0)", "[150.0, inf)"]
        for length in ["[0.0, 50.0)", "[50.0, 100.0)"]
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.1), sharey=True)
    for ax, missing in zip(axes, missing_bins, strict=True):
        rows = table[(table.missing_bin == missing) & (table.exposure_bin == "[0.8, 1.01)")]
        grid = np.full((5, 4), np.nan)
        lookup = {}
        for r, state in enumerate(STATES):
            for c, (sigma, length) in enumerate(columns):
                found = rows[
                    (rows.state == state)
                    & (rows.localization_bin_nm == sigma)
                    & (rows.length_bin == length)
                ]
                if len(found) != 1:
                    raise ValueError("Expected one saved estimate per plotted stress bin")
                if len(found) == 1:
                    row = found.iloc[0]
                    grid[r, c] = row.accuracy
                    lookup[r, c] = row
        im = ax.imshow(grid, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
        for (r, c), row in lookup.items():
            sparse = "*" if row.n < 100 else ""
            label = (
                f"{row.accuracy:.2f}{sparse}\n"
                f"[{row.accuracy_lower_95:.2f}, {row.accuracy_upper_95:.2f}]\nn={row.n}"
            )
            ax.text(
                c,
                r,
                label,
                ha="center",
                va="center",
                fontsize=10,
                color="white" if row.accuracy > 0.60 else INK,
            )
        ax.set(
            xticks=range(4),
            xticklabels=[
                "80–<150 nm\n30–49 frames",
                "80–<150 nm\n50–79 frames",
                "150–200 nm\n30–49 frames",
                "150–200 nm\n50–79 frames",
            ],
            yticks=range(5),
            yticklabels=[s.capitalize() for s in STATES],
            title="Declared missing probability "
            + ("0.20–<0.40" if missing == missing_bins[0] else "0.40–0.50"),
        )
        ax.tick_params(length=0, pad=9)
    fig.suptitle("Where does the classifier fail?", x=0.02, ha="left", fontsize=23, weight="bold")
    fig.text(
        0.02,
        0.89,
        "Stress split only • classical measurement-aware model • exposure fraction 0.80–1.00",
    )
    fig.text(
        0.02,
        0.02,
        "Cells: accuracy [95% within-bin trajectory-bootstrap CI], n tracks. "
        "* n < 100. No interpolation; other parameters vary within each bin.",
        fontsize=10,
    )
    fig.subplots_adjust(left=0.105, right=0.89, top=0.79, bottom=0.16, wspace=0.10)
    cax = fig.add_axes((0.92, 0.20, 0.016, 0.53))
    fig.colorbar(im, cax=cax, label="Accuracy")
    save(fig, "identifiability-stress")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    overview()
    trajectories()
    observation()
    results()
    identifiability()
    manifest = {
        "description": "Synthetic illustrations and frozen synthetic benchmark summaries only",
        "simulator_version": SIMULATOR_VERSION,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "pandas": pd.__version__,
        "examples": EXAMPLES,
        "inputs_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(RESULTS.iterdir())
            if p.suffix in {".csv", ".json"}
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (OUT / "figure-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"Rendered five figures as PNG + SVG in {OUT}")


if __name__ == "__main__":
    main()
