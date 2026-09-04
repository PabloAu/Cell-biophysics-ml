# Benchmark v1 candidate — 2026-09-04

Status: **candidate only; not approved for publication**

This run completed benchmark-scale generation and the paired classical baseline
ablation. It is scientifically informative but not yet a reportable release
because generation occurred from a modified working tree. The dataset manifest
records the preceding commit `a690346e892303e452c0d840070cbef555ce3838`;
the streaming and reflection fixes were present but uncommitted. Regenerate from
a clean commit before publication.

## Dataset

- Samples: 50,000
- Train / validation / matched test / stress test: 30,000 / 5,000 / 7,500 / 7,500
- Samples per state: 10,000
- Compressed bytes: 592,985,089
- SHA-256: `2a3b43f72bc6f1207af0bd749cc6a9a42d772f073a5283d64825d1c8b07c9cb4`
- Simulator version: `0.1.1`
- Root seed: `20260904`

The exact generation configuration and manifest are preserved beside this
record. The generated JSON Lines dataset remains in the ignored
`data/generated/` directory.

## Classical baseline findings

| Model / split | Accuracy | Macro-F1 | ECE | Coverage at 0.70 | Selective accuracy |
|---|---:|---:|---:|---:|---:|
| Trajectory-only, matched | 0.5733 | 0.5693 | 0.0248 | 0.2284 | 0.9130 |
| Measurement-aware, matched | 0.5783 | 0.5748 | 0.0191 | 0.2397 | 0.9194 |
| Trajectory-only, stress | 0.3423 | 0.3068 | 0.3721 | 0.5327 | 0.4053 |
| Measurement-aware, stress | 0.3483 | 0.3275 | 0.3057 | 0.4127 | 0.4310 |

Measurement metadata provides a modest matched improvement and a larger stress
improvement in macro-F1 and calibration, but it does not make the stress regime
reliable. The high-confidence subset remains wrong more often than correct under
stress.

For the measurement-aware model, a 1,000-replicate state-stratified trajectory
bootstrap gave:

- matched accuracy 0.5783, 95% CI [0.5675, 0.5885];
- matched macro-F1 0.5748, 95% CI [0.5637, 0.5853];
- stress accuracy 0.3483, 95% CI [0.3381, 0.3575];
- stress macro-F1 0.3275, 95% CI [0.3174, 0.3375];
- stress selective accuracy 0.4310, 95% CI [0.4141, 0.4473].

In stress cells containing at least 100 trajectories, confined motion reached up
to 0.744 accuracy. Several subdiffusive cells were only 0.095–0.116 accurate
despite mean confidence around 0.61–0.68. This confident failure is a primary
result, not a detail to average away.

## Environment

- Windows 11 `10.0.26200`
- Python `3.12.10`
- NumPy `2.5.2`
- pandas `3.0.5`
- scikit-learn `1.9.0`
- SciPy `1.18.1`
- joblib `1.6.0`
- PyTorch `2.14.0+cpu`; CUDA unavailable
- PyYAML `6.0.3`

## Artifacts

Ignored runtime artifacts are under `artifacts/benchmark_v1/`:

- `classical_ablation/comparison.json`
- `classical_ablation/trajectory_only.joblib`
- `classical_ablation/measurement_aware.joblib`
- `classical_evaluation/metrics.json`
- matched and stress identifiability CSV tables

## Independent simulator cross-check

`andi_crosscheck.json` records a deterministic comparison against
`andi_datasets` 2.1.13 using 1,000 independent 256-step trajectories per model.
At lags 1–64, both implementations were compared with the analytical MSD and
with each other, and lag-one increment correlations were compared with the
fractional-Gaussian-noise expectation.

All checks passed. The largest between-simulator MSD difference was 0.71% for
Brownian motion and 0.86% for fractional Brownian motion. The largest MSD error
relative to theory across either implementation was 1.19% for Brownian motion
and 0.61% for fractional Brownian motion. The check is deliberately limited to
latent Brownian and fractional-Brownian dynamics; it does not validate directed
drift, circular confinement, switching, or the acquisition model.

## Temporal baseline feasibility

This host exposes PyTorch 2.14.0 with 12 CPU training threads and no CUDA. A
benchmark-scale attempt was stopped after about five minutes without a
completed epoch, before any model artifact was written. The configured 30-epoch
run is therefore deferred to GPU-capable compute.

## Remaining gates

1. Commit and review the generator/reflection/bootstrap changes, then regenerate
   from that clean commit.
2. Extend independent validation beyond Brownian and fractional-Brownian latent
   dynamics where genuinely equivalent external implementations exist.
3. Run the temporal baseline on GPU-capable compute.
4. Produce final figures and review the failure interpretation.
5. Prepare a Hugging Face dry-run publication plan. Publication still requires
   explicit approval.
