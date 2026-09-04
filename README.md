# Cell Biophysics ML

Research software for asking a deliberately harder question than trajectory
classification alone:

> **Under what measurement conditions can intracellular dynamical states
> actually be identified?**

The first project, `01_trajectory_benchmark`, is a measurement-aware benchmark
for single-particle trajectories. It generates a latent physical process and a
separate observation process, then evaluates both physics-informed and learned
models across acquisition regimes.

## Research program

```text
01_trajectory_benchmark/
  latent dynamics -> acquisition model -> inference -> identifiability maps

02_physical_world_model/ (planned)
  trajectory history -> physical latent state -> perturbed future observables
```

The benchmark currently covers Brownian diffusion, directed transport,
reflecting circular confinement, fractional-Brownian subdiffusion, and
two-state switching. The measurement layer covers finite exposure/motion blur,
frame interval, localization uncertainty, track length, and missing detections.

## Quick start

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
python -m pip install -e ".[dev,ml,space]"
python scripts/generate_dataset.py --config configs/smoke.yaml
python scripts/train_baseline.py --data data/generated/smoke.jsonl.gz
python scripts/run_baseline_ablation.py --data data/generated/smoke.jsonl.gz
python scripts/evaluate.py --data data/generated/smoke.jsonl.gz \
  --model artifacts/classical/model.joblib
python -m pytest
```

Launch the local interactive explorer:

```bash
python space/app.py
```

The smoke configuration is only for validation. Results intended for reporting
must use `configs/benchmark_v1.yaml`, record the Git commit, and retain the
generated manifest.

## What makes the benchmark useful

- The latent path and observed localization are stored separately.
- Acquisition parameters are first-class variables, not a single generic
  "noise" level.
- Matched test data and deliberately difficult acquisition stress tests are
  reported separately.
- The baseline comparison distinguishes trajectory-only features from
  measurement-aware features.
- Selective prediction is evaluated with coverage-risk curves, so a model can
  abstain when the observation does not support a reliable claim.
- Every generated sample carries its simulator version, seed, physical
  parameters, acquisition parameters, and per-frame latent state.

## Scope and scientific caution

This is a synthetic benchmark, not evidence that a classifier trained here is
valid for biological interpretation. Fractional Brownian motion is one useful
model of subdiffusion, not a unique mechanistic explanation. Confinement is
implemented with a reflecting circular boundary. The first switching model is
a two-state Brownian/directed process. These assumptions are explicit so they
can be challenged and extended.

Real Cell-iSCAT data will enter only after provenance, consent/IP, anonymization,
release permissions, and train/test leakage risks are reviewed. See
`docs/experimental_data_governance.md`.

## Planned Hugging Face artifacts

- Dataset: `PabloAu/cell-biophysics-trajectories`
- Model: `PabloAu/cell-biophysics-trajectory-classifier`
- Space: `PabloAu/cell-biophysics-trajectory-explorer`

The repository is the source of truth; generated data and model weights are
versioned separately on the Hugging Face Hub.

After a reportable benchmark run, prepare and validate the publication plan:

```bash
python scripts/prepare_space.py --model artifacts/classical/model.joblib
python scripts/publish_hf.py \
  --dataset data/generated/cell-biophysics-trajectories-v1.jsonl.gz \
  --model artifacts/classical/model.joblib
```

The publication script is a dry run unless `--apply` is supplied. Authenticate
with `hf auth login`, review the printed repository/file plan, then add
`--apply`. Do not publish smoke artifacts.

## Status

Version `0.1.1` is the benchmark foundation and reproducible vertical slice.
The published scientific result should be an identifiability map with
uncertainty, not an isolated headline accuracy.
