---
pretty_name: Cell Biophysics Trajectories
license: cc-by-4.0
task_categories:
- time-series-classification
- tabular-classification
tags:
- biology
- biophysics
- single-particle-tracking
- anomalous-diffusion
- synthetic
size_categories:
- 10K<n<100K
---

# Cell Biophysics Trajectories

Synthetic two-dimensional single-particle trajectories with paired latent and
observed paths. The benchmark is designed to measure which physical states
remain identifiable after realistic acquisition effects are applied.

## Dataset structure

Each JSONL record contains:

- `sample_id`, `split`, `regime`, `seed`, `simulator_version`
- `state_label`: `brownian`, `directed`, `confined`, `subdiffusive`, or
  `switching`
- `t_s`: frame times in seconds
- `latent_xy_um`: latent positions sampled at the end of each frame
- `observed_xy_um`: exposure-averaged, noisy positions; missing localizations
  are represented by `null`
- `observed_mask`: whether each frame has a detection
- `latent_state`: per-frame physical-state label
- `physical_params`: diffusion, velocity, anomalous exponent, confinement, and
  switching parameters
- `acquisition_params`: frame interval, exposure, trajectory length,
  localization uncertainty, missing-detection probability, and simulator
  oversampling

See `docs/data_schema.md` for field definitions and invariants.

## Splits

- `train` and `validation`: ordinary acquisition conditions.
- `test_matched`: held-out simulations drawn from the training support.
- `test_stress`: shorter, noisier, more incomplete, and more strongly blurred
  observations. It is intentionally out of distribution with respect to the
  acquisition parameters, while using the same physical-state families.

Splits are created during generation; users should not randomly repartition the
full file because that would erase the intended acquisition shift.

## Intended uses

- Benchmark physical-state classification and parameter inference.
- Compare measurement-blind with measurement-aware estimators.
- Study calibration, abstention, and failure boundaries.
- Develop simulation-based inference and domain-shift methods.

## Out-of-scope uses

- Direct biological or clinical interpretation without experimental
  validation.
- Treating the class labels as unique mechanistic explanations.
- Claiming performance on Cell-iSCAT or fluorescence microscopy solely from
  synthetic test accuracy.

## Known limitations

The observation model currently uses Gaussian localization uncertainty and
missing-at-random detections. It does not yet generate photons, point-spread
functions, tracking/linking errors, depth-dependent localization uncertainty,
or particle-particle interactions. Subdiffusion is represented by fractional
Brownian motion. These limitations are central benchmark extension targets.

## Licensing and provenance

Generated benchmark records are released under CC BY 4.0. The generating code
is Apache-2.0. No experimental or proprietary data is included.
