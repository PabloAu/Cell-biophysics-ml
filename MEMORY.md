# Durable project memory

This file records decisions that should survive across sessions and agents. It
is intentionally concise; chronological activity belongs in `WORKLOG.md` and
the current state belongs in `CONTEXT.md`.

## Scientific decisions

- The observed trajectory is not the latent physical process. Simulation and
  evaluation must keep latent dynamics and acquisition effects separate.
- The principal deliverable is an uncertainty-aware identifiability map across
  physical and measurement regimes, not an isolated accuracy score.
- Matched and acquisition-stress test sets are distinct reporting surfaces and
  must never be merged for a primary score.
- Measurement-aware versus trajectory-only ablation is mandatory because
  acquisition metadata may improve inference or enable shortcut learning.
- Selective prediction is part of the evaluation: the system should abstain
  when measurements do not support a reliable claim.
- Fractional Brownian motion is a model of subdiffusion, not a unique mechanistic
  explanation.
- Switching-class samples are conditioned on at least one realized transition.

## Reproducibility decisions

- Gzip-compressed JSON Lines is the canonical generated-data representation.
- All physical distances use micrometres and times use seconds.
- Reportable runs use `configs/benchmark_v1.yaml`; `configs/smoke.yaml` is for
  validation only.
- Every reportable run preserves its configuration, Git commit, environment,
  simulator version, dataset manifest/hash, metrics, uncertainty tables, and a
  written interpretation including failed expectations.
- Generated data and model weights are kept out of Git and versioned separately
  on the Hugging Face Hub only after review.

## Experimental-data decisions

- No experimental data enters Git.
- Ownership, release authorization, sensitivity, provenance, transformations,
  calibration, acquisition settings, perturbation timing, hierarchy, and
  licensing must be recorded before experimental use.
- Leakage-safe splits operate at the biological experiment or acquisition-day
  level; tracks from one cell must remain in one partition.
- Synthetic-to-experimental shift must be quantified before fine-tuning or
  biological interpretation.

## Planned Hugging Face destinations

- Dataset: `PabloAu/cell-biophysics-trajectories`
- Model: `PabloAu/cell-biophysics-trajectory-classifier`
- Space: `PabloAu/cell-biophysics-trajectory-explorer`

These names are plans, not authorization to publish.
