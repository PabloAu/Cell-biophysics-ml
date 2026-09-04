# Experiments

Each reportable run should receive a timestamped directory containing:

- the exact YAML configuration;
- the Git commit and environment information;
- a dataset manifest/hash;
- scalar metrics and per-regime tables;
- figures and bootstrap confidence intervals;
- a short interpretation including failed expectations.

Generated outputs are ignored by Git; small configuration and summary files may
be committed deliberately.

## Recorded runs

- `benchmark_v1_candidate_20260904/` — benchmark-scale dataset, classical
  baseline, bootstrap intervals, and independent AnDi latent-dynamics
  cross-check; retained as a modified-working-tree preflight, not a
  publication-ready run.
