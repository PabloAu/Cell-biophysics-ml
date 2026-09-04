# Repository instructions for coding agents

## Mission

Build a scientifically defensible, measurement-aware benchmark for determining
when intracellular dynamical states are identifiable from single-particle
trajectories. The primary result is an identifiability map with uncertainty,
not a single headline accuracy value.

## Read first

Before making changes, read:

1. `CONTEXT.md` for the current project state and next milestone.
2. `MEMORY.md` for durable scientific and engineering decisions.
3. `docs/scientific_design.md` and `docs/data_schema.md` for the evaluation
   contract and data invariants.
4. `WORKLOG.md` for recent work and validation results.

## Scientific guardrails

- Keep latent dynamics separate from the observation/acquisition process.
- Treat acquisition parameters as first-class variables.
- Never merge `test_matched` and `test_stress` for a primary result.
- Never tune models on either test split.
- Report macro and per-class metrics, calibration, coverage-risk behavior, and
  uncertainty across acquisition regimes.
- Bootstrap complete trajectories, not individual time steps.
- Do not claim synthetic performance establishes biological validity.
- Any change to a generated probability distribution must increment
  `SIMULATOR_VERSION`; schema-breaking changes require a major dataset version.

## Data safety

- Do not commit experimental Cell-iSCAT data, raw or derived, to Git.
- Experimental data may enter the workflow only after the governance gate in
  `docs/experimental_data_governance.md` is satisfied.
- Keep tracks from the same cell together. Prefer experiment/day-level splits
  for train, validation, and test partitions.
- Do not publish data, models, or a Space without explicit user approval and a
  reviewed dry-run publication plan.
- Never store access tokens or credentials in the repository.

## Engineering workflow

- Preserve deterministic seeds, exact YAML configuration, Git commit, runtime
  environment, and generated manifests for every reportable run.
- Add or update tests for changes to simulators, measurement corruption,
  features, metrics, and serialization.
- Run `python -m pytest` and `ruff check .` before considering code changes
  complete.
- Use `configs/smoke.yaml` only for functional validation. Use
  `configs/benchmark_v1.yaml` for reportable benchmark output.
- Keep generated datasets and model weights outside Git. Small configs,
  manifests, summaries, and interpretation notes may be committed deliberately.

## Project records

- Update `CONTEXT.md` when the active milestone or repository state changes.
- Update `MEMORY.md` only for durable decisions that should constrain future
  work; do not use it as a transcript.
- Append a dated entry to `WORKLOG.md` after material implementation or
  validation work, including commands/results and any unresolved issue.
