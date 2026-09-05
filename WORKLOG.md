# Project work log

Append material milestones and validation results here. Keep entries factual and
link to detailed experiment records where appropriate.

## 2026-09-04 — benchmark foundation and reproducible vertical slice

- Established package version `0.1.1` and the measurement-aware trajectory
  benchmark architecture.
- Added Brownian, directed, confined, fractional-Brownian subdiffusive, and
  two-state switching simulators.
- Added finite-exposure observation, localization noise, missing detections, and
  acquisition-aware feature extraction.
- Added classical and temporal baseline scaffolds, matched/stress evaluation,
  calibration, abstention, and identifiability summaries.
- Generated a smoke dataset and smoke-scale model/evaluation artifacts.
- Corrected switching generation so a switching-labelled sample contains at
  least one observed state transition.
- Recorded Git commits `fe401e4` and `a690346` for this foundation.
- Remaining before a reportable result: benchmark-scale execution, independent
  simulator validation, and trajectory-level bootstrap confidence intervals.

## 2026-09-04 — repository coordination records

- Added root `AGENTS.md`, `CONTEXT.md`, `MEMORY.md`, and `WORKLOG.md` so future
  sessions and agents share the project mission, constraints, decisions,
  current state, and chronological handoff.
- Indexed these records from the main README.
- Next action: revalidate the baseline and assess benchmark-scale runtime before
  launching the reportable run.

## 2026-09-04 — benchmark preflight

- Revalidated the existing baseline: 13 tests passed and lint checks passed.
- Found that dataset generation retained every fully materialized trajectory in
  memory until serialization, creating avoidable peak-memory risk for the
  50,000-sample benchmark configuration.
- Changed generation to deterministically shuffle lightweight sample
  specifications and stream trajectories directly into gzip JSON Lines.
- Added a deterministic, balanced miniature-generation regression test.
- The first real-range probe exposed a confinement failure when a valid step
  required more than the solver's arbitrary 12-reflection limit. Replaced the
  fixed limit with convergence-based reflection plus a pathological safety
  guard, and strengthened the confinement test at the benchmark's extreme
  radius/diffusion/frame-interval combination.
- A corrected 800-sample real-range probe completed in 9.492 seconds and
  produced 9,847,287 bytes.
- Generated and independently verified a 50,000-record benchmark candidate
  (592,985,089 bytes; SHA-256
  `2a3b43f72bc6f1207af0bd749cc6a9a42d772f073a5283d64825d1c8b07c9cb4`).
  The manifest records balanced state counts and the intended
  30,000/5,000/7,500/7,500 split counts.
- Trained benchmark-scale trajectory-only and measurement-aware classical
  baselines and generated matched/stress identifiability tables under
  `artifacts/benchmark_v1/`.
- Measurement-aware matched accuracy was 0.5783 (macro-F1 0.5748); stress
  accuracy was 0.3483 (macro-F1 0.3275). Stress ECE was 0.3057, showing that
  severe acquisition shift remains poorly calibrated.
- In stress bins with at least 100 samples, confined motion was most identifiable
  (up to 0.744 accuracy), while several subdiffusive bins reached only
  0.095-0.116 accuracy despite mean confidence around 0.61-0.68.
- Added deterministic state-stratified trajectory bootstrap support for metric
  confidence intervals; 15 tests and lint passed.
- Completed a 1,000-replicate bootstrap. Measurement-aware matched accuracy was
  0.5783 (95% CI 0.5675-0.5885); stress accuracy was 0.3483 (95% CI
  0.3381-0.3575); stress selective accuracy was 0.4310 (95% CI
  0.4141-0.4473).
- Preserved the exact candidate configuration, manifest, environment, compact
  metrics, interpretation, and publication caveat in
  `experiments/benchmark_v1_candidate_20260904/`.

## 2026-09-04 — independent latent-simulator cross-check

- Assessed the configured benchmark-scale temporal CNN on the local CPU-only
  PyTorch runtime. The attempt was stopped after about five minutes without a
  completed epoch and before any artifact was written; the full run is deferred
  to GPU-capable compute.
- Added a deterministic optional cross-check against `andi_datasets` 2.1.13 for
  latent Brownian and fractional-Brownian ensembles, including analytical MSD,
  between-simulator MSD, and lag-one increment-correlation checks.
- Ran 1,000 independent 256-step trajectories per model at seven lags. All
  predeclared checks passed. Maximum between-simulator MSD differences were
  0.71% (Brownian) and 0.86% (fractional Brownian); maximum errors relative to
  analytical MSD were 1.19% and 0.61%, respectively.
- Recorded configuration, thresholds, environment, Git state, scope caveat,
  and full metrics in
  `experiments/benchmark_v1_candidate_20260904/andi_crosscheck.json`.
- Scope remains limited: the cross-check does not validate directed drift,
  circular confinement, switching dynamics, or camera acquisition.

## 2026-09-05 - illustrated README and public architecture guide

- Replaced the introductory README with the current synthetic/experimental scope,
  model overview, same-dataset results, executable quick start and repository map.
  Added docs/GUIDE.md with physical/measurement assumptions, classical and temporal
  architecture, CLI boundaries, uncertainty interpretation and proposed Cell-iSCAT
  optical/transport/volume workflow. Experimental models remain explicitly planned.
- Added five reproducible figures as PNG/SVG: research workflow, synthetic motion
  families, controlled observation effects, accuracy/coverage-risk and all 40
  nonempty stress identifiability bins (7,500 tracks), including intervals/counts.
  The temporal configuration was verified to contain 26,373 trainable parameters.
- Added a Python renderer, seed/environment/input-hash manifest, and compact
  synthetic summaries under docs/results/20260905. Original source and normalized
  snapshot hashes are preserved; .gitattributes pins relevant text to LF.
  Generator, models, feature extraction and evaluation behavior were unchanged.
- Reviewed release scope in docs/DOCUMENTATION_RELEASE.md. No experimental pixels,
  trajectories, labels, private inventories, weights or generated datasets are
  included. User explicitly requested the GitHub documentation commit/push;
  Hugging Face/data/model publication remains outside this authorization.
- Preserved pre-existing uncommitted experimental/planning work. Staged only the
  documentation release and this entry/public context note, rather than all local
  changes. Summary source bytes agree after documented newline normalization.
- Validation: python -m pytest -> 29 passed in the active local tree;
  python -m ruff check . -> passed; python -m pip check -> passed.
  An isolated export of the staged release passed its 18 benchmark tests and Ruff;
  all five regenerated PNG hashes exactly matched the reviewed assets.
  Verified 75 staged Markdown links/anchors, staged renderer/input hashes,
  synthetic summary hashes, model parameter count and stress-bin coverage.
  Visually inspected all five figures. git diff --cached --check passed after
  removing Matplotlib's generated SVG trailing whitespace. Release-text scan found
  no credential patterns or machine-local source paths.
- The snapshot reproduces figures from saved estimates; full run-specific temporal
  evaluation and within-bin/paired-bootstrap audits remain in local run records.
  Their absence from the generic CLI is documented rather than implying a full
  evaluation replay. Synthetic performance remains no evidence of biological validity.
