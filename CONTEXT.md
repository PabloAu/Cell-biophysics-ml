# Current project context

Last updated: 2026-09-04

## Objective

The repository implements the first stage of a cell-biophysics research
program: a measurement-aware benchmark for identifying dynamical states from
single-particle trajectories. The scientific question is when a state is
identifiable under realistic acquisition conditions, rather than which model
achieves the highest aggregate accuracy.

## Current state

- Package version: `0.1.1`.
- Current Git baseline at this snapshot: `a690346`.
- Implemented latent states: Brownian, directed, confined, subdiffusive, and
  two-state switching trajectories.
- Implemented observation effects: frame interval, finite exposure/motion blur,
  localization uncertainty, missing detections, and track length.
- Implemented models: physical-feature classical baseline and temporal baseline
  scaffold.
- Implemented evaluation: matched/stress splits, calibration, selective
  prediction, and acquisition-binned identifiability tables.
- Smoke dataset, classical artifacts, ablation output, temporal artifact, and
  local Space package have been generated for functional validation only.
- Experimental Cell-iSCAT data has not entered the repository.

## Active milestone

Complete the reportable Phase I benchmark before requesting experimental data:

1. Revalidate tests and lint from the current clean baseline. Completed on
   2026-09-04.
2. Assess the benchmark-scale generator for runtime and memory feasibility.
   Peak-memory risk from materializing all samples was identified and replaced
   with deterministic streaming. A probe then exposed an arbitrary 12-bounce
   failure in confined paths at valid benchmark extremes; the reflection solver
   now runs to convergence with a pathological safety guard. Runtime probing
   remains.
3. Generate the versioned dataset from `configs/benchmark_v1.yaml`, preserving
   its manifest and exact provenance. A verified 50,000-sample candidate was
   generated on 2026-09-04; its SHA-256 is
   `2a3b43f72bc6f1207af0bd749cc6a9a42d772f073a5283d64825d1c8b07c9cb4`.
4. Train the classical measurement-aware and trajectory-only ablation models.
   Completed for the benchmark candidate under `artifacts/benchmark_v1/`.
5. Train/evaluate the temporal baseline if local compute is adequate. A
   benchmark-scale CPU attempt was stopped after about five minutes without a
   completed epoch; no artifact was produced, and the run is deferred to
   GPU-capable compute.
6. Evaluate matched and stress splits separately.
7. Add trajectory-level bootstrap confidence intervals and an independent
   simulator cross-check before treating results as reportable. The bootstrap
   implementation and 1,000-replicate classical intervals are complete. A
   1,000-trajectory-per-model cross-check against `andi_datasets` 2.1.13 passed
   for latent Brownian motion and fractional Brownian motion; its intentionally
   limited scope does not cover directed drift, circular confinement,
   switching, or the observation model.
8. Prepare the Hugging Face publication plan as a dry run. Publishing requires
   explicit user approval and authenticated local credentials.

## Current limitations

- Benchmark-scale candidate artifacts now exist, but the run was generated from
  a modified working tree whose manifest points to its prior commit. It must be
  regenerated from a clean committed tree before publication.
- The independent AnDi cross-check covers Brownian and fractional-Brownian
  latent dynamics only; no equivalent external implementation has yet checked
  directed drift, circular confinement, switching, or camera acquisition.
- The temporal baseline remains untrained because this host has CPU-only
  PyTorch and the benchmark-scale attempt was not locally practical.
- Detailed provenance and findings for the current candidate are in
  `experiments/benchmark_v1_candidate_20260904/`.
- Photon/PSF image formation and a raw-image baseline are not yet implemented.
- No claims about transfer to experimental biology are supported yet.

## Key references within this repository

- Scientific contract: `docs/scientific_design.md`
- Dataset schema: `docs/data_schema.md`
- Roadmap: `docs/roadmap.md`
- Experimental data gate: `docs/experimental_data_governance.md`
- Run-record requirements: `experiments/README.md`
