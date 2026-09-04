# Scientific design

## Thesis

The observed trajectory is not the underlying physical process. A useful model
must therefore be evaluated jointly over physical and measurement conditions.

## Position relative to existing benchmarks

The AnDi challenges established rigorous community benchmarks for anomalous
diffusion-model classification, anomalous-exponent regression, and change-point
analysis. DeepSPT demonstrated strong temporal segmentation and diffusional
fingerprinting for heterogeneous trajectories. DeepTRACE added an accessible
feature-engineering, annotation, model-training, and segmentation workflow. This
project complements them by making acquisition parameters explicit benchmark
axes and treating failure under acquisition shift as a primary result. Notably,
the second AnDi benchmark discussed motion blur but excluded it from the released
challenge observations to keep the ground truth focused; here it is a core axis.

The design is motivated by analytical work showing that localization noise and
finite camera exposure alter trajectory statistics, and by recent evidence that
the emission model can dominate the likelihood in image-based SPT. A later
benchmark version should therefore include photon-level image formation and
raw-image baselines rather than implying that Gaussian coordinate noise is a
complete observation model.

## Questions

1. Which state pairs become indistinguishable as localization uncertainty,
   exposure fraction, missingness, and track length change?
2. Does providing acquisition metadata improve calibration and robustness, or
   merely encourage shortcut learning?
3. When should a classifier abstain?
4. Which engineered physical summaries remain reliable under each regime?
5. How far do conclusions survive a simulator-family shift?

## Models in v0.1

- Brownian: isotropic Wiener process with diffusion coefficient `D`.
- Directed: Brownian diffusion plus constant drift vector.
- Confined: Brownian diffusion inside a reflecting circle.
- Subdiffusive: isotropic fractional Brownian motion with `alpha = 2H < 1`.
- Switching: a Markov switch between Brownian and directed dynamics.

All physical quantities use micrometres and seconds.

## Observation model in v0.1

The latent process is simulated on a finer time grid than the camera frames.
Each observation averages the latent position over the exposed fraction of the
frame, then adds independent Gaussian localization noise and applies missing
detections. End-of-frame latent positions are retained for comparison.

## Evaluation contract

- Never merge `test_matched` and `test_stress` for a primary score.
- Never tune on either test split.
- Report macro and per-class metrics.
- Report calibration and a coverage-risk curve.
- Bootstrap trajectories, not individual steps, for confidence intervals.
- Publish the generator config, manifest, simulator version, and Git commit.
- Run ablations with and without acquisition metadata.

## External validation ladder

1. Unit and property tests of each generator.
2. Recovery of known Brownian and directed parameters under ideal acquisition.
3. Cross-check selected trajectories against `andi-datasets` or an independent
   simulator.
4. Synthetic image formation and localization/tracking round trip.
5. Public experimental benchmark transfer.
6. Cell-iSCAT evaluation with experiment-level splits and release approval.

## References

- Muñoz-Gil et al. (2021), *Objective comparison of methods to decode
  anomalous diffusion*, Nature Communications.
  https://doi.org/10.1038/s41467-021-26320-w
- Muñoz-Gil et al. (2025), *Quantitative evaluation of methods to analyze
  motion changes in single-particle experiments*, Nature Communications.
  https://doi.org/10.1038/s41467-025-61949-x
- Berglund (2010), *Statistics of camera-based single-particle tracking*,
  Physical Review E. https://doi.org/10.1103/PhysRevE.82.011917
- Kæstel-Hansen et al. (2025), *Deep learning-assisted analysis of single-particle
  tracking for automated correlation between diffusion and function*, Nature
  Methods. https://doi.org/10.1038/s41592-025-02665-8
- Hendrix et al. (2025), *How Easy Is It to Learn Motion Models from Widefield
  Fluorescence Single Particle Tracks?*, arXiv preprint.
  https://doi.org/10.48550/arXiv.2507.05599
- Walker et al. (2026), *DeepTRACE brings flexible machine learning to
  single-molecule track analysis*, Communications Biology.
  https://doi.org/10.1038/s42003-026-09899-y
