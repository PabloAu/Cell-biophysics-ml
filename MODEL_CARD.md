---
library_name: scikit-learn
license: apache-2.0
pipeline_tag: tabular-classification
tags:
- biology
- biophysics
- single-particle-tracking
- interpretable-ml
datasets:
- PabloAu/cell-biophysics-trajectories
---

# Cell Biophysics Trajectory Classifier

Baseline classifiers for the Cell Biophysics Trajectory Benchmark.

## Models

1. A physics-feature logistic regression using displacement, MSD, geometry,
   persistence, and confinement-related summaries.
2. The same baseline augmented with known acquisition metadata.
3. A small masked one-dimensional temporal convolutional network operating on
   displacement sequences, with an optional acquisition-metadata branch.

The feature-only and temporal comparisons are meaningful only when evaluated
on the same predefined dataset split and simulator version.

## Primary evaluation

Report results separately for `test_matched` and `test_stress`:

- macro F1 and balanced accuracy;
- per-class recall and confusion matrix;
- multiclass Brier score and expected calibration error;
- selective accuracy and coverage at declared confidence thresholds;
- accuracy/coverage over acquisition-condition bins.

## Abstention

The default demonstration abstains when maximum predicted probability is below
the configured threshold. This threshold is a communication aid, not a formal
guarantee. It must be selected on validation data and reported with coverage
and risk.

## Limitations

Predictions identify the simulator class most compatible with an observation;
they do not establish a unique cellular mechanism. Synthetic-to-experimental
transfer is not validated in version 0.1. Gaussian localization noise is an
idealization, and the benchmark does not yet include raw image formation.
