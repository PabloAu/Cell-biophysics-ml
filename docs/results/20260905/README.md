# Synthetic result snapshot â€” 5 September 2026

This directory contains compact summaries of the completed synthetic benchmark,
exported for the README and guide. It contains no experimental records,
per-trajectory predictions, generated dataset or model weights.

## Which run?

- Source commit: `54d04681a255e5e9a25a710ef41dbd6222bcffdf`.
- Simulator: `0.1.1`; configuration: [benchmark_v1.yaml](../../../configs/benchmark_v1.yaml).
- Dataset SHA-256: `882b14b54577a7a6f238bb023cc3ae4deccb99ac5f4d3fe8c2b77dc414861e47`.
- 50,000 balanced synthetic tracks: 30,000 train; 5,000 validation;
  7,500 matched test; 7,500 stress test.
- Same ordered test IDs and true labels were verified across the three models.
- Classical models fitted on train only. Temporal checkpoint selected by
  validation loss (epoch 30 of 30); neither test set was used for fitting/tuning.
- Confidence threshold 0.70; 10 ECE bins; 1,000 bootstrap resamples;
  seed 20260904; 95% pointwise intervals.

[provenance.json](provenance.json) records model hashes, summary source names and
SHA-256 hashes, source/runtime information and scope. Summary payloads preserve the completed local run records, with line endings
normalized to LF (both source and snapshot hashes are recorded); machine-local executable and
library paths were omitted from the exported provenance.

## Files

| File | Contents |
| --- | --- |
| [comparison.csv](comparison.csv) | Six model/split rows with accuracy intervals, macro-F1, calibration and selective metrics |
| [classical-metrics.json](classical-metrics.json) | Both classical models: validation and separate test reports, per-class metrics, confusion and bootstrap summaries |
| [temporal-metrics.json](temporal-metrics.json) | Temporal validation and separate test reports, per-class metrics, confusion and test bootstrap summaries |
| [paired-accuracy-differences.json](paired-accuracy-differences.json) | Same-trajectory paired model comparisons |
| [identifiability_test_matched.csv](identifiability_test_matched.csv) | Measurement-aware classical acquisition/class bins, counts, accuracy intervals and sparse flags |
| [identifiability_test_stress.csv](identifiability_test_stress.csv) | Same bin report for the stress split |
| [coverage-risk_test_matched.json](coverage-risk_test_matched.json) | Measurement-aware classical matched acceptance/error curve |
| [coverage-risk_test_stress.json](coverage-risk_test_stress.json) | Measurement-aware classical stress acceptance/error curve |

These are saved estimates, not inputs for model selection. The renderer reads them
without refitting or changing intervals. They are sufficient to reproduce the
published figures, not to independently rerun model evaluation without the
corresponding dataset, checkpoints and run-specific audit.

## Interpretation and limits

Acquisition shift lowers accuracy and undermines confidence. Classical
measurement-aware accuracy exceeds temporal accuracy in this run, but the temporal
model has lower stress ECE and negative log likelihood. The two classical metadata
ablation accuracy differences include zero in their paired intervals.

Bootstrap units are complete synthetic trajectories. Aggregate/model-comparison
resampling is state-stratified; bin intervals condition on membership in a given
class/acquisition cell. Intervals do not cover training randomness, model choice,
simulator-family uncertainty or biological replication. Bin intervals are pointwise,
not simultaneous. Coverage-risk curves are point estimates without bands.

The original 4 September candidate has a different dataset file hash. Only the
regenerated dataset above underlies this comparison. The old full data is absent,
so the file-level difference has not been localized. Strict deterministic CUDA
algorithms were not enabled in the temporal run. Full local run records are retained
separately; this documentation release does not publish their entire working tree.

No experimental classifier, volume estimator or biological mechanism is validated
by these results. [Return to the guide](../../GUIDE.md).
