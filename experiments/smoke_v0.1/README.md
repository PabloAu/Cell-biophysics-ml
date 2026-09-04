# Smoke validation — v0.1

Date: 2026-09-04
Configuration: `configs/smoke.yaml`
Dataset records: 175
Dataset SHA-256: `ce99063766041aef4ca3ddeca0b20f5effd7a3cd611ea6a5be0d4d639ff92625`

This run validates wiring and failure reporting. It is intentionally too small
for scientific comparison and must not be quoted as benchmark performance.

## Checks

- 13 simulator, measurement, serialization, feature, temporal-input, and metric
  tests passed.
- Static code-quality checks passed.
- The self-contained Gradio bundle loaded locally, returned a simulated result,
  updated after changing state, and produced no browser console errors.

## Diagnostic results

| Model / split | Accuracy | Macro F1 | ECE | Coverage at 0.70 | Selective accuracy |
|---|---:|---:|---:|---:|---:|
| Feature + acquisition / validation | 0.60 | 0.573 | 0.183 | 0.36 | 1.00 |
| Feature + acquisition / matched test | 0.52 | 0.523 | 0.121 | 0.24 | 1.00 |
| Feature + acquisition / stress test | 0.24 | 0.251 | 0.447 | 0.44 | 0.364 |
| Feature only / validation | 0.48 | 0.444 | 0.172 | 0.28 | 1.00 |
| Temporal CNN + acquisition / validation | 0.28 | 0.253 | 0.054 | 0.00 | — |

The stress split exposes a useful failure: the tiny measurement-aware baseline
is confidently wrong too often. That is evidence that the abstention threshold
cannot be treated as a guarantee and must be selected and calibrated on a
larger, representative validation set.
