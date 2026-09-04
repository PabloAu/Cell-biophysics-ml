# Smoke validation — v0.1

Date: 2026-09-04
Configuration: `configs/smoke.yaml`
Dataset records: 175
Dataset SHA-256: `81fd497aa3ffed9e456d1b2a4201521f4ddff2cffdf73fa095897a2d778b3cc7`

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
| Feature + acquisition / validation | 0.56 | 0.533 | 0.191 | 0.36 | 1.00 |
| Feature + acquisition / matched test | 0.56 | 0.567 | 0.078 | 0.24 | 1.00 |
| Feature + acquisition / stress test | 0.20 | 0.214 | 0.479 | 0.48 | 0.333 |
| Feature only / validation | 0.52 | 0.500 | 0.236 | 0.28 | 1.00 |
| Temporal CNN + acquisition / validation | 0.28 | 0.257 | 0.054 | 0.00 | — |

The stress split exposes a useful failure: the tiny measurement-aware baseline
is confidently wrong too often. That is evidence that the abstention threshold
cannot be treated as a guarantee and must be selected and calibrated on a
larger, representative validation set.
