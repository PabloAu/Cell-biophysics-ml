# Contributing

Contributions that improve physical validity, measurement realism,
reproducibility, or evaluation are welcome.

1. Open an issue describing the scientific or engineering motivation.
2. Add tests for every new simulator, corruption, feature, or metric.
3. Document units, parameter support, and assumptions.
4. Run `python -m pytest` and `ruff check .` before opening a pull request.
5. Do not add experimental Cell-iSCAT data or derived artifacts unless their
   release status and provenance have been reviewed.

Simulation changes that alter the generated distribution must increment the
simulator version and update the dataset card.
