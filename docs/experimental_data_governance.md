# Experimental data governance gate

The synthetic benchmark is intentionally separated from Cell-iSCAT data. Before
experimental data enters this repository or any Hub artifact, record:

- ownership and institutional release authorization;
- applicable collaborator, funder, and publication constraints;
- whether metadata contains personal, clinical, or sensitive information;
- original file provenance and immutable checksums;
- transformations from raw images to volume and trajectories;
- calibration, acquisition, perturbation timing, and quality-control metadata;
- experiment/day/cell/particle hierarchy for leakage-safe splitting;
- a public-data license or an explicit decision to keep the data private.

The default split unit should be the biological experiment or acquisition day,
not a randomly selected trajectory. Tracks from the same cell must not be split
across train and test. Osmotic-shock prediction should hold out entire cells and
preferably experimental days and perturbation conditions.

No experimental data should be committed to Git. Use checksummed manifests and
a controlled data store; publish only approved derived artifacts.
