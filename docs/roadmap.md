# Roadmap

## Phase I — trajectory benchmark

- [x] Explicit latent and measurement simulators.
- [x] Brownian, directed, confined, subdiffusive, and switching states.
- [x] Matched and acquisition-stress splits.
- [x] Physical features, classical baseline, temporal baseline scaffold.
- [x] Calibration, abstention, and identifiability metrics.
- [x] Interactive simulator/explorer scaffold.
- [ ] Run benchmark-scale generation and training.
- [ ] Independent simulator cross-check and bootstrap confidence intervals.
- [ ] Publish versioned dataset, model, Space, and model weights.
- [ ] Add photon/PSF image formation and a raw-image baseline.

## Phase II — experimental bridge

- [ ] Approve Cell-iSCAT governance and release boundary.
- [ ] Define cell/experiment/perturbation metadata schema.
- [ ] Register trajectory, cell-volume, and osmotic-shock time axes.
- [ ] Build leakage-safe experiment-level splits.
- [ ] Quantify synthetic-to-experimental shift before fine-tuning.

## Phase III — physical cell world model

- [ ] Encode trajectory histories into a physical latent state `z_t`.
- [ ] Predict future latent state conditioned on time and perturbation.
- [ ] Attach interpretable heads for diffusion, anomalous exponent,
  confinement, transport, volume, and transition probabilities.
- [ ] Score future physical observables and perturbation response, not visual
  realism alone.
- [ ] Fuse image features, trajectories, and cell metadata.
