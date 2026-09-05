# Documentation release scope — 5 September 2026

The user requested a README and guide explaining the purpose, model architecture,
project structure and workflow, with example visuals, committed and pushed to the
existing GitHub repository. This record defines the concrete documentation release.

## Included

- Updated README and new detailed guide.
- Five computed synthetic/schematic figures, PNG/SVG sources, Python renderer and
  seed/environment/input-hash manifest.
- Compact synthetic benchmark summaries copied from the completed 5 September
  runs; sanitized provenance and explicit uncertainty/reproduction limits.
- This scope record, public context update and a documentation work-log entry.

The existing simulator, feature extraction, training and evaluation source remain
unchanged. No probability distribution or dataset schema changes; no version bump
is required. Figures are teaching examples or saved synthetic estimates and are
explicitly labeled accordingly.

## Excluded

Experimental raw/derived data, preview sprites, source path inventories, human
annotations, credentials, trained weights, complete generated datasets, private
Article records, and unrelated pending experimental implementation/planning work.
No Cell-iSCAT image or trajectory screenshot is part of this release. Private
viewing rights do not by themselves complete an experimental asset release review.

The requested GitHub commit/push is authorized. This is not a Hugging Face
publication, experimental data release, Space deployment or model-weight release.
The existing data/model publication gate remains in force.

## Review and validation

Check local links against the actual staged Git tree, not only the richer working
copy. Audit the exact staged file list and text for experimental payloads, local
source paths and credentials. Verify summary source hashes and numeric figure
coverage, inspect rendered figures for legibility, and run tests and Ruff before
pushing. Stage only this release; preserve unrelated existing edits. Record actual
commands and outcomes in WORKLOG.md.
