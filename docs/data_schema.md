# Data schema and invariants

The canonical on-disk format is gzip-compressed JSON Lines. This preserves
nested arrays without requiring a binary dependency and is directly loadable by
the Hugging Face `datasets` library. Parquet derivatives may be published for
fast scanning but are not the source representation.

## Units

| Field suffix | Unit |
|---|---|
| `_um` | micrometre |
| `_um2_s` | square micrometre per second |
| `_um_s` | micrometre per second |
| `_s` | second |

## Invariants

- `len(t_s) == len(latent_xy_um) == len(observed_xy_um)`.
- `observed_mask[i]` is false exactly when both observed coordinates are null.
- Time is strictly increasing at the declared frame interval.
- `exposure_time_s <= frame_interval_s`.
- A sample seed identifies an independent random stream.
- `latent_state` is aligned one-to-one with frames.
- Split assignment is stored, deterministic, and never inferred from file order.

## Missing values

JSON uses `[null, null]` for a missing two-dimensional observation. In memory,
the Python API uses `NaN` coordinates and an explicit Boolean mask.

## Versioning

Any change that alters the probability distribution of generated samples must
change `SIMULATOR_VERSION`. Schema-breaking changes increment the major dataset
version.
