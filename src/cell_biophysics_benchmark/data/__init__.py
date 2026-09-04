"""Dataset generation and I/O."""

from .generation import generate_dataset, load_config
from .io import iter_samples, read_samples, write_samples

__all__ = ["generate_dataset", "iter_samples", "load_config", "read_samples", "write_samples"]
