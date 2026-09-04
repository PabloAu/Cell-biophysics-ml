"""Deterministic JSON Lines I/O."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from ..types import TrajectorySample


def write_samples(samples: Iterable[TrajectorySample], output: str | Path) -> dict[str, object]:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            for sample in samples:
                line = json.dumps(
                    sample.to_record(), sort_keys=True, separators=(",", ":"), allow_nan=False
                )
                compressed.write(line.encode("utf-8") + b"\n")
                count += 1
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": path.as_posix(), "samples": count, "sha256": sha256}


def iter_samples(path: str | Path) -> Iterator[TrajectorySample]:
    with gzip.open(path, mode="rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield TrajectorySample.from_record(json.loads(line))


def read_samples(path: str | Path, split: str | None = None) -> list[TrajectorySample]:
    return [sample for sample in iter_samples(path) if split is None or sample.split == split]
