"""Console entry points."""

from __future__ import annotations

import argparse
import json

from .data import generate_dataset, load_config


def generate_main() -> None:
    parser = argparse.ArgumentParser(description="Generate the trajectory benchmark")
    parser.add_argument("--config", required=True, help="YAML generation configuration")
    args = parser.parse_args()
    print(json.dumps(generate_dataset(load_config(args.config)), indent=2))
