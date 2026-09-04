"""Shared labels and version identifiers."""

SIMULATOR_VERSION = "0.1.0"

STATE_LABELS = (
    "brownian",
    "directed",
    "confined",
    "subdiffusive",
    "switching",
)

STATE_TO_INDEX = {label: index for index, label in enumerate(STATE_LABELS)}
