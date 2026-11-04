"""Market-event data: schema, validation, synthetic data, adapters, preprocessing."""

from eventflow.marketdata.schema import (
    SIDE_BUY,
    SIDE_NAMES,
    SIDE_SELL,
    SIDE_UNKNOWN,
    Capability,
    EventFrame,
    Provenance,
    SessionInfo,
    ValidationReport,
    validate_frame,
)
from eventflow.marketdata.synthetic import SyntheticConfig, generate_synthetic

__all__ = [
    "SIDE_BUY",
    "SIDE_NAMES",
    "SIDE_SELL",
    "SIDE_UNKNOWN",
    "Capability",
    "EventFrame",
    "Provenance",
    "SessionInfo",
    "SyntheticConfig",
    "ValidationReport",
    "generate_synthetic",
    "validate_frame",
]
