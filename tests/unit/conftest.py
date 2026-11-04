"""Shared fixtures for the library tests: small deterministic frames."""

from __future__ import annotations

import numpy as np
import pytest

from eventflow.marketdata import EventFrame, Provenance, SyntheticConfig, generate_synthetic


@pytest.fixture(scope="session")
def small_frame() -> EventFrame:
    return generate_synthetic(
        SyntheticConfig(seed=7, sessions_per_instrument=2, events_per_session=300)
    )


def make_frame(n: int = 6, **overrides: object) -> EventFrame:
    """A hand-written two-session frame for validation tests."""
    columns: dict[str, object] = {
        "timestamp": np.array([0.0, 0.5, 0.5, 2.0, 10.0, 10.25][:n]),
        "instrument": np.zeros(n, dtype=int),
        "event_type": np.array([0, 1, 2, 0, 2, 1][:n]),
        "side": np.array([0, 1, 0, 2, 1, 0][:n]),
        "price_change": np.array([0.0, -1.0, 2.0, 0.0, -3.0, 0.0][:n]),
        "quantity": np.array([10.0, 5.0, 1.0, 7.0, 300.0, 2.0][:n]),
        "session": np.array([0, 0, 0, 0, 1, 1][:n]),
    }
    columns.update(overrides)
    return EventFrame.from_columns(
        **columns,
        event_types=("add", "cancel", "trade"),
        instruments=("X",),
        provenance=Provenance(source="test", adapter="hand"),
    )
