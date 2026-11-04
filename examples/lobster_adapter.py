"""Write a synthetic frame in LOBSTER layout, read it back and inspect capabilities.

No LOBSTER data is shipped or downloaded; this only demonstrates the adapter
on files generated locally from the synthetic process.

    python examples/lobster_adapter.py /tmp/lobster_demo
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from eventflow.marketdata import SyntheticConfig, generate_synthetic
from eventflow.marketdata.lobster import (
    LOBSTER_EVENT_TYPES,
    LobsterDay,
    read_lobster,
    write_lobster_fixture,
)


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = generate_synthetic(
        SyntheticConfig(
            seed=0, instruments=("DEMO",), sessions_per_instrument=1, events_per_session=500
        )
    )
    # Synthetic types (add, cancel, trade) -> LOBSTER (submission, deletion, visible execution).
    lobster_like = frame.replace(
        event_type=np.array([0, 2, 3])[frame.event_type], event_types=LOBSTER_EVENT_TYPES
    )
    message, book = out_dir / "DEMO_message.csv", out_dir / "DEMO_orderbook.csv"
    write_lobster_fixture(lobster_like, message, book)
    loaded = read_lobster([LobsterDay("DEMO", message, book)])
    print(loaded.summary())
    print("price change matches:", np.allclose(loaded.price_change, lobster_like.price_change))


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "lobster_demo"))
