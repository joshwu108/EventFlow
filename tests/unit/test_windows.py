from __future__ import annotations

import numpy as np
import pytest
import torch

from eventflow.errors import ConfigError
from eventflow.tokenization import (
    EventTokenizer,
    EventWindowDataset,
    causal_mask,
    collate_events,
    window_starts,
)


def test_windows_stay_inside_sessions_and_cover_tails() -> None:
    session = np.array([0] * 7 + [1] * 3 + [2])
    windows = window_starts(session, window=4, stride=3)
    assert windows == [(0, 4), (3, 7), (7, 10)]
    for a, b in windows:
        assert len(set(session[a:b])) == 1


def test_window_validation() -> None:
    with pytest.raises(ConfigError):
        window_starts(np.zeros(3), window=1, stride=1)


def test_causal_mask_blocks_future_only() -> None:
    mask = causal_mask(4)
    assert mask.dtype == torch.bool
    assert not mask.diagonal().any()
    assert mask[0, 1] and not mask[1, 0]
    assert int(mask.sum()) == 6


def test_collate_pads_and_masks(small_frame) -> None:
    tok = EventTokenizer.fit(small_frame)
    data = EventWindowDataset(tok.encode(small_frame), context_length=16)
    batch = collate_events(
        [data[0], {k: v[:5] if k != "instrument" else v for k, v in data[1].items()}]
    )
    assert batch.categorical.shape == (2, 17, len(tok.layout.categorical))
    assert batch.valid[1].sum() == 5
    _, _, _, target_valid = batch.targets()
    assert target_valid.shape == (2, 16)
    assert int(target_valid[1].sum()) == 4
    assert batch.inputs().length == 16
    assert data.num_targets() == sum(b - a - 1 for a, b in data.windows)


def test_append_and_last_keep_alignment(small_frame) -> None:
    tok = EventTokenizer.fit(small_frame)
    data = EventWindowDataset(tok.encode(small_frame), context_length=4)
    batch = collate_events([data[0], data[1]])
    grown = batch.append(
        batch.categorical[:, -1],
        batch.continuous[:, -1],
        batch.continuous_mask[:, -1],
        batch.session_time[:, -1],
    )
    assert grown.length == batch.length + 1
    trimmed = grown.last(3)
    assert trimmed.length == 3
    assert torch.equal(trimmed.categorical[:, -1], batch.categorical[:, -1])
