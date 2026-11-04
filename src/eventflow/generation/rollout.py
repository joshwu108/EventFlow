"""Constrained autoregressive generation of event sequences.

Any object with ``sample_next(context, generator, temperature) -> NextEvent``
(the neural models and :class:`~eventflow.models.baseline.MarkovBaseline`)
can be rolled out. One step:

1. keep the last ``max_context`` events of each row (sliding window);
2. sample the next encoded event (categorical constraints are applied by the
   model's sampler);
3. decode it with the tokenizer, which enforces positive lot-rounded
   quantities and non-negative gaps and flags remaining invalid events;
4. advance ``session_time`` by the decoded gap and append the event.

Generated events are returned both encoded and as an :class:`EventFrame`
with one session per rollout row. Book state is not generated, so the frame
never declares ``BOOK_TOP``; order-book consistency of generated streams is
therefore *not* verified by this package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

from eventflow.errors import ConfigError
from eventflow.marketdata.schema import Capability, EventFrame, Provenance, SessionInfo
from eventflow.models.generative import NextEvent
from eventflow.tokenization.tokenizer import DecodedEvents, EncodedEvents, EventTokenizer
from eventflow.tokenization.windows import EventBatch, collate_events, window_starts


class NextEventSampler(Protocol):
    def sample_next(
        self,
        context: EventBatch,
        generator: torch.Generator | None = None,
        temperature: float = 1.0,
    ) -> NextEvent: ...


@dataclass(frozen=True)
class Prompts:
    """Real context windows and the real events that followed them."""

    context: EventBatch
    continuation: EncodedEvents | None
    start_timestamp: np.ndarray  # [B] timestamp of the last context event


@dataclass(frozen=True)
class RolloutResult:
    categorical: np.ndarray  # [B, H, K]
    continuous: np.ndarray  # [B, H, D]
    decoded: DecodedEvents  # flattened row-major [B * H]
    frame: EventFrame
    invalid: np.ndarray  # [B, H] bool

    @property
    def invalid_rate(self) -> float:
        return float(self.invalid.mean()) if self.invalid.size else 0.0

    def invalid_rate_by_step(self) -> np.ndarray:
        """Fraction of invalid events at each generation step (long-horizon validity)."""
        return self.invalid.mean(axis=0)


def sample_prompts(
    encoded: EncodedEvents, context_length: int, count: int, *, horizon: int = 0, seed: int = 0
) -> Prompts:
    """Draw ``count`` session-local windows of exactly ``context_length`` events.

    When ``horizon > 0`` only windows followed by at least ``horizon`` events
    of the same session are eligible and those events are returned as
    ``continuation`` (row-major ``[count * horizon]``).
    """
    span = context_length + horizon
    windows = [
        w for w in window_starts(encoded.session, span, 1, min_length=span) if w[1] - w[0] == span
    ]
    if not windows:
        raise ConfigError(f"no session holds {span} consecutive events")
    rng = np.random.default_rng(seed)
    chosen = [
        windows[i] for i in rng.choice(len(windows), size=count, replace=count > len(windows))
    ]
    items = []
    for a, _ in chosen:
        b = a + context_length
        items.append(
            {
                "categorical": encoded.categorical[a:b],
                "continuous": encoded.continuous[a:b],
                "continuous_mask": encoded.continuous_mask[a:b],
                "instrument": encoded.instrument[a],
                "session_time": encoded.session_time[a:b],
            }
        )
    continuation = None
    if horizon:
        rows = np.concatenate([np.arange(a + context_length, a + span) for a, _ in chosen])
        continuation = EncodedEvents(
            encoded.categorical[rows],
            encoded.continuous[rows],
            encoded.continuous_mask[rows],
            encoded.instrument[rows],
            encoded.session[rows],
            encoded.session_time[rows],
            encoded.timestamp[rows],
            encoded.price_tail[rows],
        )
    starts = np.array([encoded.timestamp[a + context_length - 1] for a, _ in chosen])
    return Prompts(collate_events(items), continuation, starts)


@torch.no_grad()
def rollout(
    model: NextEventSampler,
    tokenizer: EventTokenizer,
    prompts: Prompts,
    horizon: int,
    *,
    max_context: int,
    generator: torch.Generator | None = None,
    temperature: float = 1.0,
) -> RolloutResult:
    """Generate ``horizon`` events after each prompt row."""
    if horizon < 1 or max_context < 1:
        raise ConfigError("horizon and max_context must be >= 1")
    context = prompts.context
    if not bool(context.valid.all()):
        raise ConfigError("rollout prompts must be unpadded (equal-length) windows")
    if hasattr(model, "eval"):
        model.eval()
    instrument = context.instrument.cpu().numpy()
    cats, conts, invalid = [], [], []
    for _ in range(horizon):
        nxt = model.sample_next(context.last(max_context), generator, temperature)
        decoded = tokenizer.decode(
            nxt.categorical.cpu().numpy(), nxt.continuous.cpu().numpy(), instrument
        )
        gap = torch.as_tensor(
            decoded.interarrival,
            dtype=context.session_time.dtype,
            device=context.session_time.device,
        )
        context = context.append(
            nxt.categorical,
            nxt.continuous.to(context.continuous.dtype),
            nxt.continuous_mask,
            context.session_time[:, -1] + gap,
        )
        cats.append(nxt.categorical.cpu().numpy())
        conts.append(nxt.continuous.cpu().numpy())
        invalid.append(decoded.invalid)
    cat = np.stack(cats, axis=1)
    cont = np.stack(conts, axis=1)
    B, H = cat.shape[:2]
    decoded = tokenizer.decode(
        cat.reshape(B * H, -1), cont.reshape(B * H, -1), np.repeat(instrument, H)
    )
    frame = decoded_to_frame(decoded, tokenizer, prompts.start_timestamp, H)
    return RolloutResult(cat, cont, decoded, frame, np.stack(invalid, axis=1))


def decoded_to_frame(
    decoded: DecodedEvents,
    tokenizer: EventTokenizer,
    start: np.ndarray,
    horizon: int,
    source: str = "generated",
) -> EventFrame:
    """Assemble row-major decoded events (``B`` rows of ``horizon``) into a frame."""
    B = len(start)
    gaps = decoded.interarrival.reshape(B, horizon)
    timestamp = (start[:, None] + np.cumsum(gaps, axis=1)).reshape(-1)
    session = np.repeat(np.arange(B), horizon)
    capabilities = set(tokenizer.capabilities) - {Capability.BOOK_TOP, Capability.LABELS}
    instrument = decoded.instrument.reshape(B, horizon)[:, 0]
    return EventFrame.from_columns(
        timestamp=timestamp,
        instrument=decoded.instrument,
        event_type=decoded.event_type,
        side=decoded.side,
        price_change=decoded.price_change,
        quantity=decoded.quantity,
        session=session,
        event_types=tokenizer.event_types,
        instruments=tokenizer.instruments,
        provenance=Provenance(source=source, adapter="eventflow.generation.rollout"),
        capabilities=sorted(capabilities),
        sessions=[
            SessionInfo(i, int(instrument[i]), f"rollout{i}", float(start[i])) for i in range(B)
        ],
    )


@torch.no_grad()
def one_step_samples(
    model: NextEventSampler,
    tokenizer: EventTokenizer,
    encoded: EncodedEvents,
    context_length: int,
    count: int,
    *,
    seed: int = 0,
    temperature: float = 1.0,
) -> tuple[DecodedEvents, DecodedEvents]:
    """One generated next event per real context, plus the real next event.

    Comparing the two sets measures *conditional* (one-step) fidelity without
    compounding rollout error.
    """
    prompts = sample_prompts(encoded, context_length, count, horizon=1, seed=seed)
    assert prompts.continuation is not None
    if hasattr(model, "eval"):
        model.eval()
    generator = torch.Generator().manual_seed(seed)
    nxt = model.sample_next(prompts.context, generator, temperature)
    inst = prompts.context.instrument.cpu().numpy()
    fake = tokenizer.decode(nxt.categorical.cpu().numpy(), nxt.continuous.cpu().numpy(), inst)
    real = tokenizer.decode(prompts.continuation.categorical, prompts.continuation.continuous, inst)
    return fake, real
