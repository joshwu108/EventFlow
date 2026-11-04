"""Tiny model configurations and batches shared by the model tests."""

from __future__ import annotations

from eventflow.marketdata import EventFrame
from eventflow.models.config import DiffusionConfig, FlowConfig, ModelConfig
from eventflow.models.generative import EventGenerativeModel
from eventflow.tokenization import EventBatch, EventTokenizer, EventWindowDataset, collate_events

KINDS = ("transformer", "ddpm", "cfm", "hybrid")


def tiny_config(kind: str, **overrides: object) -> ModelConfig:
    base: dict[str, object] = dict(
        d_model=32,
        n_layers=2,
        n_heads=2,
        d_ff=64,
        dropout=0.0,
        max_context=64,
        head_width=32,
        head_depth=2,
        mdn_components=3,
        loss_repeats=2,
        diffusion=DiffusionConfig(train_steps=50, sample_steps=5),
        flow=FlowConfig(sample_steps=4),
    )
    base.update(overrides)
    return ModelConfig.preset(kind, **base)


def tiny_model(kind: str, tokenizer: EventTokenizer, **overrides: object) -> EventGenerativeModel:
    return EventGenerativeModel.from_tokenizer(tiny_config(kind, **overrides), tokenizer)


def tiny_batch(
    frame: EventFrame, tokenizer: EventTokenizer, context: int = 16, size: int = 4
) -> EventBatch:
    data = EventWindowDataset(tokenizer.encode(frame), context_length=context)
    return collate_events([data[i] for i in range(size)])
