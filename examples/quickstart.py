"""End-to-end walkthrough on synthetic data, CPU only, about two seconds.

Generates a synthetic event stream, splits it chronologically, fits the
tokenizer on the training partition, builds an *untrained* hybrid model,
computes its training loss, rolls it out and evaluates the rollout against
the real test partition.

The model is untrained, so the evaluation numbers only demonstrate the
pipeline; they say nothing about model quality.

    python examples/quickstart.py
"""

from __future__ import annotations

import torch

from eventflow.evaluation import evaluate
from eventflow.generation import rollout, sample_prompts
from eventflow.marketdata import SyntheticConfig, generate_synthetic
from eventflow.marketdata.preprocess import SplitConfig, chronological_split
from eventflow.models import EventGenerativeModel, ModelConfig
from eventflow.tokenization import EventTokenizer, EventWindowDataset, collate_events


def main() -> None:
    frame = generate_synthetic(SyntheticConfig(seed=0, events_per_session=1000))
    parts = chronological_split(frame, SplitConfig(train=0.5, validation=0.0, test=0.5))
    assert parts.test is not None
    tokenizer = EventTokenizer.fit(parts.train)
    print("layout:", tokenizer.layout.to_dict())

    torch.manual_seed(0)
    config = ModelConfig.preset(
        "hybrid", d_model=32, n_layers=2, n_heads=2, d_ff=64, head_width=64, max_context=64
    )
    model = EventGenerativeModel.from_tokenizer(config, tokenizer)
    print("parameters:", model.num_parameters())

    windows = EventWindowDataset(tokenizer.encode(parts.train), context_length=32)
    batch = collate_events([windows[i] for i in range(8)])
    loss = model.loss(batch)
    print("initial loss components:", {k: round(v, 3) for k, v in loss.components.items()})

    test = tokenizer.encode(parts.test)
    prompts = sample_prompts(test, context_length=32, count=4, seed=0)
    result = rollout(
        model,
        tokenizer,
        prompts,
        horizon=100,
        max_context=32,
        generator=torch.Generator().manual_seed(0),
    )
    print("rollout events:", len(result.frame), "invalid rate:", result.invalid_rate)

    report = evaluate(
        parts.test,
        result.frame,
        metrics=["event_type_marginal", "side_marginal", "structural_validity"],
    )
    for item in report.results:
        tv = item.values.get("tv")
        print(f"{item.category:>11}  {item.name:<22} {item.status}  tv={tv}")


if __name__ == "__main__":
    main()
