"""Wall-clock cost of drawing one next event, per continuous head.

Measures the time of ``model.sample_next`` for untrained models of a given
size. Sampling cost does not depend on training, so the measurement is
meaningful before any experiment; it is *not* a quality comparison.

This script is not run by the test suite or CI. Run it on the hardware you
care about, e.g.::

    python benchmarks/sampling_cost.py --device cuda --batch 256 --repeats 20

and record the device, batch size and library versions with the numbers.
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from eventflow.marketdata import SyntheticConfig, generate_synthetic
from eventflow.models import EventGenerativeModel, ModelConfig
from eventflow.tokenization import EventTokenizer, EventWindowDataset, collate_events
from eventflow.training import environment_info, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--context", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--d-model", type=int, default=128)
    args = parser.parse_args()

    device = resolve_device(args.device)
    frame = generate_synthetic(
        SyntheticConfig(seed=0, events_per_session=max(500, 4 * args.context))
    )
    tokenizer = EventTokenizer.fit(frame)
    windows = EventWindowDataset(tokenizer.encode(frame), context_length=args.context)
    items = [windows[i % len(windows)] for i in range(args.batch)]
    context = collate_events(items).inputs().to(device)
    results = {}
    for kind in ("transformer", "ddpm", "cfm", "hybrid"):
        cfg = ModelConfig.preset(kind, d_model=args.d_model, max_context=args.context)
        model = EventGenerativeModel.from_tokenizer(cfg, tokenizer).to(device).eval()
        generator = torch.Generator(device=device).manual_seed(0)
        model.sample_next(context, generator)  # warm-up
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(args.repeats):
            model.sample_next(context, generator)
        if device.type == "cuda":
            torch.cuda.synchronize()
        per_call = (time.perf_counter() - start) / args.repeats
        results[kind] = {
            "ms_per_batch": 1e3 * per_call,
            "us_per_event": 1e6 * per_call / args.batch,
            "parameters": model.num_parameters(),
        }
    print(
        json.dumps(
            {"environment": environment_info(device), "args": vars(args), "results": results},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
