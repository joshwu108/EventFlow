=========
EventFlow
=========

EventFlow is a research framework for generative modelling of financial
market-event sequences. It supports controlled comparisons between a
statistical autoregressive baseline, a causal Transformer with a
mixture-density head, a conditional DDPM, conditional flow matching over
relaxed categoricals, and a hybrid model that samples discrete event fields
from softmax heads and continuous fields with conditional flow matching.

.. contents::
   :local:
   :depth: 1

Research question
=================

A market event mixes fields with very different statistical character: the
event type and side are categorical, price moves are integer tick counts,
sizes are positive and heavy-tailed, and interarrival times are positive
with a point mass at zero (simultaneous messages). EventFlow asks whether a
*hybrid* model -- categorical heads for the discrete fields and a
conditional continuous generator for the rest, both conditioned on a causal
Transformer summary of the history -- reproduces the distribution of such
events more faithfully than a single-mechanism alternative.

The hybrid design follows the per-token diffusion head of Li et al. (2024)
and the event-level generator described in Jane Street's write-up on
autoregressive diffusion for market data; EventFlow makes no claim of
architectural novelty. Its contribution is infrastructure for a fair,
reproducible comparison.

Status
======

**Implemented and tested on CPU:** data schema and validation, synthetic
data, CSV/Parquet/LOBSTER adapters, preprocessing and chronological splits,
tokenization, all five model families, constrained generation, the training
loop with checkpoint/resume, the evaluation framework, experiment configs,
dry runs and the CLI. The test suite checks numerical routines against
analytic results (diffusion posterior, oracle DDPM sampling, flow-matching
transport of Gaussians, ODE solver convergence orders, distance measures).

**Not yet experimentally validated:** no model has been trained beyond
few-step smoke tests, and no comparative result exists. This repository
contains no performance claims.

Installation
============

Python 3.11 or newer.

.. code-block:: bash

   python -m venv .venv
   .venv/bin/pip install -e '.[dev]'        # or '.[parquet]' / plain '.'
   .venv/bin/python -m pytest -q            # CPU only, well under a minute

Runtime dependencies are NumPy and PyTorch; ``pyarrow`` is optional and only
needed for Parquet.

Quick start
===========

.. code-block:: bash

   python examples/quickstart.py                      # pipeline walkthrough (untrained model)
   eventflow validate configs/*.toml                  # parse and validate every config
   eventflow compare configs/{baseline,transformer,ddpm,cfm,hybrid}.toml --parameters
   eventflow dry-run configs/smoke.toml               # initialise everything, no training
   eventflow train configs/hybrid.toml --yes          # real training (GPU recommended)
   eventflow report runs/baseline runs/hybrid         # tabulate finished runs

In Python:

.. code-block:: python

   from eventflow.marketdata import SyntheticConfig, generate_synthetic
   from eventflow.tokenization import EventTokenizer
   from eventflow.models import EventGenerativeModel, ModelConfig

   frame = generate_synthetic(SyntheticConfig(seed=0))
   tokenizer = EventTokenizer.fit(frame)
   model = EventGenerativeModel.from_tokenizer(ModelConfig.preset("hybrid"), tokenizer)

Architecture
============

.. code-block:: text

   data source --> EventFrame --> preprocess --> chronological split
                                                     |
                                   tokenizer fitted on train only
                                                     |
                     session-local windows --> causal Transformer encoder
                                                     |
                     +-------------------------------+-------------------+
                     | categorical chain              continuous head    |
                     |  event type -> side ->         MDN | DDPM | flow  |
                     |  price bin -> zero-gap         (conditioned on    |
                     |                                 sampled discretes)|
                     +-------------------------------+-------------------+
                                                     |
                     constrained decoding --> next event --> context update

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - package
     - contents
   * - ``eventflow.marketdata``
     - ``EventFrame`` schema, capability flags, validation, synthetic process, CSV/Parquet/LOBSTER adapters, preprocessing, chronological splits, fingerprints
   * - ``eventflow.tokenization``
     - reversible mixed encodings, windows, masks
   * - ``eventflow.models``
     - context encoder, heads, ``EventGenerativeModel``, ``MarkovBaseline``, model configs and presets
   * - ``eventflow.diffusion``
     - noise schedules, posterior, DDPM head
   * - ``eventflow.flow``
     - OT probability path, ODE solvers, flow-matching head
   * - ``eventflow.generation``
     - constrained rollouts and one-step conditional samples
   * - ``eventflow.training``
     - trainer, checkpoints, logs, manifests, seeding
   * - ``eventflow.evaluation``
     - distances, statistics, metric registry, TSTR task
   * - ``eventflow.experiments``
     - TOML configs, fairness checks, dry run, runner
   * - ``eventflow.reporting``
     - comparison tables from finished runs

Data model
==========

An ``EventFrame`` holds one row per event: timestamp, instrument, event
type, side, relative price change in ticks, quantity, session, derived
interarrival time, optional top of book and optional evaluation-only labels.
Arrays are read-only; every transformation returns a new frame.

Each source declares **capabilities** (``side``, ``price``, ``quantity``,
``multi_event_type``, ``message_level``, ``book_top`` ...). Capabilities are
never widened beyond what the arrays support, ``message_level`` is never
inferred, and a metric or model input that needs a missing capability is
refused. In particular, a table of public trade prints is not treated as an
order-book event stream.

Adapters: ``read_csv`` / ``read_parquet`` for generic tables and
``read_lobster`` for the published LOBSTER message/orderbook layout. No data
is downloaded and no third-party data is included.

Tokenization
============

Categorical fields: event type, side, price change binned to
``[-K, K]`` ticks (outer bins are lossy tails), and a *zero-gap* flag.
Continuous fields: per-instrument standardised log quantity and log
interarrival time, the latter masked when the gap is zero or undefined.
Statistics are fitted on the training partition only. Decoding enforces
positive lot-rounded quantities and non-negative gaps, counts every
correction, and flags categorical combinations never seen in training.

At generation time the model conditions on all fields of past events of the
same session, the instrument and elapsed session time; it never sees the
timestamp of the event being generated. Order-book state is deliberately not
a model input because it cannot be updated consistently during free-running
generation without a validated matching engine.

Models
======

============  ==================  ==============  ===============================
preset        categorical fields  continuous      objective
============  ==================  ==============  ===============================
baseline      Markov chain        bootstrap       counting
transformer   softmax chain       Gaussian MDN    exact NLL
ddpm          softmax chain       DDPM head       CE + epsilon-prediction MSE
cfm           relaxed (+-1)       flow matching   CFM regression (joint)
hybrid        softmax chain       flow matching   CE + CFM regression
============  ==================  ==============  ===============================

The four neural presets share one encoder and differ in two switches, so
each pairwise comparison isolates one component. ``EventGenerativeModel``
also exposes ``sample_categorical`` and continuous sampling given true
categorical fields for component-wise ablations.

Training and reproducibility
============================

AdamW with decoupled weight decay on weight matrices, warmup plus cosine or
linear decay, gradient accumulation and clipping, bf16/fp16 autocast, fixed
seed validation, early stopping, non-finite step guards, atomic checkpoints
(``os.replace``) holding model, optimiser, scheduler, scaler and RNG state,
and exact resume (tested to reproduce uninterrupted training on CPU under an
unchanged schedule). Each run writes ``manifest.json`` with the full config,
dataset fingerprints, Git revision and library versions, and
``metrics.jsonl``. No external tracking service is used.

Evaluation
==========

Metrics are reported in five separate categories:

* **marginal** -- event-type, side, price-change, quantity and interarrival
  distributions; tail statistics (Hill index, kurtosis, extreme quantiles);
* **conditional** -- event-type transitions, per-type conditionals, price
  changes within observable volatility regimes;
* **temporal** -- return autocorrelation, bucketed volatility clustering,
  interarrival autocorrelation;
* **structural** -- constraint violations, unseen categorical combinations,
  invalid-event rates by rollout step;
* **downstream** -- train-on-synthetic / test-on-real direction prediction
  reported against train-on-real.

The statistics are inspired by LOB-Bench but this is **not** an
implementation of LOB-Bench and its numbers are not comparable with LOB-Bench
scores. Full order-book consistency of generated streams is not verified.

Experiments
===========

``configs/_base.toml`` holds every setting shared by the comparison; the five
research configs extend it and change only the model. ``eventflow compare``
reports any difference in data, partitions, tokenisation, context length,
optimiser, schedule, training-token budget, sampling or evaluation settings,
and (with ``--parameters``) parameter counts outside 25 % of the median.
Always run ``eventflow dry-run`` before training.

Limitations
===========

* No trained model and no experimental result exists yet.
* The synthetic process is a test fixture with known properties, not a
  market simulator.
* Book state is neither modelled nor generated; book-dependent statistics
  (spread, depth, imbalance) are out of scope for generated data.
* Price changes beyond ``max_price_ticks`` are clipped to tail bins.
* Resuming with a different ``max_steps`` changes the learning-rate schedule.

References
==========

* Ho, Jain, Abbeel. *Denoising Diffusion Probabilistic Models*. NeurIPS 2020.
* Nichol, Dhariwal. *Improved Denoising Diffusion Probabilistic Models*. ICML 2021.
* Lipman, Chen, Ben-Hamu, Nickel, Le. *Flow Matching for Generative
  Modeling*. ICLR 2023. arXiv:2210.02747.
* Li, Tian, Li, Deng, He. *Autoregressive Image Generation without Vector
  Quantization*. NeurIPS 2024. arXiv:2406.11838.
* Peebles, Xie. *Scalable Diffusion Models with Transformers*. ICCV 2023
  (adaptive layer-norm conditioning).
* Nagy et al. *LOB-Bench: Benchmarking Generative AI for Finance*. ICML 2025.
* Jane Street Tech Blog. *Can you use autoregressive diffusion to generate
  market data?*
* LOBSTER limit order book data, https://lobsterdata.com (format only).

All code in this repository was written for EventFlow; the methods above are
reimplemented from their papers, no third-party source code is copied.

License
========

MIT; see ``LICENSE``.
