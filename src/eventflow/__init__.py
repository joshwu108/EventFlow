"""EventFlow: generative modelling of financial market-event sequences.

The package is organised by pipeline stage:

* :mod:`eventflow.data` -- validated event frames, adapters, preprocessing.
* :mod:`eventflow.tokenization` -- reversible mixed discrete/continuous encodings.
* :mod:`eventflow.models` -- context encoder, output heads, generative models.
* :mod:`eventflow.diffusion` / :mod:`eventflow.flow` -- continuous generative heads.
* :mod:`eventflow.generation` -- constrained autoregressive rollouts.
* :mod:`eventflow.training` -- trainer, checkpoints, run manifests.
* :mod:`eventflow.evaluation` -- distributional, temporal and structural metrics.
* :mod:`eventflow.experiments` / :mod:`eventflow.reporting` -- configs, runners, tables.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
