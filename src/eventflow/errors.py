"""Exception hierarchy.

Every error raised deliberately by EventFlow derives from :class:`EventFlowError`
so that callers can distinguish validated failures from programming errors.
"""

from __future__ import annotations


class EventFlowError(Exception):
    """Base class for all EventFlow errors."""


class SchemaError(EventFlowError):
    """An event frame violates the schema (lengths, dtypes, ordering, ranges)."""


class CapabilityError(EventFlowError):
    """An operation needs a field or property the dataset does not provide."""


class ConfigError(EventFlowError):
    """A configuration is malformed, inconsistent or incompatible."""


class TokenizationError(EventFlowError):
    """Encoding or decoding failed, or a tokenizer was used before fitting."""


class CheckpointError(EventFlowError):
    """A checkpoint is missing, corrupt or incompatible with the current run."""


class NumericalError(EventFlowError):
    """A computation produced non-finite values beyond the configured tolerance."""
