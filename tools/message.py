"""Commit message rules.

The subject is ``<type>: <description>`` with a type from ``COMMIT_TYPES``.
A commit carries exactly one author, the configured identity, so co-author,
sign-off and "generated with" trailers are refused wherever they appear.
"""

from __future__ import annotations

import re

COMMIT_TYPES: tuple[str, ...] = ("feat", "fix", "refactor", "docs", "test", "chore", "perf", "ci")
MAX_SUBJECT_LENGTH = 100

_SUBJECT = re.compile(r"^(?P<type>[a-z]+): (?P<description>.*)$")
# Matched case-insensitively against every line of the message.
_FORBIDDEN_TRAILERS = re.compile(
    r"(co-authored-by\s*:|signed-off-by\s*:|generated with\b|\U0001F916)", re.IGNORECASE
)


class MessageError(ValueError):
    """The commit message violates the convention; nothing was committed."""


def _check_single_author(text: str) -> None:
    for line in text.splitlines():
        if _FORBIDDEN_TRAILERS.search(line):
            raise MessageError(
                "commits carry a single author, the configured identity; refusing the line:\n"
                f"  {line.strip()}"
            )


def compose_message(subject: str, body: str | None = None) -> str:
    """Return the full commit message after validating ``subject`` and ``body``.

    Raises:
        MessageError: the subject spans lines, lacks a known ``<type>: ``
            prefix or a description, exceeds ``MAX_SUBJECT_LENGTH``, or any
            line carries a co-author, sign-off or generation trailer.
    """
    if "\n" in subject.strip():
        raise MessageError("the subject must be a single line; put details in --body")
    subject = subject.strip()
    match = _SUBJECT.match(subject)
    if match is None or match.group("type") not in COMMIT_TYPES:
        raise MessageError(
            f"subject must be '<type>: <description>' with type in {', '.join(COMMIT_TYPES)}"
        )
    if not match.group("description").strip():
        raise MessageError("subject needs a description after the type")
    if len(subject) > MAX_SUBJECT_LENGTH:
        raise MessageError(f"subject exceeds {MAX_SUBJECT_LENGTH} characters ({len(subject)})")
    _check_single_author(subject)
    if body is None or not body.strip():
        return subject
    body = body.strip()
    _check_single_author(body)
    return f"{subject}\n\n{body}"
