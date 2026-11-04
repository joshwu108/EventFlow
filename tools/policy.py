"""Read and validate the repository commit policy (``tools/commit_policy.toml``).

The policy is the only place where commit and push autonomy is granted. A
missing file grants nothing, and a malformed file is an error rather than a
partial grant, so the tooling never acts on a policy it did not fully read.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

DEFAULT_POLICY_PATH = Path(__file__).resolve().parent / "commit_policy.toml"


class PolicyError(ValueError):
    """The policy file exists but cannot be trusted as written."""


@dataclass(frozen=True)
class Policy:
    """The maintainer's grant. Every field has a conservative default."""

    auto_commit: bool = False
    auto_push: bool = False
    granted_on: str = ""
    granted_by: str = ""
    remote: str = "origin"
    branch: str = "main"
    test_command: tuple[str, ...] = ("python", "-m", "pytest", "-q")
    lint_command: tuple[str, ...] = ()
    requires_explicit_approval: tuple[str, ...] = ()
    protected_paths: tuple[str, ...] = ()

    def grant_summary(self) -> str:
        """One line describing the grant, for the generated skill and logs."""
        if not self.auto_commit:
            return "NONE on record -- ask before every commit."
        push = (
            f"tested milestones are pushed to {self.remote}/{self.branch}"
            if self.auto_push
            else "nothing is pushed"
        )
        return (
            f"routine files may be committed without re-asking and {push} "
            f"(granted {self.granted_on or '?'} by {self.granted_by or '?'}). "
            f"Always requires separate approval: "
            f"{', '.join(self.requires_explicit_approval) or 'nothing listed'}."
        )


_BOOL_FIELDS = ("auto_commit", "auto_push")
_STR_FIELDS = ("granted_on", "granted_by", "remote", "branch")
_LIST_FIELDS = ("test_command", "lint_command", "requires_explicit_approval", "protected_paths")


def _string_tuple(raw: object, key: str) -> tuple[str, ...]:
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise PolicyError(f"{key} must be a list of strings")
    return tuple(raw)


def load_policy(path: Path = DEFAULT_POLICY_PATH) -> Policy:
    """Load ``path`` into a ``Policy``.

    Returns the default (no grant) policy when ``path`` does not exist.

    Raises:
        PolicyError: an unknown key, a wrongly typed value, an empty
            ``test_command``, or ``auto_push`` without ``auto_commit``.
    """
    if not path.is_file():
        return Policy()
    try:
        raw = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise PolicyError(f"{path}: {exc}") from exc

    known = {field.name for field in fields(Policy)}
    for key in raw:
        if key not in known:
            raise PolicyError(f"{path}: unknown key {key!r}")
    for key in _BOOL_FIELDS:
        if key in raw and not isinstance(raw[key], bool):
            raise PolicyError(f"{path}: {key} must be true or false")
    for key in _STR_FIELDS:
        if key in raw and not isinstance(raw[key], str):
            raise PolicyError(f"{path}: {key} must be a string")
    lists = {key: _string_tuple(raw[key], key) for key in _LIST_FIELDS if key in raw}

    loaded = Policy(
        **{key: raw[key] for key in _BOOL_FIELDS + _STR_FIELDS if key in raw},
        **lists,
    )
    if not loaded.test_command:
        raise PolicyError(f"{path}: test_command must name a command")
    if loaded.auto_push and not loaded.auto_commit:
        raise PolicyError(f"{path}: auto_push requires auto_commit")
    return loaded
