"""Push a tested milestone to the configured remote branch, fast-forward only.

The push is autonomous only under a recorded ``auto_push`` grant, from the
configured branch, with an empty index, and when the remote branch is an
ancestor of ``HEAD``. A diverged remote is refused: reconciling it (merge or
rebase) is one of the operations that always needs the maintainer. The
tool never passes ``--force`` or any of its variants.
"""

from __future__ import annotations

from pathlib import Path

from tools.gitwrap import GitError, current_branch, git
from tools.policy import Policy


class PushError(RuntimeError):
    """The push was refused; the remote is untouched."""


def _counts(repo: Path, remote_ref: str) -> tuple[int, int]:
    """Return ``(ahead, behind)`` of ``HEAD`` relative to ``remote_ref``."""
    left_right = git(repo, "rev-list", "--left-right", "--count", f"HEAD...{remote_ref}")
    ahead, behind = (int(part) for part in left_right.split())
    return ahead, behind


def push_milestone(repo: Path, policy: Policy) -> str:
    """Push ``HEAD`` to ``policy.remote``/``policy.branch`` and describe the result.

    Raises:
        PushError: no push grant, wrong branch, missing remote, non-empty
            index, fetch failure, or a remote branch that is not an
            ancestor of ``HEAD``.
    """
    if not policy.auto_push:
        raise PushError("no push grant on record -- pushing needs explicit approval.")
    branch = current_branch(repo)
    if branch != policy.branch:
        raise PushError(f"on branch '{branch}', not '{policy.branch}': refusing to push.")
    if git(repo, "remote", "get-url", policy.remote, check=False) == "":
        raise PushError(
            f"remote '{policy.remote}' is not configured; changing remotes needs approval."
        )
    if git(repo, "diff", "--cached", "--name-only"):
        raise PushError("index is not empty: finish or reset the staged change before pushing.")

    target = f"{policy.remote}/{policy.branch}"
    try:
        git(repo, "fetch", "-q", policy.remote, policy.branch)
    except GitError as exc:
        raise PushError(f"could not fetch {target}:\n{exc}") from exc
    ahead, behind = _counts(repo, target)
    if behind:
        raise PushError(
            f"HEAD is {behind} commit(s) behind {target}: the remote has diverged. "
            "Merging or rebasing needs explicit approval; nothing was pushed."
        )
    if not ahead:
        return f"nothing to push: {target} is at HEAD"

    try:
        git(repo, "push", "-q", policy.remote, f"HEAD:refs/heads/{policy.branch}")
    except GitError as exc:
        raise PushError(f"push to {target} failed:\n{exc}") from exc
    git(repo, "fetch", "-q", policy.remote, policy.branch)
    if git(repo, "rev-parse", target) != git(repo, "rev-parse", "HEAD"):
        raise PushError(f"{target} does not match HEAD after the push; inspect before retrying.")
    return f"pushed {ahead} commit{'s' if ahead != 1 else ''} to {target}"
