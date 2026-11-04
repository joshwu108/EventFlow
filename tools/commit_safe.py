"""Safe local commit helper.

Enforces the repository commit policy (``tools/commit_policy.toml``)
mechanically, so that it does not depend on an agent remembering it:

  * requires a recorded grant and the repository-local Git identity;
  * validates the message (``<type>: <description>``, single author);
  * stages only paths given explicitly on the command line, after the
    forbidden-path rules (never ``git add -A``);
  * runs ``git diff --check``, the lint command and the test suite
    (``--skip-tests`` is explicit and announced);
  * re-reads the index and aborts on anything outside the allowlist;
  * never merges, rebases, amends, force-pushes or changes remotes.

Usage:
    python -m tools.commit_safe --message "test: add parity cases" \\
        --path tools/x.py --path tests/test_x.py [--body ...] [--dry-run]

The exit status is 0 after a commit, a dry run or when nothing changed, and
1 when any check aborts. An aborted run resets the index.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tools.allowlist import AllowlistError, check_allowlist
from tools.gates import GateError, run_lint, run_tests, whitespace_check
from tools.gitwrap import GitError, git, local_identity
from tools.message import MessageError, compose_message
from tools.policy import DEFAULT_POLICY_PATH, Policy, PolicyError, load_policy
from tools.staging import StagingError, reset_index, stage_allowlist, verify_staged

REPO = Path(__file__).resolve().parent.parent

CommitAborted = (AllowlistError, GateError, GitError, MessageError, PolicyError, StagingError)


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser; shared with the push front end."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--message", "-m", required=True, help="<type>: <description>")
    parser.add_argument("--body", default=None, help="what changed and why")
    parser.add_argument(
        "--path", action="append", dest="paths", default=[], help="repeatable allowlist entry"
    )
    parser.add_argument("--dry-run", action="store_true", help="check and stage, then reset")
    parser.add_argument("--skip-tests", action="store_true", help="explicitly skip the test gate")
    parser.add_argument(
        "--approve-protected", action="store_true", help="maintainer approved a protected path"
    )
    parser.add_argument("--repo", type=Path, default=REPO, help=argparse.SUPPRESS)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH, help=argparse.SUPPRESS)
    return parser


def create_commit(
    repo: Path,
    policy: Policy,
    *,
    message: str,
    body: str | None,
    paths: list[str],
    dry_run: bool = False,
    skip_tests: bool = False,
    approve_protected: bool = False,
) -> str | None:
    """Run every check and create the commit; return its short hash, or ``None``.

    ``None`` means nothing was committed because the allowlist produced no
    change or ``dry_run`` was set. Any failure raises one of the
    ``CommitAborted`` exceptions after the index has been reset.
    """
    if not policy.auto_commit:
        raise PolicyError("no commit grant on record -- ask before every commit.")
    name, email = local_identity(repo)
    full_message = compose_message(message, body)
    allowlist = check_allowlist(repo, paths, policy, approve_protected=approve_protected)
    try:
        whitespace_check(repo)
        run_lint(repo, policy)
        if skip_tests:
            print("tests skipped on explicit request (--skip-tests).")
        else:
            run_tests(repo, policy)
        stage_allowlist(repo, allowlist)
        staged = verify_staged(repo, allowlist)
        print(f"author:  {name} <{email}>")
        if not staged:
            print("nothing to commit; index reset.")
            return None
        print("staged:  " + ", ".join(staged))
        print(git(repo, "diff", "--cached", "--stat"))
        if dry_run:
            reset_index(repo)
            print("dry run: index reset, no commit created.")
            return None
        git(repo, "commit", "-q", "-m", full_message)
    except CommitAborted:
        reset_index(repo)
        raise
    print("committed locally.")
    print(git(repo, "log", "-1", "--pretty=format:%h %an <%ae> %s"))
    return git(repo, "rev-parse", "--short", "HEAD")


def main(argv: list[str] | None = None) -> int:
    """Parse the arguments and create the commit; return the exit status."""
    args = build_parser().parse_args(argv)
    try:
        policy = load_policy(args.policy)
        create_commit(
            args.repo,
            policy,
            message=args.message,
            body=args.body,
            paths=args.paths,
            dry_run=args.dry_run,
            skip_tests=args.skip_tests,
            approve_protected=args.approve_protected,
        )
    except CommitAborted as exc:
        print(f"ABORTED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
