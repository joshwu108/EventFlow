"""Regenerate the Git-ignored local commit skill on a fresh checkout.

Renders ``tools/commit_skill_template.txt`` into
``.claude/skills/commit/SKILL.md`` from three inputs: the repository commit
policy (``tools/commit_policy.toml``), the repository-local Git identity, and
the commit convention found in a reference project. The prerequisites for
autonomous commits are checked on every run and reported on stdout.

Usage:
    python -m tools.bootstrap_commit_skill [--reference ~/TapeForge] [--check]

With ``--check`` the skill file is not written. The exit status is 0 when
every prerequisite is satisfied, 1 when the template is missing, and 2 when a
prerequisite is unmet (no reference reviewed, no local identity, an ignore
rule gap, or tracked paths that the policy excludes).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from tools.gitwrap import GitError, local_identity, tracked_paths
from tools.paths import forbidden_reason
from tools.policy import DEFAULT_POLICY_PATH, Policy, PolicyError, load_policy

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "tools" / "commit_skill_template.txt"
SKILL_PATH = REPO / ".claude" / "skills" / "commit" / "SKILL.md"
DEFAULT_REFERENCE = Path.home() / "TapeForge"

# Paths inside a reference project where a commit convention may live. Only
# these names are read; a blanket scan of the reference could pick up secrets.
REFERENCE_CANDIDATES: tuple[str, ...] = (
    ".agents/skills/commit/SKILL.md",
    ".claude/skills/commit/SKILL.md",
    ".claude/commands/commit.md",
    "tools/commit_skill_template.txt",
    "CLAUDE.md",
    "AGENTS.md",
)

UNSET = "<UNSET -- ask the maintainer>"


def review_reference(reference: Path) -> tuple[str, list[Path]]:
    """Report which commit-convention files exist at ``reference``.

    Returns:
        A status line rendered into the generated skill, and the files a
        maintainer should read. A missing or unrecognised reference is
        reported as such rather than treated as reviewed.
    """
    if not reference.exists():
        return (
            f"NOT FOUND at {reference} -- conventions below are derived from the "
            "policy only. Ask the maintainer to supply their commit skill.",
            [],
        )
    found = [reference / name for name in REFERENCE_CANDIDATES if (reference / name).is_file()]
    if not found:
        return (
            f"{reference} exists but contains no recognizable commit skill "
            f"(looked for: {', '.join(REFERENCE_CANDIDATES)}).",
            [],
        )
    listing = ", ".join(p.relative_to(reference).as_posix() for p in found)
    return f"reviewed {reference} -- {listing}", found


def render_skill(
    template: Path, policy: Policy, identity: tuple[str, str], reference_status: str
) -> str:
    """Fill the template's placeholders; an unset identity is marked, never invented."""
    name, email = identity
    return (
        template.read_text()
        .replace("{REFERENCE_STATUS}", reference_status)
        .replace("{AUTHORIZATION}", policy.grant_summary())
        .replace("{GIT_USER_NAME}", name or UNSET)
        .replace("{GIT_USER_EMAIL}", email or UNSET)
        .replace("{TEST_COMMAND}", " ".join(policy.test_command))
        .replace("{LINT_COMMAND}", " ".join(policy.lint_command) or "(none configured)")
        .replace("{REMOTE_BRANCH}", f"{policy.remote}/{policy.branch}")
    )


# One representative path per ignore rule that keeps local-only material out of Git.
IGNORE_PROBES: tuple[str, ...] = (
    "planning/notes.rst",
    "notes.md",
    ".claude/skills/commit/SKILL.md",
    ".claude/settings.local.json",
    ".agents/skills/commit/SKILL.md",
    ".venv/bin/python",
    "data/sample.csv.gz",
)


def ignore_gaps(repo: Path) -> list[str]:
    """Return the probe paths that the repository's own ignore rules do not cover.

    Global excludes (``core.excludesFile`` and the XDG default) are switched
    off: a rule that exists only on this machine would not protect a fresh
    clone, so it must not satisfy the check.
    """
    gaps = []
    for path in IGNORE_PROBES:
        result = subprocess.run(
            [
                "git",
                "-c",
                f"core.excludesFile={os.devnull}",
                "check-ignore",
                "-q",
                "--no-index",
                path,
            ],
            cwd=repo,
            check=False,
        )
        if result.returncode != 0:
            gaps.append(path)
    return gaps


def tracked_violations(repo: Path) -> list[str]:
    """Return tracked paths that the forbidden-path rules keep out of Git."""
    return [path for path in tracked_paths(repo) if forbidden_reason(repo, path)]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument(
        "--check", action="store_true", help="verify only; do not write the skill file"
    )
    parser.add_argument("--repo", type=Path, default=REPO, help=argparse.SUPPRESS)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH, help=argparse.SUPPRESS)
    parser.add_argument("--skill-path", type=Path, default=SKILL_PATH, help=argparse.SUPPRESS)
    return parser


def _identity(repo: Path) -> tuple[str, str]:
    try:
        return local_identity(repo)
    except GitError:
        return "", ""


def main(argv: list[str] | None = None) -> int:
    """Run the prerequisite checks and, unless ``--check`` is given, write the skill file."""
    args = _parser().parse_args(argv)
    problems: list[str] = []

    status, found = review_reference(args.reference)
    print(f"reference skill: {status}")
    for path in found:
        print(f"  read: {path}")
    if not found:
        problems.append("no commit-convention reference was reviewed")

    try:
        policy = load_policy(args.policy)
    except PolicyError as exc:
        print(f"ABORTED: {exc}", file=sys.stderr)
        return 2
    print(f"policy: {policy.grant_summary()}")

    name, email = _identity(args.repo)
    print(
        f"local git identity: {name} <{email}>" if name and email else "local git identity: NOT SET"
    )
    if not (name and email):
        problems.append(
            "set the repository-local identity before enabling auto-commit:\n"
            '    git config --local user.name "<name>" && '
            "git config --local user.email <verified email>"
        )

    gaps = ignore_gaps(args.repo)
    print("ignore check: " + ("all probe paths ignored" if not gaps else f"NOT ignored: {gaps}"))
    if gaps:
        problems.append(f"gitignore does not cover: {', '.join(gaps)}")

    violations = tracked_violations(args.repo)
    print(f"tracked policy violations: {violations or 'none'}")
    if violations:
        problems.append(
            "forbidden files are tracked. Ask the maintainer before running "
            "`git rm --cached` on them; never delete the local copies."
        )

    if not args.check:
        if not TEMPLATE.is_file():
            print(f"ABORTED: missing template {TEMPLATE}", file=sys.stderr)
            return 1
        rendered = render_skill(TEMPLATE, policy, (name, email), status)
        args.skill_path.parent.mkdir(parents=True, exist_ok=True)
        args.skill_path.write_text(rendered)
        print(f"wrote {args.skill_path} ({len(rendered.splitlines())} lines)")

    if problems:
        print("\nmissing prerequisites:")
        for item in problems:
            print(f"  - {item}")
        return 2
    print("\nall prerequisites satisfied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
