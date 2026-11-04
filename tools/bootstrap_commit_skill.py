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

from pathlib import Path

from tools.policy import Policy

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
