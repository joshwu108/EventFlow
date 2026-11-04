"""Git ``commit-msg`` hook enforcing the single-author rule.

``tools.commit_safe`` already refuses co-author, sign-off and "generated
with" lines, but a commit made with plain ``git commit`` (by a person, an
editor integration or an agent) bypasses it. Installed as
``.git/hooks/commit-msg``, this hook applies the same rule to every commit
in the checkout. It deliberately checks nothing else, so merge and revert
messages written by git itself still pass.

Usage:
    python -m tools.commit_msg_hook <message-file>     # run by git
    python -m tools.commit_msg_hook --install          # write the hook
"""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

from tools.message import MessageError, check_single_author

REPO = Path(__file__).resolve().parent.parent
HOOK_MARKER = "# installed by EventFlow tools.commit_msg_hook"


def check_message_file(path: Path) -> None:
    """Raise ``MessageError`` when the message in ``path`` breaks the rule.

    Lines starting with ``#`` are git's template comments and are skipped,
    matching what git strips before recording the commit.
    """
    lines = [line for line in path.read_text().splitlines() if not line.startswith("#")]
    check_single_author("\n".join(lines))


def install_hook(repo: Path, python: str) -> Path:
    """Write ``.git/hooks/commit-msg`` for ``repo`` and return its path.

    Raises:
        FileExistsError: a ``commit-msg`` hook exists that this tool did
            not install; it is left untouched.
    """
    hook = repo / ".git" / "hooks" / "commit-msg"
    if hook.exists() and HOOK_MARKER not in hook.read_text():
        raise FileExistsError(
            f"{hook} exists and was not installed by EventFlow; merge it by hand."
        )
    hook.parent.mkdir(parents=True, exist_ok=True)
    hook.write_text(
        "#!/bin/sh\n"
        f"{HOOK_MARKER}\n"
        'cd "$(git rev-parse --show-toplevel)" || exit 1\n'
        f'exec {shlex.quote(python)} -m tools.commit_msg_hook "$1"\n'
    )
    hook.chmod(0o755)
    return hook


def main(argv: list[str] | None = None) -> int:
    """Check the message file named in ``argv`` or install the hook."""
    args = sys.argv[1:] if argv is None else argv
    if args == ["--install"]:
        print(f"installed {install_hook(REPO, sys.executable)}")
        return 0
    if len(args) != 1:
        print("usage: python -m tools.commit_msg_hook <message-file> | --install", file=sys.stderr)
        return 2
    try:
        check_message_file(Path(args[0]))
    except MessageError as exc:
        print(f"commit-msg: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
