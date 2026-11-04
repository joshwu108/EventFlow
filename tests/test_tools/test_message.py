"""Commit messages follow <type>: <description> and carry a single author."""

import pytest

from tools.message import COMMIT_TYPES, MessageError, compose_message


def test_subject_with_known_type_passes() -> None:
    assert compose_message("feat: add the thing") == "feat: add the thing"


def test_body_is_separated_by_one_blank_line() -> None:
    assert compose_message("fix: x", "Why it broke.\n") == "fix: x\n\nWhy it broke."


@pytest.mark.parametrize(
    "subject", ["feat add thing", "Feat: thing", "feature: thing", "feat:thing"]
)
def test_subject_without_a_known_type_prefix_is_refused(subject: str) -> None:
    with pytest.raises(MessageError, match="<type>: <description>"):
        compose_message(subject)


def test_every_documented_type_is_accepted() -> None:
    for kind in COMMIT_TYPES:
        compose_message(f"{kind}: something")


def test_empty_description_is_refused() -> None:
    with pytest.raises(MessageError, match="description"):
        compose_message("feat: ")


def test_overlong_subject_is_refused() -> None:
    with pytest.raises(MessageError, match="subject"):
        compose_message("feat: " + "x" * 120)


def test_multiline_subject_is_refused() -> None:
    with pytest.raises(MessageError, match="single line"):
        compose_message("feat: a\nb")


@pytest.mark.parametrize(
    "body",
    [
        "Co-Authored-By: Claude <noreply@anthropic.com>",
        "co-authored-by: Someone <someone@example.invalid>",
        "Done.\n\nCo-authored-by: Bot <bot@example.invalid>",
        "🤖 Generated with Claude Code",
        "Generated with some assistant",
        "Signed-off-by: Claude <noreply@anthropic.com>",
    ],
)
def test_co_author_and_generation_trailers_are_refused(body: str) -> None:
    with pytest.raises(MessageError, match="single author"):
        compose_message("feat: thing", body)


def test_trailer_in_subject_is_refused() -> None:
    with pytest.raises(MessageError, match="single author"):
        compose_message("feat: thing Co-Authored-By: x")


def test_ordinary_body_mentioning_assistance_is_fine() -> None:
    assert "assistant" in compose_message("docs: note", "Describe the assistant module.")
