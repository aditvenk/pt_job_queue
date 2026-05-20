from __future__ import annotations

from unittest.mock import patch

from ptq.evaluator.reviewer_profile import (
    append_pr_feedback_to_profile,
    generate_reviewer_profile,
)


def test_generate_reviewer_profile_writes_style_markdown(tmp_path):
    def fake_gh(args):
        if args[:2] == ["search", "prs"]:
            return [
                {
                    "number": 10,
                    "title": "Fix distributed tensor bug",
                    "url": "https://github.com/pytorch/pytorch/pull/10",
                    "repository": {"nameWithOwner": "pytorch/pytorch"},
                    "updatedAt": "2026-05-01T00:00:00Z",
                }
            ]
        endpoint = args[1]
        if endpoint.endswith("/issues/10/comments"):
            return [
                {
                    "user": {"login": "aditvenk"},
                    "body": "Can you add a regression test for this edge case?",
                    "html_url": "https://github.com/pytorch/pytorch/pull/10#x",
                    "created_at": "2026-05-01T00:00:00Z",
                }
            ]
        if endpoint.endswith("/pulls/10/comments"):
            return [
                {
                    "user": {"login": "aditvenk"},
                    "body": (
                        "This looks like it changes BC semantics. Can we keep "
                        "the old path unchanged?"
                    ),
                    "path": "torch/foo.py",
                    "line": 12,
                    "html_url": "https://github.com/pytorch/pytorch/pull/10#y",
                    "created_at": "2026-05-01T00:00:00Z",
                },
                {
                    "user": {"login": "someone-else"},
                    "body": "unrelated",
                },
            ]
        if endpoint.endswith("/pulls/10/reviews"):
            return [{"user": {"login": "aditvenk"}, "state": "COMMENTED", "body": ""}]
        return []

    output = tmp_path / "aditvenk.md"
    with patch("ptq.evaluator.reviewer_profile._run_gh_json", side_effect=fake_gh):
        profile = generate_reviewer_profile(
            "aditvenk",
            repo="pytorch/pytorch",
            months=6,
            limit=25,
            output=output,
        )

    text = output.read_text()
    assert profile.comment_count == 3
    assert "# @aditvenk Review Profile" in text
    assert "Review Priorities To Emulate" in text
    assert "Regression coverage" in text
    assert "BC/API semantic changes" in text
    assert "Can you add a regression test" in text


def test_append_pr_feedback_to_profile_adds_matching_author_once(tmp_path):
    profile = tmp_path / "aditvenk.md"
    profile.write_text(
        "# @aditvenk Review Profile\n\n"
        "## Evaluator Instructions\n\n"
        "- Keep reviews direct.\n"
    )
    feedback = {
        "pr_url": "https://github.com/pytorch/pytorch/pull/123",
        "comments": [
            {
                "kind": "inline",
                "id": 10,
                "author": "aditvenk",
                "path": "torch/foo.py",
                "line": 12,
                "body": "Can we keep the old no_grad behavior unchanged?",
                "url": "https://github.com/pytorch/pytorch/pull/123#discussion_r10",
                "created_at": "2026-05-19T12:00:00Z",
            },
            {
                "kind": "conversation",
                "id": 11,
                "author": "someone-else",
                "body": "Unrelated reviewer comment.",
            },
        ],
    }

    assert (
        append_pr_feedback_to_profile(profile, username="aditvenk", feedback=feedback)
        == 1
    )
    assert (
        append_pr_feedback_to_profile(profile, username="aditvenk", feedback=feedback)
        == 0
    )

    text = profile.read_text()
    assert "Recent PR Feedback Incorporated" in text
    assert "Can we keep the old no_grad behavior unchanged?" in text
    assert "someone-else" not in text
