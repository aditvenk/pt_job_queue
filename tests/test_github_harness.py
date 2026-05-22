from __future__ import annotations

from unittest.mock import patch

import pytest

from scripts import github_harness


def test_client_run_gh_uses_harness_socket():
    with (
        patch.object(github_harness, "harness_available", return_value=True),
        patch.object(
            github_harness,
            "call_harness",
            return_value={"returncode": 0, "stdout": "ok\n", "stderr": ""},
        ) as call_harness,
    ):
        result = github_harness.client_run_gh(
            ["pr", "view", "https://github.com/pytorch/pytorch/pull/184367"],
            cwd="/tmp/worktree",
        )

    assert result["stdout"] == "ok\n"
    call_harness.assert_called_once_with(
        "gh",
        args=["pr", "view", "https://github.com/pytorch/pytorch/pull/184367"],
        cwd="/tmp/worktree",
    )


def test_client_run_gh_requires_harness_socket():
    with patch.object(github_harness, "harness_available", return_value=False):
        with pytest.raises(RuntimeError, match="GitHub harness socket not found"):
            github_harness.client_run_gh(["pr", "view", "123"])


def test_gh_cli_preserves_leading_gh_options():
    with patch.object(
        github_harness,
        "client_run_gh",
        return_value={"returncode": 0, "stdout": "", "stderr": ""},
    ) as client_run_gh:
        with pytest.raises(SystemExit) as exc:
            github_harness.main(["gh", "--repo", "pytorch/pytorch", "pr", "view", "1"])

    assert exc.value.code == 0
    client_run_gh.assert_called_once_with(
        ["--repo", "pytorch/pytorch", "pr", "view", "1"],
        cwd=None,
    )
