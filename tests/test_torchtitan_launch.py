"""Tests for torchtitan launch: worktree creation, venv setup, repo persistence."""

from __future__ import annotations

from subprocess import CompletedProcess
from unittest.mock import MagicMock, patch

from ptq.application.run_service import finalize_run, launch
from ptq.domain.models import JobRecord, RunRequest
from ptq.ssh import LocalBackend


def _ok(cmd="", **kwargs):
    return CompletedProcess(args="", returncode=0, stdout="", stderr="")


def _mock_backend(backend):
    def run_side_effect(cmd: str, check: bool = True, **kw):
        if cmd == f"test -d {backend.workspace}/pytorch/.git":
            return CompletedProcess(args="", returncode=0, stdout="", stderr="")
        if "test -x" in cmd or "test -d" in cmd or "test -f" in cmd:
            return CompletedProcess(args="", returncode=1, stdout="", stderr="")
        return _ok()

    backend.run = MagicMock(side_effect=run_side_effect)
    backend.copy_to = MagicMock()
    backend.launch_background = MagicMock(return_value=12345)
    backend.tail_log = MagicMock()


class TestLaunchTorchtitan:
    @patch("ptq.application.run_service.deploy_scripts")
    def test_torchtitan_uses_git_worktree(self, _deploy, repo, frozen_date):
        """Torchtitan should use standard git worktree for the primary repo."""
        backend = LocalBackend(workspace="/tmp/ws")
        _mock_backend(backend)

        launch(
            repo,
            backend,
            RunRequest(message="hello", local=True, follow=False, repo="torchtitan"),
        )

        run_cmds = [
            call.args[0]
            for call in backend.run.call_args_list
            if isinstance(call.args[0], str)
        ]
        assert any("git worktree add" in c and "/torchtitan" in c for c in run_cmds)
        assert not any(
            "create_worktree.py create torchtitan" in c for c in run_cmds
        )

    @patch("ptq.application.run_service.deploy_scripts")
    def test_torchtitan_creates_pytorch_support_worktree(
        self, _deploy, repo, frozen_date
    ):
        backend = LocalBackend(workspace="/tmp/ws")
        _mock_backend(backend)

        job_id = launch(
            repo,
            backend,
            RunRequest(message="hello", local=True, follow=False, repo="torchtitan"),
        )

        run_cmds = [
            call.args[0]
            for call in backend.run.call_args_list
            if isinstance(call.args[0], str)
        ]
        assert any(
            "create_worktree.py create pytorch" in c
            and f"--parent-dir /tmp/ws/jobs/{job_id}" in c
            for c in run_cmds
        )

    @patch("ptq.application.run_service.deploy_scripts")
    def test_reused_torchtitan_job_rewrites_torch_editable_paths(self, _deploy, repo):
        backend = LocalBackend(workspace="/tmp/ws")
        repo.save(
            JobRecord(
                job_id="j1",
                issue=3409,
                runs=1,
                local=True,
                workspace="/tmp/ws",
                repo="torchtitan",
            )
        )

        def run_side_effect(cmd: str, check: bool = True, **kw):
            if cmd == "test -d /tmp/ws/torchtitan/.git":
                return _ok()
            if "test -d /tmp/ws/jobs/j1/torchtitan/.git" in cmd:
                return _ok()
            if cmd == "test -d /tmp/ws/jobs/j1/.venv/bin":
                return _ok()
            if "test -d /tmp/ws/jobs/j1/pytorch/.git" in cmd:
                return _ok()
            if cmd == "test -x /tmp/ws/jobs/j1/.venv/bin/python":
                return _ok()
            if "sysconfig.get_path" in cmd:
                return CompletedProcess(
                    args="", returncode=0, stdout="/tmp/ws/jobs/j1/.venv/site\n"
                )
            if cmd == "realpath /tmp/ws/pytorch":
                return CompletedProcess(args="", returncode=0, stdout="/base/pytorch\n")
            if cmd == "realpath /tmp/ws/jobs/j1/pytorch":
                return CompletedProcess(args="", returncode=0, stdout="/job/pytorch\n")
            return _ok()

        backend.run = MagicMock(side_effect=run_side_effect)
        backend.copy_to = MagicMock()
        backend.launch_background = MagicMock(return_value=12345)
        backend.tail_log = MagicMock()

        launch(
            repo,
            backend,
            RunRequest(
                existing_job_id="j1",
                message="continue",
                local=True,
                follow=False,
                repo="torchtitan",
            ),
        )

        run_cmds = [
            call.args[0]
            for call in backend.run.call_args_list
            if isinstance(call.args[0], str)
        ]
        assert any(
            "__editable__*torch*" in c and "/tmp/ws/jobs/j1/pytorch" in c
            for c in run_cmds
        )

    @patch("ptq.application.run_service.deploy_scripts")
    def test_torchtitan_repo_persisted(self, _deploy, repo, frozen_date):
        backend = LocalBackend(workspace="/tmp/ws")
        _mock_backend(backend)

        job_id = launch(
            repo,
            backend,
            RunRequest(message="hello", local=True, follow=False, repo="torchtitan"),
        )

        job = repo.get(job_id)
        assert job.repo == "torchtitan"

    @patch("ptq.application.run_service.deploy_scripts")
    def test_pytorch_still_uses_create_worktree(self, _deploy, repo, frozen_date):
        """Pytorch should still use create_worktree.py."""
        backend = LocalBackend(workspace="/tmp/ws")
        _mock_backend(backend)

        launch(
            repo,
            backend,
            RunRequest(message="hello", local=True, follow=False, repo="pytorch"),
        )

        run_cmds = [
            call.args[0]
            for call in backend.run.call_args_list
            if isinstance(call.args[0], str)
        ]
        assert any("create_worktree.py" in c for c in run_cmds)


def test_finalize_run_writes_fallback_report_when_missing():
    backend = LocalBackend(workspace="/tmp/ws")
    writes = []

    log = (
        '{"type":"assistant","message":{"role":"assistant","content":'
        '[{"type":"text","text":"Partial diagnosis before stopping."}]}}\n'
        '{"type":"result","subtype":"error_max_turns",'
        '"terminal_reason":"max_turns","errors":["Reached maximum turns"]}\n'
    )
    worklog = "## Run 1\n\n### Reproduce\nConfirmed failure.\n"

    def run_side_effect(cmd: str, check: bool = True, **kw):
        if cmd == "cat /tmp/ws/jobs/j1/worklog.md":
            return CompletedProcess(args="", returncode=0, stdout=worklog, stderr="")
        if cmd == "cat /tmp/ws/jobs/j1/agent_logs/claude-1.log":
            return CompletedProcess(args="", returncode=0, stdout=log, stderr="")
        if cmd == "test -s /tmp/ws/jobs/j1/report.md":
            return CompletedProcess(args="", returncode=1, stdout="", stderr="")
        if "cat > /tmp/ws/jobs/j1/report.md" in cmd:
            writes.append(cmd)
        return _ok()

    backend.run = MagicMock(side_effect=run_side_effect)

    finalize_run(
        backend,
        "j1",
        JobRecord(job_id="j1", workspace="/tmp/ws", agent="claude", runs=1),
    )

    assert writes
    assert "PTQ Fallback Report" in writes[0]
    assert "error_max_turns / max_turns / Reached maximum turns" in writes[0]
    assert "Confirmed failure." in writes[0]
