from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from ptq.cli import app
from ptq.domain.models import JobRecord, RebaseInfo, RebaseState
from ptq.infrastructure.job_repository import JobRepository
from ptq.orchestrator.models import Issue, SolveResult

runner = CliRunner()


def _make_repo(tmp_path: Path, records: list[JobRecord] | None = None) -> JobRepository:
    repo = JobRepository(tmp_path / "jobs.json")
    for r in records or []:
        repo.save(r)
    return repo


class TestRunValidation:
    def test_no_issue_no_message_no_job_id(self):
        result = runner.invoke(app, ["run", "--local"])
        assert result.exit_code != 0
        assert (
            "Provide --issue, --preset, --message, or a JOB_ID to re-run."
            in result.output
        )

    def test_defaults_to_local_when_no_machine(self, tmp_path):
        repo = _make_repo(tmp_path)

        def fake_launch(r, b, req, **kw):
            repo.save(JobRecord(job_id="test-job", local=True, workspace="/tmp/ws"))
            return "test-job"

        with (
            patch("ptq.cli._repo", return_value=repo),
            patch(
                "ptq.application.run_service.launch", side_effect=fake_launch
            ) as mock_launch,
        ):
            result = runner.invoke(app, ["run", "-m", "hello", "--no-follow"])

        assert result.exit_code == 0, result.output
        mock_launch.assert_called_once()
        request = mock_launch.call_args.args[2]
        assert request.local is True

    def test_no_follow_prints_takeover_command(self, tmp_path):
        repo = _make_repo(tmp_path)

        def fake_launch(r, b, req, **kw):
            repo.save(JobRecord(job_id="test-job", local=True, workspace="/tmp/ws"))
            return "test-job"

        with (
            patch("ptq.cli._repo", return_value=repo),
            patch("ptq.application.run_service.launch", side_effect=fake_launch),
        ):
            result = runner.invoke(app, ["run", "-m", "hello", "--no-follow"])

        assert result.exit_code == 0, result.output
        flat = " ".join(result.output.split())
        assert "Take over: cd /tmp/ws/jobs/test-job && source .venv/bin/activate" in flat
        assert "Results: ptq results test-job" in flat

    def test_input_and_message_mutually_exclusive(self):
        result = runner.invoke(app, ["run", "-i", "f.md", "-m", "hello", "--local"])
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output

    def test_input_file_not_found(self):
        result = runner.invoke(app, ["run", "-i", "/nonexistent/path.md", "--local"])
        assert result.exit_code != 0
        assert "File not found" in result.output

    def test_input_file_reads_contents(self, tmp_path):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("do the thing")
            f.flush()
            tmp_file = f.name

        repo = _make_repo(tmp_path)

        def fake_launch(r, b, req, **kw):
            repo.save(JobRecord(job_id="test-job", local=True, workspace="/tmp/ws"))
            return "test-job"

        with (
            patch("ptq.cli._repo", return_value=repo),
            patch(
                "ptq.application.run_service.launch", side_effect=fake_launch
            ) as mock_launch,
        ):
            result = runner.invoke(
                app, ["run", "-i", tmp_file, "--local", "--no-follow"]
            )

        Path(tmp_file).unlink()
        assert result.exit_code == 0, result.output
        mock_launch.assert_called_once()
        request = mock_launch.call_args.args[2]
        assert request.message == "do the thing"

    def test_agent_type_passed_through(self, tmp_path):
        repo = _make_repo(tmp_path)

        def fake_launch(r, b, req, **kw):
            repo.save(
                JobRecord(
                    job_id="test-job", local=True, workspace="/tmp/ws", agent="codex"
                )
            )
            return "test-job"

        with (
            patch("ptq.cli._repo", return_value=repo),
            patch(
                "ptq.application.run_service.launch", side_effect=fake_launch
            ) as mock_launch,
        ):
            result = runner.invoke(
                app, ["run", "-m", "hello", "--agent", "codex", "--no-follow"]
            )

        assert result.exit_code == 0, result.output
        request = mock_launch.call_args.args[2]
        assert request.agent_type == "codex"

    def test_review_file_passed_through(self, tmp_path):
        review_path = tmp_path / "review.json"
        review_path.write_text('{"verdict": "needs_revision"}')
        repo = _make_repo(tmp_path)

        def fake_launch(r, b, req, **kw):
            repo.save(JobRecord(job_id="test-job", local=True, workspace="/tmp/ws"))
            return "test-job"

        with (
            patch("ptq.cli._repo", return_value=repo),
            patch(
                "ptq.application.run_service.launch", side_effect=fake_launch
            ) as mock_launch,
        ):
            result = runner.invoke(
                app,
                [
                    "run",
                    "-m",
                    "hello",
                    "--review-file",
                    str(review_path),
                    "--no-follow",
                ],
            )

        assert result.exit_code == 0, result.output
        request = mock_launch.call_args.args[2]
        assert request.review_feedback_json == '{"verdict": "needs_revision"}'

    def test_issue_url_infers_repo_for_run(self, tmp_path):
        repo = _make_repo(tmp_path)

        def fake_launch(r, b, req, **kw):
            repo.save(
                JobRecord(
                    job_id="test-job",
                    issue=req.issue_number,
                    local=True,
                    workspace="/tmp/ws",
                    repo=req.repo,
                )
            )
            return "test-job"

        cfg = type(
            "Cfg",
            (),
            {
                "default_agent": "claude",
                "default_max_turns": 100,
                "effective_model": staticmethod(lambda agent, model=None: "opus"),
                "effective_thinking": staticmethod(lambda agent, thinking=None: None),
                "prompt_preset": staticmethod(lambda _x: None),
                "prompt_preset_choices": staticmethod(lambda: []),
            },
        )()

        with (
            patch("ptq.cli._repo", return_value=repo),
            patch("ptq.config.load_config", return_value=cfg),
            patch(
                "ptq.issue.fetch_issue",
                return_value={"title": "TorchTitan bug", "body": "", "labels": []},
            ) as fetch_issue,
            patch("ptq.infrastructure.backends.create_backend"),
            patch(
                "ptq.application.run_service.launch", side_effect=fake_launch
            ) as mock_launch,
        ):
            result = runner.invoke(
                app,
                [
                    "run",
                    "--issue",
                    "https://github.com/pytorch/torchtitan/issues/3409",
                    "--no-follow",
                ],
            )

        assert result.exit_code == 0, result.output
        fetch_issue.assert_called_once_with(3409, repo="pytorch/torchtitan")
        request = mock_launch.call_args.args[2]
        assert request.issue_number == 3409
        assert request.repo == "torchtitan"

    def test_thinking_passed_through(self, tmp_path):
        repo = _make_repo(tmp_path)

        def fake_launch(r, b, req, **kw):
            repo.save(JobRecord(job_id="test-job", local=True, workspace="/tmp/ws"))
            return "test-job"

        with (
            patch("ptq.cli._repo", return_value=repo),
            patch(
                "ptq.application.run_service.launch", side_effect=fake_launch
            ) as mock_launch,
        ):
            result = runner.invoke(
                app,
                [
                    "run",
                    "-m",
                    "hello",
                    "--agent",
                    "pi",
                    "--thinking",
                    "high",
                    "--no-follow",
                ],
            )

        assert result.exit_code == 0, result.output
        request = mock_launch.call_args.args[2]
        assert request.thinking == "high"

    def test_thinking_uses_agent_default_when_omitted(self, tmp_path):
        repo = _make_repo(tmp_path)

        def fake_launch(r, b, req, **kw):
            repo.save(JobRecord(job_id="test-job", local=True, workspace="/tmp/ws"))
            return "test-job"

        with (
            patch("ptq.cli._repo", return_value=repo),
            patch(
                "ptq.config.load_config",
                return_value=type(
                    "Cfg",
                    (),
                    {
                        "default_agent": "claude",
                        "default_max_turns": 100,
                        "effective_model": staticmethod(
                            lambda agent, model=None: model or "openai-codex/gpt-5.5"
                        ),
                        "effective_thinking": staticmethod(
                            lambda agent, thinking=None: thinking or "high"
                        ),
                        "prompt_preset": staticmethod(lambda _x: None),
                        "prompt_preset_choices": staticmethod(lambda: []),
                    },
                )(),
            ),
            patch(
                "ptq.application.run_service.launch", side_effect=fake_launch
            ) as mock_launch,
        ):
            result = runner.invoke(
                app, ["run", "-m", "hello", "--agent", "pi", "--no-follow"]
            )

        assert result.exit_code == 0, result.output
        request = mock_launch.call_args.args[2]
        assert request.thinking == "high"

    def test_rerun_passes_existing_job_id(self, tmp_path):
        repo = _make_repo(
            tmp_path,
            [
                JobRecord(
                    job_id="20260217-adhoc-abc123",
                    runs=1,
                    local=True,
                    workspace="/tmp/ws",
                    agent="cursor",
                ),
            ],
        )
        with (
            patch("ptq.cli._repo", return_value=repo),
            patch("ptq.application.run_service.launch") as mock_launch,
        ):
            mock_launch.return_value = "20260217-adhoc-abc123"
            result = runner.invoke(
                app, ["run", "20260217-adhoc-abc123", "-m", "try again"]
            )

        assert result.exit_code == 0, result.output
        request = mock_launch.call_args.args[2]
        assert request.existing_job_id == "20260217-adhoc-abc123"
        assert request.agent_type == "cursor"

    def test_rerun_preserves_saved_thinking(self, tmp_path):
        repo = _make_repo(
            tmp_path,
            [
                JobRecord(
                    job_id="20260217-adhoc-abc123",
                    runs=1,
                    local=True,
                    workspace="/tmp/ws",
                    agent="pi",
                    model="openai-codex/gpt-5.5",
                    thinking="high",
                ),
            ],
        )
        with (
            patch("ptq.cli._repo", return_value=repo),
            patch("ptq.application.run_service.launch") as mock_launch,
        ):
            mock_launch.return_value = "20260217-adhoc-abc123"
            result = runner.invoke(
                app, ["run", "20260217-adhoc-abc123", "-m", "try again"]
            )

        assert result.exit_code == 0, result.output
        request = mock_launch.call_args.args[2]
        assert request.agent_type == "pi"
        assert request.thinking == "high"


def _make_clean_repo(tmp_path: Path) -> JobRepository:
    return _make_repo(
        tmp_path,
        [
            JobRecord(
                job_id="job-stopped",
                issue=100,
                runs=1,
                local=True,
                workspace="/tmp/ws",
                agent="claude",
            ),
            JobRecord(
                job_id="job-running",
                issue=200,
                runs=2,
                local=True,
                workspace="/tmp/ws",
                agent="codex",
                pid=99999,
            ),
        ],
    )


RUNNING_PID = 99999


class TestCleanSingleJob:
    def test_removes_job_from_db(self, tmp_path):
        repo = _make_clean_repo(tmp_path)
        mock_backend = MagicMock()
        mock_backend.workspace = "/tmp/ws"
        mock_backend.is_pid_alive.return_value = False

        with (
            patch("ptq.cli._repo", return_value=repo),
            patch(
                "ptq.application.job_service.backend_for_job", return_value=mock_backend
            ),
        ):
            result = runner.invoke(app, ["clean", "job-stopped"])
        assert result.exit_code == 0, result.output
        assert "job-stopped" not in repo.list_all()
        assert "removed" in result.output

    def test_unknown_target_treated_as_machine(self, tmp_path):
        repo = _make_clean_repo(tmp_path)
        mock_backend = MagicMock()
        mock_backend.workspace = "/tmp/ws"
        mock_backend.is_pid_alive.return_value = False

        with (
            patch("ptq.cli._repo", return_value=repo),
            patch(
                "ptq.infrastructure.backends.RemoteBackend", return_value=mock_backend
            ),
        ):
            result = runner.invoke(app, ["clean", "nonexistent-machine"])
        assert result.exit_code == 0
        assert "Nothing to clean" in result.output


class TestCleanMachine:
    def test_bulk_clean_removes_stopped_jobs(self, tmp_path):
        repo = _make_clean_repo(tmp_path)
        mock_backend = MagicMock()
        mock_backend.workspace = "/tmp/ws"
        mock_backend.is_pid_alive.side_effect = lambda pid: pid == RUNNING_PID

        with (
            patch("ptq.cli._repo", return_value=repo),
            patch(
                "ptq.infrastructure.backends.LocalBackend", return_value=mock_backend
            ),
        ):
            result = runner.invoke(app, ["clean", "--local"])
        assert result.exit_code == 0, result.output
        remaining = repo.list_all()
        assert "job-stopped" not in remaining
        assert "job-running" in remaining


class TestSetupValidation:
    def test_no_machine_no_local(self):
        result = runner.invoke(app, ["setup"])
        assert result.exit_code != 0


class TestList:
    def test_list_shows_pr_and_rebase_state(self, tmp_path):
        repo = _make_repo(
            tmp_path,
            [
                JobRecord(
                    job_id="job-1",
                    issue=176093,
                    local=True,
                    workspace="/tmp/ws",
                    agent="cursor",
                    runs=46,
                    pr_url="https://github.com/pytorch/pytorch/pull/176243",
                    rebase=RebaseInfo(state=RebaseState.NEEDS_HUMAN),
                )
            ],
        )
        mock_backend = MagicMock()
        mock_backend.workspace = "/tmp/ws"
        mock_backend.is_pid_alive.return_value = False

        with (
            patch("ptq.cli._repo", return_value=repo),
            patch(
                "ptq.infrastructure.backends.backend_for_job", return_value=mock_backend
            ),
            patch("ptq.application.pr_service.get_pr_state", return_value="closed"),
        ):
            result = runner.invoke(app, ["list"])

        assert result.exit_code == 0, result.output
        assert "PR" in result.output
        assert "Rebase" in result.output
        assert "closed" in result.output
        assert "human" in result.output

    def test_list_shows_dashes_when_no_pr_or_rebase(self, tmp_path):
        repo = _make_repo(
            tmp_path,
            [
                JobRecord(
                    job_id="job-2",
                    issue=176094,
                    local=True,
                    workspace="/tmp/ws",
                    agent="claude",
                )
            ],
        )
        mock_backend = MagicMock()
        mock_backend.workspace = "/tmp/ws"
        mock_backend.is_pid_alive.return_value = False

        with (
            patch("ptq.cli._repo", return_value=repo),
            patch(
                "ptq.infrastructure.backends.backend_for_job", return_value=mock_backend
            ),
        ):
            result = runner.invoke(app, ["list"])

        assert result.exit_code == 0, result.output
        assert "PR" in result.output
        assert "Rebase" in result.output
        assert "#176" in result.output


def test_evaluator_models_default_to_required_review_pair():
    from ptq.cli import _evaluator_models

    assert _evaluator_models({"model": "claude"}) == [
        "gpt-5.5",
        "claude-opus-4-7",
    ]


def _fake_orchestrate_config():
    return type(
        "Cfg",
        (),
        {
            "orchestrator": {
                "github_repo": "pytorch/pytorch",
                "issue_selection_prompt": "open bugs",
                "max_issues": 7,
                "parallel": 3,
                "max_iterations": 4,
                "approval_threshold": 0.8,
                "machine": "localhost",
            },
            "evaluator": {
                "approval_threshold": 0.8,
                "shelve_threshold": 0.3,
                "max_iterations": 4,
            },
            "default_agent": "claude",
            "default_max_turns": 100,
            "effective_model": staticmethod(lambda agent: "opus"),
            "effective_thinking": staticmethod(lambda agent: None),
        },
    )()


def test_orchestrate_defaults_follow_on_and_pr_off():
    captured = {}

    class FakeEvaluator:
        def __init__(self, **kwargs):
            pass

    class FakeOrchestrator:
        def __init__(self, config, **kwargs):
            captured["config"] = config
            captured["stream_solver"] = kwargs["stream_solver"]

        async def run(self):
            return []

    with (
        patch("ptq.config.load_config", return_value=_fake_orchestrate_config()),
        patch("ptq.evaluator.Evaluator", FakeEvaluator),
        patch("ptq.orchestrator.Orchestrator", FakeOrchestrator),
    ):
        result = runner.invoke(app, ["orchestrate", "--issue", "123"])

    assert result.exit_code == 0, result.output
    assert captured["stream_solver"] is True
    assert captured["config"].push_pr is False
    assert captured["config"].max_issues == 1


def test_orchestrate_watch_pr_implies_pr_and_sets_watch_knobs():
    captured = {}

    class FakeEvaluator:
        def __init__(self, **kwargs):
            pass

    class FakeOrchestrator:
        def __init__(self, config, **kwargs):
            captured["config"] = config

        async def run(self):
            return []

    with (
        patch("ptq.config.load_config", return_value=_fake_orchestrate_config()),
        patch("ptq.evaluator.Evaluator", FakeEvaluator),
        patch("ptq.orchestrator.Orchestrator", FakeOrchestrator),
    ):
        result = runner.invoke(
            app,
            [
                "orchestrate",
                "--issue",
                "123",
                "--watch-pr",
                "--watch-pr-interval-seconds",
                "10",
                "--watch-pr-idle-hours",
                "2",
            ],
        )

    assert result.exit_code == 0, result.output
    assert captured["config"].push_pr is True
    assert captured["config"].watch_pr is True
    assert captured["config"].watch_pr_interval_seconds == 10
    assert captured["config"].watch_pr_idle_seconds == 7200


def test_orchestrate_adds_profile_backed_evaluator(tmp_path):
    profile_path = tmp_path / "aditvenk.md"

    class FakeEvaluator:
        def __init__(self, **kwargs):
            raise AssertionError("add-evaluator must not construct Evaluator")

    class FakeOrchestrator:
        def __init__(self, config, **kwargs):
            raise AssertionError("add-evaluator must not launch Orchestrator")

        async def run(self):
            return []

    with (
        patch("ptq.config.load_config", return_value=_fake_orchestrate_config()),
        patch("ptq.evaluator.Evaluator", FakeEvaluator),
        patch("ptq.orchestrator.Orchestrator", FakeOrchestrator),
        patch(
            "ptq.evaluator.reviewer_profile.generate_reviewer_profile"
        ) as generate,
        patch(
            "ptq.evaluator.reviewer_profile.reviewer_profile_path",
            return_value=profile_path,
        ),
        patch("ptq.config.ensure_additional_evaluator", return_value=True) as persist,
    ):
        result = runner.invoke(
            app,
            [
                "orchestrate",
                "--add-evaluator",
                "aditvenk-style",
                "--profile",
                "aditvenk",
                "--agent",
                "gpt-5.5",
            ],
        )

    assert result.exit_code == 0, result.output
    generate.assert_called_once_with("aditvenk", repo="pytorch/pytorch")
    persist.assert_called_once_with(
        name="aditvenk-style",
        profile="aditvenk",
        model="gpt-5.5",
    )
    assert "saved evaluator aditvenk-style" in result.output


def test_orchestrate_add_evaluator_rejects_issue_option(tmp_path):
    profile_path = tmp_path / "aditvenk.md"

    with (
        patch("ptq.config.load_config", return_value=_fake_orchestrate_config()),
        patch(
            "ptq.evaluator.reviewer_profile.reviewer_profile_path",
            return_value=profile_path,
        ),
        patch("ptq.evaluator.reviewer_profile.generate_reviewer_profile") as generate,
        patch("ptq.config.ensure_additional_evaluator") as persist,
    ):
        result = runner.invoke(
            app,
            [
                "orchestrate",
                "--issue",
                "123",
                "--add-evaluator",
                "aditvenk-style",
                "--profile",
                "aditvenk",
            ],
        )

    assert result.exit_code != 0
    assert "--add-evaluator only adds an evaluator" in result.output
    assert "--issue" in result.output
    generate.assert_not_called()
    persist.assert_not_called()


def test_orchestrate_repo_flag_selects_supported_repo_profile():
    captured = {}

    class FakeEvaluator:
        def __init__(self, **kwargs):
            pass

    class FakeOrchestrator:
        def __init__(self, config, **kwargs):
            captured["config"] = config

        async def run(self):
            return []

    with (
        patch("ptq.config.load_config", return_value=_fake_orchestrate_config()),
        patch("ptq.evaluator.Evaluator", FakeEvaluator),
        patch("ptq.orchestrator.Orchestrator", FakeOrchestrator),
    ):
        result = runner.invoke(
            app,
            ["orchestrate", "--repo", "torchtitan", "--issue", "123"],
        )

    assert result.exit_code == 0, result.output
    assert captured["config"].repo == "torchtitan"
    assert captured["config"].github_repo == "pytorch/torchtitan"
    assert (
        captured["config"].issue_selection_prompt
        == "https://github.com/pytorch/torchtitan/issues/123"
    )


def test_orchestrate_issue_url_infers_repo_profile():
    captured = {}

    class FakeEvaluator:
        def __init__(self, **kwargs):
            pass

    class FakeOrchestrator:
        def __init__(self, config, **kwargs):
            captured["config"] = config

        async def run(self):
            return []

    with (
        patch("ptq.config.load_config", return_value=_fake_orchestrate_config()),
        patch("ptq.evaluator.Evaluator", FakeEvaluator),
        patch("ptq.orchestrator.Orchestrator", FakeOrchestrator),
    ):
        result = runner.invoke(
            app,
            [
                "orchestrate",
                "--issue",
                "https://github.com/pytorch/torchtitan/issues/3409",
            ],
        )

    assert result.exit_code == 0, result.output
    assert captured["config"].repo == "torchtitan"
    assert captured["config"].github_repo == "pytorch/torchtitan"
    assert captured["config"].max_issues == 1
    assert (
        captured["config"].issue_selection_prompt
        == "https://github.com/pytorch/torchtitan/issues/3409"
    )


def test_orchestrate_review_runs_evaluator_panel_only():
    captured = {}

    class FakeEvaluator:
        def __init__(self, **kwargs):
            captured["evaluator_kwargs"] = kwargs

        def validate_configuration(self):
            captured["validated"] = True

    class FakeOrchestrator:
        def __init__(self, config, **kwargs):
            raise AssertionError("--review must not launch Orchestrator")

        async def run(self):
            return []

    def fake_review(url, evaluator, **kwargs):
        captured["review_url"] = url
        captured["evaluator"] = evaluator
        return SimpleNamespace(
            pr=SimpleNamespace(url="https://github.com/pytorch/pytorch/pull/184746"),
            review=SimpleNamespace(verdict="needs_revision", score=0.72),
            report_path=Path("/tmp/ptq-review/report.md"),
        )

    with (
        patch("ptq.config.load_config", return_value=_fake_orchestrate_config()),
        patch("ptq.evaluator.Evaluator", FakeEvaluator),
        patch("ptq.orchestrator.Orchestrator", FakeOrchestrator),
        patch("ptq.orchestrator.review.run_pull_request_review", side_effect=fake_review),
    ):
        result = runner.invoke(
            app,
            [
                "orchestrate",
                "--review",
                "https://github.com/pytorch/pytorch/pull/184746#pullrequestreview-1",
            ],
        )

    assert result.exit_code == 0, result.output
    assert captured["validated"] is True
    assert captured["review_url"].startswith("https://github.com/pytorch/pytorch/pull/")
    assert "report.md: /tmp/ptq-review/report.md" in result.output


def test_orchestrate_message_becomes_initial_solver_guidance():
    captured = {}

    class FakeEvaluator:
        def __init__(self, **kwargs):
            pass

    class FakeOrchestrator:
        def __init__(self, config, **kwargs):
            captured["config"] = config

        async def run(self):
            return []

    with (
        patch("ptq.config.load_config", return_value=_fake_orchestrate_config()),
        patch("ptq.evaluator.Evaluator", FakeEvaluator),
        patch("ptq.orchestrator.Orchestrator", FakeOrchestrator),
    ):
        result = runner.invoke(
            app,
            [
                "orchestrate",
                "--issue",
                "123",
                "-m",
                "Start by checking optimizer state restoration.",
            ],
        )

    assert result.exit_code == 0, result.output
    assert (
        captured["config"].initial_message
        == "Start by checking optimizer state restoration."
    )
    assert captured["config"].adhoc is False


def test_orchestrate_message_without_issue_runs_adhoc_loop():
    captured = {}

    class FakeEvaluator:
        def __init__(self, **kwargs):
            pass

    class FakeOrchestrator:
        def __init__(self, config, **kwargs):
            captured["config"] = config

        async def run(self):
            return []

    with (
        patch("ptq.config.load_config", return_value=_fake_orchestrate_config()),
        patch("ptq.evaluator.Evaluator", FakeEvaluator),
        patch("ptq.orchestrator.Orchestrator", FakeOrchestrator),
    ):
        result = runner.invoke(
            app,
            [
                "orchestrate",
                "--repo",
                "torchtitan",
                "--message",
                "Investigate and fix the checkpoint load failure.",
                "--pr",
            ],
        )

    assert result.exit_code == 0, result.output
    assert captured["config"].adhoc is True
    assert captured["config"].repo == "torchtitan"
    assert captured["config"].issue_selection_prompt == ""
    assert (
        captured["config"].initial_message
        == "Investigate and fix the checkpoint load failure."
    )
    assert captured["config"].push_pr is True


def test_orchestrate_infers_repo_from_configured_github_repo():
    captured = {}
    cfg = _fake_orchestrate_config()
    cfg.orchestrator["github_repo"] = "pytorch/torchtitan"

    class FakeEvaluator:
        def __init__(self, **kwargs):
            pass

    class FakeOrchestrator:
        def __init__(self, config, **kwargs):
            captured["config"] = config

        async def run(self):
            return []

    with (
        patch("ptq.config.load_config", return_value=cfg),
        patch("ptq.evaluator.Evaluator", FakeEvaluator),
        patch("ptq.orchestrator.Orchestrator", FakeOrchestrator),
    ):
        result = runner.invoke(app, ["orchestrate", "--issue", "123"])

    assert result.exit_code == 0, result.output
    assert captured["config"].repo == "torchtitan"
    assert captured["config"].github_repo == "pytorch/torchtitan"


def test_orchestrate_final_output_points_to_report_and_pr(tmp_path):
    workspace = tmp_path / "ws"
    report = workspace / "jobs" / "job-123" / "report.md"
    report.parent.mkdir(parents=True)
    report.write_text("report")
    repo = _make_repo(
        tmp_path,
        [
            JobRecord(
                job_id="job-123",
                issue=123,
                local=True,
                workspace=str(workspace),
            )
        ],
    )

    class FakeEvaluator:
        def __init__(self, **kwargs):
            pass

    class FakeOrchestrator:
        def __init__(self, config, **kwargs):
            pass

        async def run(self):
            return [
                SolveResult(
                    issue=Issue(number=123, title="bug"),
                    verdict="approved",
                    score=0.91,
                    iterations=1,
                    job_id="job-123",
                    branch="ptq/123",
                    pr_url="https://github.com/pytorch/pytorch/pull/1",
                )
            ]

    with (
        patch("ptq.cli._repo", return_value=repo),
        patch("ptq.config.load_config", return_value=_fake_orchestrate_config()),
        patch("ptq.evaluator.Evaluator", FakeEvaluator),
        patch("ptq.orchestrator.Orchestrator", FakeOrchestrator),
    ):
        result = runner.invoke(app, ["orchestrate", "--issue", "123", "--pr"])

    assert result.exit_code == 0, result.output
    assert (
        "#123 approved score=0.91 iterations=1 job=job-123 branch=ptq/123"
        in result.output
    )
    assert f"report.md: {report}" in result.output
    assert "PR: https://github.com/pytorch/pytorch/pull/1" in result.output


def test_orchestrate_final_output_marks_missing_report(tmp_path):
    workspace = tmp_path / "ws"
    repo = _make_repo(
        tmp_path,
        [
            JobRecord(
                job_id="job-123",
                issue=123,
                local=True,
                workspace=str(workspace),
            )
        ],
    )

    class FakeEvaluator:
        def __init__(self, **kwargs):
            pass

    class FakeOrchestrator:
        def __init__(self, config, **kwargs):
            pass

        async def run(self):
            return [
                SolveResult(
                    issue=Issue(number=123, title="bug"),
                    verdict="error",
                    score=0.0,
                    iterations=0,
                    job_id="job-123",
                )
            ]

    with (
        patch("ptq.cli._repo", return_value=repo),
        patch("ptq.config.load_config", return_value=_fake_orchestrate_config()),
        patch("ptq.evaluator.Evaluator", FakeEvaluator),
        patch("ptq.orchestrator.Orchestrator", FakeOrchestrator),
    ):
        result = runner.invoke(app, ["orchestrate", "--issue", "123"])

    expected = workspace / "jobs" / "job-123" / "report.md"
    assert result.exit_code == 0, result.output
    assert f"report.md: missing (expected at {expected})" in result.output


def test_generate_review_profile_is_not_a_top_level_command():
    result = runner.invoke(app, ["generate-review-profile", "aditvenk"])
    assert result.exit_code != 0


def test_evaluate_is_not_a_top_level_command():
    result = runner.invoke(app, ["evaluate", "job-123"])
    assert result.exit_code != 0
