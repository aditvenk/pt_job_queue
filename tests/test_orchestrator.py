from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from ptq.evaluator.models import ReviewComment, ReviewResult

from ptq.orchestrator.issue_selector import (
    IssueSelector,
    extract_direct_issue_numbers,
    translate_issue_selection_prompt,
)
from ptq.orchestrator.models import Issue, OrchestratorConfig, SolveResult
from ptq.orchestrator.orchestrator import (
    Orchestrator,
    _build_orchestrator_pr_note,
    _format_pr_feedback_snapshot,
    _format_review_snapshot,
)


def test_translates_common_issue_selection_prompt():
    query = translate_issue_selection_prompt(
        "Open issues labeled 'module: nn' with a repro script, filed in the last 30 days"
    )
    assert "is:issue" in query
    assert "is:open" in query
    assert 'label:"module: nn"' in query
    assert "repro" in query
    assert "created:>=" in query


def test_extracts_direct_issue_from_url():
    assert extract_direct_issue_numbers(
        "https://github.com/pytorch/pytorch/issues/76449"
    ) == [76449]


def test_direct_issue_selection_fetches_exact_issue():
    issue_data = {
        "title": "Bug",
        "body": "Body",
        "labels": [{"name": "module: nn"}],
        "comments": [],
    }
    with patch("ptq.orchestrator.issue_selector.fetch_issue", return_value=issue_data):
        issues = IssueSelector("pytorch/pytorch").select(
            "https://github.com/pytorch/pytorch/issues/76449",
            limit=1,
        )
    assert len(issues) == 1
    assert issues[0].number == 76449
    assert issues[0].url == "https://github.com/pytorch/pytorch/issues/76449"


def test_orchestrator_dry_run_returns_selected_issues(tmp_path):
    class FakeOrchestrator(Orchestrator):
        async def select_issues(self):
            return [Issue(number=1, title="bug")]

    config = OrchestratorConfig(
        issue_selection_prompt="open bugs",
        dry_run=True,
        log_path=tmp_path / "runs.jsonl",
    )
    results = asyncio.run(FakeOrchestrator(config).run())
    assert results[0].verdict == "dry_run"
    assert results[0].issue.number == 1


def test_solve_issue_stops_after_approved_review(tmp_path):
    class FakeOrchestrator(Orchestrator):
        async def _launch_solver(self, issue, review, pr_feedback=None):
            return "job-1"

        async def _wait_for_job(self, job_id):
            return None

        async def _read_solver_output(self, *, issue, job_id, iteration):
            return object()

        async def _write_review_artifact(self, job_id, review):
            return None

    class FakeEvaluator:
        def evaluate(self, solver_output):
            return ReviewResult(
                verdict="approved",
                score=0.9,
                iteration=1,
                repro_fidelity="faithful",
                summary="Approved.",
            )

    orchestrator = FakeOrchestrator(
        OrchestratorConfig(
            issue_selection_prompt="open bugs",
            log_path=tmp_path / "runs.jsonl",
        ),
        evaluator=FakeEvaluator(),
    )
    result = asyncio.run(orchestrator.solve_issue(Issue(number=1, title="bug")))
    assert result.verdict == "approved"
    assert result.iterations == 1


def test_solve_issue_injects_pr_feedback_when_present(tmp_path):
    captured = {}

    class FakeOrchestrator(Orchestrator):
        async def _fetch_pr_feedback(self, issue):
            return {
                "source": "github_pr",
                "pr_url": "https://github.com/pytorch/pytorch/pull/123",
                "comments": [
                    {
                        "kind": "inline",
                        "author": "reviewer",
                        "path": "torch/foo.py",
                        "line": 10,
                        "body": "Please handle the no_grad case.",
                    }
                ],
            }

        async def _launch_solver(self, issue, review, pr_feedback=None):
            captured["pr_feedback"] = pr_feedback
            return "job-1"

        async def _wait_for_job(self, job_id):
            return None

        async def _read_solver_output(self, *, issue, job_id, iteration):
            return object()

        async def _write_review_artifact(self, job_id, review):
            return None

    class FakeEvaluator:
        def evaluate(self, solver_output):
            return ReviewResult(
                verdict="approved",
                score=0.9,
                iteration=1,
                repro_fidelity="faithful",
                summary="Approved.",
            )

    orchestrator = FakeOrchestrator(
        OrchestratorConfig(
            issue_selection_prompt="open bugs",
            log_path=tmp_path / "runs.jsonl",
            push_pr=True,
        ),
        evaluator=FakeEvaluator(),
    )
    result = asyncio.run(orchestrator.solve_issue(Issue(number=1, title="bug")))
    assert result.verdict == "approved"
    assert captured["pr_feedback"]["comments"][0]["body"] == (
        "Please handle the no_grad case."
    )


def test_pr_feedback_snapshot_includes_failing_ci():
    text = _format_pr_feedback_snapshot(
        Issue(number=1, title="bug"),
        {
            "pr_url": "https://github.com/pytorch/pytorch/pull/123",
            "comments": [],
            "ci_failures": [
                {
                    "name": "distributed / test",
                    "state": "FAILURE",
                    "description": "Timeout in test_dtensor",
                }
            ],
        },
    )
    assert "0 GitHub PR feedback item(s) and 1 failing PR check(s)" in text
    assert "distributed / test (FAILURE): Timeout in test_dtensor" in text


def test_review_snapshot_includes_reviewer_scores_and_blocking_comments():
    text = _format_review_snapshot(
        ReviewResult(
            verdict="needs_revision",
            score=0.74,
            iteration=1,
            repro_fidelity="faithful",
            comments=[
                ReviewComment(
                    file="test/foo.py",
                    line=12,
                    comment="Add the missing regression test.",
                    severity="blocking",
                    reviewer="gpt-5.5",
                )
            ],
            summary="Not ready yet.",
            reviewer="aggregate",
            reviewer_results=[
                {"reviewer": "gpt-5.5", "verdict": "needs_revision", "score": 0.74},
                {"reviewer": "claude-opus-4-7", "verdict": "approved", "score": 0.85},
            ],
        )
    )
    assert "evaluator review: needs_revision score=0.74 repro=faithful" in text
    assert "gpt-5.5: needs_revision score=0.74" in text
    assert "claude-opus-4-7: approved score=0.85" in text
    assert "[gpt-5.5] test/foo.py:12: Add the missing regression test." in text


def test_post_process_keeps_approved_result_when_branch_prep_fails(tmp_path):
    class FakeOrchestrator(Orchestrator):
        async def _prepare_review_branch(self, result):
            raise RuntimeError("checkout failed")

    orchestrator = FakeOrchestrator(
        OrchestratorConfig(
            issue_selection_prompt="open bugs",
            log_path=tmp_path / "runs.jsonl",
        )
    )
    result = SolveResult(
        issue=Issue(number=1, title="bug"),
        verdict="approved",
        score=0.9,
        iterations=1,
        job_id="job-1",
        state="completed",
    )
    asyncio.run(orchestrator.post_process(result))
    assert result.verdict == "approved"
    assert result.branch is None
    assert "failed to prepare review branch" in result.error


def test_post_process_pushes_draft_pr_when_enabled(tmp_path):
    class FakeOrchestrator(Orchestrator):
        async def _prepare_review_branch(self, result):
            raise AssertionError("should push PR instead of preparing local branch")

        async def _push_draft_pr(self, result):
            return SimpleNamespace(
                branch=f"ptq/{result.issue.number}",
                url="https://github.com/pytorch/pytorch/pull/123",
            )

    orchestrator = FakeOrchestrator(
        OrchestratorConfig(
            issue_selection_prompt="open bugs",
            log_path=tmp_path / "runs.jsonl",
            push_pr=True,
        )
    )
    result = SolveResult(
        issue=Issue(number=1, title="bug"),
        verdict="approved",
        score=0.9,
        iterations=1,
        job_id="job-1",
        state="completed",
    )
    asyncio.run(orchestrator.post_process(result))
    assert result.branch == "ptq/1"
    assert result.pr_url == "https://github.com/pytorch/pytorch/pull/123"


def test_push_draft_pr_uses_issue_title_without_issue_number(tmp_path):
    class FakeRepo:
        pass

    captured = {}

    def fake_create_pr(repo, job_id, **kwargs):
        captured.update(kwargs)
        return type(
            "PR",
            (),
            {"branch": "ptq/166156", "url": "https://github.com/pytorch/pytorch/pull/1"},
        )()

    orchestrator = Orchestrator(
        OrchestratorConfig(
            issue_selection_prompt="open bugs",
            log_path=tmp_path / "runs.jsonl",
            push_pr=True,
        ),
        job_repo=FakeRepo(),
    )
    result = SolveResult(
        issue=Issue(
            number=166156,
            title="Parameter.to_local() will make parameter a normal tensor",
        ),
        verdict="approved",
        score=0.9,
        iterations=1,
        job_id="job-1",
    )
    with patch("ptq.application.pr_service.create_pr", side_effect=fake_create_pr):
        asyncio.run(orchestrator._push_draft_pr(result))
    assert captured["title"] == "Parameter.to_local() will make parameter a normal tensor"
    assert "#166156" not in captured["title"]


def test_orchestrator_pr_note_excludes_evaluator_summary():
    note = _build_orchestrator_pr_note(
        SolveResult(
            issue=Issue(number=166156, title="Parameter.to_local regression"),
            verdict="approved",
            score=0.88,
            iterations=1,
            job_id="job-1",
            review=ReviewResult(
                verdict="approved",
                score=0.88,
                iteration=1,
                repro_fidelity="faithful",
                summary="All evaluator reviewers approved the diff.",
            ),
        )
    )
    assert "Evaluator score: 0.88" in note
    assert "Evaluator summary:" not in note
    assert "All evaluator reviewers approved" not in note
