from __future__ import annotations

from ptq.evaluator.models import ReviewComment, ReviewResult
from ptq.orchestrator.review import (
    PullRequestRef,
    build_pr_review_report,
    parse_diff_hunks,
    parse_pull_request_url,
)


def test_parse_pull_request_url_accepts_review_anchor():
    ref = parse_pull_request_url(
        "https://github.com/pytorch/pytorch/pull/184746#pullrequestreview-1"
    )

    assert ref.owner == "pytorch"
    assert ref.repo == "pytorch"
    assert ref.number == 184746
    assert ref.url == "https://github.com/pytorch/pytorch/pull/184746"


def test_review_report_includes_code_snapshot_for_actionable_feedback():
    diff = """diff --git a/torch/foo.py b/torch/foo.py
--- a/torch/foo.py
+++ b/torch/foo.py
@@ -8,4 +8,5 @@ def f():
     before()
     value = old_call()
+    value = new_call()
     after()
"""
    hunks = parse_diff_hunks(diff)
    review = ReviewResult(
        verdict="needs_revision",
        score=0.55,
        iteration=1,
        repro_fidelity="uncertain",
        component_scores={
            "fix_correctness": 0.55,
            "scope_minimality": 0.9,
            "test_coverage": 0.6,
            "code_quality": 0.8,
        },
        comments=[
            ReviewComment(
                file="torch/foo.py",
                line=10,
                comment="This changed path needs a regression test.",
                severity="blocking",
                reviewer="gpt-5.5",
            )
        ],
        summary="Reviewers found a test coverage gap.",
        reviewer_results=[
            {
                "reviewer": "gpt-5.5",
                "verdict": "needs_revision",
                "score": 0.55,
                "summary": "Needs test coverage.",
            },
            {
                "reviewer": "claude-opus-4-7",
                "verdict": "approved",
                "score": 0.86,
                "summary": "The change is ready.",
            }
        ],
    )

    report = build_pr_review_report(
        PullRequestRef(
            owner="pytorch",
            repo="pytorch",
            number=184746,
            url="https://github.com/pytorch/pytorch/pull/184746",
        ),
        {
            "title": "Fix foo",
            "body": (
                "## Problem\n"
                "The foo path loses alias metadata during tracing.\n\n"
                "## Solution\n"
                "Replay the wrapper view metadata instead of using the fallback."
            ),
            "files": [
                {"path": "torch/foo.py"},
                {"path": "test/test_foo.py"},
            ],
        },
        review,
        hunks,
    )

    assert "## Consolidated Summary" in report
    assert "### Intro" in report
    assert all(
        heading in report
        for heading in ("**Background**", "**Problem**", "**Fix Summary**")
    )
    assert "Not ready; 1/2 reviewers requested changes." in report
    assert "### gpt-5.5" in report
    assert "This changed path needs a regression test." in report
    assert "### 1. gpt-5.5: torch/foo.py:10" in report
    assert "@@ -8,4 +8,5 @@ def f():" in report
    assert "+    value = new_call()" in report
