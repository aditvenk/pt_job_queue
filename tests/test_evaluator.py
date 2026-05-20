from __future__ import annotations

import json
import urllib.error
from unittest.mock import patch

from ptq.evaluator import Evaluator, ReviewerSpec, SolverOutput
from ptq.evaluator.evaluator import (
    _call_openai,
    _claude_cli_model,
    _extract_claude_cli_text,
)


class FakeEvaluator(Evaluator):
    def _call_llm(self, prompt: str, model_name: str | None = None) -> str:
        return json.dumps(
            {
                "verdict": "needs_revision",
                "score": 0.75,
                "iteration": 1,
                "repro_fidelity": "faithful",
                "comments": [
                    {
                        "file": "test",
                        "line": None,
                        "comment": "Add a regression test.",
                        "severity": "blocking",
                    }
                ],
                "summary": "Close, but test coverage is missing.",
            }
        )

    def _run_lint(self, worktree_path):
        return "Not run in unit test."


def test_evaluator_parses_structured_review():
    evaluator = FakeEvaluator(
        reviewer_models=["fake-reviewer"],
        approval_threshold=0.8,
        max_iterations=5,
    )
    result = evaluator.evaluate(
        SolverOutput(
            issue_number=174923,
            issue_body="torch.foo returns the wrong shape",
            iteration=1,
            report_md="Fixed shape calculation.",
            fix_diff="diff --git a/torch/foo.py b/torch/foo.py",
            repro_script="import torch\n",
            repro_filename="repro_174923_generated.py",
            status_json={"repro_source": "generated"},
        )
    )
    assert result.verdict == "needs_revision"
    assert result.score == 0.75
    assert result.comments[0].file == "test"


def test_unfaithful_repro_forces_zero_score():
    class UnfaithfulEvaluator(FakeEvaluator):
        def _call_llm(self, prompt: str, model_name: str | None = None) -> str:
            return json.dumps(
                {
                    "verdict": "approved",
                    "score": 1.0,
                    "iteration": 1,
                    "repro_fidelity": "unfaithful",
                    "comments": [],
                    "summary": "Wrong repro.",
                }
            )

    result = UnfaithfulEvaluator(reviewer_models=["fake-reviewer"]).evaluate(
        SolverOutput(
            issue_number=174923,
            issue_body="expected RuntimeError",
            iteration=1,
            report_md="",
            fix_diff="",
            repro_script="print('unrelated')",
            repro_filename="repro_174923_generated.py",
        )
    )
    assert result.verdict == "needs_revision"
    assert result.score == 0.0


def test_component_scores_are_source_of_truth():
    class ComponentEvaluator(FakeEvaluator):
        def _call_llm(self, prompt: str, model_name: str | None = None) -> str:
            return json.dumps(
                {
                    "verdict": "approved",
                    "score": 1.0,
                    "component_scores": {
                        "fix_correctness": 0.7,
                        "scope_minimality": 0.9,
                        "test_coverage": 0.6,
                        "code_quality": 0.8,
                    },
                    "iteration": 1,
                    "repro_fidelity": "faithful",
                    "comments": [],
                    "summary": "Top-level score should be ignored.",
                }
            )

    result = ComponentEvaluator(
        reviewer_models=["fake-reviewer"],
        approval_threshold=0.8,
    ).evaluate(
        SolverOutput(
            issue_number=174923,
            issue_body="expected RuntimeError",
            iteration=1,
            report_md="",
            fix_diff="",
            repro_script="import torch",
            repro_filename="repro_174923_generated.py",
        )
    )
    assert result.verdict == "needs_revision"
    assert round(result.score, 2) == 0.6
    assert result.component_scores == {
        "fix_correctness": 0.7,
        "scope_minimality": 0.9,
        "test_coverage": 0.6,
        "code_quality": 0.8,
    }
    assert result.reviewer_results[0]["component_scores"] == result.component_scores
    assert round(result.reviewer_results[0]["score"], 2) == 0.6


def test_user_message_is_included_in_evaluator_prompt():
    prompts = []

    class PromptCapturingEvaluator(FakeEvaluator):
        def _call_llm(self, prompt: str, model_name: str | None = None) -> str:
            prompts.append(prompt)
            return super()._call_llm(prompt, model_name)

    PromptCapturingEvaluator(reviewer_models=["fake-reviewer"]).evaluate(
        SolverOutput(
            issue_number=174923,
            issue_body="expected RuntimeError",
            iteration=1,
            report_md="",
            fix_diff="",
            repro_script="import torch",
            user_message="just answer why CI is not catching these failures",
            repro_filename="repro_174923_generated.py",
        )
    )
    assert prompts
    assert "Human Task / Message Prior" in prompts[0]
    assert "just answer why CI is not catching these failures" in prompts[0]


def test_all_reviewers_must_approve():
    class SplitEvaluator(FakeEvaluator):
        def _call_llm(self, prompt: str, model_name: str | None = None) -> str:
            if model_name == "codex":
                score = 0.91
                verdict = "approved"
                summary = "Looks correct."
            else:
                score = 0.72
                verdict = "needs_revision"
                summary = "Missing edge-case coverage."
            return json.dumps(
                {
                    "verdict": verdict,
                    "score": score,
                    "iteration": 1,
                    "repro_fidelity": "faithful",
                    "comments": [
                        {
                            "file": "test",
                            "line": None,
                            "comment": summary,
                            "severity": "blocking",
                        }
                    ],
                    "summary": summary,
                }
            )

    result = SplitEvaluator(
        reviewer_models=["codex", "claude"],
        approval_threshold=0.8,
    ).evaluate(
        SolverOutput(
            issue_number=174923,
            issue_body="expected RuntimeError",
            iteration=1,
            report_md="",
            fix_diff="",
            repro_script="import torch",
            repro_filename="repro_174923_generated.py",
        )
    )
    assert result.verdict == "needs_revision"
    assert result.score == 0.72
    assert len(result.reviewer_results) == 2
    assert {comment.reviewer for comment in result.comments} == {"codex", "claude"}


def test_approved_only_when_every_reviewer_scores_high_enough():
    class ApprovingEvaluator(FakeEvaluator):
        def _call_llm(self, prompt: str, model_name: str | None = None) -> str:
            return json.dumps(
                {
                    "verdict": "approved",
                    "score": 0.88,
                    "iteration": 1,
                    "repro_fidelity": "faithful",
                    "comments": [],
                    "summary": f"{model_name} approves.",
                }
            )

    result = ApprovingEvaluator(
        reviewer_models=["codex", "claude"],
        approval_threshold=0.8,
    ).evaluate(
        SolverOutput(
            issue_number=174923,
            issue_body="expected RuntimeError",
            iteration=1,
            report_md="",
            fix_diff="",
            repro_script="import torch",
            repro_filename="repro_174923_generated.py",
        )
    )
    assert result.verdict == "approved"
    assert result.score == 0.88


def test_profile_backed_reviewer_uses_profile_prompt():
    calls = []

    class ProfileEvaluator(FakeEvaluator):
        def _call_llm(self, prompt: str, model_name: str | None = None) -> str:
            calls.append((model_name, prompt))
            return json.dumps(
                {
                    "verdict": "approved",
                    "score": 0.9,
                    "iteration": 1,
                    "repro_fidelity": "faithful",
                    "comments": [],
                    "summary": f"{model_name} approves.",
                }
            )

    result = ProfileEvaluator(
        reviewer_models=["gpt-5.5"],
        additional_reviewers=[
            ReviewerSpec(
                name="aditvenk-style",
                model="gpt-5.5",
                profile_text="Prioritize BC analysis and crisp inline comments.",
            )
        ],
    ).evaluate(
        SolverOutput(
            issue_number=174923,
            issue_body="expected RuntimeError",
            iteration=1,
            report_md="",
            fix_diff="",
            repro_script="import torch",
            repro_filename="repro_174923_generated.py",
        )
    )

    assert result.verdict == "approved"
    assert {item["reviewer"] for item in result.reviewer_results} == {
        "gpt-5.5",
        "aditvenk-style",
    }
    profile_prompts = [prompt for _model, prompt in calls if "aditvenk-style" in prompt]
    assert profile_prompts
    assert "Prioritize BC analysis" in profile_prompts[0]


def test_validate_configuration_accepts_claude_cli_fallback(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with patch("ptq.evaluator.evaluator.shutil.which", return_value="/usr/bin/claude"):
        Evaluator(reviewer_models=["claude-opus-4-7"]).validate_configuration()


def test_validate_configuration_fails_without_claude_api_or_cli(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with patch("ptq.evaluator.evaluator.shutil.which", return_value=None):
        try:
            Evaluator(reviewer_models=["claude-opus-4-7"]).validate_configuration()
        except RuntimeError as exc:
            assert "ANTHROPIC_API_KEY" in str(exc)
        else:
            raise AssertionError("Expected missing Claude credentials to fail")


def test_validate_configuration_accepts_codex_cli_fallback(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with patch("ptq.evaluator.evaluator.shutil.which", return_value="/usr/bin/codex"):
        Evaluator(reviewer_models=["gpt-5.5"]).validate_configuration()


def test_claude_cli_model_aliases():
    assert _claude_cli_model("claude-opus-4-7") == "opus"
    assert _claude_cli_model("claude-sonnet-4-5") == "sonnet"


def test_extract_claude_cli_json_result_text():
    assert _extract_claude_cli_text(
        json.dumps({"result": '{"verdict": "approved"}'})
    ) == '{"verdict": "approved"}'


def test_openai_network_failure_falls_back_to_codex_cli(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def fake_run(args, **kwargs):
        output_path = args[args.index("--output-last-message") + 1]
        with open(output_path, "w") as f:
            f.write('{"verdict": "approved"}')
        return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    with (
        patch("ptq.evaluator.evaluator.shutil.which", return_value="/usr/bin/codex"),
        patch(
            "ptq.evaluator.evaluator._http_json",
            side_effect=urllib.error.URLError("dns"),
        ),
        patch("ptq.evaluator.evaluator.subprocess.run", side_effect=fake_run),
    ):
        assert _call_openai("gpt-5.5", "prompt") == '{"verdict": "approved"}'
