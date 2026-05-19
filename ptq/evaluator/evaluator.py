from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from ptq.evaluator.models import ReviewComment, ReviewResult, ReviewerSpec
from ptq.evaluator.repro_validator import validate_repro_presence
from ptq.evaluator.rubric import build_evaluation_prompt

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


@dataclass
class SolverOutput:
    issue_number: int
    issue_body: str
    iteration: int
    report_md: str
    fix_diff: str
    repro_script: str
    repro_filename: str = ""
    status_json: dict = field(default_factory=dict)
    worktree_path: Path | None = None


@dataclass
class Evaluator:
    model: str = ""
    reviewer_models: list[str] | None = None
    additional_reviewers: list[ReviewerSpec] = field(default_factory=list)
    approval_threshold: float = 0.8
    shelve_threshold: float = 0.3
    max_iterations: int = 5

    def validate_configuration(self) -> None:
        missing: list[str] = []
        for reviewer in self._effective_reviewers():
            provider, model = _infer_provider_and_model(reviewer.model)
            if provider == "anthropic":
                if not os.environ.get("ANTHROPIC_API_KEY") and not shutil.which(
                    "claude"
                ):
                    missing.append(
                        f"{reviewer.name} requires ANTHROPIC_API_KEY or the "
                        "`claude` CLI on PATH"
                    )
            elif provider == "openai":
                if not os.environ.get("OPENAI_API_KEY") and not shutil.which("codex"):
                    missing.append(
                        f"{reviewer.name} requires OPENAI_API_KEY or the "
                        "`codex` CLI on PATH"
                    )
            else:
                missing.append(f"{reviewer.name} resolved to unsupported provider")
        if missing:
            raise RuntimeError(
                "Evaluator is not configured: " + "; ".join(sorted(set(missing)))
            )

    def evaluate(self, solver_output: SolverOutput) -> ReviewResult:
        repro_check = validate_repro_presence(
            issue_number=solver_output.issue_number,
            repro_filename=solver_output.repro_filename,
            repro_script=solver_output.repro_script,
            status_json=solver_output.status_json,
        )
        if repro_check.blocks_evaluation:
            return ReviewResult(
                verdict="needs_revision",
                score=0.0,
                iteration=solver_output.iteration,
                repro_fidelity="uncertain",
                comments=repro_check.comments,
                summary=(
                    "The evaluator could not find a repro script. Produce the "
                    "repro artifact before attempting a fix."
                ),
            )

        lint_output = self._run_lint(solver_output.worktree_path)
        prompt = build_evaluation_prompt(
            issue_number=solver_output.issue_number,
            issue_body=solver_output.issue_body,
            repro_filename=solver_output.repro_filename,
            repro_script=solver_output.repro_script,
            fix_diff=solver_output.fix_diff,
            report_md=solver_output.report_md,
            status_json=solver_output.status_json,
            iteration=solver_output.iteration,
            max_iterations=self.max_iterations,
            approval_threshold=self.approval_threshold,
            shelve_threshold=self.shelve_threshold,
            lint_output=lint_output,
        )
        reviewer_results = self._evaluate_all_reviewers(
            self._effective_reviewers(),
            prompt,
            iteration=solver_output.iteration,
        )
        return self._aggregate_reviewer_results(
            reviewer_results,
            repro_comments=repro_check.comments,
            iteration=solver_output.iteration,
        )

    def _evaluate_all_reviewers(
        self,
        reviewers: list[ReviewerSpec],
        prompt: str,
        *,
        iteration: int,
    ) -> list[ReviewResult]:
        if len(reviewers) <= 1:
            return [
                self._evaluate_with_reviewer(
                    reviewers[0],
                    prompt,
                    iteration=iteration,
                )
            ] if reviewers else []
        with ThreadPoolExecutor(max_workers=len(reviewers)) as executor:
            return list(
                executor.map(
                    lambda reviewer: self._evaluate_with_reviewer(
                        reviewer,
                        prompt,
                        iteration=iteration,
                    ),
                    reviewers,
                )
            )

    def _effective_reviewer_models(self) -> list[str]:
        if self.reviewer_models is not None:
            models = [
                model.strip() for model in self.reviewer_models if model.strip()
            ]
            if models:
                return models
        if self.model.strip():
            return [self.model.strip()]
        return ["gpt-5.5", "claude-opus-4-7"]

    def _effective_reviewers(self) -> list[ReviewerSpec]:
        reviewers = [
            ReviewerSpec.from_model(model)
            for model in self._effective_reviewer_models()
        ]
        for reviewer in self.additional_reviewers:
            try:
                reviewers.append(reviewer.with_loaded_profile())
            except OSError as exc:
                raise RuntimeError(
                    f"Evaluator profile for {reviewer.name} is not readable: "
                    f"{reviewer.profile_path}"
                ) from exc
        return reviewers

    def _evaluate_with_reviewer(
        self, reviewer: ReviewerSpec, prompt: str, *, iteration: int
    ) -> ReviewResult:
        raw = self._call_llm(_prompt_for_reviewer(prompt, reviewer), reviewer.model)
        result = ReviewResult.from_dict(_extract_json_object(raw), iteration=iteration)
        result.reviewer = reviewer.name
        for comment in result.comments:
            comment.reviewer = comment.reviewer or reviewer.name
        return self._apply_verdict_rules(result)

    def _aggregate_reviewer_results(
        self,
        reviewer_results: list[ReviewResult],
        *,
        repro_comments: list[ReviewComment],
        iteration: int,
    ) -> ReviewResult:
        comments = [
            *repro_comments,
            *[
                comment
                for reviewer_result in reviewer_results
                for comment in reviewer_result.comments
            ],
        ]
        if not reviewer_results:
            return ReviewResult(
                verdict="needs_revision",
                score=0.0,
                iteration=iteration,
                repro_fidelity="uncertain",
                comments=comments,
                summary="No evaluator reviewers were configured.",
            )

        min_score = min(result.score for result in reviewer_results)
        reviewer_data = [result.to_dict() for result in reviewer_results]
        repro_fidelity = _aggregate_repro_fidelity(reviewer_results)
        if any(result.repro_fidelity == "unfaithful" for result in reviewer_results):
            verdict = "needs_revision"
            score = 0.0
        elif all(
            result.verdict == "approved" and result.score >= self.approval_threshold
            for result in reviewer_results
        ):
            verdict = "approved"
            score = min_score
        elif (
            iteration >= self.max_iterations
            and min_score < self.shelve_threshold
            and all(result.score < self.approval_threshold for result in reviewer_results)
        ):
            verdict = "shelve"
            score = min_score
        else:
            verdict = "needs_revision"
            score = min_score

        summary_lines = [
            f"{result.reviewer}: {result.verdict} "
            f"score={result.score:.2f}. {result.summary}".strip()
            for result in reviewer_results
        ]
        if verdict != "approved":
            summary = (
                "The diff is not ready for human review because not every "
                f"reviewer reached {self.approval_threshold:.2f}. "
                + " ".join(summary_lines)
            )
        else:
            summary = (
                "All evaluator reviewers approved the diff. "
                + " ".join(summary_lines)
            )
        return ReviewResult(
            verdict=verdict,  # type: ignore[arg-type]
            score=score,
            iteration=iteration,
            repro_fidelity=repro_fidelity,  # type: ignore[arg-type]
            comments=comments,
            summary=summary,
            reviewer="aggregate",
            reviewer_results=reviewer_data,
        )

    def _apply_verdict_rules(self, result: ReviewResult) -> ReviewResult:
        if result.repro_fidelity == "unfaithful":
            result.score = 0.0
            result.verdict = "needs_revision"
            return result
        if result.score >= self.approval_threshold:
            result.verdict = "approved"
        elif (
            result.iteration >= self.max_iterations
            and result.score < self.shelve_threshold
        ):
            result.verdict = "shelve"
        else:
            result.verdict = "needs_revision"
        return result

    def _run_lint(self, worktree_path: Path | None) -> str:
        if worktree_path is None:
            return "Not run: no local worktree path was available to the evaluator."
        if not worktree_path.exists():
            return f"Not run: worktree path does not exist locally: {worktree_path}"
        try:
            result = subprocess.run(
                ["lintrunner", "-m", "origin/main"],
                cwd=worktree_path,
                capture_output=True,
                text=True,
                timeout=180,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            return f"Not run: {exc}"
        output = (result.stdout or "") + (result.stderr or "")
        return f"exit_code={result.returncode}\n{output.strip()}"

    def _call_llm(self, prompt: str, model_name: str | None = None) -> str:
        provider, model = _infer_provider_and_model(model_name or self.model)
        if provider == "anthropic":
            return _call_anthropic(model, prompt)
        if provider == "openai":
            return _call_openai(model, prompt)
        raise RuntimeError(f"Unsupported evaluator model/provider: {self.model}")


def _prompt_for_reviewer(prompt: str, reviewer: ReviewerSpec) -> str:
    if not reviewer.profile_text:
        return prompt
    return f"""\
{prompt}

## Additional Reviewer Profile: {reviewer.name}

You are the `{reviewer.name}` evaluator. Apply the same required JSON schema and
the same scoring rules above. Use the profile below to shape the review focus,
comment phrasing, and approval strictness, while grounding every comment in the
actual issue, repro, report, diff, and lint evidence. Do not claim to be this
person; this is a style and review-priority profile for an automated evaluator.

```markdown
{reviewer.profile_text}
```
"""


def _infer_provider_and_model(model: str) -> tuple[str, str]:
    normalized = model.strip()
    lower = normalized.lower()
    if lower in {"claude", "anthropic"}:
        return "anthropic", os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")
    if lower.startswith("claude"):
        return "anthropic", normalized
    if lower in {"openai", "gpt"}:
        return "openai", os.environ.get("OPENAI_MODEL", "gpt-5.1")
    if lower.startswith("gpt") or lower.startswith("o"):
        return "openai", normalized
    return "anthropic", normalized


def _aggregate_repro_fidelity(results: list[ReviewResult]) -> str:
    values = {result.repro_fidelity for result in results}
    if "unfaithful" in values:
        return "unfaithful"
    if values == {"from_issue"}:
        return "from_issue"
    if values <= {"faithful", "from_issue"}:
        return "faithful"
    return "uncertain"


def _extract_json_object(text: str) -> dict:
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    match = _JSON_BLOCK_RE.search(stripped)
    if match:
        return json.loads(match.group(1))

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(stripped[start : end + 1])
    raise ValueError("Evaluator LLM response did not contain a JSON object.")


def _http_json(url: str, payload: dict, headers: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Evaluator LLM request failed: {exc.code} {body}") from exc


def _call_anthropic(model: str, prompt: str) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        if shutil.which("claude"):
            return _call_claude_cli(model, prompt)
        raise RuntimeError(
            "Claude evaluator requires ANTHROPIC_API_KEY or the `claude` CLI on PATH."
        )
    response = _http_json(
        "https://api.anthropic.com/v1/messages",
        {
            "model": model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
        },
        {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
    )
    parts = response.get("content", [])
    return "\n".join(
        part.get("text", "") for part in parts if isinstance(part, dict)
    ).strip()


def _call_claude_cli(model: str, prompt: str) -> str:
    cli_model = _claude_cli_model(model)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", prefix="ptq-evaluator-", delete=False
    ) as f:
        f.write(prompt)
        prompt_path = f.name
    try:
        result = subprocess.run(
            [
                "claude",
                "-p",
                "Return only the ReviewResult JSON object for the evaluator "
                "instructions in the appended system prompt.",
                "--model",
                cli_model,
                "--max-turns",
                "1",
                "--output-format",
                "json",
                "--append-system-prompt-file",
                prompt_path,
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
    finally:
        try:
            os.unlink(prompt_path)
        except OSError:
            pass
    if result.returncode != 0:
        output = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"Claude CLI evaluator failed: {output}")
    return _extract_claude_cli_text(result.stdout)


def _claude_cli_model(model: str) -> str:
    lower = model.lower()
    if "opus" in lower:
        return "opus"
    if "sonnet" in lower:
        return "sonnet"
    if "haiku" in lower:
        return "haiku"
    return model


def _extract_claude_cli_text(stdout: str) -> str:
    stripped = stdout.strip()
    if not stripped:
        raise RuntimeError("Claude CLI evaluator returned no output.")
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        return stripped
    if isinstance(data, dict):
        for key in ("result", "text", "content"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, list):
                parts = [
                    item.get("text", "")
                    for item in value
                    if isinstance(item, dict) and item.get("text")
                ]
                if parts:
                    return "\n".join(parts).strip()
        message = data.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                parts = [
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and item.get("text")
                ]
                if parts:
                    return "\n".join(parts).strip()
    return stripped


def _call_openai(model: str, prompt: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        if shutil.which("codex"):
            return _call_codex_cli(model, prompt)
        raise RuntimeError(
            "OpenAI evaluator requires OPENAI_API_KEY or the `codex` CLI on PATH."
        )
    try:
        response = _http_json(
            "https://api.openai.com/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
            },
            {"Authorization": f"Bearer {api_key}"},
        )
    except urllib.error.URLError as exc:
        if shutil.which("codex"):
            return _call_codex_cli(model, prompt)
        raise RuntimeError(f"OpenAI evaluator request failed: {exc}") from exc
    choices = response.get("choices", [])
    if not choices:
        raise RuntimeError("OpenAI evaluator returned no choices.")
    message = choices[0].get("message", {})
    return str(message.get("content") or "").strip()


def _call_codex_cli(model: str, prompt: str) -> str:
    with tempfile.TemporaryDirectory(prefix="ptq-codex-evaluator-") as tmpdir:
        output_path = Path(tmpdir) / "review.txt"
        result = subprocess.run(
            [
                "codex",
                "exec",
                "--model",
                model,
                "--sandbox",
                "read-only",
                "--cd",
                tmpdir,
                "--skip-git-repo-check",
                "--ignore-rules",
                "--output-last-message",
                str(output_path),
                "-",
            ],
            input=(
                "Return only the ReviewResult JSON object requested by the "
                "following evaluator instructions. Do not include markdown.\n\n"
                f"{prompt}"
            ),
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            output = (result.stderr or result.stdout).strip()
            raise RuntimeError(f"Codex CLI evaluator failed: {output}")
        if output_path.exists():
            text = output_path.read_text().strip()
            if text:
                return text
        return result.stdout.strip()
