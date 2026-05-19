from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ptq.evaluator.models import ReviewResult


@dataclass
class Issue:
    number: int
    title: str
    body: str = ""
    labels: list[str] = field(default_factory=list)
    url: str = ""
    raw: dict = field(default_factory=dict)


@dataclass
class OrchestratorConfig:
    issue_selection_prompt: str
    repo: str = "pytorch"
    github_repo: str = "pytorch/pytorch"
    max_issues: int = 20
    parallel: int = 4
    max_iterations: int = 5
    approval_threshold: float = 0.8
    machine: str = "localhost"
    dry_run: bool = False
    solver_agent: str = "claude"
    solver_model: str = "opus"
    solver_thinking: str | None = None
    solver_max_turns: int = 100
    initial_message: str | None = None
    push_pr: bool = False
    watch_pr: bool = False
    watch_pr_interval_seconds: float = 300.0
    watch_pr_idle_seconds: float = 86400.0
    log_path: Path = Path.home() / ".ptq" / "orchestrator" / "runs.jsonl"

    @property
    def local(self) -> bool:
        return self.machine in {"", "local", "localhost", "127.0.0.1"}


@dataclass
class SolveResult:
    issue: Issue
    verdict: str
    score: float
    iterations: int
    job_id: str | None = None
    branch: str | None = None
    pr_url: str | None = None
    review: ReviewResult | None = None
    state: str = "unknown"
    error: str = ""
    completed_at: str = field(default_factory=lambda: datetime.now().isoformat())
