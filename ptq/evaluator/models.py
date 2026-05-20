from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ReviewVerdict = Literal["approved", "needs_revision", "shelve"]
ReviewSeverity = Literal["blocking", "suggestion", "nit"]
ReproFidelity = Literal["faithful", "unfaithful", "uncertain", "from_issue"]

COMPONENT_SCORE_KEYS = (
    "fix_correctness",
    "scope_minimality",
    "test_coverage",
    "code_quality",
)


@dataclass
class ReviewComment:
    file: str
    line: int | None
    comment: str
    severity: ReviewSeverity
    reviewer: str = ""

    def to_dict(self) -> dict:
        return {
            "file": self.file,
            "line": self.line,
            "comment": self.comment,
            "severity": self.severity,
            "reviewer": self.reviewer,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ReviewComment:
        severity = str(data.get("severity", "blocking"))
        if severity not in {"blocking", "suggestion", "nit"}:
            severity = "blocking"
        line = data.get("line")
        return cls(
            file=str(data.get("file") or "general"),
            line=line if isinstance(line, int) else None,
            comment=str(data.get("comment") or ""),
            severity=severity,  # type: ignore[arg-type]
            reviewer=str(data.get("reviewer") or ""),
        )


@dataclass
class ReviewResult:
    verdict: ReviewVerdict
    score: float
    iteration: int
    repro_fidelity: ReproFidelity
    component_scores: dict[str, float] = field(default_factory=dict)
    comments: list[ReviewComment] = field(default_factory=list)
    summary: str = ""
    reviewer: str = ""
    reviewer_results: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "score": self.score,
            "iteration": self.iteration,
            "repro_fidelity": self.repro_fidelity,
            "component_scores": self.component_scores,
            "comments": [comment.to_dict() for comment in self.comments],
            "summary": self.summary,
            "reviewer": self.reviewer,
            "reviewer_results": self.reviewer_results,
        }

    @classmethod
    def from_dict(cls, data: dict, *, iteration: int) -> ReviewResult:
        verdict = str(data.get("verdict", "needs_revision"))
        if verdict not in {"approved", "needs_revision", "shelve"}:
            verdict = "needs_revision"

        repro_fidelity = str(data.get("repro_fidelity", "uncertain"))
        if repro_fidelity not in {"faithful", "unfaithful", "uncertain", "from_issue"}:
            repro_fidelity = "uncertain"

        component_scores = _parse_component_scores(data)
        raw_score = data.get("score", 0.0)
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            score = 0.0
        score = max(0.0, min(1.0, score))
        if component_scores:
            score = _weakest_component_score(component_scores)

        comments_data = data.get("comments", [])
        comments = [
            ReviewComment.from_dict(comment)
            for comment in comments_data
            if isinstance(comment, dict)
        ]
        return cls(
            verdict=verdict,  # type: ignore[arg-type]
            score=score,
            iteration=int(data.get("iteration") or iteration),
            repro_fidelity=repro_fidelity,  # type: ignore[arg-type]
            component_scores=component_scores,
            comments=comments,
            summary=str(data.get("summary") or ""),
            reviewer=str(data.get("reviewer") or ""),
        )


@dataclass(frozen=True)
class ReviewerSpec:
    name: str
    model: str
    profile_path: str = ""
    profile_text: str = ""

    @classmethod
    def from_model(cls, model: str) -> "ReviewerSpec":
        model = model.strip()
        return cls(name=model, model=model)

    def with_loaded_profile(self) -> "ReviewerSpec":
        if self.profile_text or not self.profile_path:
            return self
        from pathlib import Path

        path = Path(self.profile_path).expanduser()
        return ReviewerSpec(
            name=self.name,
            model=self.model,
            profile_path=str(path),
            profile_text=path.read_text(),
        )


def _parse_component_scores(data: dict) -> dict[str, float]:
    raw = data.get("component_scores")
    if not isinstance(raw, dict):
        raw = {}
    parsed: dict[str, float] = {}
    for key in COMPONENT_SCORE_KEYS:
        value = raw.get(key, data.get(f"{key}_score"))
        if value is None:
            continue
        try:
            parsed[key] = max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            continue
    if len(parsed) != len(COMPONENT_SCORE_KEYS):
        return {}
    return parsed


def _weakest_component_score(component_scores: dict[str, float]) -> float:
    return min(component_scores[key] for key in COMPONENT_SCORE_KEYS)
