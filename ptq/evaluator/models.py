from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ReviewVerdict = Literal["approved", "needs_revision", "shelve"]
ReviewSeverity = Literal["blocking", "suggestion", "nit"]
ReproFidelity = Literal["faithful", "unfaithful", "uncertain", "from_issue"]


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

        raw_score = data.get("score", 0.0)
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            score = 0.0
        score = max(0.0, min(1.0, score))

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
            comments=comments,
            summary=str(data.get("summary") or ""),
            reviewer=str(data.get("reviewer") or ""),
        )
