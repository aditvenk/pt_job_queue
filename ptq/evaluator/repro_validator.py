from __future__ import annotations

from dataclasses import dataclass

from ptq.evaluator.models import ReviewComment


@dataclass(frozen=True)
class ReproCheck:
    source: str
    fidelity_hint: str
    comments: list[ReviewComment]
    blocks_evaluation: bool = False


def infer_repro_source(
    *,
    issue_number: int,
    repro_filename: str,
    status_json: dict | None = None,
) -> str:
    status_json = status_json or {}
    source = str(status_json.get("repro_source") or "").strip().lower()
    if source in {"generated", "from_issue", "issue", "extracted"}:
        return "from_issue" if source in {"from_issue", "issue", "extracted"} else source

    if repro_filename == f"repro_{issue_number}.py":
        return "from_issue"
    if repro_filename == f"repro_{issue_number}_generated.py":
        return "generated"
    if repro_filename.endswith("_generated.py"):
        return "generated"
    return "unknown"


def validate_repro_presence(
    *,
    issue_number: int,
    repro_filename: str,
    repro_script: str,
    status_json: dict | None = None,
) -> ReproCheck:
    source = infer_repro_source(
        issue_number=issue_number,
        repro_filename=repro_filename,
        status_json=status_json,
    )
    comments: list[ReviewComment] = []
    blocks = False

    if not repro_script.strip():
        blocks = True
        comments.append(
            ReviewComment(
                file="repro",
                line=None,
                comment=(
                    "No repro script was provided. Write either "
                    f"`repro_{issue_number}.py` from the issue or "
                    f"`repro_{issue_number}_generated.py` if you generated one."
                ),
                severity="blocking",
            )
        )

    if source == "unknown" and repro_script.strip():
        comments.append(
            ReviewComment(
                file="repro",
                line=None,
                comment=(
                    "Repro source is unclear from the filename/status.json. Use "
                    f"`repro_{issue_number}.py` for issue-provided repros or "
                    f"`repro_{issue_number}_generated.py` for agent-generated repros."
                ),
                severity="suggestion",
            )
        )

    fidelity_hint = "from_issue" if source == "from_issue" else "uncertain"
    return ReproCheck(
        source=source,
        fidelity_hint=fidelity_hint,
        comments=comments,
        blocks_evaluation=blocks,
    )
