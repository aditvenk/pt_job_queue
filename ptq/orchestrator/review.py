from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from subprocess import CompletedProcess

from ptq.application.pr_service import _run_github
from ptq.domain.models import PtqError
from ptq.evaluator import Evaluator, PullRequestReviewInput
from ptq.evaluator.models import COMPONENT_SCORE_KEYS, ReviewComment, ReviewResult
from ptq.ssh import LocalBackend

_PR_URL_RE = re.compile(
    r"^https?://github\.com/([^/\s]+)/([^/\s]+)/pull/(\d+)(?:[/?#].*)?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PullRequestRef:
    owner: str
    repo: str
    number: int
    url: str

    @property
    def github_repo(self) -> str:
        return f"{self.owner}/{self.repo}"


@dataclass
class PullRequestReviewReport:
    pr: PullRequestRef
    review: ReviewResult
    report_path: Path


@dataclass
class DiffHunk:
    file: str
    new_start: int
    new_count: int
    text: str

    def contains(self, line: int) -> bool:
        if self.new_count <= 0:
            return line == self.new_start
        return self.new_start <= line < self.new_start + self.new_count


def parse_pull_request_url(url: str) -> PullRequestRef:
    match = _PR_URL_RE.match(url.strip())
    if not match:
        raise PtqError(
            "--review must be a GitHub pull request or review URL like "
            "https://github.com/pytorch/pytorch/pull/184746"
        )
    owner, repo, number = match.groups()
    return PullRequestRef(
        owner=owner,
        repo=repo,
        number=int(number),
        url=f"https://github.com/{owner}/{repo}/pull/{number}",
    )


def run_pull_request_review(
    review_url: str,
    evaluator: Evaluator,
    *,
    output_dir: Path | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> PullRequestReviewReport:
    progress = on_progress or (lambda _msg: None)
    pr = parse_pull_request_url(review_url)
    backend = LocalBackend()

    progress(f"Fetching PR metadata for {pr.github_repo}#{pr.number}")
    metadata = _fetch_pr_metadata(backend, pr)
    progress("Fetching PR diff")
    diff = _fetch_pr_diff(backend, pr)
    if not diff.strip():
        raise PtqError(f"Could not fetch a non-empty diff for {pr.url}")

    review_input = PullRequestReviewInput(
        pr_url=pr.url,
        title=str(metadata.get("title") or ""),
        body=str(metadata.get("body") or ""),
        author=_author_login(metadata.get("author")),
        base_ref=str(metadata.get("baseRefName") or ""),
        head_ref=str(metadata.get("headRefName") or ""),
        files=_files_metadata(metadata.get("files")),
        diff=diff,
    )

    progress("Running evaluator panel")
    review = evaluator.evaluate_pull_request(review_input)
    report_dir = output_dir or _default_report_dir(pr)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "report.md"
    review_path = report_dir / "review.json"

    hunks = parse_diff_hunks(diff)
    report_path.write_text(build_pr_review_report(pr, metadata, review, hunks))
    review_path.write_text(json.dumps(review.to_dict(), indent=2))
    progress(f"Wrote review report to {report_path}")
    return PullRequestReviewReport(pr=pr, review=review, report_path=report_path)


def _fetch_pr_metadata(backend: LocalBackend, pr: PullRequestRef) -> dict:
    result = _run_github(
        backend,
        [
            "pr",
            "view",
            pr.url,
            "--repo",
            pr.github_repo,
            "--json",
            ",".join(
                [
                    "number",
                    "title",
                    "body",
                    "author",
                    "url",
                    "baseRefName",
                    "headRefName",
                    "files",
                    "additions",
                    "deletions",
                    "changedFiles",
                    "isDraft",
                    "reviewDecision",
                    "state",
                ]
            ),
        ],
        check=False,
    )
    if result.returncode != 0:
        raise PtqError(_github_error("fetch PR metadata", result))
    try:
        data = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise PtqError("gh pr view returned invalid JSON.") from exc
    return data if isinstance(data, dict) else {}


def _fetch_pr_diff(backend: LocalBackend, pr: PullRequestRef) -> str:
    result = _run_github(
        backend,
        ["pr", "diff", pr.url, "--repo", pr.github_repo, "--patch"],
        check=False,
    )
    if result.returncode != 0:
        raise PtqError(_github_error("fetch PR diff", result))
    return result.stdout


def _github_error(action: str, result: CompletedProcess[str]) -> str:
    detail = (result.stderr or result.stdout or "").strip()
    return f"Failed to {action} with gh: {detail or result.returncode}"


def _author_login(author: object) -> str:
    if isinstance(author, dict):
        return str(author.get("login") or author.get("name") or "")
    return ""


def _files_metadata(files: object) -> list[dict]:
    if not isinstance(files, list):
        return []
    return [item for item in files if isinstance(item, dict)]


def _default_report_dir(pr: PullRequestRef) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    repo = pr.github_repo.replace("/", "-")
    return Path.home() / ".ptq" / "reviews" / f"{repo}-{pr.number}-{stamp}"


def parse_diff_hunks(diff: str) -> list[DiffHunk]:
    hunks: list[DiffHunk] = []
    current_file = ""
    current_lines: list[str] = []
    new_start = 0
    new_count = 0

    def flush() -> None:
        nonlocal current_lines, new_start, new_count
        if current_file and current_lines:
            hunks.append(
                DiffHunk(
                    file=current_file,
                    new_start=new_start,
                    new_count=new_count,
                    text="\n".join(current_lines),
                )
            )
        current_lines = []
        new_start = 0
        new_count = 0

    for line in diff.splitlines():
        if line.startswith("diff --git "):
            flush()
            current_file = _file_from_diff_header(line)
            continue
        if line.startswith("+++ b/"):
            current_file = line.removeprefix("+++ b/")
            continue
        if line.startswith("@@ "):
            flush()
            new_start, new_count = _parse_hunk_range(line)
            current_lines = [line]
            continue
        if current_lines:
            current_lines.append(line)
    flush()
    return hunks


def _file_from_diff_header(line: str) -> str:
    parts = line.split()
    if len(parts) >= 4 and parts[3].startswith("b/"):
        return parts[3][2:]
    return ""


def _parse_hunk_range(header: str) -> tuple[int, int]:
    match = re.search(r"\+(\d+)(?:,(\d+))?", header)
    if not match:
        return 0, 0
    return int(match.group(1)), int(match.group(2) or "1")


def build_pr_review_report(
    pr: PullRequestRef,
    metadata: dict,
    review: ReviewResult,
    hunks: list[DiffHunk],
) -> str:
    lines = [
        "# PR Evaluator Report",
        "",
        f"- PR: {pr.url}",
        f"- Title: {metadata.get('title') or ''}",
        f"- Verdict: {review.verdict}",
        f"- Score: {review.score:.2f}",
    ]
    components = _format_component_scores(review.component_scores)
    if components:
        lines.append(f"- Component lows: {components}")
    _append_consolidated_summary(lines, metadata, review)

    lines.extend(["", "## Reviewer Results", ""])
    if review.reviewer_results:
        for reviewer in review.reviewer_results:
            if not isinstance(reviewer, dict):
                continue
            name = str(reviewer.get("reviewer") or "reviewer")
            verdict = str(reviewer.get("verdict") or "")
            score = _float_text(reviewer.get("score"))
            comps = _format_component_scores(reviewer.get("component_scores"))
            summary = " ".join(str(reviewer.get("summary") or "").split())
            lines.extend([f"### {name}", "", f"- Verdict: `{verdict}`"])
            lines.append(f"- Score: `{score}`")
            if comps:
                lines.append(f"- Scores: {comps}")
            if summary:
                lines.extend(["", _wrap_summary(summary), ""])
    else:
        lines.append("- No per-reviewer results were returned.")

    actionable = _actionable_comments(review.comments)
    lines.extend(["", "## Actionable Feedback", ""])
    if not actionable:
        lines.append("No blocking or suggestion comments.")
    for index, comment in enumerate(actionable, 1):
        location = _comment_location(comment)
        reviewer = f"{comment.reviewer}: " if comment.reviewer else ""
        lines.extend(
            [
                f"### {index}. {reviewer}{location}",
                "",
                f"- Severity: `{comment.severity}`",
                "",
                comment.comment.strip() or "(empty comment)",
                "",
                "#### Code Snapshot",
                "",
                "```diff",
                _snapshot_for_comment(comment, hunks),
                "```",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _append_consolidated_summary(
    lines: list[str], metadata: dict, review: ReviewResult
) -> None:
    lines.extend(["", "## Consolidated Summary", ""])
    lines.extend(_intro_lines(metadata, review))
    reviewer_results = [
        result for result in review.reviewer_results if isinstance(result, dict)
    ]
    if not reviewer_results:
        lines.append(review.summary or "(none)")
        return

    total = len(reviewer_results)
    needs_revision = [
        result
        for result in reviewer_results
        if str(result.get("verdict") or "") != "approved"
    ]
    if needs_revision:
        lines.append(
            f"- **Readiness**: Not ready; {len(needs_revision)}/{total} "
            "reviewers requested changes."
        )
    else:
        lines.append(f"- **Readiness**: Ready; all {total} reviewers approved.")

    components = _format_component_scores(review.component_scores)
    if components:
        lines.append(f"- **Lowest component scores**: {components}")

    if needs_revision:
        lines.append("- **Reviewers requesting changes**:")
        for result in needs_revision:
            name = str(result.get("reviewer") or "reviewer")
            score = _float_text(result.get("score"))
            summary = _first_sentences(str(result.get("summary") or ""), limit=1)
            suffix = f" - {summary}" if summary else ""
            lines.append(f"  - `{name}` score={score}{suffix}")

    blockers = _actionable_comments(review.comments)
    if blockers:
        lines.append("- **Top actionable items**:")
        for comment in blockers[:5]:
            reviewer = f"{comment.reviewer}: " if comment.reviewer else ""
            text = _first_sentences(comment.comment, limit=1)
            lines.append(f"  - {reviewer}{_comment_location(comment)} - {text}")


def _intro_lines(metadata: dict, review: ReviewResult) -> list[str]:
    lines = ["### Intro", ""]
    background = _infer_background(metadata, review)
    problem = _infer_problem_detail(metadata)
    fix = _infer_fix_detail(metadata, review)
    if background:
        lines.extend(["**Background**", "", background, ""])
    if problem:
        lines.extend(["**Problem**", "", problem, ""])
    if fix:
        lines.extend(["**Fix Summary**", "", fix, ""])
    if len(lines) == 2:
        lines.extend(["No PR background was available from the metadata.", ""])
    return lines


def _infer_background(metadata: dict, review: ReviewResult) -> str:
    text = _context_text(metadata, review).lower()
    paths = _changed_file_paths(metadata)
    parts: list[str] = []
    domain = _infer_domain(metadata)
    if domain:
        parts.append(domain)
    if "aotautograd" in text or "aot_autograd" in text or "aot eager" in text:
        parts.append(
            "AOTAutograd is part of PyTorch's compilation path: it traces a "
            "program ahead of time and must record enough metadata for compiled "
            "outputs to behave like eager outputs."
        )
    if "dtensor" in text:
        parts.append(
            "DTensor is a tensor wrapper used for distributed tensors; in addition "
            "to local tensor data, it carries placement and sharding metadata that "
            "must survive tracing and view operations."
        )
    if "wrapper subclass" in text or "traceable wrapper" in text:
        parts.append(
            "Traceable wrapper subclasses wrap one or more inner tensors. Compiler "
            "metadata has to describe both the outer wrapper and the inner tensor "
            "views so replaying a view does not drop subclass-specific state."
        )
    if "alias" in text or "view" in text or "as_strided" in text:
        parts.append(
            "View and alias outputs are especially sensitive because they are not "
            "independent tensors: the compiled graph needs to preserve the original "
            "relationship to the base tensor instead of reconstructing an invalid "
            "or unsupported view."
        )
    if not parts and paths:
        parts.append(
            "This PR changes implementation and test files in the areas listed "
            "below; the review should be read in that codebase context."
        )
    return " ".join(parts)


def _infer_domain(metadata: dict) -> str:
    paths = _changed_file_paths(metadata)
    if not paths:
        return ""
    roots = []
    for path in paths:
        parts = path.split("/")
        if len(parts) >= 2 and parts[0] in {"torch", "test", "c10", "aten"}:
            roots.append("/".join(parts[:2]))
        elif parts:
            roots.append(parts[0])
    unique = list(dict.fromkeys(roots))
    if not unique:
        return ""
    shown = ", ".join(f"`{root}`" for root in unique[:5])
    if len(unique) > 5:
        shown += f", and {len(unique) - 5} more area(s)"
    return f"The PR changes code in {shown}."


def _changed_file_paths(metadata: dict) -> list[str]:
    files = metadata.get("files")
    if not isinstance(files, list):
        return []
    paths = []
    for item in files:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "").strip()
        if path:
            paths.append(path)
    return paths


def _infer_problem_detail(metadata: dict) -> str:
    body = str(metadata.get("body") or "")
    title = str(metadata.get("title") or "").strip()
    section = _extract_markdown_section(
        body,
        ["Problem", "Motivation", "Summary", "Description"],
    )
    problem = _sentences(section or body, limit=4, max_chars=900)
    title = _strip_issue_footer(title)
    if title and problem:
        return (
            f"The PR is titled \"{title}\". The problem described by the PR is: "
            f"{problem}"
        )
    if title:
        return f"The PR is titled \"{title}\"."
    if problem:
        return f"The problem described by the PR is: {problem}"
    return ""


def _infer_fix_detail(metadata: dict, review: ReviewResult) -> str:
    body = str(metadata.get("body") or "")
    section = _extract_markdown_section(
        body,
        ["Fix", "Solution", "Summary", "Test Plan"],
    )
    pieces: list[str] = []
    if section:
        pieces.append(_sentences(section, limit=4, max_chars=900))
    summaries = []
    for result in review.reviewer_results:
        if not isinstance(result, dict):
            continue
        summary = str(result.get("summary") or "")
        if summary:
            summaries.append(summary)
    if summaries:
        pieces.append(_best_fix_sentences(summaries))
    elif review.summary:
        pieces.append(_sentences(review.summary, limit=3, max_chars=700))
    test_paths = [
        path for path in _changed_file_paths(metadata) if path.startswith("test/")
    ]
    if test_paths:
        shown = ", ".join(f"`{path}`" for path in test_paths[:4])
        if len(test_paths) > 4:
            shown += f", and {len(test_paths) - 4} more test file(s)"
        pieces.append(
            f"The PR also changes tests in {shown}, which is the evidence the "
            "evaluators use when judging whether the new behavior is covered."
        )
    return " ".join(piece for piece in pieces if piece).strip()


def _best_fix_sentences(summaries: list[str]) -> str:
    selected: list[str] = []
    keywords = (
        "fix",
        "adds",
        "routes",
        "extends",
        "preserves",
        "replay",
        "implements",
        "changes",
    )
    for summary in summaries:
        for sentence in re.split(r"(?<=[.!?])\s+", " ".join(summary.split())):
            if not sentence:
                continue
            lowered = sentence.lower()
            if any(keyword in lowered for keyword in keywords):
                selected.append(sentence)
                break
        if len(selected) >= 3:
            break
    if not selected:
        selected = [
            _sentences(summary, limit=1, max_chars=260)
            for summary in summaries[:3]
            if summary.strip()
        ]
    return _truncate_text(" ".join(selected), 1000)


def _context_text(metadata: dict, review: ReviewResult) -> str:
    chunks = [
        str(metadata.get("title") or ""),
        str(metadata.get("body") or ""),
        " ".join(_changed_file_paths(metadata)),
        review.summary,
    ]
    for result in review.reviewer_results:
        if isinstance(result, dict):
            chunks.append(str(result.get("summary") or ""))
    return "\n".join(chunks)


def _strip_issue_footer(text: str) -> str:
    text = re.sub(r"\bFixe?s?\s+#\d+\b\.?", "", text, flags=re.IGNORECASE)
    return " ".join(text.split()).strip(" .")


def _extract_markdown_section(text: str, names: list[str]) -> str:
    if not text.strip():
        return ""
    escaped = "|".join(re.escape(name) for name in names)
    pattern = re.compile(
        rf"^#+\s*(?:{escaped})\s*$\n(?P<body>.*?)(?=^#+\s+|\Z)",
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(text)
    if match:
        return match.group("body").strip()
    return ""


def _actionable_comments(comments: list[ReviewComment]) -> list[ReviewComment]:
    return [
        comment
        for comment in comments
        if comment.severity in {"blocking", "suggestion"}
    ]


def _comment_location(comment: ReviewComment) -> str:
    if comment.file in {"", "general", "repro", "test"}:
        return comment.file or "general"
    if comment.line is not None:
        return f"{comment.file}:{comment.line}"
    return comment.file


def _snapshot_for_comment(comment: ReviewComment, hunks: list[DiffHunk]) -> str:
    file = _normalize_comment_file(comment.file)
    if not file:
        return "No file-specific diff hunk is available for this comment."
    candidates = [hunk for hunk in hunks if hunk.file == file]
    if not candidates:
        return f"No diff hunk found for `{file}`."
    if comment.line is not None:
        for hunk in candidates:
            if hunk.contains(comment.line):
                return _truncate_snapshot(hunk.text)
    return _truncate_snapshot(candidates[0].text)


def _normalize_comment_file(file: str) -> str:
    file = file.strip()
    if file in {"", "general", "repro", "test"}:
        return ""
    if file.startswith("a/") or file.startswith("b/"):
        return file[2:]
    return file


def _truncate_snapshot(text: str, *, max_lines: int = 80) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    head = lines[: max_lines - 1]
    head.append(f"... truncated {len(lines) - len(head)} line(s) ...")
    return "\n".join(head)


def _format_component_scores(component_scores: object) -> str:
    if not isinstance(component_scores, dict):
        return ""
    parts = []
    for key in COMPONENT_SCORE_KEYS:
        value = component_scores.get(key)
        try:
            parts.append(f"{key}={float(value):.2f}")
        except (TypeError, ValueError):
            continue
    return ", ".join(parts)


def _float_text(value: object) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "0.00"


def _first_sentences(text: str, *, limit: int) -> str:
    return _sentences(text, limit=limit, max_chars=260)


def _sentences(text: str, *, limit: int, max_chars: int) -> str:
    compact = " ".join(text.split())
    if not compact:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", compact)
    selected = " ".join(sentence for sentence in sentences[:limit] if sentence).strip()
    return _truncate_text(selected, max_chars)


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) > max_chars:
        return text[: max_chars - 3].rstrip() + "..."
    return text


def _wrap_summary(text: str) -> str:
    compact = " ".join(text.split())
    return compact or "(empty summary)"
