from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import time
from hashlib import sha1
from collections.abc import Callable
from pathlib import Path

from ptq.domain.models import PRResult, PtqError
from ptq.github_harness import call_github_harness, harness_available
from ptq.infrastructure.backends import backend_for_job
from ptq.infrastructure.job_repository import JobRepository
from ptq.repo_profiles import get_profile
from ptq.ssh import Backend, LocalBackend, RemoteBackend

_HTTPS_TO_SSH = {
    "https://github.com/": "git@github.com:",
}

_PR_STATE_TTL_SECONDS = 45.0
_REPRO_COMMENT_MARKER = "<!-- ptq:repro-script -->"
_BOT_COMMENT_PREFIX = "[bot]"
_RESOLUTION_MARKER_PREFIX = "<!-- ptq:resolution:"
_CI_LOG_EXCERPT_LIMIT = 6000
_CI_LOG_RUN_LIMIT = 3
_BOT_REPLY_RESOLUTION_LIMIT = 220
_DEFAULT_GITHUB_ENV = {
    "https_proxy": "http://fwdproxy:8080",
    "http_proxy": "http://fwdproxy:8080",
    "no_proxy": (
        ".fbcdn.net,.facebook.com,.thefacebook.com,.tfbnw.net,.fb.com,"
        ".fburl.com,.facebook.net,.sb.fbsbx.com,localhost"
    ),
}
_pr_state_cache: dict[str, tuple[float, str]] = {}


def _read_file(backend: Backend, path: str) -> str:
    result = backend.run(f"cat {path}", check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def _github_env_prefix() -> str:
    env = {
        key: os.environ.get(key) or value
        for key, value in _DEFAULT_GITHUB_ENV.items()
    }
    env["HTTPS_PROXY"] = os.environ.get("HTTPS_PROXY") or env["https_proxy"]
    env["HTTP_PROXY"] = os.environ.get("HTTP_PROXY") or env["http_proxy"]
    env["NO_PROXY"] = (
        os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or env["no_proxy"]
    )
    return " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items()) + " "


def _gh(command: str) -> str:
    return f"{_github_env_prefix()}gh {command}"


def _run_github(
    backend: Backend,
    args: list[str],
    *,
    cwd: str | None = None,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    if harness_available() and isinstance(backend, LocalBackend):
        harness_cwd = None
        if cwd:
            harness_cwd = str(Path(cwd.replace("~", str(Path.home()))))
        try:
            data = call_github_harness("gh", args=args, cwd=harness_cwd)
        except PtqError as exc:
            # Older harness servers only support issue fetch/search. Let direct
            # backend execution handle tests, stale sockets, and non-Codex shells
            # until the harness is restarted with the gh action.
            if (
                "Unknown action: gh" not in str(exc)
                and "GitHub harness unavailable" not in str(exc)
            ):
                raise
        else:
            if not isinstance(data, dict):
                raise PtqError("GitHub harness returned invalid gh response.")
            result = subprocess.CompletedProcess(
                ["gh", *args],
                int(data.get("returncode", 1)),
                str(data.get("stdout") or ""),
                str(data.get("stderr") or ""),
            )
            if check and result.returncode != 0:
                raise subprocess.CalledProcessError(
                    result.returncode,
                    result.args,
                    output=result.stdout,
                    stderr=result.stderr,
                )
            return result

    quoted_args = " ".join(shlex.quote(arg) for arg in args)
    cmd = _gh(quoted_args)
    if cwd:
        cmd = f"cd {cwd} && {cmd}"
    return backend.run(cmd, check=check)


def _read_json_file(backend: Backend, path: str) -> dict:
    text = _read_file(backend, path)
    if not text:
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _clean_summary_lines(text: str, *, limit: int = 4) -> list[str]:
    lines: list[str] = []
    in_fence = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or not line:
            continue
        if line.startswith("#"):
            continue
        line = line.lstrip("-*0123456789. ").strip()
        if not line:
            continue
        if len(line) > 140:
            line = f"{line[:137].rstrip()}..."
        lines.append(line)
        if len(lines) >= limit:
            break
    return lines


def _build_commit_message(
    title: str,
    *,
    issue_number: int | None,
    human_note: str,
    report: str,
    status: dict,
) -> str:
    body_lines: list[str] = []
    status_summary = str(status.get("summary") or "").strip()
    if status_summary:
        body_lines.append(status_summary)
    body_lines.extend(_clean_summary_lines(report, limit=3))
    if not body_lines:
        body_lines.extend(_clean_summary_lines(human_note, limit=3))

    parts = [title.strip()]
    if body_lines:
        parts.append("")
        parts.append("Summary:")
        parts.extend(f"- {line}" for line in body_lines[:4])
    if issue_number is not None:
        parts.append("")
        parts.append(f"Fixes #{issue_number}")
    return "\n".join(parts).strip() + "\n"


def _clean_pr_title(text: str) -> str:
    title = " ".join(text.split()).strip()
    title = re.sub(r"\s*\bFixes?\s+#\d+\b\.?\s*$", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\s*\(#\d+\)\s*$", "", title)
    title = title.strip(" .")
    if len(title) > 90:
        title = title[:87].rstrip(" .") + "..."
    return title


def _first_sentence(text: str) -> str:
    compact = " ".join(text.split())
    if not compact:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", compact)
    return sentences[0].strip() if sentences else compact


def _build_pr_title(
    requested_title: str | None,
    *,
    status: dict,
    report: str,
) -> str:
    candidates = [
        requested_title,
        str(status.get("pr_title") or ""),
        _first_sentence(str(status.get("summary") or "")),
        _first_sentence(_extract_markdown_section(report, ["Summary"])),
        _first_sentence(_extract_markdown_section(report, ["Fix"])),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        title = _clean_pr_title(str(candidate))
        if title:
            return title
    return "Apply PTQ-generated PyTorch fix"


def _read_repro_script(
    backend: Backend,
    job_dir: str,
    *,
    issue_number: int | None,
    status: dict,
) -> tuple[str, str]:
    candidates: list[str] = []
    repro_file = str(status.get("repro_file") or "").strip()
    if repro_file:
        candidates.append(repro_file)
    if issue_number is not None:
        candidates.extend(
            [
                f"repro_{issue_number}.py",
                f"repro_{issue_number}_generated.py",
            ]
        )
    candidates.append("repro.py")

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        script = _read_file(backend, f"{job_dir}/{candidate}")
        if script:
            return candidate, script
    return "", ""


def _parse_github_pr_url(pr_url: str) -> tuple[str, str, str] | None:
    match = re.match(r"https://github\.com/([^/]+)/([^/]+)/pull/([0-9]+)", pr_url)
    if not match:
        return None
    return match.group(1), match.group(2), match.group(3)


def _json_items_from_stdout(stdout: str) -> list[dict]:
    text = stdout.strip()
    if not text:
        return []

    def flatten(value: object) -> list[dict]:
        if isinstance(value, dict):
            return [value]
        if not isinstance(value, list):
            return []
        items: list[dict] = []
        for entry in value:
            if isinstance(entry, dict):
                items.append(entry)
            elif isinstance(entry, list):
                items.extend(item for item in entry if isinstance(item, dict))
        return items

    try:
        return flatten(json.loads(text))
    except json.JSONDecodeError:
        pass

    items: list[dict] = []
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        try:
            value, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            break
        items.extend(flatten(value))
        idx = end
    return items


def _is_ptq_managed_comment(body: str) -> bool:
    return (
        _REPRO_COMMENT_MARKER in body
        or _RESOLUTION_MARKER_PREFIX in body
        or "This PR was generated by [ptq]" in body
        or "Automated draft PR generated by ptq" in body
    )


def _normalize_pr_state(raw_state: str, merged_at: str) -> str:
    match raw_state.upper():
        case "OPEN":
            return "open"
        case "MERGED":
            return "merged"
        case "CLOSED":
            return "merged" if merged_at else "closed"
        case _:
            return "unknown"


def get_pr_state(
    backend: Backend,
    pr_url: str,
    *,
    force_refresh: bool = False,
    ttl_seconds: float = _PR_STATE_TTL_SECONDS,
) -> str:
    if not pr_url:
        return "unknown"

    now = time.monotonic()
    cached = _pr_state_cache.get(pr_url)
    if not force_refresh and cached and now - cached[0] < ttl_seconds:
        return cached[1]

    result = _run_github(
        backend,
        [
            "pr",
            "view",
            pr_url,
            "--json",
            "state,mergedAt",
            "--jq",
            '[.state, (.mergedAt // "")] | @tsv',
        ],
        check=False,
    )
    if result.returncode != 0:
        state = "unknown"
    else:
        raw_state, _, merged_at = result.stdout.strip().partition("\t")
        state = _normalize_pr_state(raw_state.strip(), merged_at.strip())
    _pr_state_cache[pr_url] = (now, state)
    return state


def _build_pr_body(
    report: str,
    worklog: str,
    repro: str,
    issue_number: int | None,
    human_note: str,
    status: dict | None = None,
    repro_filename: str = "",
) -> str:
    _ = human_note
    status = status or {}
    parts: list[str] = []
    agent_report = _build_agent_report_section(
        report,
        repro,
        repro_filename or str(status.get("repro_file") or ""),
        status,
    )
    if agent_report:
        parts.append(f"\n## Agent Report\n{agent_report}")
    files_changed = status.get("files_changed")
    if isinstance(files_changed, list) and files_changed:
        files = [
            str(path)
            for path in files_changed
            if isinstance(path, str) and path.strip()
        ]
        if files:
            parts.append(
                "\n## Files Changed\n" + "\n".join(f"- `{path}`" for path in files)
            )
    if issue_number is not None:
        parts.append(f"\n\nFixes #{issue_number}")
    parts.append(
        "\n---\n*This PR was generated by [ptq](https://github.com/drisspg/pt_job_queue) "
        "with human review.*"
    )
    return "\n".join(parts).lstrip()


def _truncate_text(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _concise_status_summary(text: str) -> str:
    compact = " ".join(text.split())
    if not compact:
        return ""
    for marker in (
        " Run ",
        " In-tree regression",
        " spin fixlint",
        " Lintrunner",
        " Lint",
    ):
        if marker in compact:
            compact = compact.split(marker, 1)[0].strip()
    sentences = re.split(r"(?<=[.!?])\s+", compact)
    concise = " ".join(sentence for sentence in sentences[:2] if sentence).strip()
    return _truncate_text(concise or compact, 500)


def _build_agent_report_section(
    report: str,
    repro: str,
    repro_filename: str,
    status: dict,
) -> str:
    sections: list[str] = []
    summary = _extract_markdown_section(report, ["Summary"])
    if not summary:
        summary = _concise_status_summary(str(status.get("summary") or ""))
    _append_report_subsection(sections, "Summary", summary, limit=900)
    _append_report_subsection(
        sections,
        "Root Cause",
        _extract_markdown_section(report, ["Root cause", "Root Cause"]),
        limit=1600,
    )
    _append_report_subsection(
        sections,
        "Fix",
        _extract_markdown_section(report, ["Fix"]),
        limit=2200,
    )
    repro_section = _build_repro_report_section(
        _strip_details_blocks(_extract_markdown_section(report, ["Repro"])),
        repro,
        repro_filename,
    )
    _append_report_subsection(sections, "Repro", repro_section, limit=5000)
    _append_report_subsection(
        sections,
        "Testing",
        _extract_markdown_section(report, ["Test results", "Testing", "Tests"]),
        limit=2600,
    )
    return "\n\n".join(sections).strip()


def _append_report_subsection(
    sections: list[str],
    title: str,
    content: str,
    *,
    limit: int,
) -> None:
    content = content.strip()
    if not content:
        return
    sections.append(f"### {title}\n{_truncate_markdown(content, limit)}")


def _extract_markdown_section(markdown: str, titles: list[str]) -> str:
    wanted = {_normalize_heading(title) for title in titles}
    lines = markdown.splitlines()
    collecting = False
    start_level = 0
    collected: list[str] = []
    for line in lines:
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if match:
            level = len(match.group(1))
            heading = _normalize_heading(match.group(2))
            if collecting and level <= start_level:
                break
            if not collecting and heading in wanted:
                collecting = True
                start_level = level
                continue
        if collecting:
            collected.append(line)
    return "\n".join(collected).strip()


def _normalize_heading(text: str) -> str:
    text = re.sub(r"[^\w\s-]", "", text.strip().lower())
    return re.sub(r"\s+", " ", text)


def _strip_details_blocks(markdown: str) -> str:
    return re.sub(r"<details>.*?</details>", "", markdown, flags=re.DOTALL).strip()


def _build_repro_report_section(
    intro: str,
    repro: str,
    repro_filename: str,
) -> str:
    parts: list[str] = []
    if intro:
        parts.append(intro)
    parts.extend(["Run:", "", "```bash", _public_repro_command(repro_filename), "```"])
    if repro:
        fence = "````" if "```" in repro else "```"
        parts.extend(
            [
                "",
                "<details>",
                "<summary>Repro Script</summary>",
                "",
                f"{fence}python",
                repro,
                fence,
                "",
                "</details>",
            ]
        )
    return "\n".join(parts).strip()


def _public_repro_command(filename: str) -> str:
    return f"python {shlex.quote(filename)}" if filename else "python repro.py"


def _truncate_markdown(text: str, limit: int) -> str:
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[: limit - 3].rstrip() + "..."


def _ensure_ssh_remote(
    backend: Backend, worktree: str, _log: Callable[[str], None]
) -> None:
    result = backend.run(f"cd {worktree} && git remote get-url origin", check=False)
    url = result.stdout.strip()
    for https_prefix, ssh_prefix in _HTTPS_TO_SSH.items():
        if url.startswith(https_prefix):
            ssh_url = url.replace(https_prefix, ssh_prefix)
            if not ssh_url.endswith(".git"):
                ssh_url += ".git"
            _log(f"Switching origin to SSH: {ssh_url}")
            backend.run(f"cd {worktree} && git remote set-url origin '{ssh_url}'")
            return


def _delete_repro_comment_if_present(
    backend: Backend,
    worktree: str,
    pr_url: str,
    _log: Callable[[str], None],
) -> None:
    parsed = _parse_github_pr_url(pr_url)
    if not parsed:
        return
    owner, repo, number = parsed
    comments = _run_github(
        backend,
        ["api", f"repos/{owner}/{repo}/issues/{number}/comments", "--paginate"],
        cwd=worktree,
        check=False,
    )
    if comments.returncode != 0:
        return
    for item in _json_items_from_stdout(comments.stdout):
        if _REPRO_COMMENT_MARKER not in str(item.get("body") or ""):
            continue
        comment_id = str(item.get("id") or "")
        if not comment_id:
            continue
        _log("Removing old repro script PR comment...")
        _run_github(
            backend,
            ["api", "-X", "DELETE", f"repos/{owner}/{repo}/issues/comments/{comment_id}"],
            cwd=worktree,
            check=False,
        )


def _existing_open_pr_url(
    repo: JobRepository,
    job_id: str,
    *,
    backend: Backend,
    worktree: str,
    branch: str,
) -> str:
    job = repo.get(job_id)
    if job.pr_url and get_pr_state(backend, job.pr_url, force_refresh=True) == "open":
        return job.pr_url

    result = _run_github(
        backend,
        [
            "pr",
            "list",
            "--head",
            branch,
            "--state",
            "open",
            "--json",
            "url",
            "--jq",
            ".[0].url",
        ],
        cwd=worktree,
        check=False,
    )
    url = result.stdout.strip() if result.returncode == 0 else ""
    if url and url != job.pr_url:
        job.pr_url = url
        repo.save(job)
    return url


def _review_comment_entry(item: dict) -> dict | None:
    body = str(item.get("body") or "").strip()
    if not body or _is_ptq_managed_comment(body):
        return None
    user = item.get("user")
    author = ""
    if isinstance(user, dict):
        author = str(user.get("login") or "")
    return {
        "kind": "review",
        "id": item.get("id"),
        "author": author,
        "state": str(item.get("state") or ""),
        "body": body,
        "url": str(item.get("html_url") or item.get("url") or ""),
        "submitted_at": str(item.get("submitted_at") or ""),
    }


def _conversation_comment_entry(item: dict) -> dict | None:
    body = str(item.get("body") or "").strip()
    if not body or _is_ptq_managed_comment(body):
        return None
    user = item.get("user")
    author = ""
    if isinstance(user, dict):
        author = str(user.get("login") or "")
    return {
        "kind": "conversation",
        "id": item.get("id"),
        "author": author,
        "body": body,
        "url": str(item.get("html_url") or item.get("url") or ""),
        "created_at": str(item.get("created_at") or ""),
    }


def _inline_comment_entry(item: dict) -> dict | None:
    body = str(item.get("body") or "").strip()
    if not body or _is_ptq_managed_comment(body):
        return None
    user = item.get("user")
    author = ""
    if isinstance(user, dict):
        author = str(user.get("login") or "")
    line = item.get("line")
    if line is None:
        line = item.get("original_line")
    return {
        "kind": "inline",
        "id": item.get("id"),
        "author": author,
        "path": str(item.get("path") or ""),
        "line": line,
        "side": str(item.get("side") or ""),
        "body": body,
        "url": str(item.get("html_url") or item.get("url") or ""),
        "created_at": str(item.get("created_at") or ""),
    }


def _ci_failure_entry(item: dict) -> dict | None:
    bucket = str(item.get("bucket") or "").strip().lower()
    state = str(item.get("state") or "").strip().upper()
    if bucket not in {"fail"} and state not in {
        "FAILURE",
        "FAILED",
        "ERROR",
        "TIMED_OUT",
        "ACTION_REQUIRED",
    }:
        return None
    return {
        "name": str(item.get("name") or ""),
        "state": state,
        "bucket": bucket,
        "workflow": str(item.get("workflow") or ""),
        "description": str(item.get("description") or ""),
        "link": str(item.get("link") or ""),
        "started_at": str(item.get("startedAt") or ""),
        "completed_at": str(item.get("completedAt") or ""),
        "event": str(item.get("event") or ""),
    }


def _actions_run_id(link: str) -> str:
    match = re.search(r"/actions/runs/([0-9]+)", link)
    return match.group(1) if match else ""


def _fetch_failed_run_log_excerpt(
    backend: Backend,
    worktree: str,
    *,
    owner: str,
    repo: str,
    run_id: str,
) -> str:
    result = _run_github(
        backend,
        [
            "run",
            "view",
            run_id,
            "--repo",
            f"{owner}/{repo}",
            "--log-failed",
        ],
        cwd=worktree,
        check=False,
    )
    text = result.stdout.strip()
    if not text:
        return ""
    return _truncate_markdown(text, _CI_LOG_EXCERPT_LIMIT)


def _fetch_ci_failures(
    backend: Backend,
    worktree: str,
    pr_url: str,
    *,
    owner: str,
    repo: str,
    _log: Callable[[str], None],
) -> list[dict]:
    result = _run_github(
        backend,
        [
            "pr",
            "checks",
            pr_url,
            "--json",
            "bucket,completedAt,description,event,link,name,startedAt,state,workflow",
        ],
        cwd=worktree,
        check=False,
    )
    if result.returncode != 0 and not result.stdout.strip():
        detail = (result.stderr or result.stdout or "").strip()
        _log(f"Could not read PR checks: {detail}")
        return []
    failures: list[dict] = []
    for item in _json_items_from_stdout(result.stdout):
        entry = _ci_failure_entry(item)
        if entry is not None:
            failures.append(entry)
    log_excerpts: dict[str, str] = {}
    for failure in failures:
        run_id = _actions_run_id(str(failure.get("link") or ""))
        if not run_id:
            continue
        if run_id not in log_excerpts and len(log_excerpts) < _CI_LOG_RUN_LIMIT:
            excerpt = _fetch_failed_run_log_excerpt(
                backend,
                worktree,
                owner=owner,
                repo=repo,
                run_id=run_id,
            )
            if excerpt:
                log_excerpts[run_id] = excerpt
        if run_id in log_excerpts:
            failure["failed_log_excerpt"] = log_excerpts[run_id]
    return failures


def fetch_pr_feedback(
    repo: JobRepository,
    job_id: str,
    *,
    log: Callable[[str], None] | None = None,
) -> dict | None:
    """Return reviewer feedback and CI failures for an existing open draft PR."""
    _log = log or (lambda _: None)
    job = repo.get(job_id)
    if job.issue is None:
        return None

    backend = backend_for_job(job)
    job_dir = f"{backend.workspace}/jobs/{job_id}"
    profile = get_profile(job.repo)
    worktree = f"{job_dir}/{profile.dir_name}"
    branch = f"ptq/{job.issue}"
    pr_url = _existing_open_pr_url(
        repo,
        job_id,
        backend=backend,
        worktree=worktree,
        branch=branch,
    )
    if not pr_url:
        return None

    parsed = _parse_github_pr_url(pr_url)
    if not parsed:
        return None
    owner, gh_repo, number = parsed

    endpoints = [
        (
            "conversation",
            ["api", f"repos/{owner}/{gh_repo}/issues/{number}/comments", "--paginate"],
            _conversation_comment_entry,
        ),
        (
            "inline",
            ["api", f"repos/{owner}/{gh_repo}/pulls/{number}/comments", "--paginate"],
            _inline_comment_entry,
        ),
        (
            "review",
            ["api", f"repos/{owner}/{gh_repo}/pulls/{number}/reviews", "--paginate"],
            _review_comment_entry,
        ),
    ]

    comments: list[dict] = []
    for label, args, formatter in endpoints:
        result = _run_github(backend, args, cwd=worktree, check=False)
        if result.returncode != 0:
            _log(f"Could not read {label} PR feedback: {result.stderr.strip()}")
            continue
        for item in _json_items_from_stdout(result.stdout):
            entry = formatter(item)
            if entry is not None:
                comments.append(entry)

    ci_failures = _fetch_ci_failures(
        backend,
        worktree,
        pr_url,
        owner=owner,
        repo=gh_repo,
        _log=_log,
    )

    if not comments and not ci_failures:
        return None

    summary_parts = []
    if comments:
        summary_parts.append(f"{len(comments)} GitHub PR feedback item(s)")
    if ci_failures:
        summary_parts.append(f"{len(ci_failures)} failing PR check(s)")
    summary = (
        f"Found {' and '.join(summary_parts)}. Address reviewer comments and inspect "
        "failing CI. If a CI failure is caused by this PR, fix it before finishing."
    )
    return {
        "source": "github_pr",
        "pr_url": pr_url,
        "summary": summary,
        "comments": comments,
        "ci_failures": ci_failures,
    }


def _resolved_pr_comments(status: dict) -> list[dict]:
    raw = status.get("resolved_pr_comments")
    if raw is None:
        raw = status.get("pr_comment_resolutions")
    if not isinstance(raw, list):
        return []

    resolved: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        resolution = str(item.get("resolution") or item.get("reply") or "").strip()
        if not resolution:
            continue
        comment_id = item.get("comment_id")
        if comment_id is None:
            comment_id = item.get("id")
        comment_url = str(item.get("comment_url") or item.get("url") or "").strip()
        if comment_id is None and not comment_url:
            continue
        entry = {
            "comment_id": comment_id,
            "kind": str(item.get("kind") or "").strip().lower(),
            "comment_url": comment_url,
            "resolution": resolution,
        }
        resolved.append(entry)
    return resolved


def _resolution_key(entry: dict) -> str:
    comment_id = entry.get("comment_id")
    if comment_id is not None and str(comment_id).strip():
        return str(comment_id).strip()
    comment_url = str(entry.get("comment_url") or "").strip()
    return sha1(comment_url.encode("utf-8")).hexdigest()[:16]


def _resolution_marker(entry: dict) -> str:
    return f"{_RESOLUTION_MARKER_PREFIX}{_resolution_key(entry)} -->"


def _brief_resolution(text: str) -> str:
    compact = " ".join(text.split())
    if not compact:
        return "Addressed."
    sentences = re.split(r"(?<=[.!?])\s+", compact)
    first_sentence = sentences[0].strip() if sentences else compact
    return _truncate_text(first_sentence or compact, _BOT_REPLY_RESOLUTION_LIMIT)


def _resolution_reply_body(entry: dict, *, include_reference: bool) -> str:
    body = (
        f"{_BOT_COMMENT_PREFIX} Resolved in the latest update. "
        f"{_brief_resolution(str(entry['resolution']))}"
    )
    comment_url = str(entry.get("comment_url") or "").strip()
    if include_reference and comment_url:
        body += f"\n\nReference: {comment_url}"
    return f"{body}\n\n{_resolution_marker(entry)}"


def _existing_resolution_markers(
    backend: Backend,
    worktree: str,
    *,
    owner: str,
    repo: str,
    number: str,
) -> set[str]:
    markers: set[str] = set()
    for args in (
        ["api", f"repos/{owner}/{repo}/issues/{number}/comments", "--paginate"],
        ["api", f"repos/{owner}/{repo}/pulls/{number}/comments", "--paginate"],
    ):
        result = _run_github(backend, args, cwd=worktree, check=False)
        if result.returncode != 0:
            continue
        for item in _json_items_from_stdout(result.stdout):
            body = str(item.get("body") or "")
            for marker in re.findall(
                rf"{re.escape(_RESOLUTION_MARKER_PREFIX)}[^>]+-->", body
            ):
                markers.add(marker)
    return markers


def _reply_to_resolved_pr_comments(
    backend: Backend,
    worktree: str,
    pr_url: str,
    status: dict,
    _log: Callable[[str], None],
) -> None:
    resolved = _resolved_pr_comments(status)
    if not resolved:
        return
    parsed = _parse_github_pr_url(pr_url)
    if not parsed:
        return
    owner, repo, number = parsed
    existing_markers = _existing_resolution_markers(
        backend,
        worktree,
        owner=owner,
        repo=repo,
        number=number,
    )

    for entry in resolved:
        marker = _resolution_marker(entry)
        if marker in existing_markers:
            continue
        comment_id = entry.get("comment_id")
        kind = str(entry.get("kind") or "")
        if kind == "inline" and comment_id is not None:
            _log(f"Replying to resolved inline PR comment {comment_id}...")
            result = _run_github(
                backend,
                [
                    "api",
                    "-X",
                    "POST",
                    f"repos/{owner}/{repo}/pulls/{number}/comments/{comment_id}/replies",
                    "-f",
                    f"body={_resolution_reply_body(entry, include_reference=False)}",
                ],
                cwd=worktree,
                check=False,
            )
        else:
            target = comment_id if comment_id is not None else entry.get("comment_url")
            _log(f"Replying to resolved PR feedback {target}...")
            result = _run_github(
                backend,
                [
                    "pr",
                    "comment",
                    pr_url,
                    "--body",
                    _resolution_reply_body(entry, include_reference=True),
                ],
                cwd=worktree,
                check=False,
            )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            _log(f"Could not reply to resolved PR comment: {detail}")


def create_pr(
    repo: JobRepository,
    job_id: str,
    *,
    human_note: str,
    title: str | None = None,
    draft: bool = True,
    log: Callable[[str], None] | None = None,
) -> PRResult:
    if not human_note.strip():
        raise PtqError(
            "A human note is required. Describe what this PR does, "
            "why you believe it's correct, and how the reviewer should approach it."
        )
    # Kept for API compatibility; ptq-created PRs are always drafts.
    _ = draft

    _log = log or (lambda _: None)
    job = repo.get(job_id)
    backend = backend_for_job(job)
    job_dir = f"{backend.workspace}/jobs/{job_id}"
    profile = get_profile(job.repo)
    worktree = f"{job_dir}/{profile.dir_name}"
    existing_open_pr_url = ""

    if job.pr_url:
        pr_state = get_pr_state(backend, job.pr_url, force_refresh=True)
        match pr_state:
            case "open":
                existing_open_pr_url = job.pr_url
                _log(f"Existing PR is open: {existing_open_pr_url}")
            case "closed" | "merged":
                _log(f"Stored PR is {pr_state}. Creating a new PR from current branch.")
            case _:
                _log("Stored PR state is unknown. Proceeding with create-or-find flow.")

    report = _read_file(backend, f"{job_dir}/report.md")
    worklog = _read_file(backend, f"{job_dir}/worklog.md")
    status = _read_json_file(backend, f"{job_dir}/status.json")
    branch = f"ptq/{job.issue}" if job.issue is not None else f"ptq/{job_id}"
    pr_title = _build_pr_title(title, status=status, report=report)

    _log(f"Branch: {branch}")
    _log(f"Title: {pr_title}")

    repro_filename, repro = _read_repro_script(
        backend,
        job_dir,
        issue_number=job.issue,
        status=status,
    )
    body = _build_pr_body(
        report,
        worklog,
        repro,
        job.issue,
        human_note,
        status=status,
        repro_filename=repro_filename,
    )
    _log(
        f"PR body: report.md {'found' if report else 'missing'}, "
        f"worklog.md {'found' if worklog else 'missing'}, "
        f"repro {'found' if repro else 'missing'}"
    )

    commit_message = _build_commit_message(
        pr_title,
        issue_number=job.issue,
        human_note=human_note,
        report=report,
        status=status,
    )
    commit_file = ".ptq_commit_message"
    quoted_commit_file = shlex.quote(commit_file)
    _log(f"Checking out branch {branch}...")
    backend.run(f"cd {worktree} && git checkout -B '{branch}'")
    _log("Staging changes...")
    backend.run(f"cd {worktree} && git add -A")
    has_staged_changes = (
        backend.run(
            f"cd {worktree} && git diff --cached --quiet", check=False
        ).returncode
        != 0
    )
    if has_staged_changes:
        _log("Creating commit...")
        backend.run(
            f"cd {worktree} && "
            f"printf %s {shlex.quote(commit_message)} > {quoted_commit_file}"
        )
        commit_result = backend.run(
            f"cd {worktree} && git commit -F {quoted_commit_file}",
            check=False,
        )
        backend.run(f"cd {worktree} && rm -f {quoted_commit_file}", check=False)
        if commit_result.returncode != 0:
            stderr = commit_result.stderr.strip() if commit_result.stderr else ""
            stdout = commit_result.stdout.strip() if commit_result.stdout else ""
            raise PtqError(f"git commit failed: {stderr or stdout or 'unknown error'}")
    else:
        _log("No staged changes to commit.")

    _ensure_ssh_remote(backend, worktree, _log)
    _log("Pushing branch...")
    push_result = backend.run(
        f"cd {worktree} && git push -u origin '{branch}'",
        check=False,
    )
    if push_result.returncode != 0:
        stderr = push_result.stderr.strip() if push_result.stderr else ""
        stdout = push_result.stdout.strip() if push_result.stdout else ""
        raise PtqError(f"git push failed: {stderr or stdout or 'unknown error'}")
    url = ""
    if existing_open_pr_url:
        _log("Updating existing PR...")
        edit_result = _run_github(
            backend,
            [
                "pr",
                "edit",
                existing_open_pr_url,
                "--title",
                pr_title,
                "--body",
                body,
            ],
            cwd=worktree,
            check=False,
        )
        if edit_result.returncode == 0:
            url = existing_open_pr_url
        else:
            _log("Could not update existing PR, falling back to create-or-find flow.")

    result = None
    if not url:
        _log("Creating PR...")
        result = _run_github(
            backend,
            [
                "pr",
                "create",
                "--title",
                pr_title,
                "--body",
                body,
                "--head",
                branch,
                "--draft",
            ],
            cwd=worktree,
            check=False,
        )
        for line in result.stdout.strip().splitlines():
            if line.startswith("http"):
                url = line.strip()
                break

    if not url and result and result.returncode != 0:
        stderr = result.stderr.strip() if result.stderr else ""
        if "already exists" in stderr:
            url = _run_github(
                backend,
                [
                    "pr",
                    "list",
                    "--head",
                    branch,
                    "--state",
                    "open",
                    "--json",
                    "url",
                    "--jq",
                    ".[0].url",
                ],
                cwd=worktree,
                check=False,
            ).stdout.strip()
            if url:
                _log("Updating existing PR body...")
                _run_github(
                    backend,
                    ["pr", "edit", branch, "--body", body],
                    cwd=worktree,
                    check=False,
                )
        if not url:
            raise PtqError(f"gh pr create failed: {stderr or result.stdout}")

    if url:
        _run_github(backend, ["pr", "ready", url, "--undo"], cwd=worktree, check=False)
        _delete_repro_comment_if_present(backend, worktree, url, _log)
        _reply_to_resolved_pr_comments(backend, worktree, url, status, _log)
        job.pr_url = url
        job.human_note = human_note
        job.pr_title = pr_title
        repo.save(job)

    return PRResult(url=url, branch=branch)
