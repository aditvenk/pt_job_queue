from __future__ import annotations

import json
import os
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha1
from pathlib import Path
from typing import Any

from ptq.domain.models import PtqError
from ptq.github_harness import call_github_harness, harness_available

DEFAULT_PROFILE_DIR = Path.home() / ".ptq" / "evaluator_profiles"
DEFAULT_PROFILE_MODEL = "gpt-5.5"
_MAX_EXCERPT_CHARS = 700

_DEFAULT_GITHUB_ENV = {
    "https_proxy": "http://fwdproxy:8080",
    "http_proxy": "http://fwdproxy:8080",
    "no_proxy": (
        ".fbcdn.net,.facebook.com,.thefacebook.com,.tfbnw.net,.fb.com,"
        ".fburl.com,.facebook.net,.sb.fbsbx.com,localhost"
    ),
}


@dataclass(frozen=True)
class ReviewerProfile:
    username: str
    path: Path
    pr_count: int
    comment_count: int


def reviewer_profile_path(profile: str, *, base_dir: Path | None = None) -> Path:
    raw = Path(profile).expanduser()
    if raw.suffix == ".md" or raw.is_absolute() or len(raw.parts) > 1:
        return raw
    return (base_dir or DEFAULT_PROFILE_DIR) / f"{profile}.md"


def generate_reviewer_profile(
    username: str,
    *,
    repo: str = "pytorch/pytorch",
    months: int = 6,
    limit: int = 100,
    output: Path | None = None,
) -> ReviewerProfile:
    username = username.lstrip("@").strip()
    if not username:
        raise PtqError("GitHub username is required.")

    since = _since_date(months)
    prs = _search_reviewed_prs(username, repo=repo, since=since, limit=limit)
    comments = _fetch_user_review_comments(username, prs)
    path = output or reviewer_profile_path(username)
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        _build_profile_markdown(
            username=username,
            repo=repo,
            since=since,
            prs=prs,
            comments=comments,
        )
    )
    return ReviewerProfile(
        username=username,
        path=path,
        pr_count=len(prs),
        comment_count=len(comments),
    )


def profile_username(profile_path: str | Path) -> str:
    path = Path(profile_path).expanduser()
    if path.exists():
        match = re.search(
            r"^#\s+@([A-Za-z0-9-]+)\s+Review Profile\s*$",
            path.read_text(errors="replace"),
            re.MULTILINE,
        )
        if match:
            return match.group(1)
    return path.stem


def append_pr_feedback_to_profile(
    profile_path: str | Path,
    *,
    username: str,
    feedback: dict,
) -> int:
    path = Path(profile_path).expanduser()
    if not path.exists():
        return 0
    comments = _matching_feedback_comments(username, feedback)
    if not comments:
        return 0

    text = path.read_text()
    blocks = []
    for comment in comments:
        marker = _feedback_marker(comment)
        if marker in text:
            continue
        blocks.append(_feedback_block(comment, marker))
    if not blocks:
        return 0

    if "## Recent PR Feedback Incorporated" not in text:
        if not text.endswith("\n"):
            text += "\n"
        text += "\n## Recent PR Feedback Incorporated\n\n"
        text += (
            "These are fresh human-review comments observed by ptq while "
            "watching draft PRs. Treat them as high-signal examples of this "
            "reviewer's current expectations.\n\n"
        )
    elif not text.endswith("\n"):
        text += "\n"

    text += "\n".join(blocks)
    if not text.endswith("\n"):
        text += "\n"
    path.write_text(text)
    return len(blocks)


def _since_date(months: int) -> str:
    months = max(1, months)
    # Good enough for GitHub search's date filter without adding dateutil.
    since = datetime.now(timezone.utc) - timedelta(days=31 * months)
    return since.date().isoformat()


def _search_reviewed_prs(
    username: str,
    *,
    repo: str,
    since: str,
    limit: int,
) -> list[dict]:
    args = [
        "search",
        "prs",
        "--reviewed-by",
        username,
        "--updated",
        f">={since}",
        "--limit",
        str(max(1, limit)),
        "--json",
        "number,title,url,repository,updatedAt",
    ]
    if repo:
        args.extend(["--repo", repo])
    data = _run_gh_json(args)
    if not isinstance(data, list):
        raise PtqError("GitHub PR search returned an unexpected response.")
    return [item for item in data if isinstance(item, dict)]


def _fetch_user_review_comments(username: str, prs: list[dict]) -> list[dict]:
    comments: list[dict] = []
    for pr in prs:
        repo_name = _repo_full_name(pr)
        number = pr.get("number")
        if not repo_name or not isinstance(number, int):
            continue
        owner, repo = repo_name.split("/", 1)
        for kind, endpoint in (
            ("conversation", f"repos/{owner}/{repo}/issues/{number}/comments"),
            ("inline", f"repos/{owner}/{repo}/pulls/{number}/comments"),
            ("review", f"repos/{owner}/{repo}/pulls/{number}/reviews"),
        ):
            try:
                rows = _run_gh_json(["api", endpoint, "--paginate"])
            except PtqError:
                continue
            if not isinstance(rows, list):
                continue
            for item in rows:
                if not isinstance(item, dict):
                    continue
                entry = _comment_entry(username, kind, repo_name, number, pr, item)
                if entry is not None:
                    comments.append(entry)
    return comments


def _comment_entry(
    username: str,
    kind: str,
    repo: str,
    number: int,
    pr: dict,
    item: dict,
) -> dict | None:
    user = item.get("user")
    author = str(user.get("login") or "") if isinstance(user, dict) else ""
    if author.lower() != username.lower():
        return None
    body = str(item.get("body") or "").strip()
    state = str(item.get("state") or "")
    if not body and not state:
        return None
    return {
        "kind": kind,
        "repo": repo,
        "pr_number": number,
        "pr_title": str(pr.get("title") or ""),
        "pr_url": str(pr.get("url") or ""),
        "path": str(item.get("path") or ""),
        "line": item.get("line") or item.get("original_line"),
        "state": state,
        "body": body,
        "url": str(item.get("html_url") or item.get("url") or ""),
        "created_at": str(item.get("created_at") or item.get("submitted_at") or ""),
    }


def _matching_feedback_comments(username: str, feedback: dict) -> list[dict]:
    comments = feedback.get("comments")
    if not isinstance(comments, list):
        return []
    username = username.lstrip("@").lower()
    matched = []
    for item in comments:
        if not isinstance(item, dict):
            continue
        author = str(item.get("author") or "").lstrip("@").lower()
        body = str(item.get("body") or "").strip()
        if author == username and body:
            matched.append(item)
    return matched


def _feedback_marker(comment: dict) -> str:
    raw = "|".join(
        str(comment.get(key) or "")
        for key in ("id", "url", "author", "path", "line", "body")
    )
    digest = sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"<!-- ptq:profile-feedback:{digest} -->"


def _feedback_block(comment: dict, marker: str) -> str:
    kind = str(comment.get("kind") or "comment")
    author = str(comment.get("author") or "reviewer")
    url = str(comment.get("url") or "")
    created_at = str(
        comment.get("created_at")
        or comment.get("submitted_at")
        or datetime.now(timezone.utc).date().isoformat()
    )
    loc = str(comment.get("path") or kind)
    line = comment.get("line")
    if line is not None:
        loc = f"{loc}:{line}"
    body = _quote_feedback(str(comment.get("body") or ""))
    url_suffix = f" ({url})" if url else ""
    return (
        f"{marker}\n"
        f"- {created_at} @{author} on `{loc}` [{kind}]{url_suffix}\n"
        f"{body}\n"
    )


def _quote_feedback(text: str) -> str:
    cleaned = _clean_markdown(text)
    if not cleaned:
        return "  > (empty comment)"
    wrapped = _truncate(cleaned, _MAX_EXCERPT_CHARS)
    return "\n".join(f"  > {line}" for line in wrapped.splitlines())


def _repo_full_name(pr: dict) -> str:
    repo = pr.get("repository")
    if isinstance(repo, dict):
        for key in ("nameWithOwner", "fullName"):
            value = repo.get(key)
            if isinstance(value, str) and "/" in value:
                return value
        owner = repo.get("owner")
        name = repo.get("name")
        if isinstance(owner, dict):
            owner = owner.get("login")
        if isinstance(owner, str) and isinstance(name, str):
            return f"{owner}/{name}"
    return ""


def _run_gh_json(args: list[str]) -> Any:
    if harness_available():
        try:
            data = call_github_harness("gh", args=args)
        except PtqError:
            data = None
        else:
            if isinstance(data, dict) and "returncode" in data:
                if int(data.get("returncode") or 1) != 0:
                    detail = data.get("stderr") or data.get("stdout") or "gh failed"
                    raise PtqError(str(detail))
                return json.loads(str(data.get("stdout") or "null"))
            return data

    result = subprocess.run(
        ["gh", *args],
        env=_github_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise PtqError((result.stderr or result.stdout).strip() or "gh failed")
    return json.loads(result.stdout or "null")


def _github_env() -> dict[str, str]:
    env = os.environ.copy()
    for key, value in _DEFAULT_GITHUB_ENV.items():
        env.setdefault(key, value)
    env.setdefault("HTTPS_PROXY", env.get("https_proxy", ""))
    env.setdefault("HTTP_PROXY", env.get("http_proxy", ""))
    env.setdefault("NO_PROXY", env.get("no_proxy", ""))
    return env


def _build_profile_markdown(
    *,
    username: str,
    repo: str,
    since: str,
    prs: list[dict],
    comments: list[dict],
) -> str:
    stats = _comment_stats(comments)
    themes = _theme_counts(comments)
    examples = _representative_examples(comments)
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    lines = [
        f"# @{username} Review Profile",
        "",
        "This profile is generated from public GitHub PR review activity and is "
        "intended for a ptq evaluator. Use it to approximate review preferences "
        "and commenting style; do not claim to be this reviewer.",
        "",
        "## Source",
        "",
        f"- GitHub user: @{username}",
        f"- Repository filter: {repo or 'all repositories'}",
        f"- Window start: {since}",
        f"- PRs sampled: {len(prs)}",
        f"- Review/comment items by @{username}: {len(comments)}",
        f"- Generated at: {generated_at}",
        "",
        "## Voice And Commenting Style",
        "",
        *_style_bullets(stats),
        "",
        "## Review Priorities To Emulate",
        "",
        *_priority_bullets(themes),
        "",
        "## Evaluator Instructions",
        "",
        "- Apply the normal ptq evaluator rubric first: repro fidelity, correctness, "
        "scope, tests, and code quality remain binding.",
        "- Use this profile only to shape what you pay attention to and how you phrase "
        "comments.",
        "- Prefer concrete, file-oriented comments with a clear requested action.",
        "- Do not add comments just to imitate style; only comment on real technical "
        "risks visible in the issue, report, repro, diff, or lint output.",
        "- If approving, keep the summary concise and mention any residual risk that "
        "this reviewer would likely care about.",
        "",
        "## Representative Comment Excerpts",
        "",
    ]
    if examples:
        for example in examples:
            loc = example["path"] or example["kind"]
            if example["line"]:
                loc = f"{loc}:{example['line']}"
            body = _truncate(_clean_markdown(example["body"]), _MAX_EXCERPT_CHARS)
            lines.extend(
                [
                    f"- {example['repo']}#{example['pr_number']} {loc}",
                    f"  > {body}",
                ]
            )
    else:
        lines.append(
            "- No non-empty review comments were found in the sampled window. "
            "Use the normal ptq rubric and keep comments direct and technical."
        )
    lines.append("")
    return "\n".join(lines)


def _comment_stats(comments: list[dict]) -> dict:
    bodies = [
        str(comment.get("body") or "")
        for comment in comments
        if comment.get("body")
    ]
    words = [_word_count(body) for body in bodies]
    return {
        "count": len(comments),
        "body_count": len(bodies),
        "inline": sum(1 for comment in comments if comment.get("kind") == "inline"),
        "conversation": sum(
            1 for comment in comments if comment.get("kind") == "conversation"
        ),
        "review": sum(1 for comment in comments if comment.get("kind") == "review"),
        "avg_words": int(sum(words) / len(words)) if words else 0,
        "question_count": sum(1 for body in bodies if "?" in body),
        "code_count": sum(1 for body in bodies if "`" in body),
    }


def _style_bullets(stats: dict) -> list[str]:
    count = int(stats["count"])
    if count == 0:
        return [
            "- Not enough review comments were found to infer a strong personal style.",
            "- Default to concise, direct, technical feedback.",
        ]

    bullets = []
    avg_words = int(stats["avg_words"])
    if avg_words <= 35:
        bullets.append("- Comments tend to be short and action-oriented.")
    elif avg_words <= 90:
        bullets.append("- Comments tend to be moderately detailed without long essays.")
    else:
        bullets.append("- Comments tend to include detailed rationale and context.")

    inline = int(stats["inline"])
    if inline >= max(2, count // 2):
        bullets.append("- Prefer inline comments tied to specific files or code paths.")
    else:
        bullets.append("- Use both summary-level and code-specific feedback as needed.")

    if int(stats["question_count"]) >= max(1, count // 4):
        bullets.append(
            "- Questions are a normal way to flag uncertainty or request rationale."
        )
    if int(stats["code_count"]) >= max(1, count // 4):
        bullets.append("- Refer directly to identifiers, commands, and code snippets.")
    bullets.append("- Keep the tone direct, pragmatic, and focused on what must change.")
    return bullets


def _theme_counts(comments: list[dict]) -> Counter[str]:
    themes = {
        "tests": r"\b(test|coverage|regression|ci)\b",
        "correctness": r"\b(correct|incorrect|bug|fail|failure|wrong|edge case)\b",
        "compatibility": r"\b(bc|backward|compat|api|semantics)\b",
        "distributed": r"\b(distributed|dtensor|ddp|rank|shard|replicate)\b",
        "performance": r"\b(perf|performance|slow|fast|memory|allocation)\b",
        "clarity": r"\b(confusing|unclear|readability|simpler|rationale)\b",
        "scope": r"\b(scope|unrelated|refactor|minimal|cleanup)\b",
    }
    counts: Counter[str] = Counter()
    for comment in comments:
        body = str(comment.get("body") or "").lower()
        for name, pattern in themes.items():
            if re.search(pattern, body):
                counts[name] += 1
    return counts


def _priority_bullets(themes: Counter[str]) -> list[str]:
    defaults = [
        "- Check whether the repro and tests cover the reported failure directly.",
        "- Look for correctness gaps, edge cases, and unintended behavior changes.",
        "- Ask for narrower changes when the diff includes unrelated cleanup.",
    ]
    if not themes:
        return defaults

    bullets: list[str] = []
    labels = {
        "tests": "Regression coverage and CI evidence matter; call out missing or weak tests.",
        "correctness": (
            "Prioritize concrete correctness issues and edge cases over "
            "style-only feedback."
        ),
        "compatibility": "Watch for BC/API semantic changes and ask for explicit rationale.",
        "distributed": (
            "Pay close attention to distributed tensor/DDP semantics and "
            "multi-rank behavior."
        ),
        "performance": "Flag avoidable overhead, extra allocations, and scalability risks.",
        "clarity": "Ask for clearer rationale when the code path or invariant is not obvious.",
        "scope": "Push back on unrelated refactors or broad changes outside the bug fix.",
    }
    for name, _count in themes.most_common(5):
        bullets.append(f"- {labels[name]}")
    for default in defaults:
        if len(bullets) >= 5:
            break
        if default not in bullets:
            bullets.append(default)
    return bullets


def _representative_examples(comments: list[dict]) -> list[dict]:
    candidates = [
        comment
        for comment in comments
        if 20 <= len(_clean_markdown(str(comment.get("body") or ""))) <= 1200
    ]
    candidates.sort(
        key=lambda item: (
            item.get("kind") != "inline",
            abs(_word_count(str(item.get("body") or "")) - 55),
        )
    )
    return candidates[:8]


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9_]+", text))


def _clean_markdown(text: str) -> str:
    text = re.sub(r"```.*?```", " code block ", text, flags=re.DOTALL)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."
