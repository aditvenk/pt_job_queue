from __future__ import annotations

import json
import os
import re
import subprocess

from ptq.domain.models import PtqError
from ptq.github_harness import call_github_harness, harness_available

_DEFAULT_GITHUB_ENV = {
    "https_proxy": "http://fwdproxy:8080",
    "http_proxy": "http://fwdproxy:8080",
    "no_proxy": (
        ".fbcdn.net,.facebook.com,.thefacebook.com,.tfbnw.net,.fb.com,"
        ".fburl.com,.facebook.net,.sb.fbsbx.com,localhost"
    ),
}


def github_cli_env() -> dict[str, str]:
    env = os.environ.copy()
    for key, value in _DEFAULT_GITHUB_ENV.items():
        env.setdefault(key, value)
    return env


def fetch_issue(issue_number: int, repo: str = "pytorch/pytorch") -> dict:
    if harness_available():
        try:
            data = call_github_harness(
                "fetch_issue",
                repo=repo,
                issue_number=issue_number,
            )
            return data if isinstance(data, dict) else {}
        except PtqError as exc:
            if "GitHub harness unavailable" not in str(exc):
                raise

    try:
        result = subprocess.run(
            [
                "gh",
                "issue",
                "view",
                str(issue_number),
                "--repo",
                repo,
                "--json",
                "title,body,comments,labels",
            ],
            env=github_cli_env(),
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else ""
        stdout = exc.stdout.strip() if exc.stdout else ""
        raise PtqError(
            f"Failed to fetch {repo}#{issue_number} with gh: "
            f"{stderr or stdout or exc}. If this environment cannot reach GitHub, "
            "start the external harness with "
            "`python scripts/github_harness.py serve` from a normal shell."
        ) from exc
    return json.loads(result.stdout)


def search_issues(
    query: str, *, repo: str = "pytorch/pytorch", limit: int = 20
) -> list[dict]:
    if harness_available():
        try:
            data = call_github_harness(
                "search_issues",
                repo=repo,
                query=query,
                limit=limit,
            )
            return data if isinstance(data, list) else []
        except PtqError as exc:
            if "GitHub harness unavailable" not in str(exc):
                raise

    try:
        result = subprocess.run(
            [
                "gh",
                "search",
                "issues",
                query,
                "--repo",
                repo,
                "--json",
                "number,title,labels,url",
                "--limit",
                str(limit),
            ],
            env=github_cli_env(),
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else ""
        stdout = exc.stdout.strip() if exc.stdout else ""
        raise PtqError(
            f"Failed to search {repo} issues with gh: {stderr or stdout or exc}. "
            "If this environment cannot reach GitHub, start the external harness "
            "with `python scripts/github_harness.py serve` from a normal shell."
        ) from exc
    return json.loads(result.stdout or "[]")


def extract_repro_script(
    issue_data: dict, import_hint: str = "import torch"
) -> str | None:
    body = issue_data.get("body", "") or ""
    for comment in issue_data.get("comments", []):
        body += "\n" + (comment.get("body", "") or "")

    code_blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", body, re.DOTALL)
    for block in code_blocks:
        if import_hint in block or "import torch" in block:
            return block.strip()
    return None


def format_issue_context(issue_data: dict, issue_number: int) -> str:
    title = issue_data.get("title", "")
    body = issue_data.get("body", "") or ""
    labels = [label.get("name", "") for label in issue_data.get("labels", [])]

    lines = [
        f"# Issue #{issue_number}: {title}",
        "",
        f"**Labels**: {', '.join(labels) if labels else 'none'}",
        "",
        "## Description",
        "",
        body,
    ]

    comments = issue_data.get("comments", [])
    if comments:
        lines.extend(["", "## Comments", ""])
        for i, comment in enumerate(comments, 1):
            author = comment.get("author", {}).get("login", "unknown")
            comment_body = comment.get("body", "")
            lines.extend([f"### Comment {i} by @{author}", "", comment_body, ""])

    return "\n".join(lines)
