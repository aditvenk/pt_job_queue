from __future__ import annotations

import re
from datetime import datetime, timedelta

from ptq.issue import fetch_issue, format_issue_context, search_issues
from ptq.orchestrator.models import Issue

_LABEL_RE = re.compile(r"label(?:ed)?\s+['\"]([^'\"]+)['\"]", re.IGNORECASE)
_LAST_DAYS_RE = re.compile(
    r"(?:filed|created)\s+in\s+the\s+last\s+(\d+)\s+days", re.IGNORECASE
)
_REACTIONS_RE = re.compile(r"more\s+than\s+(\d+)\s+thumbs?\s+up", re.IGNORECASE)
_ISSUE_URL_RE = re.compile(r"github\.com/[^/\s]+/[^/\s]+/issues/(\d+)")
_ISSUE_REF_RE = re.compile(r"(?:^|[\s#])(\d{4,})(?:\s|$)")


def extract_direct_issue_numbers(prompt: str) -> list[int]:
    numbers = [int(match) for match in _ISSUE_URL_RE.findall(prompt)]
    numbers.extend(int(match) for match in _ISSUE_REF_RE.findall(prompt))
    return list(dict.fromkeys(numbers))


def translate_issue_selection_prompt(prompt: str) -> str:
    lowered = prompt.lower()
    tokens = ["is:issue"]

    if "open" in lowered and "is:open" not in lowered:
        tokens.append("is:open")
    if "closed" in lowered and "is:closed" not in lowered:
        tokens.append("is:closed")
    if "no assignee" in lowered or "unassigned" in lowered:
        tokens.append("no:assignee")
    if "good first issue" in lowered and "label:" not in lowered:
        tokens.append('label:"good first issue"')
    if "bug" in lowered and "label:bug" not in lowered:
        tokens.append("label:bug")
    if "repro" in lowered or "reproduction" in lowered:
        tokens.append("repro")

    tokens.extend(
        re.findall(
            r"\b(?:is|label|created|updated|no|assignee|reactions):[^\s]+",
            prompt,
        )
    )

    for label in _LABEL_RE.findall(prompt):
        tokens.append(f'label:"{label}"')
    for quoted in re.findall(r"['\"]([^'\"]+)['\"]", prompt):
        if ":" in quoted:
            tokens.append(f'label:"{quoted}"')

    last_days = _LAST_DAYS_RE.search(prompt)
    if last_days:
        cutoff = datetime.now() - timedelta(days=int(last_days.group(1)))
        tokens.append(f"created:>={cutoff.date().isoformat()}")

    reactions = _REACTIONS_RE.search(prompt)
    if reactions:
        tokens.append(f"reactions:>={int(reactions.group(1)) + 1}")

    existing_search_terms = [
        word
        for word in re.findall(
            r"[A-Za-z_][A-Za-z0-9_:-]*",
            re.sub(r"['\"][^'\"]+['\"]", "", prompt),
        )
        if word.lower()
        not in {
            "open",
            "issues",
            "issue",
            "labeled",
            "label",
            "with",
            "a",
            "the",
            "filed",
            "created",
            "in",
            "last",
            "days",
            "all",
            "that",
            "mention",
            "more",
            "than",
            "thumbs",
            "up",
            "no",
            "assignee",
            "script",
        }
        and not word.lower().startswith("module:")
    ]
    if "incorrect output" in lowered:
        tokens.append('"incorrect output"')
    elif existing_search_terms and not any(":" in term for term in existing_search_terms):
        tokens.extend(existing_search_terms[:4])

    return " ".join(dict.fromkeys(tokens))


class IssueSelector:
    def __init__(self, github_repo: str):
        self.github_repo = github_repo

    def select(self, prompt: str, *, limit: int) -> list[Issue]:
        direct_numbers = extract_direct_issue_numbers(prompt)
        if direct_numbers:
            return [self._fetch_issue(number) for number in direct_numbers[:limit]]

        query = translate_issue_selection_prompt(prompt)
        rows = search_issues(query, repo=self.github_repo, limit=limit)
        issues: list[Issue] = []
        for row in rows:
            number = int(row["number"])
            issues.append(self._fetch_issue(number, fallback=row))
        return issues

    def _fetch_issue(self, number: int, fallback: dict | None = None) -> Issue:
        fallback = fallback or {}
        full = fetch_issue(number, repo=self.github_repo)
        return Issue(
            number=number,
            title=str(full.get("title") or fallback.get("title") or ""),
            body=format_issue_context(full, number),
            labels=[
                str(label.get("name") or "")
                for label in full.get("labels", fallback.get("labels", []))
            ],
            url=str(
                fallback.get("url")
                or f"https://github.com/{self.github_repo}/issues/{number}"
            ),
            raw=full,
        )
