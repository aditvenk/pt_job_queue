from __future__ import annotations

import json

REVIEW_JSON_SCHEMA = {
    "verdict": "approved | needs_revision | shelve",
    "score": "float from 0.0 to 1.0",
    "iteration": "integer",
    "repro_fidelity": "faithful | unfaithful | uncertain | from_issue",
    "comments": [
        {
            "file": "filename, general, repro, or test",
            "line": "integer or null",
            "comment": "review comment",
            "severity": "blocking | suggestion | nit",
        }
    ],
    "summary": "one paragraph summary for the solver",
}


def build_evaluation_prompt(
    *,
    issue_number: int,
    issue_body: str,
    repro_filename: str,
    repro_script: str,
    fix_diff: str,
    report_md: str,
    status_json: dict,
    iteration: int,
    max_iterations: int,
    approval_threshold: float,
    shelve_threshold: float,
    lint_output: str,
) -> str:
    return f"""\
You are evaluating a PyTorch issue-fixing agent's output. Return only valid JSON
matching this schema:

```json
{json.dumps(REVIEW_JSON_SCHEMA, indent=2)}
```

## Decision Rules

Step 0 - Repro Fidelity (BLOCKING, gate check)
- If the repro was agent-generated (`repro_<issue>_generated.py`), verify that it reproduces the EXACT failure described in the issue:
  - same error message / traceback as the reporter described
  - same API surface mentioned in the issue
  - minimal but faithful to the reporter's scenario
  - cannot pass/fail for reasons unrelated to the reported bug
- If the repro was extracted directly from the issue (`repro_<issue>.py`), check only that it appears faithfully transcribed and set `repro_fidelity` to `from_issue`.
- If the repro is unfaithful, immediately return `verdict="needs_revision"`, `score=0.0`, and comments telling the solver to fix the repro first. Do not evaluate the fix.

Step 1 - Fix Correctness (weight: 0.4)
- Does the fix address the root cause described in report.md?
- Does the repro pass after the fix is applied?
- Are obvious edge cases missed?

Step 2 - Scope and Minimality (weight: 0.2)
- Is the change minimal and focused?
- No unrelated refactoring?

Step 3 - Test Coverage (weight: 0.2)
- Are new/modified tests included?
- Do they cover the fix and relevant edge cases?

Step 4 - Code Quality (weight: 0.2)
- Style matches PyTorch conventions?
- No regressions introduced?
- Treat `lintrunner -m origin/main` output as part of the evidence when present.

## Scoring Logic

- If `repro_fidelity == "unfaithful"`: `score = 0.0`, `verdict = "needs_revision"`.
- Otherwise score is the weighted average of Steps 1-4.
- If `score >= {approval_threshold}`: `verdict = "approved"`.
- If `iteration >= max_iterations` and `score < {shelve_threshold}`: `verdict = "shelve"`.
- Else: `verdict = "needs_revision"`.

## Evaluation Context

- Issue: pytorch/pytorch#{issue_number}
- Iteration: {iteration}
- Max iterations: {max_iterations}
- Approval threshold: {approval_threshold}
- Shelve threshold: {shelve_threshold}
- Repro file: {repro_filename or "(missing)"}

### Solver status.json

```json
{json.dumps(status_json, indent=2, sort_keys=True)}
```

### GitHub Issue Body

{issue_body}

### Repro Script

```python
{repro_script}
```

### Solver report.md

```markdown
{report_md}
```

### fix.diff

```diff
{fix_diff}
```

### Lint Output

```text
{lint_output}
```
"""
