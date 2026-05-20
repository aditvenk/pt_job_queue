from __future__ import annotations

import json

REVIEW_JSON_SCHEMA = {
    "verdict": "approved | needs_revision | shelve",
    "score": "weakest component score from 0.0 to 1.0",
    "component_scores": {
        "fix_correctness": "float from 0.0 to 1.0",
        "scope_minimality": "float from 0.0 to 1.0",
        "test_coverage": "float from 0.0 to 1.0",
        "code_quality": "float from 0.0 to 1.0",
    },
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
    user_message: str,
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

Step 1 - Fix Correctness
- Does the fix address the root cause described in report.md?
- Does the repro pass after the fix is applied?
- Are obvious edge cases missed?
- Does the solver output satisfy the human task/message, if one was provided?
- Return this as `component_scores.fix_correctness`.

Step 2 - Scope and Minimality
- Is the change minimal and focused?
- No unrelated refactoring?
- Is the change within the scope requested by the human task/message?
- Return this as `component_scores.scope_minimality`.

Step 3 - Test Coverage
- Are new/modified tests included?
- Do they cover the fix and relevant edge cases?
- Return this as `component_scores.test_coverage`.

Step 4 - Code Quality
- Style matches PyTorch conventions?
- No regressions introduced?
- Treat `lintrunner -m origin/main` output as part of the evidence when present.
- Return this as `component_scores.code_quality`.

## Scoring Logic

- If `repro_fidelity == "unfaithful"`: `score = 0.0`, `verdict = "needs_revision"`.
- Otherwise provide all four `component_scores`; `score` should be the lowest
  component score.
- The evaluator implementation recomputes `score` from `component_scores`, so
  make the component scores the source of truth.
- If every component score is >= {approval_threshold}: `verdict = "approved"`.
- If `iteration >= max_iterations` and the weakest component score is < {shelve_threshold}: `verdict = "shelve"`.
- Else: `verdict = "needs_revision"`.

## Evaluation Context

- Issue: pytorch/pytorch#{issue_number}
- Iteration: {iteration}
- Max iterations: {max_iterations}
- Approval threshold: {approval_threshold}
- Shelve threshold: {shelve_threshold}
- Repro file: {repro_filename or "(missing)"}

### Human Task / Message Prior

The user may have provided an additional message when launching orchestrate.
Treat this as task context and a constraint on what the solver should do.
If the message asks for analysis, explanation, or a narrow investigation rather
than a code fix, evaluate whether the solver respected that request. Code
changes outside the requested scope should lower correctness and/or
scope_minimality and should receive blocking feedback.

{user_message or "(none provided)"}

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
