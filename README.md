# pt_job_queue fork: PyTorch issue hill-climbing

This fork adds an agentic hill-climbing loop on top of PTQ for working through
PyTorch GitHub issues:

- **PTQ solver**: launches an agent in an isolated PyTorch worktree.
- **Evaluator**: reviews the solver output with multiple reviewer models.
- **Orchestrator**: selects issues, runs solver/evaluator iterations in parallel,
  and can update a draft PR for human review.

For general PTQ setup, workspace management, dashboard usage, and base solver
behavior, use the upstream project documentation:

https://github.com/drisspg/pt_job_queue

This README focuses on the features added in this fork.

## Quick Start

Run all commands from this repo with `uv run ptq ...`.

```bash
# One-time local workspace setup, if you have not already done it.
uv run ptq setup --local

# Include supported add-on repos when you want to orchestrate them.
uv run ptq setup --local --extras torchtitan

# Dry-run issue selection without launching agents.
uv run ptq orchestrate \
  --issue 76449 \
  --parallel 1 \
  --machine localhost \
  --dry-run

# Run one issue through one solver/evaluator iteration.
uv run ptq orchestrate \
  --issue 76449 \
  --parallel 1 \
  --max-iterations 1 \
  --machine localhost
```

If the issue is approved and you want PTQ to push or update a draft PR, add
`--pr`:

```bash
uv run ptq orchestrate \
  --issue 76449 \
  --parallel 1 \
  --machine localhost \
  --pr
```

PRs created by this fork are always draft PRs.

## Orchestrator

The orchestrator manages the full issue loop:

1. Select GitHub issues from a prompt or explicit issue number.
2. Launch PTQ solver jobs in isolated git worktrees.
3. Run the evaluator on `report.md`, `fix.diff`, `status.json`, and the repro.
4. Feed evaluator feedback back into the next solver iteration.
5. Stop when approved, shelved, or max iterations is reached.
6. Optionally push or update a draft PR with `--pr`.

Examples:

```bash
# Run one explicit issue.
uv run ptq orchestrate --issue 166156 --machine localhost

# Run one explicit issue from another supported repo profile.
uv run ptq orchestrate --repo torchtitan --issue 1234 --machine localhost

# Run one explicit issue with initial solver guidance.
uv run ptq orchestrate \
  --issue 166156 \
  --machine localhost \
  -m "Start by checking whether the wrapper drops custom tensor metadata."

# Run one explicit issue and update/create a draft PR after approval.
uv run ptq orchestrate --issue 166156 --machine localhost --pr

# Push a draft PR, then keep watching it for review comments or CI failures.
uv run ptq orchestrate --issue 166156 --machine localhost --watch-pr

# Select issues from natural-language criteria.
uv run ptq orchestrate \
  --prompt "open issues labeled 'module: nn' with a repro script" \
  --parallel 4 \
  --machine my-gpu-box

# Preview what would be selected.
uv run ptq orchestrate \
  --prompt "good first issue bugs mentioning incorrect output" \
  --dry-run
```

Prompt-based selection uses `[orchestrator].max_issues` from
`~/.ptq/config.toml` as the selection cap. Explicit `--issue` runs always select
one issue.

Useful flags:

| Flag | Meaning |
| --- | --- |
| `--repo NAME` | PTQ repo profile to run against, such as `pytorch` or `torchtitan`; default comes from config. |
| `--issue N` | Run one explicit issue in the selected repo. |
| `--prompt TEXT` | Natural-language GitHub issue selection criteria. |
| `--message/-m TEXT` | Initial solver guidance to include alongside the issue. |
| `--parallel N` | Number of concurrent solver/evaluator loops. |
| `--max-iterations N` | Max solver/evaluator iterations per issue. |
| `--machine NAME` | Remote machine, or `localhost`/`local` for local worktrees. |
| `--follow/--no-follow` | Stream solver activity in the orchestrator console; default is `--follow`. |
| `--poll-seconds N` | Poll interval while waiting on solver jobs. |
| `--dry-run` | Select issues without launching solver jobs. |
| `--pr` | Push/create or update a draft PR after approval; default is no PR push. |
| `--watch-pr` | Implies `--pr`; keep polling the draft PR for review/CI activity until the PR closes or 24h pass with no activity. |

View orchestrator history:

```bash
uv run ptq orchestrate-results
```

The JSONL log is written to:

```text
~/.ptq/orchestrator/runs.jsonl
```

## Evaluator

The evaluator runs inline and does not require GPU access. It reviews the solver
artifacts and returns structured feedback:

- `verdict`: `approved`, `needs_revision`, or `shelve`
- `score`: `0.0` to `1.0`
- component scores for fix correctness, scope/minimality, test coverage, and
  code quality
- repro fidelity status
- blocking/suggestion/nit comments
- reviewer-specific scores and component scores

Each reviewer scores the same four rubric components:

- `fix_correctness`
- `scope_minimality`
- `test_coverage`
- `code_quality`

The evaluator treats those component scores as the source of truth. The
aggregate review records the lowest score seen for each component across all
active reviewers, so the solver can see whether the weak point was correctness,
scope, tests, or quality. The legacy scalar `score` is kept for CLI summaries
and compatibility, but it is derived as the weakest component rather than as a
weighted average.

By default every diff is first reviewed by two reviewer models:

- `gpt-5.5`
- `claude-opus-4-7`

Profile-backed reviewers run as a second stage only after those agent reviewers
approve. If any profile-backed reviewer requests revision, that issue escalates
into full-review mode, and every later iteration runs all reviewers together.
The diff is not considered ready for human review unless every reviewer in the
active stage scores it above the configured approval threshold.

You can add a profile-backed reviewer as another required evaluator:

```bash
uv run ptq orchestrate \
  --add-evaluator aditvenk-style \
  --profile aditvenk \
  --agent gpt-5.5
```

If `~/.ptq/evaluator_profiles/aditvenk.md` does not exist, orchestrate scans
recent GitHub PRs reviewed by `aditvenk` and creates it first. The added
evaluator is also saved to `~/.ptq/config.toml`, so later orchestrate runs reuse
it without passing `--add-evaluator` again.

## Repro Gate

The evaluator has a blocking Step 0 repro-fidelity check.

If the solver generated a repro, the evaluator checks that it matches the issue:

- same failure
- same API surface
- same traceback or error behavior where available
- minimal but faithful scenario
- no unrelated pass/fail reason

If the repro is unfaithful, the evaluator returns `needs_revision` with score
`0.0`, and the next solver run is told to fix the repro first.

Solver repro naming convention:

```text
repro_<issue>.py             # extracted directly from the issue
repro_<issue>_generated.py   # generated by the solver
```

## Solver Artifacts

Solver runs are expected to write:

```text
report.md
fix.diff
status.json
```

`status.json` drives evaluator/orchestrator behavior. Example:

```json
{
  "state": "ready_for_review",
  "iteration": 1,
  "repro_source": "generated",
  "repro_file": "repro_166156_generated.py",
  "repro_passes_before_fix": false,
  "repro_passes_after_fix": true,
  "how_repro_was_run": "~/.ptq_workspace/jobs/<job>/.venv/bin/python ~/.ptq_workspace/jobs/<job>/repro_166156_generated.py",
  "files_changed": [
    "torch/distributed/tensor/_api.py",
    "test/distributed/tensor/test_dtensor.py"
  ],
  "pr_title": "Preserve Parameter markers in DTensor.to_local",
  "summary": "Preserve Parameter-ness when DTensor.to_local() is called on Parameter(DTensor).",
  "resolved_pr_comments": []
}
```

If the solver cannot reproduce the issue, it should still write `report.md`, an
empty `fix.diff`, and `status.json` with:

```json
{
  "state": "not_reproducible",
  "repro_passes_before_fix": true,
  "repro_passes_after_fix": null,
  "files_changed": [],
  "summary": "Could not reproduce the reported failure"
}
```

## Draft PR Flow

When `--pr` is set and the evaluator approves the fix, PTQ will:

1. Check out branch `ptq/<issue>`.
2. Commit the solver changes with a detailed commit message.
3. Include `Fixes #<issue>` in the commit footer.
4. Push the branch.
5. Create or update a draft PR.
6. Keep the PR in draft state.

The PR title is derived from `status.json["pr_title"]` when present, otherwise
from the solver summary/report. It should be a one-line summary of the code
change, not the original GitHub issue title and not an issue-number reference.

Example:

```bash
uv run ptq orchestrate --issue 166156 --machine localhost --pr
```

PR description format:

- agent report summary
- root cause
- fix
- repro command and repro script
- testing
- backward compatibility analysis, when the solver report includes it
- files changed
- `Fixes #<issue>`

The repro is included in the PR body. PTQ no longer posts a separate repro
comment.

## PR Feedback Loop

If a draft PR already exists, rerunning orchestrate will find it by stored PR URL
or by branch name `ptq/<issue>`.

Before launching the solver, PTQ reads:

- PR conversation comments
- inline review comments
- review bodies
- review-bot comments, including Claude or other bots
- failing PR checks
- bounded failed GitHub Actions log excerpts, when available

The feedback is injected into the solver context as `github_pr_feedback`.

Rerun example:

```bash
uv run ptq orchestrate \
  --issue 166156 \
  --parallel 1 \
  --machine localhost \
  --pr
```

The solver should fix PR-caused feedback and CI failures. If a CI failure is
unrelated, flaky, or infrastructure-caused, it should document that in
`report.md` rather than making speculative code changes.

When the solver resolves a specific PR comment, it can list that comment in
`status.json`:

```json
{
  "resolved_pr_comments": [
    {
      "comment_id": 123456789,
      "kind": "inline",
      "resolution": "Added no_grad coverage for the Parameter(DTensor) path."
    }
  ]
}
```

After updating the draft PR, PTQ replies to those comments with a short bot
message:

```text
[bot] Resolved in the latest update. Added no_grad coverage for the Parameter(DTensor) path.
```

PTQ adds hidden markers to avoid posting duplicate resolution replies on later
runs. PTQ-managed comments are filtered out of future solver feedback, while
review comments from other bots are preserved.

## Watching Solver Progress

`--follow` streams solver events through the orchestrator and is enabled by default.
You can also inspect a job directly:

```bash
uv run ptq peek 20260518-pytorch-76449 --log 80
uv run ptq watch 20260518-pytorch-76449
```

`peek` shows the worklog and recent agent log events. `watch` streams an
existing job until it stops.

## GitHub Harness

Normal command-line use does not need the harness. If `gh auth status` and
GitHub network access work in your shell, PTQ runs `gh` directly.

The harness exists for environments where the agent process cannot access
GitHub directly. Start it from a normal shell:

```bash
python scripts/github_harness.py serve
```

It listens on:

```text
~/.ptq/github_harness.sock
```

PTQ auto-detects the socket. If the socket is stale or refused, PTQ falls back to
direct `gh` execution for normal shell use.

Stop and remove the harness socket:

```bash
pkill -f "scripts/github_harness.py serve"
rm -f ~/.ptq/github_harness.sock
```

## Configuration

`~/.ptq/config.toml` supports these fork-specific sections:

```toml
[orchestrator]
repo = "pytorch"
github_repo = "pytorch/pytorch"
issue_selection_prompt = "open issues labeled 'module: nn' with a repro script, filed in the last 30 days"
max_issues = 20
parallel = 4
max_iterations = 10
approval_threshold = 0.8
machine = "localhost"
watch_pr = false
watch_pr_interval_seconds = 300
watch_pr_idle_hours = 24

[evaluator]
models = ["gpt-5.5", "claude-opus-4-7"]
# additional_reviewers = [
#   { name = "aditvenk-style", profile = "aditvenk", model = "gpt-5.5" },
# ]
approval_threshold = 0.8
shelve_threshold = 0.3
max_iterations = 10
```

The orchestrator uses the default solver agent/model from the normal PTQ
`[defaults]` and `[models.*]` settings.

Example solver defaults:

```toml
[defaults]
agent = "claude"
max_turns = 100

[models.claude]
default = "opus"

[models.pi]
default = "openai-codex/gpt-5.5"
thinking = "high"
```

## Results and Artifacts

Fetch artifacts for a job:

```bash
uv run ptq results 20260518-pytorch-76449
```

Fetched artifacts include:

- `report.md`
- `fix.diff`
- `worklog.md`
- `status.json`
- `review.json`
- issue-specific repro scripts
- the latest agent log

PTQ-managed worktrees live under:

```text
~/.ptq_workspace/jobs/<job-id>/pytorch/
```

The per-job venv is:

```text
~/.ptq_workspace/jobs/<job-id>/.venv/
```

## Development

Run tests with the repo convention:

```bash
uv run --extra dev pytest
```

Focused tests for this fork:

```bash
uv run --extra dev pytest \
  tests/test_evaluator.py \
  tests/test_repro_validator.py \
  tests/test_orchestrator.py \
  tests/test_pr.py \
  tests/test_issue.py \
  tests/test_cli.py
```

## Added Modules

```text
ptq/evaluator/
  models.py          # ReviewResult, ReviewComment
  rubric.py          # Structured evaluator prompt
  repro_validator.py # Repro-fidelity gate
  evaluator.py       # Multi-reviewer evaluator

ptq/orchestrator/
  models.py          # OrchestratorConfig, Issue, SolveResult
  issue_selector.py  # GitHub issue query translation/fetching
  orchestrator.py    # Hill-climbing loop
  reporter.py        # JSONL result logging

scripts/github_harness.py
  Optional external gh execution harness.
```
