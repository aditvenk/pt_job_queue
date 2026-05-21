from __future__ import annotations

import json
import logging
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

from ptq.agent import (
    DEFAULT_MESSAGE,
    build_adhoc_prompt,
    build_system_prompt,
)
from ptq.agents import RunContext, get_agent
from ptq.application.job_context import write_job_context
from ptq.application.venv_service import (
    ProgressCallback,
    _noop_progress,
    _setup_job_venv,
    rewrite_torch_editable_to_worktree,
)
from ptq.domain.models import JobRecord, PtqError, RunRequest
from ptq.domain.policies import make_job_id
from ptq.infrastructure.job_repository import JobRepository
from ptq.issue import extract_repro_script
from ptq.repo_profiles import get_profile
from ptq.ssh import Backend, RemoteBackend
from ptq.workspace import deploy_scripts

log = logging.getLogger("ptq.run")


@contextmanager
def _timed(label: str, progress: ProgressCallback):
    t0 = time.monotonic()
    yield
    progress(f"  {label}: {time.monotonic() - t0:.1f}s")


def _validate_workspace(
    backend: Backend, workspace: str, repo: str = "pytorch"
) -> None:
    profile = get_profile(repo)
    result = backend.run(f"test -d {workspace}/{profile.dir_name}/.git", check=False)
    if result.returncode != 0:
        raise PtqError(
            f"Workspace broken: {workspace}/{profile.dir_name}/.git missing. Re-run: ptq setup"
        )


def _ensure_pytorch_support_worktree(
    backend: Backend,
    workspace: str,
    job_dir: str,
    *,
    verbose: bool = False,
    progress: ProgressCallback = _noop_progress,
) -> str | None:
    """Create an isolated PyTorch worktree for add-on repo jobs.

    TorchTitan issues often bottom out in PyTorch distributed/DTensor code.
    Agents may edit PyTorch when the root cause is there, but those edits must
    stay inside the PTQ job directory so they are isolated and diffable.
    """
    pytorch_path = f"{job_dir}/pytorch"
    exists = backend.run(
        f"test -d {pytorch_path}/.git || test -f {pytorch_path}/.git",
        check=False,
    )
    if exists.returncode == 0:
        return pytorch_path

    base = backend.run(f"test -d {workspace}/pytorch/.git", check=False)
    if base.returncode != 0:
        raise PtqError(
            f"Workspace broken: {workspace}/pytorch/.git missing. "
            "Re-run: ptq setup --extras torchtitan"
        )

    progress("Creating PyTorch support worktree...")
    with _timed("pytorch worktree creation", progress):
        backend.run(
            f"cd {workspace}/pytorch && "
            f"{workspace}/.venv/bin/python tools/create_worktree.py create pytorch "
            f"--parent-dir {job_dir} --commit HEAD",
            stream=verbose,
        )
    return pytorch_path


def _stamp_worklog_header(
    backend: Backend, job_dir: str, run_number: int, message: str | None
) -> None:
    lines = ["", "", f"## Run {run_number}", ""]
    if message:
        lines.append(f"> **User:** {message}")
        lines.append("")
    header = "\n".join(lines)
    backend.run(
        f"cat >> {job_dir}/worklog.md << 'WORKLOG_STAMP_EOF'\n{header}\nWORKLOG_STAMP_EOF"
    )


def _build_prior_context(backend: Backend, job_dir: str, run_number: int) -> str:
    worklog = backend.run(f"cat {job_dir}/worklog.md", check=False)
    report = backend.run(f"cat {job_dir}/report.md", check=False)

    worklog_content = worklog.stdout.strip() if worklog.returncode == 0 else ""
    report_content = report.stdout.strip() if report.returncode == 0 else ""

    if not worklog_content and not report_content:
        return ""

    sections = [
        "\n\n## Prior Run Context\n",
        "The following is from a previous investigation attempt on this issue. "
        "Use it to avoid repeating work and to build on what was already found.\n",
    ]
    if worklog_content:
        sections.append(f"### Previous Worklog\n{worklog_content}\n")
    if report_content:
        sections.append(f"### Previous Report\n{report_content}\n")

    sections.append(
        f"\n## Continuation Instructions\n"
        f"This is **run {run_number}**. A `## Run {run_number}` section (with the "
        f"user's steering message) has already been appended to the worklog. You MUST:\n"
        f"1. Append your findings, analysis, or changes under that section before "
        f"you finish — even if the user's message was a question rather than a fix request.\n"
        f"2. If you made any code changes, run `spin fixlint`, regenerate `fix.diff`, "
        f"and update `report.md`.\n"
        f"3. If the user asked an analytical question, update `report.md` with your "
        f"findings as a new section.\n"
        f"\nEvery run must leave a trace in the worklog and artifacts.\n"
    )
    return "\n".join(sections)


def _build_review_context(review_feedback_json: str | None) -> str:
    if not review_feedback_json:
        return ""
    return (
        "\n\n## Review Feedback\n"
        "The previous solver attempt may have been reviewed by automated evaluators "
        "or by reviewers on a draft GitHub PR, including human reviewers and "
        "review bots such as Claude. Treat this as structured blocking feedback "
        "for the current run. Address blocking comments and GitHub PR comments "
        "first, especially repro-fidelity comments, and update artifacts "
        "when done. If you address a specific GitHub PR comment from "
        "`github_pr_feedback.comments`, add it to `status.json` as "
        "`resolved_pr_comments` with its `comment_id`, `kind`, and a very short "
        "`resolution` explaining how it was fixed in one sentence. PTQ will reply to those "
        "specific PR comments after updating the draft PR. If "
        "`github_pr_feedback.ci_failures` is present, review the failing CI checks "
        "and any included `failed_log_excerpt` fields. Follow linked logs when "
        "the excerpt is insufficient. Determine whether each failure is caused "
        "by this PR; fix PR-caused failures, and document unrelated or "
        "infrastructure failures in `report.md`.\n\n"
        "```json\n"
        f"{review_feedback_json.strip()}\n"
        "```\n"
    )


def launch(
    repo: JobRepository,
    backend: Backend,
    request: RunRequest,
    *,
    on_progress: ProgressCallback | None = None,
) -> str:
    """Launch an agent job. Returns job_id."""
    progress = on_progress or _noop_progress
    agent = get_agent(request.agent_type)
    workspace = backend.workspace
    is_adhoc = request.issue_number is None
    issue_number = request.issue_number
    issue_data = request.issue_data
    repo_name = request.repo
    profile = get_profile(repo_name)

    if not is_adhoc and (issue_number is None or issue_data is None):
        raise PtqError("Issue runs require both issue number and issue data.")

    if request.existing_job_id:
        job_id = request.existing_job_id
        run_number = repo.increment_run(
            job_id,
            agent_type=request.agent_type,
            model=request.model,
            thinking=request.thinking,
        )
        label = f"issue #{request.issue_number}" if request.issue_number else "adhoc"
        progress(f"Job {job_id} — {label} (run {run_number})")
        existing = job_id
    elif is_adhoc:
        existing = None
        job_id = make_job_id(message=request.message, repo=repo_name)
        run_number = 1
        progress(f"Job {job_id} — adhoc (run 1)")
    else:
        assert issue_number is not None
        existing = repo.find_by_issue(
            issue_number,
            machine=request.machine,
            local=request.local,
            repo=repo_name,
        )
        if existing:
            job_id = existing
            run_number = repo.increment_run(
                job_id,
                agent_type=request.agent_type,
                model=request.model,
                thinking=request.thinking,
            )
            progress(f"Job {job_id} — issue #{issue_number} (run {run_number})")
        else:
            job_id = make_job_id(issue_number, repo=repo_name)
            run_number = 1
            progress(f"Job {job_id} — issue #{issue_number} (run 1)")

    job_dir = f"{workspace}/jobs/{job_id}"
    worktree_path = f"{job_dir}/{profile.dir_name}"

    if existing:
        _validate_workspace(backend, workspace, repo_name)

    backend.run(f"mkdir -p {job_dir}")

    if not existing:
        repo.save(
            JobRecord(
                job_id=job_id,
                issue=request.issue_number,
                runs=run_number,
                agent=request.agent_type,
                model=request.model,
                thinking=request.thinking,
                machine=request.machine,
                local=request.local,
                workspace=workspace,
                initializing=True,
                name=request.name,
                repo=repo_name,
            )
        )
    elif request.name:
        repo.save_name(job_id, request.name)

    deploy_scripts(backend)

    worktree_exists = backend.run(
        f"test -d {worktree_path}/.git || test -f {worktree_path}/.git", check=False
    )
    venv_exists = backend.run(f"test -d {job_dir}/.venv/bin", check=False)
    if worktree_exists.returncode == 0 and venv_exists.returncode == 0:
        progress("Reusing existing worktree.")
    else:
        if worktree_exists.returncode != 0:
            if profile.uses_custom_worktree_tool:
                progress("Creating worktree with submodules...")
                with _timed("worktree creation", progress):
                    backend.run(
                        f"cd {workspace}/pytorch && {workspace}/.venv/bin/python tools/create_worktree.py create pytorch "
                        f"--parent-dir {job_dir} --commit HEAD",
                        stream=request.verbose,
                    )
            else:
                progress(f"Creating {profile.name} worktree...")
                with _timed("worktree creation", progress):
                    branch = f"ptq-{job_id}"
                    backend.run(
                        f"cd {workspace}/{profile.dir_name} && "
                        f"git worktree add -b {branch} {worktree_path} HEAD",
                        stream=request.verbose,
                    )
        if repo_name != "pytorch":
            _ensure_pytorch_support_worktree(
                backend,
                workspace,
                job_dir,
                verbose=request.verbose,
                progress=progress,
            )
        if venv_exists.returncode != 0:
            progress("Creating per-job venv...")
            from ptq.config import load_config

            _setup_job_venv(
                backend,
                job_dir,
                worktree_path,
                verbose=request.verbose,
                progress=progress,
                build_env_prefix=load_config().build_env_prefix(),
                repo=repo_name,
            )

    if repo_name != "pytorch":
        pytorch_support_worktree = _ensure_pytorch_support_worktree(
            backend,
            workspace,
            job_dir,
            verbose=request.verbose,
            progress=progress,
        )
        if pytorch_support_worktree:
            rewrite_torch_editable_to_worktree(
                backend,
                job_dir,
                pytorch_support_worktree,
                progress=progress,
            )

    write_job_context(
        backend,
        job_id=job_id,
        workspace=workspace,
        repo=repo_name,
        name=request.name,
    )

    if is_adhoc:
        system_prompt = build_adhoc_prompt(
            request.message or DEFAULT_MESSAGE, job_id, workspace, repo=repo_name
        )
    else:
        assert issue_number is not None
        assert issue_data is not None
        system_prompt = build_system_prompt(
            issue_data, issue_number, job_id, workspace, repo=repo_name
        )

    if existing:
        prior_context = _build_prior_context(backend, job_dir, run_number)
        if prior_context:
            system_prompt += prior_context
            progress("Loaded prior run context (worklog/report).")

    review_context = _build_review_context(request.review_feedback_json)
    if review_context:
        system_prompt += review_context
        progress("Loaded review feedback.")

    _stamp_worklog_header(backend, job_dir, run_number, request.message)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(system_prompt)
        prompt_tmp = Path(f.name)

    prompt_remote = f"{job_dir}/system_prompt.md"
    backend.copy_to(prompt_tmp, prompt_remote)
    prompt_tmp.unlink()

    progress("Configuring agent workspace...")
    agent.setup_workspace(backend, worktree_path, job_dir, workspace, prompt_remote)

    if not is_adhoc:
        assert issue_data is not None
        repro = extract_repro_script(issue_data, import_hint=profile.repro_import_hint)
        if repro:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(repro)
                repro_tmp = Path(f.name)
            backend.copy_to(repro_tmp, f"{job_dir}/repro_{issue_number}.py")
            repro_tmp.unlink()
            progress("Extracted and uploaded repro script.")
        else:
            progress("No repro script found in issue — agent will write one.")

    if is_adhoc or existing:
        agent_message = request.message or DEFAULT_MESSAGE
    elif request.message:
        agent_message = f"{DEFAULT_MESSAGE}\n\nAdditional context: {request.message}"
    else:
        agent_message = DEFAULT_MESSAGE

    if request.review_feedback_json:
        agent_message += (
            "\n\nReview feedback is included in the system prompt. "
            "Revise the attempt according to that feedback before finishing."
        )

    log_file = f"{job_dir}/{agent.log_filename(run_number)}"
    unbuffer = "stdbuf -oL " if isinstance(backend, RemoteBackend) else ""
    ctx = RunContext(
        worktree_path=worktree_path,
        job_dir=job_dir,
        message=agent_message,
        model=request.model,
        thinking=request.thinking,
        max_turns=request.max_turns,
        system_prompt_file=prompt_remote,
        unbuffer_prefix=unbuffer,
    )
    agent_cmd = agent.build_cmd(ctx)

    progress(
        f"Launching {agent.name} agent ({'local' if request.local else request.machine})..."
    )
    backend.run(f"mkdir -p {job_dir}/agent_logs && touch {log_file}")
    pid = backend.launch_background(agent_cmd, log_file)
    repo.save_pid(job_id, pid)

    return job_id


def finalize_run(backend: Backend, job_id: str, job: JobRecord) -> None:
    """Extract agent summary from log and append to worklog if the agent didn't."""
    ws = backend.workspace
    job_dir = f"{ws}/jobs/{job_id}"
    run_number = job.runs
    agent = get_agent(job.agent)
    log_file = f"{job_dir}/{agent.log_filename(run_number)}"

    worklog_result = backend.run(f"cat {job_dir}/worklog.md", check=False)

    run_header = f"## Run {run_number}"
    worklog_text = worklog_result.stdout if worklog_result.returncode == 0 else ""
    header_pos = worklog_text.rfind(run_header)
    section = ""
    if header_pos != -1:
        next_header = worklog_text.find("\n## Run ", header_pos + len(run_header))
        section = (
            worklog_text[header_pos:next_header]
            if next_header != -1
            else worklog_text[header_pos:]
        )

    log_result = backend.run(f"cat {log_file}", check=False)
    log_text = log_result.stdout if log_result.returncode == 0 else ""

    if section:
        has_content = False
        for line in section.splitlines()[1:]:
            stripped = line.strip()
            if stripped and not stripped.startswith("> **User:**"):
                has_content = True
                break
        if not has_content and log_text.strip():
            summary = agent.extract_summary(log_text)
            if summary:
                entry_text = f"\n### Agent Summary (auto-extracted)\n{summary}\n"
                backend.run(
                    f"cat >> {job_dir}/worklog.md << 'WORKLOG_AUTO_EOF'\n{entry_text}\nWORKLOG_AUTO_EOF"
                )
                section += entry_text

    _ensure_report_artifact(
        backend,
        job_id=job_id,
        job_dir=job_dir,
        run_number=run_number,
        section=section,
        log_text=log_text,
    )


def _ensure_report_artifact(
    backend: Backend,
    *,
    job_id: str,
    job_dir: str,
    run_number: int,
    section: str,
    log_text: str,
) -> None:
    if backend.run(f"test -s {job_dir}/report.md", check=False).returncode == 0:
        return

    terminal = _terminal_reason_from_log(log_text)
    summary = _fallback_summary_from_log(log_text)
    if not section.strip():
        section = "(No worklog entries were written before the agent stopped.)"
    if not summary:
        summary = "(No assistant summary could be extracted from the agent log.)"
    terminal_text = terminal or "unknown"
    report = f"""# PTQ Fallback Report

The solver stopped before writing `report.md`. PTQ generated this fallback
report from the worklog and agent log so the run still has an inspectable
artifact.

- Job: `{job_id}`
- Run: {run_number}
- Solver terminal reason: {terminal_text}

## Last Worklog Section

```markdown
{section.strip()}
```

## Agent Log Summary

{summary}
"""
    backend.run(
        f"cat > {job_dir}/report.md << 'PTQ_FALLBACK_REPORT_EOF'\n"
        f"{report}\n"
        f"PTQ_FALLBACK_REPORT_EOF"
    )


def _terminal_reason_from_log(log_text: str) -> str:
    for line in reversed(log_text.splitlines()):
        if '"type":"result"' not in line and '"type": "result"' not in line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        subtype = str(payload.get("subtype") or "")
        terminal_reason = str(payload.get("terminal_reason") or "")
        errors = payload.get("errors")
        error_text = ""
        if isinstance(errors, list) and errors:
            error_text = "; ".join(str(error) for error in errors[:2])
        return " / ".join(
            part for part in (subtype, terminal_reason, error_text) if part
        )
    return ""


def _fallback_summary_from_log(log_text: str) -> str:
    if not log_text.strip():
        return ""
    try:
        for line in reversed(log_text.splitlines()):
            data = json.loads(line)
            message = data.get("message", {})
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            content = message.get("content", [])
            if not isinstance(content, list):
                continue
            texts = [
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            ]
            text = "\n".join(texts).strip()
            if text:
                return text
    except (json.JSONDecodeError, TypeError):
        pass
    return ""
