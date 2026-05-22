from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from ptq.agent import _clean, _indent, _truncate
from ptq.agents import StreamEvent, get_agent
from ptq.domain.models import JobRecord, JobStatus, PtqError, RebaseState, RunRequest

app = typer.Typer(
    name="ptq",
    help="PyTorch Job Queue — dispatch AI agents to fix issues in PyTorch and add-on repos.",
)
console = Console()

_GITHUB_ISSUE_URL_RE = re.compile(
    r"^https?://github\.com/([^/\s]+)/([^/\s]+)/issues/(\d+)(?:[/?#].*)?$",
    re.IGNORECASE,
)


def _parse_issue_reference(value: int | str | None) -> tuple[int | None, str | None]:
    if value is None:
        return None, None
    if isinstance(value, int):
        return value, None
    text = value.strip()
    if text.isdigit():
        return int(text), None
    match = _GITHUB_ISSUE_URL_RE.match(text)
    if match:
        owner, repo, number = match.groups()
        return int(number), f"{owner}/{repo}"
    raise typer.BadParameter(
        "--issue must be an issue number or GitHub issue URL like "
        "https://github.com/pytorch/pytorch/issues/184746"
    )


def _repo_for_issue_url(issue_github_repo: str, explicit_repo: str | None) -> str:
    from ptq.repo_profiles import get_profile, profile_name_for_github_repo

    inferred_repo = profile_name_for_github_repo(issue_github_repo)
    if inferred_repo is None:
        raise typer.BadParameter(
            f"GitHub repo '{issue_github_repo}' from --issue does not match any "
            "configured PTQ repo profile. Add it under [repos.<name>] in "
            "~/.ptq/config.toml first."
        )
    if explicit_repo and explicit_repo != inferred_repo:
        explicit_profile = get_profile(explicit_repo)
        if profile_name_for_github_repo(explicit_profile.github_repo) != inferred_repo:
            raise typer.BadParameter(
                f"--issue points at {issue_github_repo}, but --repo is "
                f"{explicit_repo} ({explicit_profile.github_repo})."
            )
    return inferred_repo


def _handle_error(e: PtqError) -> None:
    console.print(f"[red]{e}[/red]")
    raise typer.Exit(1)


def _render_event(ev: StreamEvent) -> None:
    match ev.kind:
        case "text":
            console.print(_clean(ev.text), end="", highlight=False)
        case "tool_use":
            console.print()
            inp = ev.tool_input
            match ev.tool_name:
                case "Bash":
                    console.print(
                        f"  [bold cyan]$[/bold cyan] [dim]{inp.get('command', '')}[/dim]"
                    )
                case "Read":
                    console.print(
                        f"  [cyan]read[/cyan] [dim]{inp.get('file_path', '') or inp.get('path', '')}[/dim]"
                    )
                case "Edit":
                    console.print(
                        f"  [yellow]edit[/yellow] [dim]{inp.get('file_path', '') or inp.get('path', '')}[/dim]"
                    )
                case "Write":
                    console.print(
                        f"  [green]write[/green] [dim]{inp.get('file_path', '') or inp.get('path', '')}[/dim]"
                    )
                case "Grep":
                    console.print(
                        f"  [cyan]grep[/cyan] [dim]{inp.get('pattern', '')}[/dim]"
                    )
                case "Glob":
                    console.print(
                        f"  [cyan]glob[/cyan] [dim]{inp.get('pattern', '')}[/dim]"
                    )
                case "ToolSearch":
                    console.print(
                        f"  [cyan]tool search[/cyan] [dim]{inp.get('query', '')}[/dim]"
                    )
                case "mcp__plugin_meta_mux__search_files":
                    pattern = inp.get("pattern", "")
                    dirs = inp.get("target_directories", "")
                    console.print(
                        f"  [cyan]search files[/cyan] [dim]{pattern} {dirs}[/dim]"
                    )
                case "mcp__plugin_meta_mux__knowledge_filtered_search":
                    query = inp.get("natural_language_query") or inp.get("keywords", "")
                    console.print(f"  [cyan]knowledge search[/cyan] [dim]{query}[/dim]")
                case "mcp__plugin_meta_mux__knowledge_load":
                    console.print(
                        f"  [cyan]load[/cyan] [dim]{inp.get('url', '')}[/dim]"
                    )
                case _:
                    console.print(f"  [dim]{ev.tool_name}[/dim]")
        case "tool_result":
            if ev.text.strip():
                console.print(f"[dim]{_indent(_truncate(_clean(ev.text)))}[/dim]")
            console.print()
        case "error":
            console.print(f"[red]{_indent(_truncate(_clean(ev.text)))}[/red]")
            console.print()


def _follow_logs(backend, log_file: str, agent, job_id: str) -> None:
    time.sleep(2)
    tail = backend.tail_log(log_file)
    try:
        for line in tail.stdout:
            stripped = _clean(line.strip())
            if stripped.startswith("{"):
                try:
                    for ev in agent.parse_stream_line(stripped):
                        _render_event(ev)
                except (json.JSONDecodeError, ValueError):
                    pass
    except KeyboardInterrupt:
        tail.terminate()
        tail.wait()
        console.print(
            "\n[bold yellow]Detached. Agent still running on remote.[/bold yellow]"
        )
        console.print(f"  ptq results {job_id}      [dim]# view results[/dim]")
        console.print(
            f"  $(ptq takeover {job_id})   [dim]# ssh into remote worktree[/dim]"
        )
        return
    tail.wait()
    console.print("\n[bold]Agent finished.[/bold]")
    console.print(f"  ptq results {job_id}")


def _repo():
    from ptq.infrastructure.job_repository import JobRepository

    return JobRepository()


def _rebase_list_label(state: RebaseState) -> str:
    match state:
        case RebaseState.IDLE:
            return "[dim]-[/dim]"
        case RebaseState.RUNNING:
            return "[blue]run[/blue]"
        case RebaseState.SUCCEEDED:
            return "[green]ok[/green]"
        case RebaseState.NEEDS_HUMAN:
            return "[yellow]human[/yellow]"
        case RebaseState.FAILED:
            return "[red]fail[/red]"


def _pr_list_label(pr_url: str | None, backend) -> str:
    from ptq.application.pr_service import get_pr_state

    if not pr_url:
        return "[dim]-[/dim]"

    match get_pr_state(backend, pr_url):
        case "open":
            return "[green]open[/green]"
        case "closed":
            return "[yellow]closed[/yellow]"
        case "merged":
            return "[cyan]merged[/cyan]"
        case _:
            return "[dim]saved[/dim]"


@app.command()
def setup(
    machine: Annotated[
        str | None, typer.Argument(help="Remote machine to set up.")
    ] = None,
    local: Annotated[
        bool, typer.Option("--local", help="Set up local workspace instead.")
    ] = False,
    build: Annotated[
        bool, typer.Option("--build", help="Also compile PyTorch from source.")
    ] = False,
    with_re_cc: Annotated[
        int | None,
        typer.Option(
            "--with-re-cc", help="Use re-cc distributed compiler with N parallel jobs."
        ),
    ] = None,
    workspace: Annotated[
        str | None, typer.Option(help="Custom workspace path.")
    ] = None,
    extras: Annotated[
        list[str] | None,
        typer.Option(
            "--extras", help="Additional repos to clone (e.g. --extras torchtitan)."
        ),
    ] = None,
) -> None:
    """One-time workspace setup: clone PyTorch with submodules, create venv, install build deps.

    Use --build to also compile PyTorch from source (needed for C++ edit support).
    Use --extras to also clone add-on repos (e.g. --extras torchtitan).
    """
    if not machine and not local:
        raise typer.BadParameter("Provide a machine name or use --local.")

    from ptq.config import load_config
    from ptq.infrastructure.backends import create_backend
    from ptq.workspace import setup_workspace

    backend = create_backend(machine=machine, local=local, workspace=workspace)
    setup_workspace(
        backend,
        build=build,
        re_cc_jobs=with_re_cc or 0,
        build_env_prefix=load_config().build_env_prefix(),
        extras=extras or [],
    )


@app.command()
def run(
    job_id: Annotated[
        str | None, typer.Argument(help="Job ID or issue number to re-run.")
    ] = None,
    issue: Annotated[
        str | None,
        typer.Option(help="GitHub issue number or GitHub issue URL."),
    ] = None,
    machine: Annotated[
        str | None, typer.Option(help="Remote machine to run on.")
    ] = None,
    local: Annotated[bool, typer.Option("--local", help="Run locally.")] = False,
    follow: Annotated[
        bool, typer.Option(help="Stream agent output to terminal.")
    ] = True,
    model: Annotated[str | None, typer.Option(help="Model to use.")] = None,
    max_turns: Annotated[int | None, typer.Option(help="Max agent turns.")] = None,
    thinking: Annotated[
        str | None,
        typer.Option(help="Reasoning/thinking level when supported by the agent."),
    ] = None,
    agent: Annotated[
        str | None, typer.Option(help="Agent type: claude, codex, cursor, or pi.")
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Stream build output and show timings."),
    ] = False,
    workspace: Annotated[
        str | None, typer.Option(help="Custom workspace path.")
    ] = None,
    message: Annotated[
        str | None,
        typer.Option("--message", "-m", help="Custom instruction for the agent."),
    ] = None,
    preset: Annotated[
        str | None,
        typer.Option(
            "--preset",
            "-p",
            help="Prompt preset key/title (combine with -m to append extra instructions).",
        ),
    ] = None,
    input_file: Annotated[
        Path | None,
        typer.Option("--input", "-i", help="Read task description from a file."),
    ] = None,
    review_file: Annotated[
        Path | None,
        typer.Option(
            "--review-file",
            help="Inject evaluator feedback JSON into the solver context.",
        ),
    ] = None,
    name: Annotated[
        str | None,
        typer.Option("--name", "-n", help="Display name for this job."),
    ] = None,
    repo: Annotated[
        str | None,
        typer.Option("--repo", help="Repo the issue is filed in (default: pytorch)."),
    ] = None,
) -> None:
    """Launch an AI agent to investigate a GitHub issue or run an adhoc task.

    Provide --issue for GitHub issue investigation, or --message for a freeform task.
    Re-run an existing job by passing its JOB_ID (or issue number) as a positional arg.

    Examples:
        ptq run --issue 174923 --machine aws-gpu-dev
        ptq run --agent codex -m "investigate OOM" --machine gpu-dev
        ptq run -p diagnose_and_plan --issue 174923 --machine gpu-dev
        ptq run -p fix_and_verify -m "focus on stride handling" --issue 174923
        ptq run -i task.md --machine gpu-dev --agent cursor
        ptq run -m "triage the repro" --agent pi
        ptq run 20260214-174923 -m "look at flex_attention.py instead"
        ptq run 174923 -m "try a different approach"
    """
    if input_file is not None and message is not None:
        raise typer.BadParameter("--input and --message are mutually exclusive.")
    if input_file is not None:
        if not input_file.exists():
            raise typer.BadParameter(f"File not found: {input_file}")
        message = input_file.read_text()
    review_feedback_json = None
    if review_file is not None:
        if not review_file.exists():
            raise typer.BadParameter(f"Review file not found: {review_file}")
        review_feedback_json = review_file.read_text()

    from ptq.application import run_service
    from ptq.config import load_config
    from ptq.infrastructure.backends import create_backend
    from ptq.issue import fetch_issue, format_issue_context

    cfg = load_config()
    if preset:
        selected_preset = cfg.prompt_preset(preset)
        if selected_preset is None:
            choices = ", ".join(cfg.prompt_preset_choices())
            raise typer.BadParameter(
                f"Unknown preset '{preset}'. Available presets: {choices}"
            )
        if message:
            message = f"{selected_preset.body}\n\n{message.strip()}"
        else:
            message = selected_preset.body

    job_repo = _repo()

    resolved_job_id: str | None = None
    if job_id is not None:
        try:
            resolved_job_id = job_repo.resolve_id(job_id)
        except PtqError as e:
            _handle_error(e)
        job = job_repo.get(resolved_job_id)
        issue = issue or job.issue
        machine = machine or job.machine
        local = local or job.local
        workspace = workspace or job.workspace
        repo = job.repo
        if agent is None:
            agent = job.agent
        if thinking is None:
            thinking = job.thinking

    if issue is None and message is None and job_id is None:
        raise typer.BadParameter(
            "Provide --issue, --preset, --message, or a JOB_ID to re-run."
        )
    if not machine and not local:
        local = True
    agent = agent or cfg.default_agent
    model = cfg.effective_model(agent, model)
    thinking = cfg.effective_thinking(agent, thinking)
    max_turns = max_turns or cfg.default_max_turns

    issue_number, issue_github_repo = _parse_issue_reference(issue)
    if issue_github_repo:
        repo = _repo_for_issue_url(issue_github_repo, repo)
    repo = repo or "pytorch"

    from ptq.repo_profiles import get_profile

    profile = get_profile(repo)

    issue_data = None
    if issue_number is not None:
        console.print(f"Fetching {profile.github_repo}#{issue_number}...")
        issue_data = fetch_issue(issue_number, repo=profile.github_repo)
        console.print(f"[bold]{issue_data['title']}[/bold]")

    backend = create_backend(machine=machine, local=local, workspace=workspace)
    request = RunRequest(
        issue_data=issue_data,
        issue_number=issue_number,
        message=message,
        machine=machine,
        local=local,
        follow=follow,
        model=model,
        thinking=thinking,
        max_turns=max_turns,
        agent_type=agent,
        existing_job_id=resolved_job_id,
        verbose=verbose,
        name=name,
        repo=repo,
        review_feedback_json=review_feedback_json,
    )

    try:
        launched_id = run_service.launch(
            job_repo, backend, request, on_progress=lambda msg: console.print(msg)
        )
    except PtqError as e:
        _handle_error(e)

    job = job_repo.get(launched_id)
    if follow:
        agent_impl = get_agent(job.agent)
        log_file = f"{backend.workspace}/jobs/{launched_id}/{agent_impl.log_filename(job.runs)}"
        _follow_logs(backend, log_file, agent_impl, launched_id)
    else:
        from ptq.takeover import for_job as takeover_for_job

        console.print()
        console.print(f"[bold green]Launched {launched_id}.[/bold green]")
        console.print(f"  Take over: {takeover_for_job(launched_id, job)}")
        console.print(f"  Results:   ptq results {launched_id}")


@app.command("presets")
def list_presets() -> None:
    """List available prompt presets (built-in + custom from config)."""
    from ptq.config import load_config

    cfg = load_config()
    console.print("[bold]Available presets[/bold]")
    for preset in cfg.prompt_presets:
        console.print(f"- [cyan]{preset.key}[/cyan] — {preset.title}")


@app.command()
def results(
    job_id: Annotated[str, typer.Argument(help="Job ID or issue number.")],
    output_dir: Annotated[
        Path | None, typer.Option(help="Custom output directory.")
    ] = None,
) -> None:
    """Fetch and display results from a completed job."""
    from rich.markdown import Markdown

    from ptq.application.artifact_service import fetch_results

    repo = _repo()
    try:
        job_id = repo.resolve_id(job_id)
    except PtqError as e:
        _handle_error(e)

    console.print(f"Fetching results for {job_id}...")
    results_dir, fetched, missing = fetch_results(repo, job_id, output_dir)
    for name in fetched:
        console.print(f"  fetched {name}")
    for name in missing:
        console.print(f"  {name} not found")

    repro_path = results_dir / "repro.py"
    if repro_path.exists():
        from rich.syntax import Syntax

        console.print()
        console.print("[bold]Repro Script[/bold]")
        console.print(Syntax(repro_path.read_text(), "python", theme="monokai"))

    for name, label in [("worklog.md", "Worklog"), ("report.md", "Report")]:
        path = results_dir / name
        if path.exists():
            console.print()
            console.print(f"[bold]{label}[/bold]")
            console.print(Markdown(path.read_text()))

    diff_path = results_dir / "fix.diff"
    if diff_path.exists():
        diff_text = diff_path.read_text()
        if diff_text.strip():
            console.print()
            console.print("[bold]Diff[/bold]")
            console.print(diff_text)
        else:
            console.print("[yellow]fix.diff is empty — no changes made.[/yellow]")
    else:
        console.print("[yellow]No fix.diff found.[/yellow]")

    console.print(f"\nArtifacts saved to: {results_dir}")


@app.command()
def apply(
    job_id: Annotated[str, typer.Argument(help="Job ID or issue number.")],
    pytorch_path: Annotated[
        Path, typer.Option(help="Path to local pytorch checkout.")
    ] = Path("~/meta/pytorch"),
) -> None:
    """Apply a job's diff to a local PyTorch checkout."""
    from ptq.application.artifact_service import apply_diff

    repo = _repo()
    try:
        job_id = repo.resolve_id(job_id)
        branch = apply_diff(repo, job_id, pytorch_path.expanduser())
    except PtqError as e:
        _handle_error(e)

    console.print(
        f"\n[bold green]Diff applied to {pytorch_path} on branch {branch}[/bold green]"
    )
    console.print(f"\nTo create a PR, run: [bold]ptq pr {job_id}[/bold]")


@app.command()
def clean(
    target: Annotated[
        str | None, typer.Argument(help="Job ID, issue number, or machine name.")
    ] = None,
    local: Annotated[
        bool, typer.Option("--local", help="Clean local workspace.")
    ] = False,
    workspace: Annotated[
        str | None, typer.Option(help="Custom workspace path.")
    ] = None,
    keep: Annotated[int, typer.Option(help="Number of most recent jobs to keep.")] = 0,
    all_jobs: Annotated[
        bool, typer.Option("--all", help="Include running jobs.")
    ] = False,
) -> None:
    """Remove jobs: kill agent, delete remote files, drop from tracking DB.

    Pass a JOB_ID to clean a single job, or a MACHINE name to bulk clean.

    Examples:
        ptq clean 20260214-174923          # clean one job
        ptq clean 174923                   # clean by issue number
        ptq clean aws-gpu-dev              # clean all stopped jobs on machine
        ptq clean aws-gpu-dev --keep 2     # keep 2 most recent
        ptq clean aws-gpu-dev --all        # include running jobs
    """
    from ptq.application.job_service import clean_machine, clean_single_job
    from ptq.infrastructure.backends import create_backend

    repo = _repo()

    if target is not None:
        all_jobs_db = repo.list_all()
        is_job = target in all_jobs_db or (
            target.isdigit()
            and any(j.issue == int(target) for j in all_jobs_db.values())
        )
        if is_job:
            try:
                resolved = repo.resolve_id(target)
                job = clean_single_job(repo, resolved)
            except PtqError as e:
                _handle_error(e)
            label = f"issue #{job.issue}" if job.issue is not None else "adhoc"
            console.print(f"  removed {resolved} ({label})")
            return

    if not target and not local:
        raise typer.BadParameter("Provide a job ID, machine name, or --local.")

    backend = create_backend(machine=target, local=local, workspace=workspace)
    removed, skipped = clean_machine(
        repo,
        backend,
        machine=target,
        local=local,
        keep=keep,
        include_running=all_jobs,
    )
    if skipped:
        console.print(f"Skipping {skipped} running job(s). Use --all to include them.")
    if not removed:
        console.print("Nothing to clean.")
        return
    console.print(f"Removing {len(removed)} job(s) (keeping {keep})...")
    for jid in removed:
        console.print(f"  removed {jid}")
    console.print("[bold green]Clean complete.[/bold green]")


@app.command(name="list")
def list_jobs() -> None:
    """List all tracked jobs."""
    from rich.table import Table

    from ptq.application.job_service import get_status
    from ptq.infrastructure.backends import backend_for_job

    repo = _repo()
    all_jobs = repo.list_all()
    if not all_jobs:
        console.print("No jobs.")
        return

    table = Table(
        show_header=True, header_style="bold", show_lines=False, pad_edge=False
    )
    table.add_column("Status", width=9)
    table.add_column("Job ID")
    table.add_column("Name")
    table.add_column("Issue", style="cyan")
    table.add_column("Agent", width=7)
    table.add_column("Runs", justify="right")
    table.add_column("PR", width=7)
    table.add_column("Rebase", width=8)
    table.add_column("Target")

    for job_id, job in sorted(all_jobs.items()):
        issue_display = f"#{job.issue}" if job.issue is not None else "[dim]adhoc[/dim]"
        backend = backend_for_job(job)
        status = get_status(job, backend)
        status_str = (
            "[bold green]running[/bold green]"
            if status == JobStatus.RUNNING
            else "[dim]stopped[/dim]"
        )
        pr_display = _pr_list_label(job.pr_url, backend)
        rebase_display = _rebase_list_label(job.rebase_info.state)
        table.add_row(
            status_str,
            job_id,
            job.name or "[dim]-[/dim]",
            issue_display,
            job.agent,
            str(job.runs),
            pr_display,
            rebase_display,
            job.target,
        )

    console.print(table)
    console.print()
    console.print("[dim]Actions:[/dim]")
    console.print(
        "[dim]  ptq run --issue NUM --machine TARGET  # new run from issue[/dim]"
    )
    console.print("[dim]  ptq run -m 'task' --machine TARGET    # new adhoc run[/dim]")
    console.print(
        "[dim]  ptq run JOB_ID                        # re-run existing job[/dim]"
    )
    console.print(
        "[dim]  ptq run JOB_ID -m 'look at X instead' # re-run with steering[/dim]"
    )
    console.print("[dim]  ptq peek JOB_ID                       # check progress[/dim]")
    console.print("[dim]  ptq watch JOB_ID                      # stream progress[/dim]")
    console.print("[dim]  ptq results JOB_ID                    # fetch results[/dim]")
    console.print(
        "[dim]  ptq pr JOB_ID                         # create GitHub PR[/dim]"
    )
    console.print(
        "[dim]  ptq takeover JOB_ID                   # drop into worktree[/dim]"
    )
    console.print("[dim]  ptq kill JOB_ID                       # stop agent[/dim]")
    console.print(
        "[dim]  ptq clean JOB_ID                      # remove job entirely[/dim]"
    )
    console.print(
        "[dim]  ptq clean MACHINE                     # bulk clean stopped jobs[/dim]"
    )
    console.print(
        "[dim]  ptq web                               # start web dashboard[/dim]"
    )


@app.command()
def peek(
    job_id: Annotated[str, typer.Argument(help="Job ID or issue number.")],
    log_lines: Annotated[
        int, typer.Option("--log", help="Number of log lines to show.")
    ] = 0,
) -> None:
    """Peek at an agent's progress (worklog + optional log tail)."""
    from rich.markdown import Markdown

    from ptq.application.job_service import get_status
    from ptq.infrastructure.backends import backend_for_job

    repo = _repo()
    try:
        job_id = repo.resolve_id(job_id)
    except PtqError as e:
        _handle_error(e)
    job = repo.get(job_id)
    backend = backend_for_job(job)
    ws = backend.workspace
    status = get_status(job, backend)
    status_str = (
        "[bold green]running[/bold green]"
        if status == JobStatus.RUNNING
        else "[dim]stopped[/dim]"
    )
    issue_label = f"issue #{job.issue}" if job.issue is not None else "adhoc"
    console.print(
        f"{status_str}  {job_id}  {issue_label}  (run {job.runs}, {job.target})"
    )
    console.print()

    worklog_path = f"{ws}/jobs/{job_id}/worklog.md"
    result = backend.run(f"cat {worklog_path}", check=False)
    if result.returncode == 0 and result.stdout.strip():
        console.print("[bold]Worklog[/bold]")
        console.print(Markdown(result.stdout))
    else:
        console.print("[yellow]No worklog yet.[/yellow]")

    if log_lines > 0:
        agent_impl = get_agent(job.agent)
        log_file = f"{ws}/jobs/{job_id}/{agent_impl.log_filename(job.runs)}"
        tail_result = backend.run(f"tail -{log_lines} {log_file}", check=False)
        if tail_result.stdout.strip():
            console.print()
            console.print(f"[bold]Last {log_lines} log lines[/bold]")
            for line in tail_result.stdout.strip().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    for ev in agent_impl.parse_stream_line(line):
                        match ev.kind:
                            case "text":
                                console.print(f"  [dim]{ev.text[:200]}[/dim]")
                            case "tool_use":
                                console.print(f"  [cyan]{ev.tool_name}[/cyan]")
                            case _:
                                pass
                except (json.JSONDecodeError, ValueError):
                    console.print(f"  [dim]{line[:200]}[/dim]")


@app.command()
def watch(
    job_id: Annotated[str, typer.Argument(help="Job ID or issue number.")],
    lines: Annotated[
        int,
        typer.Option("--lines", help="Existing log lines to show before following."),
    ] = 80,
    interval: Annotated[
        float, typer.Option("--interval", help="Seconds between log checks.")
    ] = 2.0,
    raw: Annotated[
        bool,
        typer.Option("--raw", help="Print raw agent log lines instead of parsed events."),
    ] = False,
    exit_when_stopped: Annotated[
        bool,
        typer.Option(
            "--exit-when-stopped/--keep-watching",
            help="Exit once the tracked job stops.",
        ),
    ] = True,
) -> None:
    """Stream an existing job's agent output."""
    from ptq.application.job_service import get_status
    from ptq.infrastructure.backends import backend_for_job

    repo = _repo()
    try:
        job_id = repo.resolve_id(job_id)
    except PtqError as e:
        _handle_error(e)
    job = repo.get(job_id)
    backend = backend_for_job(job)
    agent_impl = get_agent(job.agent)
    log_file = f"{backend.workspace}/jobs/{job_id}/{agent_impl.log_filename(job.runs)}"
    console.print(
        f"[bold]Watching {job_id}[/bold] "
        f"[dim](run {job.runs}, {job.target}, {log_file})[/dim]"
    )

    def read_log() -> str:
        result = backend.run(f"cat {log_file}", check=False)
        return result.stdout if result.returncode == 0 else ""

    text = read_log()
    if text and lines > 0:
        _render_log_lines(agent_impl, text.splitlines()[-lines:], raw=raw)
    offset = len(text)

    try:
        while True:
            job = repo.get(job_id)
            status = get_status(job, backend)
            text = read_log()
            if len(text) > offset:
                _render_log_lines(agent_impl, text[offset:].splitlines(), raw=raw)
                offset = len(text)
            if exit_when_stopped and status == JobStatus.STOPPED:
                console.print(f"\n[bold]Job stopped.[/bold]  ptq results {job_id}")
                return
            time.sleep(interval)
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Stopped watching; job was not killed.[/bold yellow]")
        console.print(f"  uv run ptq peek {job_id} --log 80")


def _render_log_lines(agent_impl, lines: list[str], *, raw: bool = False) -> None:
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if raw:
            console.print(stripped)
            continue
        try:
            for ev in agent_impl.parse_stream_line(stripped):
                _render_event(ev)
        except (json.JSONDecodeError, ValueError):
            continue


@app.command()
def takeover(
    job_id: Annotated[str, typer.Argument(help="Job ID or issue number.")],
) -> None:
    """Print the shell command to drop into a job's worktree."""
    from ptq.takeover import for_job as takeover_for_job

    repo = _repo()
    try:
        job_id = repo.resolve_id(job_id)
    except PtqError as e:
        _handle_error(e)
    job = repo.get(job_id)
    console.print(takeover_for_job(job_id, job))


@app.command()
def status(
    job_id: Annotated[str, typer.Argument(help="Job ID or issue number.")],
) -> None:
    """Check if an agent is still running for a job."""
    from ptq.application.job_service import get_status as _get_status
    from ptq.infrastructure.backends import backend_for_job

    repo = _repo()
    try:
        job_id = repo.resolve_id(job_id)
    except PtqError as e:
        _handle_error(e)
    job = repo.get(job_id)
    backend = backend_for_job(job)
    ws = backend.workspace

    if _get_status(job, backend) == JobStatus.RUNNING:
        console.print(
            f"[bold green]running[/bold green]  {job_id}  (run {job.runs}, {job.target}, pid {job.pid})"
        )
        if job.issue is not None:
            console.print(
                f"  uv run ptq peek {job_id} --log 80  # inspect progress"
            )
        else:
            console.print(f"  uv run ptq peek {job_id} --log 80  # inspect progress")
    else:
        console.print(
            f"[bold dim]stopped[/bold dim]  {job_id}  (run {job.runs}, {job.target})"
        )
        console.print(f"  ptq results {job_id}")

    agent_impl = get_agent(job.agent)
    log_file = f"{ws}/jobs/{job_id}/{agent_impl.log_filename(job.runs)}"
    tail = backend.run(f"tail -1 {log_file}", check=False)
    if tail.stdout.strip():
        console.print(f"\n  last log: [dim]{tail.stdout.strip()[:120]}[/dim]")


@app.command()
def pr(
    job_id: Annotated[str, typer.Argument(help="Job ID or issue number.")],
    note: Annotated[
        str | None,
        typer.Option(
            "--note",
            "-n",
            help="Your description of the PR: what it does, why it's correct, "
            "and how the reviewer should approach it. Opens $EDITOR if omitted.",
        ),
    ] = None,
    title: Annotated[str | None, typer.Option(help="PR title override.")] = None,
    draft: Annotated[
        bool,
        typer.Option(
            "--draft",
            help="Accepted for compatibility; PRs are always created as draft.",
        ),
    ] = False,
) -> None:
    """Create a draft GitHub PR from a job's worktree changes.

    Requires a human note describing the change. This is embedded at the top
    of the PR body so reviewers see the author's own assessment first.
    """
    from ptq.application.pr_service import create_pr

    repo = _repo()
    try:
        job_id = repo.resolve_id(job_id)
    except PtqError as e:
        _handle_error(e)

    if not note:
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", prefix="ptq-pr-note-", delete=False
        ) as f:
            f.write(
                "# Describe this PR for the reviewer\n"
                "#\n"
                "# What does this change do?\n"
                "# Why do you believe it's correct?\n"
                "# How should the reviewer approach it? (e.g. trivial fix, RFC, etc.)\n"
                "#\n"
                "# Lines starting with # will be stripped.\n"
            )
            note_path = f.name
        editor = os.environ.get("EDITOR", "vim")
        os.system(f"{editor} {note_path}")
        with open(note_path) as f:
            raw = f.read()
        os.unlink(note_path)
        note = "\n".join(
            line for line in raw.splitlines() if not line.startswith("#")
        ).strip()

    if not note:
        console.print("[red]No note provided — PR creation aborted.[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]Creating PR for {job_id}[/bold]")
    try:
        result = create_pr(
            repo,
            job_id,
            human_note=note,
            title=title,
            draft=True,
            log=lambda msg: console.print(f"  [dim]{msg}[/dim]"),
        )
    except PtqError as e:
        _handle_error(e)
    console.print(f"\n[bold green]PR created:[/bold green] {result.url}")


def _evaluator_models(evaluator_section: dict) -> list[str]:
    raw_models = evaluator_section.get("models")
    if isinstance(raw_models, list):
        models = [str(model).strip() for model in raw_models if str(model).strip()]
        if models:
            return models
    return ["gpt-5.5", "claude-opus-4-7"]


def _additional_evaluator_specs(
    evaluator_section: dict,
    *,
    add_evaluator: str | None = None,
    profile: str | None = None,
    agent: str | None = None,
    github_repo: str = "pytorch/pytorch",
    persist: bool = False,
):
    from ptq.evaluator import ReviewerSpec
    from ptq.evaluator.reviewer_profile import (
        DEFAULT_PROFILE_MODEL,
        generate_reviewer_profile,
        reviewer_profile_path,
    )

    specs: list[ReviewerSpec] = []
    raw_reviewers = evaluator_section.get("additional_reviewers")
    if isinstance(raw_reviewers, list):
        for item in raw_reviewers:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("profile") or "").strip()
            profile_value = str(item.get("profile") or "").strip()
            model = str(item.get("model") or DEFAULT_PROFILE_MODEL).strip()
            if name and profile_value and model:
                profile_path = reviewer_profile_path(profile_value)
                if _profile_should_be_generated(profile_value, profile_path):
                    console.print(
                        f"[dim]orchestrate:[/dim] generating evaluator profile "
                        f"@{profile_value} -> {profile_path}",
                        soft_wrap=True,
                    )
                    generate_reviewer_profile(profile_value, repo=github_repo)
                specs.append(
                    ReviewerSpec(
                        name=name,
                        model=model,
                        profile_path=str(profile_path),
                    )
                )

    if add_evaluator:
        if not profile:
            raise typer.BadParameter("--add-evaluator requires --profile.")
        model = agent or DEFAULT_PROFILE_MODEL
        profile_path = reviewer_profile_path(profile)
        if _profile_should_be_generated(profile, profile_path):
            console.print(
                f"[dim]orchestrate:[/dim] generating evaluator profile "
                f"@{profile} -> {profile_path}",
                soft_wrap=True,
            )
            generate_reviewer_profile(profile, repo=github_repo)
        if persist:
            from ptq.config import ensure_additional_evaluator

            if ensure_additional_evaluator(
                name=add_evaluator,
                profile=profile,
                model=model,
            ):
                console.print(
                    f"[dim]orchestrate:[/dim] saved evaluator "
                    f"{add_evaluator} to ~/.ptq/config.toml",
                    soft_wrap=True,
                )
        specs.append(
            ReviewerSpec(
                name=add_evaluator,
                model=model,
                profile_path=str(profile_path),
            )
        )
    return specs


def _profile_should_be_generated(profile: str, path: Path) -> bool:
    if path.exists():
        return False
    raw = Path(profile).expanduser()
    return raw.suffix != ".md" and not raw.is_absolute() and len(raw.parts) == 1


def _persist_additional_evaluator(
    *,
    name: str,
    profile: str | None,
    agent: str | None,
    github_repo: str,
) -> None:
    if not profile:
        raise typer.BadParameter("--add-evaluator requires --profile.")
    from ptq.config import ensure_additional_evaluator
    from ptq.evaluator.reviewer_profile import (
        DEFAULT_PROFILE_MODEL,
        generate_reviewer_profile,
        reviewer_profile_path,
    )

    model = agent or DEFAULT_PROFILE_MODEL
    profile_path = reviewer_profile_path(profile)
    if _profile_should_be_generated(profile, profile_path):
        console.print(
            f"[dim]orchestrate:[/dim] generating evaluator profile "
            f"@{profile} -> {profile_path}",
            soft_wrap=True,
        )
        generate_reviewer_profile(profile, repo=github_repo)
    changed = ensure_additional_evaluator(
        name=name,
        profile=profile,
        model=model,
    )
    if changed:
        console.print(
            f"[dim]orchestrate:[/dim] saved evaluator "
            f"{name} to ~/.ptq/config.toml",
            soft_wrap=True,
        )
    else:
        console.print(
            f"[dim]orchestrate:[/dim] evaluator {name} already configured",
            soft_wrap=True,
        )


def _validate_add_evaluator_only(
    *,
    prompt: str | None,
    parallel: int | None,
    machine: str | None,
    max_iterations: int | None,
    issue: str | None,
    review: str | None,
    message: str | None,
    dry_run: bool,
    follow: bool,
    poll_seconds: float,
    push_pr: bool,
    watch_pr: bool,
    watch_pr_interval_seconds: float | None,
    watch_pr_idle_hours: float | None,
) -> None:
    conflicts = []
    if prompt is not None:
        conflicts.append("--prompt")
    if parallel is not None:
        conflicts.append("--parallel")
    if machine is not None:
        conflicts.append("--machine")
    if max_iterations is not None:
        conflicts.append("--max-iterations")
    if issue is not None:
        conflicts.append("--issue")
    if review is not None:
        conflicts.append("--review")
    if message is not None:
        conflicts.append("--message")
    if dry_run:
        conflicts.append("--dry-run")
    if not follow:
        conflicts.append("--no-follow")
    if poll_seconds != 5.0:
        conflicts.append("--poll-seconds")
    if push_pr:
        conflicts.append("--pr")
    if watch_pr:
        conflicts.append("--watch-pr")
    if watch_pr_interval_seconds is not None:
        conflicts.append("--watch-pr-interval-seconds")
    if watch_pr_idle_hours is not None:
        conflicts.append("--watch-pr-idle-hours")
    if conflicts:
        joined = ", ".join(conflicts)
        raise typer.BadParameter(
            f"--add-evaluator only adds an evaluator; remove: {joined}"
        )


def _orchestrate_report_display(job_id: str | None) -> str | None:
    if not job_id:
        return None
    try:
        job = _repo().get(job_id)
    except PtqError:
        return None

    path = f"{job.workspace}/jobs/{job_id}/report.md"
    if job.local:
        local_path = Path(job.workspace).expanduser() / "jobs" / job_id / "report.md"
        if local_path.exists():
            return str(local_path)
        return f"missing (expected at {local_path})"

    remote_path = f"{job.machine or 'remote'}:{path}"
    try:
        from ptq.infrastructure.backends import backend_for_job

        backend = backend_for_job(job)
        if backend.run(f"test -f {path}", check=False).returncode == 0:
            return remote_path
    except Exception:
        pass
    return f"missing (expected at {remote_path})"



def _print_orchestrate_results(results) -> None:
    for result in results:
        issue_number = result.issue.number
        label = f"#{issue_number}" if issue_number > 0 else "adhoc"
        verdict = result.verdict
        score = result.score
        iterations = result.iterations
        job_id = result.job_id or "-"
        branch = result.branch or "-"
        pr_url = getattr(result, "pr_url", None)
        console.print(
            f"{label} {verdict} "
            f"score={score:.2f} iterations={iterations} "
            f"job={job_id} branch={branch}"
        )
        report_display = _orchestrate_report_display(result.job_id)
        if report_display:
            console.print(f"  report.md: {report_display}", soft_wrap=True)
        if pr_url:
            console.print(f"  PR: {pr_url}", soft_wrap=True)


@app.command()
def orchestrate(
    prompt: Annotated[
        str | None,
        typer.Option(
            "--prompt",
            help="Natural-language GitHub issue selection criteria.",
        ),
    ] = None,
    repo: Annotated[
        str | None,
        typer.Option(
            "--repo",
            help="PTQ repo profile to run against, e.g. pytorch or torchtitan.",
        ),
    ] = None,
    parallel: Annotated[int | None, typer.Option(help="Concurrent solver jobs.")] = None,
    machine: Annotated[
        str | None,
        typer.Option(help="Machine to run on; localhost/local runs locally."),
    ] = None,
    max_iterations: Annotated[
        int | None, typer.Option(help="Maximum hill-climbing iterations per issue.")
    ] = None,
    issue: Annotated[
        str | None,
        typer.Option(
            help="Run the orchestrator on one explicit issue number or GitHub issue URL."
        ),
    ] = None,
    review: Annotated[
        str | None,
        typer.Option(
            "--review",
            help=(
                "Review a GitHub PR/review URL with the evaluator panel only; "
                "skips solver."
            ),
        ),
    ] = None,
    message: Annotated[
        str | None,
        typer.Option(
            "--message",
            "-m",
            help="Initial solver guidance to include alongside the issue.",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Select issues without launching agents."),
    ] = False,
    follow: Annotated[
        bool,
        typer.Option(
            "--follow/--no-follow",
            help="Stream solver output while waiting.",
        ),
    ] = True,
    poll_seconds: Annotated[
        float,
        typer.Option(
            "--poll-seconds",
            help="Seconds between solver status checks.",
        ),
    ] = 5.0,
    push_pr: Annotated[
        bool,
        typer.Option(
            "--pr",
            help="After approval, push/create or update a draft GitHub PR.",
        ),
    ] = False,
    watch_pr: Annotated[
        bool,
        typer.Option(
            "--watch-pr",
            help=(
                "After pushing a draft PR, keep watching for review/CI activity; "
                "implies --pr."
            ),
        ),
    ] = False,
    watch_pr_interval_seconds: Annotated[
        float | None,
        typer.Option(
            "--watch-pr-interval-seconds",
            help="Seconds between PR activity polls when --watch-pr is set.",
        ),
    ] = None,
    watch_pr_idle_hours: Annotated[
        float | None,
        typer.Option(
            "--watch-pr-idle-hours",
            help="Stop --watch-pr after this many hours without PR activity.",
        ),
    ] = None,
    add_evaluator: Annotated[
        str | None,
        typer.Option(
            "--add-evaluator",
            help=(
                "Generate/persist a profile-backed evaluator reviewer, then exit. "
                "Do not combine with issue-solving options."
            ),
        ),
    ] = None,
    evaluator_profile: Annotated[
        str | None,
        typer.Option(
            "--profile",
            help=(
                "GitHub username or markdown profile path for --add-evaluator. "
                "Username profiles resolve under ~/.ptq/evaluator_profiles/."
            ),
        ),
    ] = None,
    evaluator_agent: Annotated[
        str | None,
        typer.Option(
            "--agent",
            help="Model used by --add-evaluator; defaults to gpt-5.5 via Codex.",
        ),
    ] = None,
) -> None:
    """Run the issue-selection, solver, evaluator hill-climbing loop."""
    from ptq.config import load_config
    from ptq.evaluator import Evaluator
    from ptq.orchestrator import Orchestrator, OrchestratorConfig
    from ptq.repo_profiles import get_profile, profile_name_for_github_repo

    cfg = load_config()
    orch = cfg.orchestrator
    evaluator_cfg = cfg.evaluator
    configured_github_repo = str(orch.get("github_repo") or "")
    configured_repo = str(orch.get("repo") or "")
    repo_name = (
        repo
        or configured_repo
        or profile_name_for_github_repo(configured_github_repo)
        or "pytorch"
    )
    try:
        profile = get_profile(repo_name)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    github_repo = (
        profile.github_repo
        if repo is not None or configured_repo
        else configured_github_repo or profile.github_repo
    )
    if add_evaluator:
        _validate_add_evaluator_only(
            prompt=prompt,
            parallel=parallel,
            machine=machine,
            max_iterations=max_iterations,
            issue=issue,
            review=review,
            message=message,
            dry_run=dry_run,
            follow=follow,
            poll_seconds=poll_seconds,
            push_pr=push_pr,
            watch_pr=watch_pr,
            watch_pr_interval_seconds=watch_pr_interval_seconds,
            watch_pr_idle_hours=watch_pr_idle_hours,
        )
        _persist_additional_evaluator(
            name=add_evaluator,
            profile=evaluator_profile,
            agent=evaluator_agent,
            github_repo=github_repo,
        )
        return

    if review is not None:
        if issue is not None:
            raise typer.BadParameter("--review cannot be combined with --issue.")
        if prompt is not None:
            raise typer.BadParameter("--review cannot be combined with --prompt.")
        if message is not None:
            raise typer.BadParameter("--review cannot be combined with --message.")
        if dry_run:
            raise typer.BadParameter("--review cannot be combined with --dry-run.")
        if push_pr or watch_pr:
            raise typer.BadParameter("--review cannot be combined with --pr/--watch-pr.")

        from ptq.orchestrator.review import parse_pull_request_url
        from ptq.orchestrator.review import run_pull_request_review

        pr_ref = parse_pull_request_url(review)
        evaluator = Evaluator(
            reviewer_models=_evaluator_models(evaluator_cfg),
            additional_reviewers=_additional_evaluator_specs(
                evaluator_cfg,
                github_repo=pr_ref.github_repo,
            ),
            approval_threshold=float(evaluator_cfg.get("approval_threshold", 0.8)),
            shelve_threshold=float(evaluator_cfg.get("shelve_threshold", 0.3)),
            max_iterations=int(evaluator_cfg.get("max_iterations", 10)),
        )
        try:
            evaluator.validate_configuration()
            report = run_pull_request_review(
                review,
                evaluator,
                on_progress=lambda msg: console.print(
                    f"[dim]orchestrate:[/dim] {msg}"
                ),
            )
        except PtqError as e:
            _handle_error(e)
        except RuntimeError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1) from e

        console.print(
            f"review {report.review.verdict} score={report.review.score:.2f}"
        )
        console.print(f"  PR: {report.pr.url}", soft_wrap=True)
        console.print(f"  report.md: {report.report_path}", soft_wrap=True)
        return

    issue_number, issue_github_repo = _parse_issue_reference(issue)
    if issue_github_repo:
        repo_name = _repo_for_issue_url(issue_github_repo, repo)
        profile = get_profile(repo_name)
        github_repo = profile.github_repo

    adhoc = issue_number is None and prompt is None and message is not None
    issue_prompt = ""
    if adhoc:
        issue_prompt = ""
    elif issue_number is not None:
        issue_prompt = f"https://github.com/{github_repo}/issues/{issue_number}"
    else:
        issue_prompt = prompt or str(orch.get("issue_selection_prompt") or "")
    if not adhoc and not issue_prompt:
        raise typer.BadParameter(
            "Provide --issue, --prompt, or --message for an adhoc orchestrator task."
        )

    agent = cfg.default_agent
    model = cfg.effective_model(agent)
    thinking = cfg.effective_thinking(agent)
    max_issues_value = 1 if issue_number is not None else int(orch.get("max_issues", 20))
    parallel_value = int(parallel or orch.get("parallel", 4))
    stream_solver = bool(follow)
    watch_pr_enabled = bool(watch_pr or orch.get("watch_pr", False))
    watch_interval = float(
        watch_pr_interval_seconds
        if watch_pr_interval_seconds is not None
        else orch.get("watch_pr_interval_seconds", 300.0)
    )
    watch_idle_hours = float(
        watch_pr_idle_hours
        if watch_pr_idle_hours is not None
        else orch.get("watch_pr_idle_hours", 24.0)
    )

    config = OrchestratorConfig(
        issue_selection_prompt=issue_prompt,
        adhoc=adhoc,
        repo=repo_name,
        github_repo=github_repo,
        max_issues=max_issues_value,
        parallel=parallel_value,
        max_iterations=int(max_iterations or orch.get("max_iterations", 10)),
        approval_threshold=float(orch.get("approval_threshold", 0.8)),
        machine=str(machine or orch.get("machine", "localhost")),
        dry_run=dry_run,
        solver_agent=agent,
        solver_model=model,
        solver_thinking=thinking,
        solver_max_turns=cfg.default_max_turns,
        initial_message=message,
        push_pr=push_pr or watch_pr_enabled,
        watch_pr=watch_pr_enabled,
        watch_pr_interval_seconds=watch_interval,
        watch_pr_idle_seconds=watch_idle_hours * 3600.0,
    )

    evaluator = Evaluator(
        reviewer_models=_evaluator_models(evaluator_cfg),
        additional_reviewers=_additional_evaluator_specs(
            evaluator_cfg,
            add_evaluator=add_evaluator,
            profile=evaluator_profile,
            agent=evaluator_agent,
            github_repo=github_repo,
            persist=bool(add_evaluator),
        ),
        approval_threshold=float(evaluator_cfg.get("approval_threshold", 0.8)),
        shelve_threshold=float(evaluator_cfg.get("shelve_threshold", 0.3)),
        max_iterations=int(evaluator_cfg.get("max_iterations", config.max_iterations)),
    )
    try:
        orchestrator = Orchestrator(
            config,
            evaluator=evaluator,
            on_progress=lambda msg: console.print(f"[dim]orchestrate:[/dim] {msg}"),
            on_solver_event=lambda _job_id, event: _render_event(event),
            stream_solver=stream_solver,
            poll_seconds=poll_seconds,
        )
        results = asyncio.run(orchestrator.run())
    except PtqError as e:
        _handle_error(e)
    _print_orchestrate_results(results)


@app.command("orchestrate-results")
def orchestrate_results(
    log_path: Annotated[
        Path | None,
        typer.Option(help="Path to orchestrator JSONL log."),
    ] = None,
) -> None:
    """Show orchestrator results/history."""
    from rich.table import Table

    from ptq.orchestrator.reporter import read_results

    path = log_path or (Path.home() / ".ptq" / "orchestrator" / "runs.jsonl")
    rows = read_results(path)
    table = Table(show_header=True, header_style="bold", pad_edge=False)
    table.add_column("Event")
    table.add_column("Issue")
    table.add_column("Verdict")
    table.add_column("Score", justify="right")
    table.add_column("Job")
    table.add_column("Branch")
    for row in rows[-50:]:
        result = row.get("result", {}) if isinstance(row.get("result"), dict) else {}
        issue = (
            result.get("issue", {}) if isinstance(result.get("issue"), dict) else {}
        )
        review = (
            result.get("review", {}) if isinstance(result.get("review"), dict) else {}
        )
        table.add_row(
            str(row.get("event", "")),
            str(issue.get("number") or row.get("issue") or ""),
            str(result.get("verdict") or review.get("verdict") or ""),
            str(result.get("score") or review.get("score") or ""),
            str(result.get("job_id") or row.get("job_id") or ""),
            str(result.get("branch") or ""),
        )
    console.print(table)
    console.print(f"[dim]{path}[/dim]")


@app.command()
def web(
    port: Annotated[int, typer.Option(help="Port to listen on.")] = 8000,
    host: Annotated[str, typer.Option(help="Host to bind to.")] = "127.0.0.1",
    debug: Annotated[bool, typer.Option(help="Enable debug logging.")] = False,
) -> None:
    """Start the web dashboard (auto-reloads on code changes)."""
    try:
        import uvicorn
    except ModuleNotFoundError:
        console.print(
            "[red]Missing web dependencies.[/red] Install with: [bold]pip install -e .[/bold]"
        )
        raise typer.Exit(1)  # noqa: B904

    console.print(f"Starting ptq web at http://{host}:{port}")
    factory = "ptq.web.app:create_debug_app" if debug else "ptq.web.app:create_app"
    uvicorn.run(
        factory,
        factory=True,
        host=host,
        port=port,
        log_level="debug" if debug else "info",
        reload=True,
        reload_dirs=[str(Path(__file__).resolve().parent)],
    )


@app.command()
def kill(
    job_id: Annotated[str, typer.Argument(help="Job ID or issue number.")],
) -> None:
    """Kill a running agent for a job."""
    from ptq.application.job_service import kill_job

    repo = _repo()
    try:
        job_id = repo.resolve_id(job_id)
    except PtqError as e:
        _handle_error(e)

    job = repo.get(job_id)
    killed = kill_job(repo, job_id)
    if killed:
        console.print(f"[bold]Killed agent for {job_id} (pid {job.pid})[/bold]")
    else:
        console.print(f"[dim]Agent already stopped for {job_id}[/dim]")


@app.command()
def rename(
    job_id: Annotated[str, typer.Argument(help="Job ID or issue number.")],
    name: Annotated[str, typer.Argument(help="New display name for the job.")],
) -> None:
    """Set or change the display name of a job."""
    repo = _repo()
    try:
        job_id = repo.resolve_id(job_id)
    except PtqError as e:
        _handle_error(e)

    repo.save_name(job_id, name)
    console.print(f"[bold]{job_id}[/bold] → {name}")


@app.command()
def worktree(
    name: Annotated[str, typer.Argument(help="Display name for this worktree.")],
    machine: Annotated[
        str | None, typer.Option(help="Remote machine to create worktree on.")
    ] = None,
    local: Annotated[
        bool, typer.Option("--local", help="Create in local workspace.")
    ] = False,
    workspace: Annotated[
        str | None, typer.Option(help="Custom workspace path.")
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Stream build output and show timings."),
    ] = False,
    repo: Annotated[
        str,
        typer.Option(
            "--repo", help="Repo to create a worktree for (default: pytorch)."
        ),
    ] = "pytorch",
) -> None:
    """Create a named worktree with a ready-to-use venv.

    Sets up a git worktree and per-worktree venv without launching an agent.
    Use `ptq run <name>` later to launch an agent in this worktree.

    Examples:
        ptq worktree flex-attn --machine gpu-dev
        ptq worktree my-fix --local
        ptq run flex-attn -m "optimize the CPU codegen"
    """
    if not machine and not local:
        local = True

    from ptq.application.job_context import write_job_context
    from ptq.application.worktree_service import provision_worktree, validate_workspace
    from ptq.domain.policies import make_job_id
    from ptq.infrastructure.backends import create_backend
    from ptq.repo_profiles import get_profile
    from ptq.takeover import shell_command as takeover_command
    from ptq.workspace import deploy_scripts

    profile = get_profile(repo)
    job_repo = _repo()
    existing = job_repo.find_by_name(name)
    if existing:
        console.print(
            f"[yellow]Worktree '{name}' already exists as {existing}[/yellow]"
        )
        raise typer.Exit(1)

    backend = create_backend(machine=machine, local=local, workspace=workspace)
    try:
        validate_workspace(backend, backend.workspace, repo=repo)
    except PtqError as e:
        _handle_error(e)

    job_id = make_job_id(message=name, repo=repo)
    job_repo.save(
        JobRecord(
            job_id=job_id,
            runs=0,
            agent="claude",
            model="",
            machine=machine,
            local=local,
            workspace=backend.workspace,
            name=name,
            repo=repo,
        )
    )

    deploy_scripts(backend)
    try:
        provision_worktree(
            backend,
            job_id,
            verbose=verbose,
            progress=lambda msg: console.print(msg),
            repo=repo,
        )
        write_job_context(
            backend,
            job_id=job_id,
            workspace=backend.workspace,
            repo=repo,
            name=name,
        )
    except PtqError as e:
        _handle_error(e)

    ws = backend.workspace
    job_dir = f"{ws}/jobs/{job_id}"
    dir_name = profile.dir_name
    console.print()
    console.print(f"[bold green]Worktree '{name}' ready.[/bold green]")
    console.print(f"  Job ID:   {job_id}")
    console.print(f"  Worktree: {job_dir}/{dir_name}")
    console.print(
        f"\n  Take over: {takeover_command(workspace=ws, job_id=job_id, repo=repo, local=local, machine=machine)}",
        soft_wrap=True,
    )
    console.print(f"  Shortcut: ptq takeover {job_id}")
    console.print(f"\n  To launch an agent: ptq run {name} -m 'your task'")


@app.command()
def rebase(
    job_id: Annotated[str, typer.Argument(help="Job ID or issue number.")],
    onto: Annotated[
        str, typer.Option("--onto", help="Target ref to rebase onto.")
    ] = "origin/main",
    agent: Annotated[str | None, typer.Option(help="Agent type override.")] = None,
    model: Annotated[str | None, typer.Option(help="Model override.")] = None,
    max_attempts: Annotated[
        int, typer.Option(help="Max conflict resolution attempts.")
    ] = 3,
) -> None:
    """Rebase a job's worktree onto a newer commit (default: origin/main).

    If conflicts arise, an agent is launched to resolve them automatically.
    Escalates to human takeover if conflicts remain after max attempts.

    Examples:
        ptq rebase 174923
        ptq rebase 20260214-174923 --onto origin/main
        ptq rebase 174923 --agent codex --model o3 --max-attempts 2
    """
    from ptq.application.rebase_service import rebase as do_rebase

    repo = _repo()
    try:
        job_id = repo.resolve_id(job_id)
    except PtqError as e:
        _handle_error(e)

    console.print(f"[bold]Rebasing {job_id} onto {onto}[/bold]")
    try:
        result = do_rebase(
            repo,
            job_id,
            target_ref=onto,
            agent_name=agent,
            model=model,
            max_attempts=max_attempts,
            on_progress=lambda msg: console.print(f"  {msg}"),
        )
    except PtqError as e:
        _handle_error(e)

    match result.state:
        case RebaseState.SUCCEEDED:
            console.print(
                f"\n[bold green]Rebase complete.[/bold green] "
                f"{result.before_sha[:10]} → {result.after_sha[:10]}"
            )
        case RebaseState.NEEDS_HUMAN:
            console.print("\n[bold yellow]Needs human intervention.[/bold yellow]")
            console.print(f"  {result.error}")
            from ptq.takeover import for_job as takeover_for_job

            job = repo.get(job_id)
            console.print(f"\n  {takeover_for_job(job_id, job)}")
        case _:
            console.print(f"\n[red]Rebase failed: {result.error}[/red]")


if __name__ == "__main__":
    app()
