from __future__ import annotations

import asyncio
import json
import shlex
import time
from pathlib import Path
from typing import Callable

from ptq.application import run_service
from ptq.application.job_service import get_status
from ptq.agents import get_agent
from ptq.domain.models import JobStatus, RunRequest
from ptq.evaluator import Evaluator, SolverOutput
from ptq.evaluator.models import ReviewResult
from ptq.infrastructure.backends import backend_for_job, create_backend
from ptq.infrastructure.job_repository import JobRepository
from ptq.orchestrator.issue_selector import IssueSelector
from ptq.orchestrator.models import Issue, OrchestratorConfig, SolveResult
from ptq.orchestrator.reporter import JsonlReporter
from ptq.repo_profiles import get_profile


class Orchestrator:
    def __init__(
        self,
        config: OrchestratorConfig,
        *,
        job_repo: JobRepository | None = None,
        evaluator: Evaluator | None = None,
        reporter: JsonlReporter | None = None,
        on_progress: Callable[[str], None] | None = None,
        on_solver_event: Callable[[str, object], None] | None = None,
        stream_solver: bool = False,
        poll_seconds: float = 10.0,
    ):
        self.config = config
        self.job_repo = job_repo or JobRepository()
        self.evaluator = evaluator or Evaluator(
            approval_threshold=config.approval_threshold,
            max_iterations=config.max_iterations,
        )
        self.reporter = reporter or JsonlReporter(config.log_path)
        self.on_progress = on_progress
        self.on_solver_event = on_solver_event
        self.stream_solver = stream_solver
        self.poll_seconds = poll_seconds
        self._log_offsets: dict[str, int] = {}
        self._job_launch_times: dict[str, int] = {}

    async def run(self) -> list[SolveResult]:
        self._progress("Selecting issues...")
        issues = await self.select_issues()
        issue_list = ", ".join(f"#{issue.number}" for issue in issues) or "none"
        self._progress(f"Selected {len(issues)} issue(s): {issue_list}")
        self.reporter.log(
            "selected_issues",
            prompt=self.config.issue_selection_prompt,
            issues=[issue.number for issue in issues],
            dry_run=self.config.dry_run,
        )
        if self.config.dry_run:
            self._progress("Dry run: not launching solver jobs.")
            return [
                SolveResult(
                    issue=issue,
                    verdict="dry_run",
                    score=0.0,
                    iterations=0,
                    state="dry_run",
                )
                for issue in issues
            ]

        self._progress("Checking evaluator configuration...")
        await asyncio.to_thread(self.evaluator.validate_configuration)

        semaphore = asyncio.Semaphore(self.config.parallel)

        async def guarded(issue: Issue) -> SolveResult:
            async with semaphore:
                self._progress(f"Starting #{issue.number}")
                result = await self.solve_issue(issue)
                await self.post_process(result)
                self._progress(
                    f"Finished #{issue.number}: {result.verdict} "
                    f"score={result.score:.2f}"
                )
                return result

        return await asyncio.gather(*(guarded(issue) for issue in issues))

    async def select_issues(self) -> list[Issue]:
        selector = IssueSelector(self.config.github_repo)
        return await asyncio.to_thread(
            selector.select,
            self.config.issue_selection_prompt,
            limit=self.config.max_issues,
        )

    async def solve_issue(self, issue: Issue) -> SolveResult:
        review = None
        job_id = None
        try:
            for iteration in range(1, self.config.max_iterations + 1):
                self._progress(f"#{issue.number} iteration {iteration}: launching solver")
                pr_feedback = await self._fetch_pr_feedback(issue)
                if pr_feedback:
                    self._progress(_format_pr_feedback_snapshot(issue, pr_feedback))
                self.reporter.log(
                    "iteration_start",
                    issue=issue.number,
                    iteration=iteration,
                    initial_message=self.config.initial_message,
                    previous_review=review.to_dict() if review else None,
                    pr_feedback=pr_feedback,
                )
                job_id = await self._launch_solver(issue, review, pr_feedback)
                self._progress(
                    f"#{issue.number} iteration {iteration}: solver job {job_id} "
                    f"launched; inspect with `uv run ptq peek {job_id} --log 80`"
                )
                await self._wait_for_job(job_id)
                self._progress(
                    f"#{issue.number} iteration {iteration}: solver stopped, "
                    "reading artifacts"
                )
                solver_output = await self._read_solver_output(
                    issue=issue, job_id=job_id, iteration=iteration
                )
                self._progress(
                    f"#{issue.number} iteration {iteration}: running evaluator"
                )
                review = await asyncio.to_thread(self.evaluator.evaluate, solver_output)
                await self._write_review_artifact(job_id, review)
                self._progress(_format_review_snapshot(review))
                self.reporter.log(
                    "iteration_review",
                    issue=issue.number,
                    job_id=job_id,
                    review=review.to_dict(),
                )
                if review.verdict in {"approved", "shelve"}:
                    self._progress(
                        f"#{issue.number} iteration {iteration}: "
                        f"{review.verdict} score={review.score:.2f}"
                    )
                    return SolveResult(
                        issue=issue,
                        verdict=review.verdict,
                        score=review.score,
                        iterations=iteration,
                        job_id=job_id,
                        review=review,
                        state="completed",
                    )
            assert review is not None
            return SolveResult(
                issue=issue,
                verdict=review.verdict,
                score=review.score,
                iterations=self.config.max_iterations,
                job_id=job_id,
                review=review,
                state="max_iterations",
            )
        except Exception as exc:
            self._progress(f"#{issue.number}: error: {exc}")
            return SolveResult(
                issue=issue,
                verdict="error",
                score=0.0,
                iterations=0,
                job_id=job_id,
                review=review,
                state="error",
                error=str(exc),
            )

    async def post_process(self, result: SolveResult) -> None:
        if result.verdict == "approved" and result.job_id:
            if self.config.push_pr:
                self._progress(
                    f"#{result.issue.number}: approved, pushing draft PR"
                )
                try:
                    pr_result = await self._push_draft_pr(result)
                    result.branch = pr_result.branch
                    result.pr_url = pr_result.url
                    self._progress(
                        f"#{result.issue.number}: draft PR ready: {pr_result.url}"
                    )
                    if self.config.watch_pr:
                        await self._watch_pr(result)
                except Exception as exc:
                    result.error = f"Approved, but failed to push draft PR: {exc}"
                    self._progress(f"#{result.issue.number}: {result.error}")
                self.reporter.log_result(result)
                return
            self._progress(
                f"#{result.issue.number}: approved, preparing local review branch"
            )
            try:
                result.branch = await self._prepare_review_branch(result)
            except Exception as exc:
                result.error = f"Approved, but failed to prepare review branch: {exc}"
                self._progress(f"#{result.issue.number}: {result.error}")
        self.reporter.log_result(result)

    async def _push_draft_pr(self, result: SolveResult):
        from ptq.application.pr_service import create_pr

        note = _build_orchestrator_pr_note(result)
        return await asyncio.to_thread(
            create_pr,
            self.job_repo,
            result.job_id,
            human_note=note,
            draft=True,
            log=lambda msg: self._progress(
                f"#{result.issue.number}: pr: {msg}"
            ),
        )

    async def _watch_pr(self, result: SolveResult) -> None:
        if not result.job_id or not result.pr_url:
            return

        idle_seconds = max(0.0, float(self.config.watch_pr_idle_seconds))
        interval = max(0.0, float(self.config.watch_pr_interval_seconds))
        initial_state = await self._pr_state(result)
        if initial_state in {"closed", "merged"}:
            self._progress(
                f"#{result.issue.number}: PR is {initial_state}; stopping PR watch"
            )
            self.reporter.log(
                "pr_watch_stop",
                issue=result.issue.number,
                pr_url=result.pr_url,
                reason=initial_state,
            )
            return
        last_activity = time.monotonic()
        baseline_feedback = await self._fetch_pr_feedback(result.issue)
        last_fingerprint = _pr_feedback_fingerprint(baseline_feedback)
        self._progress(
            f"#{result.issue.number}: watching PR for review activity "
            f"(idle timeout {idle_seconds / 3600:.1f}h)"
        )
        self.reporter.log(
            "pr_watch_start",
            issue=result.issue.number,
            job_id=result.job_id,
            pr_url=result.pr_url,
            baseline_feedback=baseline_feedback,
            idle_seconds=idle_seconds,
            interval_seconds=interval,
        )

        while True:
            state = await self._pr_state(result)
            if state in {"closed", "merged"}:
                self._progress(
                    f"#{result.issue.number}: PR is {state}; stopping PR watch"
                )
                self.reporter.log(
                    "pr_watch_stop",
                    issue=result.issue.number,
                    pr_url=result.pr_url,
                    reason=state,
                )
                return

            idle_for = time.monotonic() - last_activity
            if idle_for >= idle_seconds:
                self._progress(
                    f"#{result.issue.number}: no PR activity for "
                    f"{idle_for / 3600:.1f}h; stopping PR watch"
                )
                self.reporter.log(
                    "pr_watch_stop",
                    issue=result.issue.number,
                    pr_url=result.pr_url,
                    reason="idle",
                    idle_seconds=idle_for,
                )
                return

            if interval:
                await asyncio.sleep(interval)
            else:
                await asyncio.sleep(0)

            feedback = await self._fetch_pr_feedback(result.issue)
            fingerprint = _pr_feedback_fingerprint(feedback)
            if fingerprint == last_fingerprint:
                continue

            last_activity = time.monotonic()
            last_fingerprint = fingerprint
            if not feedback:
                self._progress(
                    f"#{result.issue.number}: PR feedback/CI cleared; continuing watch"
                )
                self.reporter.log(
                    "pr_watch_activity",
                    issue=result.issue.number,
                    pr_url=result.pr_url,
                    activity="feedback_cleared",
                )
                continue

            self._progress(_format_pr_feedback_snapshot(result.issue, feedback))
            self.reporter.log(
                "pr_watch_activity",
                issue=result.issue.number,
                pr_url=result.pr_url,
                activity="new_feedback",
                feedback=feedback,
            )
            followup = await self.solve_issue(result.issue)
            if followup.verdict == "approved" and followup.job_id:
                try:
                    pr_result = await self._push_draft_pr(followup)
                    followup.branch = pr_result.branch
                    followup.pr_url = pr_result.url
                    result.job_id = followup.job_id
                    result.branch = followup.branch
                    result.pr_url = followup.pr_url
                    self._progress(
                        f"#{result.issue.number}: draft PR updated: {pr_result.url}"
                    )
                except Exception as exc:
                    followup.error = (
                        f"Approved, but failed to update draft PR: {exc}"
                    )
                    self._progress(f"#{result.issue.number}: {followup.error}")
            self.reporter.log_result(followup)

    async def _pr_state(self, result: SolveResult) -> str:
        if not result.job_id:
            return "unknown"
        job = await asyncio.to_thread(self.job_repo.get, result.job_id)
        pr_url = result.pr_url or job.pr_url
        if not pr_url:
            return "unknown"
        backend = backend_for_job(job)
        from ptq.application.pr_service import get_pr_state

        return await asyncio.to_thread(
            get_pr_state,
            backend,
            pr_url,
            force_refresh=True,
            ttl_seconds=0.0,
        )

    async def _prepare_review_branch(self, result: SolveResult) -> str:
        job = await asyncio.to_thread(self.job_repo.get, result.job_id)
        backend = backend_for_job(job)
        profile = get_profile(job.repo)
        job_dir = f"{backend.workspace}/jobs/{result.job_id}"
        worktree = f"{job_dir}/{profile.dir_name}"
        branch = f"ptq/{result.issue.number}"
        quoted_branch = shlex.quote(branch)
        quoted_worktree = shlex.quote(worktree)
        checkout = await asyncio.to_thread(
            backend.run,
            f"cd {quoted_worktree} && git checkout -B {quoted_branch}",
            False,
        )
        if checkout.returncode != 0:
            detail = (checkout.stderr or checkout.stdout or "").strip()
            raise RuntimeError(detail or "git checkout failed")
        return branch

    async def _launch_solver(
        self,
        issue: Issue,
        review,
        pr_feedback: dict | None = None,
    ) -> str:
        backend = create_backend(
            machine=None if self.config.local else self.config.machine,
            local=self.config.local,
        )
        existing_job_id = self._existing_job_id(issue)
        review_feedback_json = _combined_review_feedback(review, pr_feedback)
        request = RunRequest(
            issue_data=issue.raw,
            issue_number=issue.number,
            message=self.config.initial_message,
            machine=None if self.config.local else self.config.machine,
            local=self.config.local,
            follow=False,
            model=self.config.solver_model,
            thinking=self.config.solver_thinking,
            max_turns=self.config.solver_max_turns,
            agent_type=self.config.solver_agent,
            existing_job_id=existing_job_id,
            name=f"orchestrator-{issue.number}",
            repo=self.config.repo,
            review_feedback_json=review_feedback_json,
        )
        launch_time_result = await asyncio.to_thread(backend.run, "date +%s", False)
        try:
            launched_after = int(launch_time_result.stdout.strip())
        except ValueError:
            launched_after = 0
        job_id = await asyncio.to_thread(
            run_service.launch,
            self.job_repo,
            backend,
            request,
            on_progress=lambda msg: self._progress(
                f"#{issue.number}: solver setup: {msg}"
            ),
        )
        self._job_launch_times[job_id] = launched_after
        return job_id

    def _existing_job_id(self, issue: Issue) -> str | None:
        return self.job_repo.find_by_issue(
            issue.number,
            machine=None if self.config.local else self.config.machine,
            local=self.config.local,
            repo=self.config.repo,
        )

    async def _fetch_pr_feedback(self, issue: Issue) -> dict | None:
        existing_job_id = self._existing_job_id(issue)
        if not existing_job_id:
            return None
        from ptq.application.pr_service import fetch_pr_feedback

        return await asyncio.to_thread(
            fetch_pr_feedback,
            self.job_repo,
            existing_job_id,
            log=lambda msg: self._progress(f"#{issue.number}: pr: {msg}"),
        )

    async def _wait_for_job(self, job_id: str) -> None:
        polls = 0
        while True:
            job = await asyncio.to_thread(self.job_repo.get, job_id)
            backend = backend_for_job(job)
            status = await asyncio.to_thread(get_status, job, backend)
            if self.stream_solver:
                await self._stream_solver_log(job_id)
            if status == JobStatus.STOPPED:
                if self.stream_solver:
                    await self._stream_solver_log(job_id)
                await asyncio.to_thread(run_service.finalize_run, backend, job_id, job)
                return
            if await self._fresh_terminal_status_exists(job_id):
                self._progress(
                    f"Solver job {job_id} wrote terminal status.json while its "
                    "process was still alive; stopping leftover process."
                )
                if job.pid is not None:
                    await asyncio.to_thread(backend.kill_pid, job.pid)
                    await asyncio.to_thread(self.job_repo.save_pid, job_id, None)
                await asyncio.to_thread(run_service.finalize_run, backend, job_id, job)
                return
            polls += 1
            if polls == 1 or polls % 6 == 0:
                self._progress(
                    f"Solver job {job_id} still running; "
                    f"peek with `uv run ptq peek {job_id} --log 80`"
                )
            await asyncio.sleep(self.poll_seconds)

    async def _stream_solver_log(self, job_id: str) -> None:
        if self.on_solver_event is None:
            return
        job = await asyncio.to_thread(self.job_repo.get, job_id)
        backend = backend_for_job(job)
        agent = get_agent(job.agent)
        log_file = f"{backend.workspace}/jobs/{job_id}/{agent.log_filename(job.runs)}"
        result = await asyncio.to_thread(backend.run, f"cat {log_file}", False)
        if result.returncode != 0 or not result.stdout:
            return
        offset = self._log_offsets.get(job_id, 0)
        text = result.stdout
        if len(text) <= offset:
            return
        self._log_offsets[job_id] = len(text)
        for line in text[offset:].splitlines():
            stripped = line.strip()
            if not stripped.startswith("{"):
                continue
            try:
                events = agent.parse_stream_line(stripped)
            except (json.JSONDecodeError, ValueError):
                continue
            for event in events:
                self.on_solver_event(job_id, event)

    async def _fresh_terminal_status_exists(self, job_id: str) -> bool:
        job = await asyncio.to_thread(self.job_repo.get, job_id)
        backend = backend_for_job(job)
        job_dir = f"{backend.workspace}/jobs/{job_id}"
        status_path = f"{job_dir}/status.json"
        mtime_result = await asyncio.to_thread(
            backend.run,
            f"stat -c %Y {status_path}",
            False,
        )
        if mtime_result.returncode != 0:
            return False
        try:
            status_mtime = int(mtime_result.stdout.strip())
        except ValueError:
            return False
        min_mtime = self._job_launch_times.get(job_id, 0)
        if min_mtime and status_mtime < min_mtime:
            return False
        status_json = _loads_json(await self._read_remote_text(backend, status_path))
        state = status_json.get("state")
        if state == "ready_for_review":
            return (
                await self._remote_exists(backend, f"{job_dir}/report.md")
                and await self._remote_exists(backend, f"{job_dir}/fix.diff")
            )
        return state == "not_reproducible"

    async def _read_solver_output(
        self, *, issue: Issue, job_id: str, iteration: int
    ) -> SolverOutput:
        job = await asyncio.to_thread(self.job_repo.get, job_id)
        backend = backend_for_job(job)
        job_dir = f"{backend.workspace}/jobs/{job_id}"
        profile = get_profile(job.repo)
        status_json = _loads_json(
            await self._read_remote_text(backend, f"{job_dir}/status.json")
        )
        report_md = await self._read_remote_text(backend, f"{job_dir}/report.md")
        fix_diff = await self._read_remote_text(backend, f"{job_dir}/fix.diff")
        if not status_json and not report_md.strip() and not fix_diff.strip():
            raise RuntimeError(
                f"Solver job {job_id} stopped without status.json, report.md, "
                "or fix.diff. Inspect the agent log with "
                f"`uv run ptq peek {job_id} --log 120`."
            )
        repro_filename = str(status_json.get("repro_file") or "")
        if not repro_filename:
            for candidate in (
                f"repro_{issue.number}.py",
                f"repro_{issue.number}_generated.py",
                "repro.py",
            ):
                if await self._remote_exists(backend, f"{job_dir}/{candidate}"):
                    repro_filename = candidate
                    break
        repro_script = (
            await self._read_remote_text(backend, f"{job_dir}/{repro_filename}")
            if repro_filename
            else ""
        )
        worktree_path = None
        if job.local:
            worktree_path = (
                Path(job_dir.replace("~", str(Path.home()))) / profile.dir_name
            )
        return SolverOutput(
            issue_number=issue.number,
            issue_body=issue.body,
            iteration=iteration,
            report_md=report_md,
            fix_diff=fix_diff,
            repro_script=repro_script,
            repro_filename=repro_filename,
            status_json=status_json,
            worktree_path=worktree_path,
        )

    async def _write_review_artifact(self, job_id: str, review) -> None:
        job = await asyncio.to_thread(self.job_repo.get, job_id)
        backend = backend_for_job(job)
        job_dir = f"{backend.workspace}/jobs/{job_id}"
        text = json.dumps(review.to_dict(), indent=2)
        await asyncio.to_thread(
            backend.run,
            f"cat > {job_dir}/review.json << 'PTQ_REVIEW_EOF'\n{text}\nPTQ_REVIEW_EOF",
        )

    async def _read_remote_text(self, backend, path: str) -> str:
        result = await asyncio.to_thread(backend.run, f"cat {path}", False)
        if result.returncode != 0:
            return ""
        return result.stdout

    async def _remote_exists(self, backend, path: str) -> bool:
        result = await asyncio.to_thread(backend.run, f"test -f {path}", False)
        return result.returncode == 0

    def _progress(self, message: str) -> None:
        if self.on_progress is not None:
            self.on_progress(message)


def _loads_json(text: str) -> dict:
    if not text.strip():
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _pr_feedback_fingerprint(feedback: dict | None) -> str:
    if not feedback:
        return ""
    comments = []
    for item in feedback.get("comments", []):
        if not isinstance(item, dict):
            continue
        comments.append(
            {
                "kind": item.get("kind"),
                "id": item.get("id"),
                "author": item.get("author"),
                "body": item.get("body"),
                "url": item.get("url"),
                "created_at": item.get("created_at") or item.get("submitted_at"),
            }
        )
    ci_failures = []
    for item in feedback.get("ci_failures", []):
        if not isinstance(item, dict):
            continue
        ci_failures.append(
            {
                "name": item.get("name"),
                "state": item.get("state"),
                "workflow": item.get("workflow"),
                "description": item.get("description"),
                "link": item.get("link"),
                "started_at": item.get("started_at"),
                "completed_at": item.get("completed_at"),
            }
        )
    payload = {"comments": comments, "ci_failures": ci_failures}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _format_review_snapshot(review: ReviewResult) -> str:
    lines = [
        "evaluator review: "
        f"{review.verdict} score={review.score:.2f} "
        f"repro={review.repro_fidelity}"
    ]
    for reviewer in review.reviewer_results:
        reviewer_name = reviewer.get("reviewer", "reviewer")
        verdict = reviewer.get("verdict", "")
        score = reviewer.get("score", 0.0)
        try:
            score_text = f"{float(score):.2f}"
        except (TypeError, ValueError):
            score_text = str(score)
        lines.append(f"  - {reviewer_name}: {verdict} score={score_text}")

    blocking = [comment for comment in review.comments if comment.severity == "blocking"]
    if blocking:
        lines.append("  blocking feedback:")
        for comment in blocking[:5]:
            loc = comment.file or "general"
            if comment.line is not None:
                loc = f"{loc}:{comment.line}"
            reviewer = f"[{comment.reviewer}] " if comment.reviewer else ""
            lines.append(f"  - {reviewer}{loc}: {_one_line(comment.comment, 260)}")
        if len(blocking) > 5:
            lines.append(f"  - ... {len(blocking) - 5} more blocking comments")

    if review.summary:
        lines.append(f"  summary: {_one_line(review.summary, 500)}")
    return "\n".join(lines)


def _combined_review_feedback(
    review: ReviewResult | None,
    pr_feedback: dict | None,
) -> str | None:
    if review is None and not pr_feedback:
        return None
    if review is not None and not pr_feedback:
        return json.dumps(review.to_dict(), indent=2)

    feedback: dict = {}
    if review is not None:
        feedback["evaluator_review"] = review.to_dict()
    if pr_feedback:
        feedback["github_pr_feedback"] = pr_feedback
    return json.dumps(feedback, indent=2)


def _format_pr_feedback_snapshot(issue: Issue, feedback: dict) -> str:
    comments = feedback.get("comments")
    count = len(comments) if isinstance(comments, list) else 0
    ci_failures = feedback.get("ci_failures")
    ci_count = len(ci_failures) if isinstance(ci_failures, list) else 0
    lines = [
        f"#{issue.number}: found {count} GitHub PR feedback item(s) "
        f"and {ci_count} failing PR check(s) "
        f"from {feedback.get('pr_url', 'existing PR')}"
    ]
    if isinstance(comments, list):
        for comment in comments[:5]:
            if not isinstance(comment, dict):
                continue
            loc = str(comment.get("path") or comment.get("kind") or "comment")
            line = comment.get("line")
            if line is not None:
                loc = f"{loc}:{line}"
            author = str(comment.get("author") or "reviewer")
            body = _one_line(str(comment.get("body") or ""), 260)
            lines.append(f"  - [{author}] {loc}: {body}")
        if count > 5:
            lines.append(f"  - ... {count - 5} more PR feedback item(s)")
    if isinstance(ci_failures, list) and ci_failures:
        lines.append("  failing CI:")
        for check in ci_failures[:5]:
            if not isinstance(check, dict):
                continue
            name = str(check.get("name") or "check")
            state = str(check.get("state") or check.get("bucket") or "failed")
            desc = _one_line(str(check.get("description") or ""), 180)
            suffix = f": {desc}" if desc else ""
            lines.append(f"  - {name} ({state}){suffix}")
        if ci_count > 5:
            lines.append(f"  - ... {ci_count - 5} more failing PR check(s)")
    return "\n".join(lines)


def _build_orchestrator_pr_note(result: SolveResult) -> str:
    lines = [
        "Automated draft PR generated by ptq after solver and evaluator approval.",
        "",
        f"Issue: #{result.issue.number} {result.issue.title}",
        f"Evaluator score: {result.score:.2f}",
    ]
    return "\n".join(lines)


def _one_line(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."
