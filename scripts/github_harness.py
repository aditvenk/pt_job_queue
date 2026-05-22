#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import socket
import socketserver
import subprocess
import sys
from pathlib import Path
from typing import Any

DEFAULT_SOCKET = Path.home() / ".ptq" / "github_harness.sock"
DEFAULT_ENV = {
    "https_proxy": "http://fwdproxy:8080",
    "http_proxy": "http://fwdproxy:8080",
    "no_proxy": (
        ".fbcdn.net,.facebook.com,.thefacebook.com,.tfbnw.net,.fb.com,"
        ".fburl.com,.facebook.net,.sb.fbsbx.com,localhost"
    ),
}


def github_env() -> dict[str, str]:
    env = os.environ.copy()
    for key, value in DEFAULT_ENV.items():
        env.setdefault(key, value)
    return env


def harness_socket_path() -> Path:
    raw = os.environ.get("PTQ_GITHUB_HARNESS_SOCKET")
    return Path(raw).expanduser() if raw else DEFAULT_SOCKET


def harness_available(socket_path: Path | None = None) -> bool:
    return (socket_path or harness_socket_path()).exists()


def call_harness(
    action: str,
    *,
    socket_path: Path | None = None,
    timeout: int = 120,
    **payload: Any,
) -> Any:
    path = (socket_path or harness_socket_path()).expanduser()
    request = json.dumps({"action": action, **payload}).encode("utf-8") + b"\n"
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(timeout)
            client.connect(str(path))
            client.sendall(request)
            chunks: list[bytes] = []
            while True:
                chunk = client.recv(65536)
                if not chunk:
                    break
                chunks.append(chunk)
    except OSError as exc:
        raise RuntimeError(f"GitHub harness unavailable at {path}: {exc}") from exc

    try:
        response = json.loads(b"".join(chunks).decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError("GitHub harness returned invalid JSON.") from exc

    if not response.get("ok"):
        error = response.get("error") or "GitHub harness request failed."
        raise RuntimeError(str(error))
    return response.get("data")


def run_gh(args: list[str]) -> Any:
    result = subprocess.run(
        ["gh", *args],
        env=github_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        raise RuntimeError(stderr or stdout or f"gh exited {result.returncode}")
    return json.loads(result.stdout or "null")


def run_gh_raw(args: list[str], cwd: str | None = None) -> dict[str, Any]:
    result = subprocess.run(
        ["gh", *args],
        cwd=cwd or None,
        env=github_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def fetch_issue(repo: str, issue: int) -> dict:
    return run_gh(
        [
            "issue",
            "view",
            str(issue),
            "--repo",
            repo,
            "--json",
            "title,body,comments,labels",
        ]
    )


def search_issues(repo: str, query: str, limit: int) -> list[dict]:
    return run_gh(
        [
            "search",
            "issues",
            query,
            "--repo",
            repo,
            "--json",
            "number,title,labels,url",
            "--limit",
            str(limit),
        ]
    )


def client_fetch_issue(repo: str, issue: int) -> dict:
    if harness_available():
        try:
            return call_harness("fetch_issue", repo=repo, issue_number=issue)
        except RuntimeError as exc:
            if "GitHub harness unavailable" not in str(exc):
                raise
    return fetch_issue(repo, issue)


def client_search_issues(repo: str, query: str, limit: int) -> list[dict]:
    if harness_available():
        try:
            return call_harness("search_issues", repo=repo, query=query, limit=limit)
        except RuntimeError as exc:
            if "GitHub harness unavailable" not in str(exc):
                raise
    return search_issues(repo, query, limit)


def client_run_gh(args: list[str], cwd: str | None = None) -> dict[str, Any]:
    if not harness_available():
        path = harness_socket_path()
        raise RuntimeError(
            f"GitHub harness socket not found at {path}. "
            "Start it outside the agent with `python scripts/github_harness.py serve`."
        )
    data = call_harness("gh", args=args, cwd=cwd)
    if not isinstance(data, dict):
        raise RuntimeError("GitHub harness returned an invalid gh response.")
    return data


def handle_request(request: dict) -> Any:
    action = request.get("action")
    if action == "fetch_issue":
        return fetch_issue(str(request["repo"]), int(request["issue_number"]))
    if action == "search_issues":
        return search_issues(
            str(request["repo"]),
            str(request["query"]),
            int(request.get("limit", 20)),
        )
    if action == "gh":
        args = request.get("args")
        if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
            raise RuntimeError("gh action requires string list 'args'.")
        cwd = request.get("cwd")
        if cwd is not None and not isinstance(cwd, str):
            raise RuntimeError("gh action 'cwd' must be a string when provided.")
        return run_gh_raw(args, cwd=cwd)
    raise RuntimeError(f"Unknown action: {action}")


class Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        try:
            raw = self.rfile.readline()
            request = json.loads(raw.decode("utf-8"))
            data = handle_request(request)
            response = {"ok": True, "data": data}
        except Exception as exc:
            response = {"ok": False, "error": str(exc)}
        self.wfile.write(json.dumps(response).encode("utf-8"))
        self.wfile.write(b"\n")


class UnixServer(socketserver.ThreadingUnixStreamServer):
    allow_reuse_address = True


def serve(socket_path: Path) -> None:
    socket_path = socket_path.expanduser()
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    if socket_path.exists():
        socket_path.unlink()
    with UnixServer(str(socket_path), Handler) as server:
        socket_path.chmod(0o600)
        print(f"github_harness_socket={socket_path}", flush=True)
        server.serve_forever()


def _main_gh(argv: list[str]) -> None:
    cwd = None
    gh_args = list(argv)
    if gh_args and gh_args[0] == "--cwd":
        if len(gh_args) < 2:
            raise RuntimeError("gh --cwd requires a path.")
        cwd = gh_args[1]
        gh_args = gh_args[2:]
    elif gh_args and gh_args[0].startswith("--cwd="):
        cwd = gh_args[0].split("=", 1)[1]
        gh_args = gh_args[1:]
    if not gh_args:
        raise RuntimeError(
            "gh requires arguments, for example: gh pr view <url> --json title"
        )
    result = client_run_gh(gh_args, cwd=cwd)
    sys.stdout.write(str(result.get("stdout") or ""))
    sys.stderr.write(str(result.get("stderr") or ""))
    raise SystemExit(int(result.get("returncode") or 0))


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "gh":
        _main_gh(argv[1:])
        return

    parser = argparse.ArgumentParser(description="PTQ GitHub access harness.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    serve_parser = sub.add_parser("serve", help="Serve JSON requests on a Unix socket.")
    serve_parser.add_argument("--socket", type=Path, default=DEFAULT_SOCKET)

    fetch_parser = sub.add_parser("fetch-issue", help="Fetch one issue as JSON.")
    fetch_parser.add_argument("issue", type=int)
    fetch_parser.add_argument("--repo", default="pytorch/pytorch")

    search_parser = sub.add_parser("search-issues", help="Search issues as JSON.")
    search_parser.add_argument("query")
    search_parser.add_argument("--repo", default="pytorch/pytorch")
    search_parser.add_argument("--limit", type=int, default=20)

    sub.add_parser("gh", help="Run gh through the harness socket.")

    args = parser.parse_args(argv)
    if args.cmd == "serve":
        serve(args.socket)
    elif args.cmd == "fetch-issue":
        print(json.dumps(client_fetch_issue(args.repo, args.issue), indent=2))
    elif args.cmd == "search-issues":
        data = client_search_issues(args.repo, args.query, args.limit)
        print(json.dumps(data, indent=2))


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
