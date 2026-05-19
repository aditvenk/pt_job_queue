#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import socketserver
import subprocess
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


def main() -> None:
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

    args = parser.parse_args()
    if args.cmd == "serve":
        serve(args.socket)
    elif args.cmd == "fetch-issue":
        print(json.dumps(fetch_issue(args.repo, args.issue), indent=2))
    elif args.cmd == "search-issues":
        print(json.dumps(search_issues(args.repo, args.query, args.limit), indent=2))


if __name__ == "__main__":
    main()
