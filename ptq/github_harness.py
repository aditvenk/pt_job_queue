from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from typing import Any

from ptq.domain.models import PtqError

DEFAULT_SOCKET_PATH = Path.home() / ".ptq" / "github_harness.sock"


def harness_socket_path() -> Path:
    raw = os.environ.get("PTQ_GITHUB_HARNESS_SOCKET")
    return Path(raw).expanduser() if raw else DEFAULT_SOCKET_PATH


def harness_available() -> bool:
    return harness_socket_path().exists()


def call_github_harness(action: str, **payload: Any) -> Any:
    path = harness_socket_path()
    request = json.dumps({"action": action, **payload}).encode("utf-8") + b"\n"
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(120)
            client.connect(str(path))
            client.sendall(request)
            chunks: list[bytes] = []
            while True:
                chunk = client.recv(65536)
                if not chunk:
                    break
                chunks.append(chunk)
    except OSError as exc:
        raise PtqError(f"GitHub harness unavailable at {path}: {exc}") from exc

    try:
        response = json.loads(b"".join(chunks).decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise PtqError("GitHub harness returned invalid JSON.") from exc

    if not response.get("ok"):
        raise PtqError(str(response.get("error") or "GitHub harness request failed."))
    return response.get("data")
