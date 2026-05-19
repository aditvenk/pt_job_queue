from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from ptq.orchestrator.models import SolveResult


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


class JsonlReporter:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, **payload: Any) -> None:
        record = {"event": event, **payload}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=_json_default, sort_keys=True) + "\n")

    def log_result(self, result: SolveResult) -> None:
        self.log("solve_result", result=result)


def read_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        out.append(json.loads(line))
    return out
