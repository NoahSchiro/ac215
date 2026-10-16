"""Run logging: terminal output plus a persistent copy on disk."""

import json
from pathlib import Path
from typing import Any, Protocol


class RunLogger(Protocol):
    """Minimal surface a tracker needs. A wandb run satisfies this as-is."""

    def log(self, data: dict[str, Any]) -> None: ...


class ConsoleLogger:
    """Prints each record and appends it to a JSON-lines file."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def log(self, data: dict[str, Any]) -> None:
        epoch = data.get("epoch")
        summary = "  ".join(
            f"{k} {v:.5f}" if isinstance(v, float) else f"{k} {v}"
            for k, v in data.items()
            if k != "epoch"
        )
        print(f"epoch {epoch:3d}  {summary}" if epoch is not None else summary)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(data, sort_keys=True) + "\n")
