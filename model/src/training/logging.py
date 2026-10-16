"""Run logging: terminal output, a persistent copy on disk, and wandb."""

import json
import time
from pathlib import Path
from typing import Any, Protocol

import wandb


class RunLogger(Protocol):
    """Minimal surface the training loop needs to report progress."""

    def log(self, data: dict[str, Any]) -> None: ...
    def summarize(self, data: dict[str, Any]) -> None: ...
    def finish(self) -> None: ...


class ConsoleLogger(RunLogger):
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

    def summarize(self, data: dict[str, Any]) -> None:
        self.log(data)

    def finish(self) -> None: ... # noop for this class


class WandbLogger(RunLogger):
    """Starts a run and streams each record to it, one curve per key."""

    def __init__(self, project: str, config: dict[str, Any]) -> None:
        self.run = wandb.init(
            entity="noahschiro-harvard-university",
            project=project,
            config={
                key: str(value) if isinstance(value, Path) else value
                for key, value in config.items()
            },
            # turns off stats I don't care about
            settings=wandb.Settings(x_disable_stats=True),
        )
        # makes x-axis = epoch
        self.run.define_metric("epoch", hidden=True)
        self.run.define_metric("*", step_metric="epoch")

    def log(self, data: dict[str, Any]) -> None:
        """Log one data point to the current epoch."""
        self.run.log(data, step=data.get("epoch"))

    def summarize(self, data: dict[str, Any]) -> None:
        """Record final results (best validation, test) on the run summary."""
        self.run.summary.update(data)

    def finish(self) -> None:
        """Cleanup."""
        self.run.finish()

def resolve_run_dir(parent: Path, resume: bool) -> Path:
    """Where this run writes: a fresh <timestamp> directory under `parent`.

    Weights (best.pt, last.pt), metrics and the local run log all land flat
    in that one directory, so successive runs never overwrite each other.
    With --resume, the newest run under `parent` that still has weights is
    reused instead, so its checkpoints keep updating in place.
    """
    if resume:
        for run_dir in sorted(parent.glob("*"), reverse=True):
            if (run_dir / "last.pt").is_file():
                return run_dir
        print("no previous run found to resume; starting a new one")
    return parent / time.strftime("%Y%m%d-%H%M%S")
