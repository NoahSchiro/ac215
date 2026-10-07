"""Train and evaluate loops, with checkpointing.

Checkpointing is written in from the start rather than retrofitted: training
runs on a rented VM that can be preempted, so a run has to survive losing its
machine. Every epoch writes `last.pt`, and improvements write `best.pt`; both
are plain files, so pointing them at a mounted bucket is all the cloud
integration needed.

Experiment tracking goes through `RunLogger`, which is a protocol rather than
a hard dependency. A W&B run object satisfies it as-is, so wiring it in is an
import in the entrypoint, not a change here.
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.training.metrics import (
    ErrorStats,
    angular_error_deg,
    screen_fraction_error,
    summarize,
)
from src.training.model import GazeNet, prepare_images


class RunLogger(Protocol):
    """Minimal surface we need from an experiment tracker (wandb fits)."""

    def log(self, data: dict[str, Any]) -> None: ...


@dataclass
class TrainConfig:
    epochs: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 64
    num_workers: int = 4
    seed: int = 0
    grad_clip: float | None = 1.0


def subject_of(sample_id: str) -> str:
    """ "mpiifacegaze/p03/day01/0005" -> "p03"."""
    parts = sample_id.split("/")
    if len(parts) < 2:
        raise ValueError(f"cannot read a subject from sample_id {sample_id!r}")
    return parts[1]


def loss_for(output_mode: str) -> nn.Module:
    """Screen fractions are coordinates (MSE); directions are angles (cosine)."""
    if output_mode == "screen_fraction":
        return nn.MSELoss()
    if output_mode == "gaze_direction":
        return nn.CosineEmbeddingLoss()
    raise ValueError(f"no loss defined for output_mode {output_mode!r}")


def _forward(
    model: GazeNet, batch: dict[str, Any], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move one batch to device and run it. Returns (prediction, target)."""
    face = prepare_images(batch["face"], device)
    if model.use_eyes:
        prediction = model(
            face,
            prepare_images(batch["left_eye"], device),
            prepare_images(batch["right_eye"], device),
        )
    else:
        prediction = model(face)
    return prediction, batch["label"].to(device, non_blocking=True).float()


def _criterion_value(
    criterion: nn.Module, prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    if isinstance(criterion, nn.CosineEmbeddingLoss):
        ones = torch.ones(prediction.shape[0], device=prediction.device)
        return criterion(prediction, target, ones)
    return criterion(prediction, target)


def train_one_epoch(
    model: GazeNet,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float | None = None,
) -> float:
    """One pass over the training set. Returns mean loss."""
    model.train()
    total = 0.0
    seen = 0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        prediction, target = _forward(model, batch, device)
        loss = _criterion_value(criterion, prediction, target)
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        count = prediction.shape[0]
        total += float(loss.item()) * count
        seen += count
    return total / max(seen, 1)


@torch.no_grad()
def evaluate(
    model: GazeNet,
    loader: DataLoader[Any],
    device: torch.device,
) -> ErrorStats:
    """Error over the whole loader, broken down per participant."""
    model.eval()
    errors: list[torch.Tensor] = []
    subjects: list[str] = []
    for batch in loader:
        prediction, target = _forward(model, batch, device)
        if model.output_mode == "screen_fraction":
            errors.append(screen_fraction_error(prediction, target).cpu())
        else:
            errors.append(angular_error_deg(prediction, target).cpu())
        subjects.extend(subject_of(sample_id) for sample_id in batch["sample_id"])
    if not errors:
        raise ValueError("evaluation loader produced no batches")
    return summarize(torch.cat(errors), subjects)


def save_checkpoint(
    path: Path,
    model: GazeNet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_error: float,
    config: TrainConfig,
) -> None:
    """Everything needed to resume on a different machine."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_error": best_error,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "output_mode": model.output_mode,
            "use_eyes": model.use_eyes,
            "config": asdict(config),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: GazeNet,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[int, float]:
    """Restore weights (and optimizer, if given). Returns (next_epoch, best)."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["output_mode"] != model.output_mode:
        raise ValueError(
            f"checkpoint is {state['output_mode']}, model is {model.output_mode}"
        )
    model.load_state_dict(state["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer_state"])
    return int(state["epoch"]) + 1, float(state["best_error"])


def fit(
    model: GazeNet,
    train_loader: DataLoader[Any],
    val_loader: DataLoader[Any],
    device: torch.device,
    config: TrainConfig,
    checkpoint_dir: Path,
    logger: RunLogger | None = None,
    resume: bool = False,
) -> ErrorStats:
    """Train for `config.epochs`, returning the best validation stats."""
    torch.manual_seed(config.seed)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = loss_for(model.output_mode)

    start_epoch = 0
    best_error = float("inf")
    last_path = checkpoint_dir / "last.pt"
    if resume and last_path.is_file():
        start_epoch, best_error = load_checkpoint(last_path, model, optimizer)
        print(f"resumed from {last_path} at epoch {start_epoch}")

    best_stats: ErrorStats | None = None
    last_stats: ErrorStats | None = None
    for epoch in range(start_epoch, config.epochs):
        started = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, config.grad_clip
        )
        stats = evaluate(model, val_loader, device)
        last_stats = stats
        elapsed = time.time() - started

        improved = stats.mean < best_error
        if improved:
            best_error = stats.mean
            best_stats = stats
            save_checkpoint(
                checkpoint_dir / "best.pt", model, optimizer, epoch, best_error, config
            )
        save_checkpoint(last_path, model, optimizer, epoch, best_error, config)

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_mean": stats.mean,
            "val_median": stats.median,
            "val_p95": stats.p95,
            "seconds": elapsed,
        }
        if logger is not None:
            logger.log(
                record | {f"val_{k}": v for k, v in stats.per_participant.items()}
            )
        marker = " *" if improved else ""
        print(
            f"epoch {epoch:3d}  loss {train_loss:.5f}"
            f"  val {stats.mean:.5f} (p95 {stats.p95:.5f})  {elapsed:.0f}s{marker}"
        )

    if best_stats is None:
        # A resumed run that never beat the earlier best is a normal outcome,
        # so report the final epoch and leave the previous best_metrics.json
        # (which describes a better checkpoint) alone.
        if last_stats is None:
            raise RuntimeError("training ran zero epochs; nothing to report")
        return last_stats

    (checkpoint_dir / "best_metrics.json").write_text(
        json.dumps(asdict(best_stats), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return best_stats
