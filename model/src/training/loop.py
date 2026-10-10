"""Train and evaluate loops, with checkpointing and resume."""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.utils import prepare_images
from src.training.logging import RunLogger
from src.training.metrics import (
    ErrorStats,
    angular_error_deg,
    screen_fraction_error,
    summarize,
)
from src.training.model import GazeNet


@dataclass
class TrainConfig:
    epochs: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 256
    num_workers: int = 8
    seed: int = 0
    grad_clip: float | None = 1.0


def forward_batch(
    model: GazeNet, batch: dict[str, Any], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move one collated batch to device and run it. Returns (prediction, target).

    `batch` is a dict of stacked tensors, not a list of per-sample dicts:
    torch's default collate turns a list of dicts into a dict of batches.
    """
    face = prepare_images(batch["face"], device, imagenet=model.pretrained)
    if model.use_eyes:
        prediction = model(
            face,
            prepare_images(batch["left_eye"], device, imagenet=model.pretrained),
            prepare_images(batch["right_eye"], device, imagenet=model.pretrained),
        )
    else:
        prediction = model(face)
    return prediction, batch["label"].to(device, non_blocking=True).float()


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
        prediction, target = forward_batch(model, batch, device)
        if model.output_mode == "gaze_direction":
            ones = torch.ones(prediction.shape[0], device=device)
            loss = criterion(prediction, target, ones)
        else:
            loss = criterion(prediction, target)
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
    model: GazeNet, loader: DataLoader[Any], device: torch.device
) -> ErrorStats:
    """Error over the whole loader, broken down per participant."""
    model.eval()
    errors: list[torch.Tensor] = []
    subjects: list[str] = []
    for batch in loader:
        prediction, target = forward_batch(model, batch, device)
        if model.output_mode == "screen_fraction":
            errors.append(screen_fraction_error(prediction, target).cpu())
        else:
            errors.append(angular_error_deg(prediction, target).cpu())
        # sample_id looks like "<dataset>/<subject>/..."
        subjects.extend(sample_id.split("/")[1] for sample_id in batch["sample_id"])
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
    """Everything needed to resume, including on another machine."""
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
    path: Path, model: GazeNet, optimizer: torch.optim.Optimizer | None = None
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
    logger: RunLogger,
    resume: bool = False,
) -> ErrorStats:
    """Train for `config.epochs`, returning the best validation stats."""
    torch.manual_seed(config.seed)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion: nn.Module = (
        nn.CosineEmbeddingLoss()
        if model.output_mode == "gaze_direction"
        else nn.MSELoss()
    )

    start_epoch = 0
    best_error = float("inf")
    last_path = checkpoint_dir / "last.pt"
    if resume and last_path.is_file():
        start_epoch, best_error = load_checkpoint(last_path, model, optimizer)
        print(f"resumed from {last_path} at epoch {start_epoch}")

    best_stats: ErrorStats | None = None
    last_stats: ErrorStats | None = None
    for epoch in range(start_epoch, config.epochs):
        started = time.monotonic()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, config.grad_clip
        )
        stats = evaluate(model, val_loader, device)
        last_stats = stats

        improved = stats.mean < best_error
        if improved:
            best_error = stats.mean
            best_stats = stats
            save_checkpoint(
                checkpoint_dir / "best.pt", model, optimizer, epoch, best_error, config
            )
        save_checkpoint(last_path, model, optimizer, epoch, best_error, config)

        logger.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_mean": stats.mean,
                "val_median": stats.median,
                "val_p95": stats.p95,
                "improved": improved,
                "seconds": time.monotonic() - started,
                **{
                    f"val_{subject}": s.mean
                    for subject, s in stats.per_participant.items()
                },
            }
        )

    if best_stats is None:
        # A resumed run that never beat the earlier best is a normal outcome;
        # report the final epoch and leave the previous best_metrics.json.
        if last_stats is None:
            raise RuntimeError("training ran zero epochs; nothing to report")
        return last_stats

    (checkpoint_dir / "best_metrics.json").write_text(
        json.dumps(asdict(best_stats), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return best_stats
