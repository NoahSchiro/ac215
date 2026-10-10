"""Train the gaze model.

    uv run python train.py --val 13 --test 14 --epochs 30

Splits are participant-disjoint: a subject never appears in both train and
eval, because generalizing to an unseen face is what we are measuring.
"""

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.data.mpiifacegaze import MPII
from src.training.logging import ConsoleLogger
from src.training.loop import TrainConfig, evaluate, fit
from src.training.model import GazeNet

NUM_PARTICIPANTS = 15


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    data = parser.add_argument_group("data")
    data.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "MPIIFaceGaze",
    )
    data.add_argument("--val", type=int, default=13)
    data.add_argument("--test", type=int, default=14)

    model = parser.add_argument_group("model")
    model.add_argument(
        "--output-mode",
        choices=["screen_fraction", "gaze_direction"],
        default="screen_fraction",
    )
    model.add_argument("--no-eyes", action="store_true", help="ablate the eye towers")
    model.add_argument("--no-pretrained", action="store_true", help="random init")

    optim = parser.add_argument_group("optimization")
    optim.add_argument("--epochs", type=int, default=30)
    optim.add_argument("--lr", type=float, default=1e-4)
    optim.add_argument("--weight-decay", type=float, default=1e-4)
    optim.add_argument("--batch-size", type=int, default=256)
    optim.add_argument("--workers", type=int, default=8)
    optim.add_argument("--seed", type=int, default=0)

    run = parser.add_argument_group("run")
    run.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    run.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    run.add_argument("--resume", action="store_true")
    run.add_argument("--wandb", metavar="PROJECT", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for name, participant in (("--val", args.val), ("--test", args.test)):
        if not 0 <= participant < NUM_PARTICIPANTS:
            raise SystemExit(f"{name} must be in 0..{NUM_PARTICIPANTS - 1}")
    if args.val == args.test:
        raise SystemExit("--val and --test must be different participants")

    train_ids = [p for p in range(NUM_PARTICIPANTS) if p not in (args.val, args.test)]
    print(f"train {train_ids}  val [{args.val}]  test [{args.test}]")

    def loader(participants: int | list[int], *, shuffle: bool) -> DataLoader[Any]:
        return DataLoader(
            MPII(participants, args.data_root),
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.workers,
            drop_last=shuffle,
            persistent_workers=args.workers > 0,
        )

    if args.device != "auto":
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = GazeNet(
        output_mode=args.output_mode,
        pretrained=not args.no_pretrained,
        use_eyes=not args.no_eyes,
    )
    print(f"{model.parameter_count() / 1e6:.2f}M parameters on {device}")

    config = TrainConfig(
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        num_workers=args.workers,
        seed=args.seed,
    )

    logger: Any = ConsoleLogger(args.checkpoint_dir / "run.jsonl")
    if args.wandb is not None:
        # Imported lazily so wandb stays an optional dependency.
        import wandb

        logger = wandb.init(project=args.wandb, config=vars(args))

    best = fit(
        model,
        loader(train_ids, shuffle=True),
        loader(args.val, shuffle=False),
        device,
        config,
        args.checkpoint_dir,
        logger,
        resume=args.resume,
    )
    print(f"\nbest val mean {best.mean:.5f} (median {best.median:.5f})")
    worst = best.worst_participant()
    if worst is not None:
        print(f"worst participant: {worst[0]} at {worst[1].mean:.5f}")

    stats, test_loss = evaluate(model, loader(args.test, shuffle=False), device)
    print(
        f"test mean {stats.mean:.5f} (loss {test_loss:.5f}) over {stats.count} samples"
    )


if __name__ == "__main__":
    main()
