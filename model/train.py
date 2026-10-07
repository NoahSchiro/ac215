"""Train the gaze model on MPIIFaceGaze.

    uv run python train.py --val 13 --test 14 --epochs 30

Splits are participant-disjoint: a subject never appears in both train and
eval, because generalizing to a new face is the thing we are measuring.
"""

import argparse
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from src.data.mpiifacegaze import MPII
from src.training.loop import TrainConfig, evaluate, fit
from src.training.model import GazeNet, select_device
from src.training.splits import Split, holdout_split


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    data = parser.add_argument_group("data")
    data.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "MPIIFaceGaze",
    )
    data.add_argument("--val", type=int, nargs="+", default=[13])
    data.add_argument("--test", type=int, nargs="+", default=None)

    model = parser.add_argument_group("model")
    model.add_argument(
        "--output-mode",
        choices=["screen_fraction", "gaze_direction"],
        default="screen_fraction",
    )
    model.add_argument("--no-eyes", action="store_true", help="ablate the eye towers")
    model.add_argument(
        "--no-pretrained",
        action="store_true",
        help="random init (expect this to be worse)",
    )

    optim = parser.add_argument_group("optimization")
    optim.add_argument("--epochs", type=int, default=30)
    optim.add_argument("--lr", type=float, default=1e-4)
    optim.add_argument("--weight-decay", type=float, default=1e-4)
    optim.add_argument("--batch-size", type=int, default=64)
    optim.add_argument("--workers", type=int, default=4)
    optim.add_argument("--seed", type=int, default=0)

    run = parser.add_argument_group("run")
    run.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    run.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    run.add_argument("--resume", action="store_true")
    run.add_argument("--wandb", metavar="PROJECT", default=None)
    return parser


def make_logger(project: str | None, config: dict[str, Any]) -> Any:
    """Optional W&B. Kept out of the training code so it stays an optional dep."""
    if project is None:
        return None
    # Imported lazily so wandb stays an optional dependency.
    import wandb

    return wandb.init(project=project, config=config)


def loaders(
    split: Split, args: argparse.Namespace
) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any] | None]:
    """Build one loader per split half. Only train is shuffled."""

    def loader(participants: tuple[int, ...], *, shuffle: bool) -> DataLoader[Any]:
        dataset = MPII(list(participants), args.data_root)
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.workers,
            drop_last=shuffle,
            persistent_workers=args.workers > 0,
        )

    test = loader(split.test, shuffle=False) if split.test else None
    return loader(split.train, shuffle=True), loader(split.val, shuffle=False), test


def main() -> None:
    args = build_parser().parse_args()
    split = holdout_split(val=args.val, test=args.test)
    print(f"train {list(split.train)}  val {list(split.val)}  test {list(split.test)}")

    model = GazeNet(
        args.output_mode,
        pretrained=not args.no_pretrained,
        use_eyes=not args.no_eyes,
    )
    device = select_device(args.device)
    print(f"{model.parameter_count() / 1e6:.2f}M parameters on {device}")

    config = TrainConfig(
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        num_workers=args.workers,
        seed=args.seed,
    )
    train_loader, val_loader, test_loader = loaders(split, args)
    logger = make_logger(args.wandb, vars(args) | {"split": list(split.train)})

    best = fit(
        model,
        train_loader,
        val_loader,
        device,
        config,
        args.checkpoint_dir,
        logger=logger,
        resume=args.resume,
    )
    print(f"\nbest val mean {best.mean:.5f} (median {best.median:.5f})")
    worst = best.worst_participant()
    if worst is not None:
        print(f"worst participant: {worst[0]} at {worst[1]:.5f}")

    if test_loader is not None:
        stats = evaluate(model, test_loader, device)
        print(f"test mean {stats.mean:.5f} over {stats.count} samples")


if __name__ == "__main__":
    main()
