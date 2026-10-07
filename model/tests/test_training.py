"""Tests for splits, metrics and the training loop.

No MediaPipe here: these use a synthetic in-memory dataset, so the whole file
runs on macOS where MediaPipe aborts in its Metal path.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.training.loop import (
    TrainConfig,
    evaluate,
    fit,
    load_checkpoint,
    loss_for,
    save_checkpoint,
    subject_of,
)
from src.training.metrics import (
    angular_error_deg,
    cm_to_degrees,
    fraction_to_cm,
    screen_fraction_error,
    summarize,
)
from src.training.model import GazeNet, prepare_images, select_device
from src.training.splits import (
    ALL_PARTICIPANTS,
    Split,
    holdout_split,
    leave_one_out_folds,
)


class FakeGaze(Dataset[dict[str, Any]]):
    """Tiny stand-in for MPII with the same keys and dtypes."""

    def __init__(self, subjects: tuple[int, ...], per_subject: int = 4) -> None:
        self.items: list[dict[str, Any]] = []
        rng = np.random.default_rng(0)
        for subject in subjects:
            for i in range(per_subject):
                self.items.append(
                    {
                        "sample_id": f"mpiifacegaze/p{subject:02d}/day01/{i:04d}",
                        "face": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
                        "left_eye": rng.integers(0, 255, (36, 60, 3), dtype=np.uint8),
                        "right_eye": rng.integers(0, 255, (36, 60, 3), dtype=np.uint8),
                        "label": np.array([0.4, 0.6], dtype=np.float32),
                        "R_norm": np.eye(3),
                    }
                )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.items[idx]


# --- splits ---------------------------------------------------------------


def test_holdout_puts_everyone_else_in_train() -> None:
    split = holdout_split(val=13, test=14)
    assert split.val == (13,)
    assert split.test == (14,)
    assert split.train == tuple(p for p in ALL_PARTICIPANTS if p not in (13, 14))
    assert len(split.train) == 13


def test_rejects_a_participant_in_two_splits() -> None:
    with pytest.raises(ValueError, match="participant-disjoint"):
        Split(train=(0, 1, 2), val=(2,))


def test_rejects_out_of_range_and_duplicates() -> None:
    with pytest.raises(ValueError, match="out of range"):
        Split(train=(0,), val=(99,))
    with pytest.raises(ValueError, match="duplicate"):
        Split(train=(0, 0), val=(1,))


def test_leave_one_out_covers_every_participant_exactly_once() -> None:
    folds = leave_one_out_folds()
    assert len(folds) == 15
    assert sorted(fold.val[0] for fold in folds) == list(ALL_PARTICIPANTS)
    for fold in folds:
        assert len(fold.train) == 14
        assert not set(fold.train) & set(fold.val)


# --- metrics --------------------------------------------------------------


def test_screen_fraction_error_is_euclidean() -> None:
    predicted = torch.tensor([[0.0, 0.0], [0.5, 0.5]])
    target = torch.tensor([[3.0, 4.0], [0.5, 0.5]])
    assert screen_fraction_error(predicted, target).tolist() == [5.0, 0.0]


def test_angular_error_on_known_angles() -> None:
    predicted = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    target = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    assert angular_error_deg(predicted, target).tolist() == pytest.approx(
        [0.0, 90.0], abs=1e-4
    )


def test_fraction_converts_to_cm_then_degrees() -> None:
    # 290x180 mm screen -> 34.1 cm diagonal; 10% error -> 3.4 cm -> ~3.5 deg
    cm = fraction_to_cm(0.1, 290.0, 180.0)
    assert cm == pytest.approx(3.41, abs=0.02)
    assert cm_to_degrees(cm, 55.0) == pytest.approx(3.55, abs=0.05)


def test_summarize_breaks_down_per_participant() -> None:
    errors = torch.tensor([0.1, 0.3, 0.2, 0.4])
    stats = summarize(errors, ["p00", "p00", "p01", "p01"])
    assert stats.count == 4
    assert stats.mean == pytest.approx(0.25)
    assert stats.per_participant == pytest.approx({"p00": 0.2, "p01": 0.3})
    assert stats.worst_participant() == ("p01", pytest.approx(0.3))


def test_summarize_rejects_mismatched_subject_ids() -> None:
    with pytest.raises(ValueError, match="subject ids"):
        summarize(torch.tensor([0.1, 0.2]), ["p00"])


# --- model ----------------------------------------------------------------


@pytest.mark.parametrize("mode,dim", [("screen_fraction", 2), ("gaze_direction", 3)])
def test_output_shape_and_range_per_mode(mode: str, dim: int) -> None:
    model = GazeNet(mode, pretrained=False)  # type: ignore[arg-type]
    out = model(
        torch.zeros(2, 3, 224, 224),
        torch.zeros(2, 3, 36, 60),
        torch.zeros(2, 3, 36, 60),
    )
    assert out.shape == (2, dim)
    if mode == "screen_fraction":
        assert torch.all((out >= 0) & (out <= 1))
    else:
        assert torch.linalg.vector_norm(out, dim=-1).tolist() == pytest.approx(
            [1.0, 1.0]
        )


def test_eye_towers_are_shared_not_duplicated() -> None:
    with_eyes = GazeNet("screen_fraction", pretrained=False, use_eyes=True)
    without = GazeNet("screen_fraction", pretrained=False, use_eyes=False)
    # One shared tower (~0.17M), not two: the delta stays well under 0.4M.
    assert with_eyes.parameter_count() - without.parameter_count() < 400_000


def test_rejects_unknown_output_mode() -> None:
    with pytest.raises(ValueError, match="output_mode"):
        GazeNet("sideways")  # type: ignore[arg-type]


def test_prepare_images_converts_hwc_uint8_to_normalized_chw() -> None:
    batch = torch.full((2, 36, 60, 3), 255, dtype=torch.uint8)
    out = prepare_images(batch, torch.device("cpu"))
    assert out.shape == (2, 3, 36, 60)
    assert out.dtype == torch.float32
    # 1.0 normalized by ImageNet stats, not raw 255
    assert out.max().item() == pytest.approx((1.0 - 0.406) / 0.225, abs=1e-4)


def test_prepare_images_rejects_wrong_layout() -> None:
    with pytest.raises(ValueError, match="B, H, W, 3"):
        prepare_images(
            torch.zeros(2, 3, 36, 60, dtype=torch.uint8), torch.device("cpu")
        )


# --- loop -----------------------------------------------------------------


def test_subject_parses_out_of_sample_id() -> None:
    assert subject_of("mpiifacegaze/p07/day01/0005") == "p07"
    with pytest.raises(ValueError, match="subject"):
        subject_of("nope")


def test_loss_matches_the_output_mode() -> None:
    assert isinstance(loss_for("screen_fraction"), torch.nn.MSELoss)
    assert isinstance(loss_for("gaze_direction"), torch.nn.CosineEmbeddingLoss)
    with pytest.raises(ValueError, match="no loss defined"):
        loss_for("sideways")


def test_checkpoint_round_trips_weights_and_optimizer(tmp_path: Path) -> None:
    model = GazeNet("screen_fraction", pretrained=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "last.pt"
    save_checkpoint(
        path, model, optimizer, epoch=4, best_error=0.25, config=TrainConfig()
    )

    restored = GazeNet("screen_fraction", pretrained=False)
    next_epoch, best = load_checkpoint(path, restored)
    assert (next_epoch, best) == (5, 0.25)
    for a, b in zip(
        model.state_dict().values(), restored.state_dict().values(), strict=True
    ):
        assert torch.equal(a, b)


def test_checkpoint_refuses_a_mismatched_output_mode(tmp_path: Path) -> None:
    path = tmp_path / "last.pt"
    model = GazeNet("screen_fraction", pretrained=False)
    save_checkpoint(
        path, model, torch.optim.AdamW(model.parameters()), 0, 1.0, TrainConfig()
    )
    with pytest.raises(ValueError, match="checkpoint is screen_fraction"):
        load_checkpoint(path, GazeNet("gaze_direction", pretrained=False))


def test_fit_runs_writes_checkpoints_and_logs(tmp_path: Path) -> None:
    split = Split(train=(0, 1), val=(2,))
    train_loader = DataLoader(FakeGaze(split.train), batch_size=2)
    val_loader = DataLoader(FakeGaze(split.val), batch_size=2)
    model = GazeNet("screen_fraction", pretrained=False)

    logged: list[dict[str, Any]] = []

    class Recorder:
        def log(self, data: dict[str, Any]) -> None:
            logged.append(data)

    best = fit(
        model,
        train_loader,
        val_loader,
        torch.device("cpu"),
        TrainConfig(epochs=2, batch_size=2, num_workers=0),
        tmp_path,
        logger=Recorder(),
    )

    assert best.count == 4
    assert (tmp_path / "last.pt").is_file()
    assert (tmp_path / "best.pt").is_file()
    assert (tmp_path / "best_metrics.json").is_file()
    assert len(logged) == 2
    assert "val_p02" in logged[0]  # per-participant breakdown reaches the tracker


def test_fit_resumes_from_the_last_checkpoint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    split = Split(train=(0,), val=(1,))
    train_loader = DataLoader(FakeGaze(split.train), batch_size=2)
    val_loader = DataLoader(FakeGaze(split.val), batch_size=2)
    config = TrainConfig(epochs=1, batch_size=2, num_workers=0)

    model = GazeNet("screen_fraction", pretrained=False)
    fit(model, train_loader, val_loader, torch.device("cpu"), config, tmp_path)

    resumed = GazeNet("screen_fraction", pretrained=False)
    fit(
        resumed,
        train_loader,
        val_loader,
        torch.device("cpu"),
        TrainConfig(epochs=2, batch_size=2, num_workers=0),
        tmp_path,
        resume=True,
    )
    assert "resumed from" in capsys.readouterr().out


def test_evaluate_reports_per_participant_error() -> None:
    model = GazeNet("screen_fraction", pretrained=False)
    loader = DataLoader(FakeGaze((3, 4)), batch_size=4)
    stats = evaluate(model, loader, torch.device("cpu"))
    assert stats.count == 8
    assert sorted(stats.per_participant) == ["p03", "p04"]


def test_select_device_honours_an_explicit_choice() -> None:
    assert select_device("cpu") == torch.device("cpu")
    assert select_device("auto").type in {"cpu", "cuda", "mps"}
