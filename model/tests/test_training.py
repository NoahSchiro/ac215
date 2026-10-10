"""Tests for metrics, the model and the training loop.

Uses a synthetic in-memory dataset, so nothing here touches the data pipeline.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.utils import prepare_images
from src.training.logging import ConsoleLogger
from src.training.loop import (
    TrainConfig,
    evaluate,
    fit,
    load_checkpoint,
    save_checkpoint,
)
from src.training.metrics import (
    angular_error_deg,
    cm_to_degrees,
    fraction_to_cm,
    screen_fraction_error,
    summarize,
)
from src.training.model import GazeNet


class FakeGaze(Dataset[dict[str, Any]]):
    """Stand-in with the same keys and dtypes a real Dataset emits."""

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


# --- metrics --------------------------------------------------------------


def test_screen_fraction_error_is_euclidean() -> None:
    predicted = torch.tensor([[0.0, 0.0], [0.5, 0.5]])
    target = torch.tensor([[3.0, 4.0], [0.5, 0.5]])
    assert screen_fraction_error(predicted, target).tolist() == [5.0, 0.0]


def test_screen_fraction_error_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        screen_fraction_error(torch.zeros(2, 2), torch.zeros(3, 2))


def test_angular_error_on_known_angles() -> None:
    predicted = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    target = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    assert angular_error_deg(predicted, target).tolist() == pytest.approx(
        [0.0, 90.0], abs=1e-4
    )


def test_fraction_converts_to_cm_then_degrees() -> None:
    # 286x179mm screen -> 33.8cm diagonal; 10% error -> 3.4cm -> ~3.2deg at 60cm
    cm = fraction_to_cm(0.1, 286.0, 179.0)
    assert cm == pytest.approx(3.37, abs=0.02)
    assert cm_to_degrees(cm, 60.0) == pytest.approx(3.22, abs=0.05)


def test_conversions_reject_nonsense_geometry() -> None:
    with pytest.raises(ValueError, match="screen dimensions"):
        fraction_to_cm(0.1, 0.0, 179.0)
    with pytest.raises(ValueError, match="viewing distance"):
        cm_to_degrees(3.0, 0.0)


def test_summarize_nests_stats_per_participant() -> None:
    stats = summarize(torch.tensor([0.1, 0.3, 0.2, 0.4]), ["p00", "p00", "p01", "p01"])
    assert stats.count == 4
    assert stats.mean == pytest.approx(0.25)
    assert stats.per_participant["p00"].mean == pytest.approx(0.2)
    assert stats.per_participant["p01"].count == 2

    worst, worst_stats = stats.worst_participant()  # type: ignore[misc]
    assert worst == "p01"
    assert worst_stats.mean == pytest.approx(0.3)


def test_summarize_rejects_mismatched_subject_ids() -> None:
    with pytest.raises(ValueError, match="subject ids"):
        summarize(torch.tensor([0.1, 0.2]), ["p00"])


def test_summarize_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        summarize(torch.tensor([]))


# --- model ----------------------------------------------------------------


@pytest.mark.parametrize("mode,dim", [("screen_fraction", 2), ("gaze_direction", 3)])
def test_output_shape_and_range_per_mode(mode: str, dim: int) -> None:
    model = GazeNet(output_mode=mode, pretrained=False)  # type: ignore[arg-type]
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


def test_eye_tower_is_shared_across_both_eyes() -> None:
    with_eyes = GazeNet(pretrained=False, use_eyes=True)
    without = GazeNet(pretrained=False, use_eyes=False)
    # One shared tower, not two: the delta stays well under 0.4M parameters.
    assert with_eyes.parameter_count() - without.parameter_count() < 400_000


def test_ablating_eyes_requires_only_the_face() -> None:
    model = GazeNet(pretrained=False, use_eyes=False)
    assert model(torch.zeros(2, 3, 224, 224)).shape == (2, 2)


def test_eye_crops_are_required_when_enabled() -> None:
    model = GazeNet(pretrained=False, use_eyes=True)
    with pytest.raises(ValueError, match="requires both eye crops"):
        model(torch.zeros(2, 3, 224, 224))


def test_rejects_unknown_output_mode() -> None:
    with pytest.raises(ValueError, match="output_mode"):
        GazeNet(output_mode="sideways")  # type: ignore[arg-type]


def test_output_mode_is_keyword_only() -> None:
    with pytest.raises(TypeError):
        GazeNet("screen_fraction")  # type: ignore[misc]


# --- image prep -----------------------------------------------------------


def test_prepare_images_applies_imagenet_statistics() -> None:
    batch = torch.full((2, 36, 60, 3), 255, dtype=torch.uint8)
    out = prepare_images(batch, torch.device("cpu"), imagenet=True)
    assert out.shape == (2, 3, 36, 60)
    assert out.max().item() == pytest.approx((1.0 - 0.406) / 0.225, abs=1e-4)


def test_prepare_images_can_skip_imagenet_statistics() -> None:
    batch = torch.full((2, 36, 60, 3), 255, dtype=torch.uint8)
    out = prepare_images(batch, torch.device("cpu"), imagenet=False)
    assert out.max().item() == pytest.approx(1.0)


def test_prepare_images_rejects_wrong_layout() -> None:
    with pytest.raises(ValueError, match="B, H, W, 3"):
        prepare_images(
            torch.zeros(2, 3, 36, 60, dtype=torch.uint8), torch.device("cpu")
        )


# --- logging --------------------------------------------------------------


def test_console_logger_appends_json_lines(tmp_path: Path) -> None:
    logger = ConsoleLogger(tmp_path / "run.jsonl")
    logger.log({"epoch": 0, "val_mean": 0.5})
    logger.log({"epoch": 1, "val_mean": 0.4})
    lines = (tmp_path / "run.jsonl").read_text().strip().split("\n")
    assert [json.loads(line)["val_mean"] for line in lines] == [0.5, 0.4]


# --- loop -----------------------------------------------------------------


def test_checkpoint_round_trips_weights_and_optimizer(tmp_path: Path) -> None:
    model = GazeNet(pretrained=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "last.pt"
    save_checkpoint(path, model, optimizer, 4, 0.25, TrainConfig())

    restored = GazeNet(pretrained=False)
    assert load_checkpoint(path, restored) == (5, 0.25)
    for a, b in zip(
        model.state_dict().values(), restored.state_dict().values(), strict=True
    ):
        assert torch.equal(a, b)


def test_checkpoint_refuses_a_mismatched_output_mode(tmp_path: Path) -> None:
    path = tmp_path / "last.pt"
    model = GazeNet(pretrained=False)
    save_checkpoint(
        path, model, torch.optim.AdamW(model.parameters()), 0, 1.0, TrainConfig()
    )
    with pytest.raises(ValueError, match="checkpoint is screen_fraction"):
        load_checkpoint(path, GazeNet(output_mode="gaze_direction", pretrained=False))


def test_fit_runs_checkpoints_and_logs(tmp_path: Path) -> None:
    train_loader = DataLoader(FakeGaze((0, 1)), batch_size=2)
    val_loader = DataLoader(FakeGaze((2,)), batch_size=2)
    logged: list[dict[str, Any]] = []

    class Recorder:
        def log(self, data: dict[str, Any]) -> None:
            logged.append(data)

    best = fit(
        GazeNet(pretrained=False),
        train_loader,
        val_loader,
        torch.device("cpu"),
        TrainConfig(epochs=2, batch_size=2, num_workers=0),
        tmp_path,
        Recorder(),
    )

    assert best.count == 4
    assert (tmp_path / "last.pt").is_file()
    assert (tmp_path / "best.pt").is_file()
    assert (tmp_path / "best_metrics.json").is_file()
    assert len(logged) == 2
    assert "val_p02" in logged[0]


def test_fit_resumes_from_the_last_checkpoint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    train_loader = DataLoader(FakeGaze((0,)), batch_size=2)
    val_loader = DataLoader(FakeGaze((1,)), batch_size=2)

    class Null:
        def log(self, data: dict[str, Any]) -> None: ...

    args = (torch.device("cpu"),)
    fit(
        GazeNet(pretrained=False),
        train_loader,
        val_loader,
        *args,
        TrainConfig(epochs=1, batch_size=2, num_workers=0),
        tmp_path,
        Null(),
    )
    fit(
        GazeNet(pretrained=False),
        train_loader,
        val_loader,
        *args,
        TrainConfig(epochs=2, batch_size=2, num_workers=0),
        tmp_path,
        Null(),
        resume=True,
    )
    assert "resumed from" in capsys.readouterr().out


def test_evaluate_reports_per_participant_error() -> None:
    stats = evaluate(
        GazeNet(pretrained=False),
        DataLoader(FakeGaze((3, 4)), batch_size=4),
        torch.device("cpu"),
    )
    assert stats.count == 8
    assert sorted(stats.per_participant) == ["p03", "p04"]
