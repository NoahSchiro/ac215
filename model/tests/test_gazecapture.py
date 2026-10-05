"""Tests for the GazeCapture dataset."""

import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.data.gazecapture import (
    GazeCapture,
    parse_gazecapture,
    parse_gazecapture_sample,
)
from src.data.schema import DeviceType, PointPx, SizePx, SourceDataset


def make_recording(
    root: Path,
    name: str = "00002",
    *,
    frame_files: tuple[str, ...] = ("00000.jpg", "00001.jpg"),
    device: str = "iPhone 6",
    split: str = "train",
    real_image: Path | None = None,
    xp: tuple[float, ...] | None = None,
    yp: tuple[float, ...] | None = None,
    screen: tuple[tuple[int, int], ...] | None = None,
    orientations: tuple[int, ...] | None = None,
) -> Path:
    """Build a minimal recording directory with touch-only frame images.

    Per-frame columns default to portrait iPhone 6 values for every frame.
    `real_image` (e.g. tests/data/face.jpg) replaces the touch-only files
    with a real, detectable face image.
    """
    n = len(frame_files)
    xp = xp if xp is not None else tuple(160.0 for _ in range(n))
    yp = yp if yp is not None else tuple(284.0 for _ in range(n))
    screen = screen if screen is not None else tuple((320, 568) for _ in range(n))
    orientations = (
        orientations if orientations is not None else tuple(1 for _ in range(n))
    )
    subject_dir = root / name
    frames_dir = subject_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for frame in frame_files:
        if real_image is None:
            (frames_dir / frame).touch()
        else:
            shutil.copy(real_image, frames_dir / frame)
    info: dict[str, Any] = {
        "TotalFrames": len(frame_files),
        "NumFaceDetections": len(frame_files),
        "NumEyeDetections": len(frame_files),
        "Dataset": split,
        "DeviceName": device,
    }
    per_frame = {
        "dotInfo.json": {
            "DotNum": list(range(len(frame_files))),
            "XPts": list(xp),
            "YPts": list(yp),
        },
        "screen.json": {
            "W": [w for w, _ in screen],
            "H": [h for _, h in screen],
            "Orientation": list(orientations),
        },
    }
    (subject_dir / "info.json").write_text(json.dumps(info))
    (subject_dir / "frames.json").write_text(json.dumps(list(frame_files)))
    for filename, payload in per_frame.items():
        (subject_dir / filename).write_text(json.dumps(payload))
    return subject_dir


def test_parse_sample_builds_sample(tmp_path: Path) -> None:
    subject_dir = make_recording(tmp_path)

    sample = parse_gazecapture_sample(subject_dir, 0)

    assert sample is not None
    assert sample.sample_id == "gazecapture/00002/train/frame-000000"
    assert sample.subject_id == "00002"
    assert sample.source_dataset is SourceDataset.GazeCapture
    assert sample.device_type is DeviceType.Phone
    assert sample.gaze_target_px == PointPx(x=160.0, y=284.0)
    assert sample.screen_size_px == SizePx(width=320, height=568)
    assert sample.screen_size_mm.width == pytest.approx(58.4)
    assert sample.screen_size_mm.height == pytest.approx(103.9)
    # iTracker's 62.5 deg FOV on a 480x640 frame, averaged across axes
    expected_focal = (480 / 2 + 640 / 2) / 2 / math.tan(math.radians(62.5) / 2)
    assert sample.camera_intrinsics.focal_length_px == pytest.approx(expected_focal)
    assert sample.camera_intrinsics.principal_point == PointPx(x=240.0, y=320.0)
    assert sample.camera_intrinsics.is_estimate is True
    assert sample.image_path == (subject_dir / "frames" / "00000.jpg").resolve()
    assert sample.image is None


def test_parse_sample_handles_landscape_frames(tmp_path: Path) -> None:
    subject_dir = make_recording(
        tmp_path,
        xp=(280.0,),
        yp=(140.0,),
        screen=((568, 320),),
        orientations=(3,),
        frame_files=("00000.jpg",),
    )

    sample = parse_gazecapture_sample(subject_dir, 0)

    assert sample is not None
    # landscape: point dimensions and physical mm swap with the frame
    assert sample.screen_size_px == SizePx(width=568, height=320)
    assert sample.screen_size_mm.width == pytest.approx(103.9)
    assert sample.screen_size_mm.height == pytest.approx(58.4)
    assert sample.camera_intrinsics.principal_point == PointPx(x=320.0, y=240.0)
    assert sample.camera_intrinsics.focal_length_px == pytest.approx(
        (480 / 2 + 640 / 2) / 2 / math.tan(math.radians(62.5) / 2)
    )


def test_parse_sample_maps_ipad_to_tablet(tmp_path: Path) -> None:
    subject_dir = make_recording(tmp_path, device="iPad Air 2")

    sample = parse_gazecapture_sample(subject_dir, 0)

    assert sample is not None
    assert sample.device_type is DeviceType.Tablet
    assert sample.screen_size_mm.width == pytest.approx(147.7)


def test_parse_sample_drops_invalid_frames(tmp_path: Path, capsys) -> None:
    make_recording(tmp_path, name="00002")  # frame 0 valid, frame 1 valid

    missing_image = make_recording(tmp_path, name="00003")
    (missing_image / "frames" / "00001.jpg").unlink()
    make_recording(tmp_path, name="00005", device="Nokia 3310")
    make_recording(tmp_path, name="00008", split="dev")  # unknown official split
    (tmp_path / "00006").mkdir()
    short_arrays = make_recording(tmp_path, name="00007")
    (short_arrays / "dotInfo.json").write_text(
        json.dumps({"XPts": [1.0], "YPts": [1.0]})
    )

    assert parse_gazecapture_sample(tmp_path / "00003", 1) is None
    assert "image not found" in capsys.readouterr().out
    assert parse_gazecapture_sample(tmp_path / "00005", 0) is None
    assert "unknown DeviceName" in capsys.readouterr().out
    assert parse_gazecapture_sample(tmp_path / "00008", 0) is None
    assert "Dataset must be one of" in capsys.readouterr().out
    assert parse_gazecapture_sample(tmp_path / "00006", 0) is None
    assert "annotation not found" in capsys.readouterr().out
    assert parse_gazecapture_sample(tmp_path / "00007", 0) is None
    assert "must be a list of length" in capsys.readouterr().out
    assert parse_gazecapture_sample(tmp_path / "00002", 99) is None
    assert "out of range" in capsys.readouterr().out
    # valid frames still parse despite their broken siblings
    assert parse_gazecapture_sample(tmp_path / "00002", 0) is not None


def test_parse_sample_drops_non_finite_gaze(tmp_path: Path, capsys) -> None:
    subject_dir = make_recording(
        tmp_path, xp=(float("nan"),), frame_files=("00000.jpg",)
    )

    assert parse_gazecapture_sample(subject_dir, 0) is None
    assert "gaze target is not finite" in capsys.readouterr().out


def test_parse_dataset_drops_and_prints_total(tmp_path: Path, capsys) -> None:
    make_recording(tmp_path, name="00002")
    make_recording(tmp_path, name="00003")  # frame 1 has no image file
    (tmp_path / "00003" / "frames" / "00001.jpg").unlink()
    (tmp_path / "00005").mkdir()  # no annotations at all
    (tmp_path / "README.md").touch()  # non-subject entries are ignored

    samples = parse_gazecapture(tmp_path)

    assert len(samples) == 3
    assert {s.subject_id for s in samples} == {"00002", "00003"}
    out = capsys.readouterr().out
    assert "dropping sample in 00003" in out
    assert "dropping subject 00005" in out
    assert "parsed 3 datapoints, dropped 1 of 4" in out


def test_parse_dataset_is_deterministic(tmp_path: Path) -> None:
    make_recording(tmp_path, name="00002")

    first = parse_gazecapture(tmp_path)
    second = parse_gazecapture(tmp_path)

    assert first == second


_FACE_IMAGE = Path(__file__).parent / "data" / "face.jpg"


def make_mini_dataset(root: Path) -> Path:
    """Two subjects (00002 train, 00003 val) with one detectable frame each."""
    for name, split in (("00002", "train"), ("00003", "val")):
        make_recording(
            root,
            name=name,
            split=split,
            frame_files=("00000.jpg",),
            real_image=_FACE_IMAGE,
        )
    return root


@pytest.fixture
def mini_dataset(tmp_path: Path) -> Path:
    """Synthetic two-subject dataset; the filtered manifest lands inside it."""
    return make_mini_dataset(tmp_path / "GazeCapture")


def test_builds_manifest_on_first_use(mini_dataset: Path) -> None:
    manifest = mini_dataset / "gazecapture_filtered.json"
    assert not manifest.is_file()

    ds = GazeCapture(["train", "val"], dataset_root=mini_dataset)

    assert manifest.is_file()  # built and cached for later runs
    assert len(ds) == 2


def test_loads_existing_manifest_without_raw_data(mini_dataset: Path) -> None:
    GazeCapture("train", dataset_root=mini_dataset)  # first use builds the manifest
    for child in mini_dataset.iterdir():  # raw data disappears, manifest stays
        if child.name != "gazecapture_filtered.json":
            shutil.rmtree(child)

    ds = GazeCapture("train", dataset_root=mini_dataset)  # manifest only

    assert len(ds) == 1


def test_getitem_computes_crops_and_label(mini_dataset: Path) -> None:
    ds = GazeCapture("train", dataset_root=mini_dataset)

    item = ds[0]

    assert item["sample_id"] == "gazecapture/00002/train/frame-000000"
    assert item["face"].shape == (224, 224, 3)
    assert item["left_eye"].shape == (36, 60, 3)
    assert item["right_eye"].shape == (36, 60, 3)
    # label = raw gaze target / screen size, from the annotation JSONs
    assert item["label"].dtype == np.float32
    assert item["label"] == pytest.approx(
        np.array([160 / 320, 284 / 568], dtype=np.float32)
    )
    assert np.allclose(item["R_norm"] @ item["R_norm"].T, np.eye(3), atol=1e-9)


def test_splits_select_subjects(mini_dataset: Path) -> None:
    ds_train = GazeCapture("train", dataset_root=mini_dataset)
    ds_val = GazeCapture(["val"], dataset_root=mini_dataset)

    assert len(ds_train) == 1
    assert len(ds_val) == 1
    assert {s.subject_id for s in ds_train.samples} == {"00002"}
    assert {s.subject_id for s in ds_val.samples} == {"00003"}


def test_rejects_unknown_splits(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="train, val, test"):
        GazeCapture("dev", dataset_root=tmp_path)
    with pytest.raises(ValueError, match="train, val, test"):
        GazeCapture(["train", "benchmark"], dataset_root=tmp_path)
