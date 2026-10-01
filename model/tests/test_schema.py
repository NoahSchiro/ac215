"""Tests for the canonical per-sample schema (PLAN.md Step 0)."""

import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.data.schema import (
    CameraIntrinsics,
    DeviceType,
    PointPx,
    Sample,
    SizeMm,
    SizePx,
    SourceDataset,
    make_sample_id,
    write_manifest,
    read_manifest
)


def make_sample(**overrides: Any) -> Sample:
    """Just a helper function for my tests."""
    values: dict[str, Any] = {
        "sample_id": make_sample_id(
            SourceDataset.MPIIFaceGaze, "123456789", "rec-0", "frame-000001"
        ),
        "source_dataset": SourceDataset.MPIIFaceGaze,
        "subject_id": "123456789",
        "image_path": Path("raw/mpiifacegaze/123456789/face/0.jpg"),
        "camera_intrinsics": CameraIntrinsics(
            focal_length_px=1200.0,
            principal_point=PointPx(x=640.0, y=360.0),
        ),
        "screen_size_mm": SizeMm(width=375.0, height=812.0),
        "screen_size_px": SizePx(width=1125, height=2436),
        "gaze_target_px": PointPx(x=562.5, y=1218.0),
        "device_type": DeviceType.Phone,
    }
    values.update(overrides)
    return Sample(**values)


def test_make_sample_id_prefixes_source_tag() -> None:
    sid = make_sample_id(SourceDataset.MPIIFaceGaze, "p01", "day01", "0001")
    assert sid == "mpiifacegaze/p01/day01/0001"


def test_make_sample_id_rejects_empty_parts() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        make_sample_id(SourceDataset.MPIIFaceGaze, "")


def test_sample_requires_enum_members() -> None:
    with pytest.raises(TypeError, match="SourceDataset"):
        make_sample(source_dataset="mpiifacegaze")
    with pytest.raises(TypeError, match="DeviceType"):
        make_sample(device_type="phone")


def test_sample_id_must_carry_source_tag() -> None:
    with pytest.raises(ValueError, match="source dataset tag"):
        make_sample(
            sample_id="notmpii/123456789/frame-000001",
            source_dataset=SourceDataset.MPIIFaceGaze,
        )


def test_sample_id_must_be_non_empty() -> None:
    with pytest.raises(ValueError):
        make_sample(sample_id="mpiifacegaze/")


def test_subject_id_must_be_non_empty() -> None:
    with pytest.raises(ValueError, match="subject_id"):
        make_sample(subject_id="")


def test_sample_is_frozen() -> None:
    sample = make_sample()
    with pytest.raises(FrozenInstanceError):
        sample.subject_id = "other"  # type: ignore[misc]


def test_equality_ignores_image_pixels() -> None:
    a = make_sample(image=np.zeros((8, 8, 3), dtype=np.uint8))
    b = make_sample(image=np.ones((8, 8, 3), dtype=np.uint8))
    assert a == b
    assert a != make_sample(subject_id="999999")


def test_image_shape_and_dtype_are_validated() -> None:
    with pytest.raises(ValueError, match="shape"):
        make_sample(image=np.zeros((4, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match="dtype"):
        make_sample(image=np.zeros((4, 4, 3), dtype=np.float32))
    make_sample(image=np.zeros((4, 4, 3), dtype=np.uint8))  # OK


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_focal_length_must_be_positive(bad: float) -> None:
    with pytest.raises(ValueError):
        CameraIntrinsics(focal_length_px=bad, principal_point=PointPx(x=0.0, y=0.0))


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_screen_size_mm_must_be_positive(bad: float) -> None:
    with pytest.raises(ValueError):
        SizeMm(width=bad, height=100.0)


def test_size_px_requires_positive_ints() -> None:
    with pytest.raises(TypeError, match="int"):
        SizePx(width=1125.5, height=2436)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="int"):
        SizePx(width="hello", height="world")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive"):
        SizePx(width=0, height=2436)


def test_manifest_entry_is_json_serializable_and_excludes_pixels() -> None:
    sample = make_sample(image=np.zeros((2, 2, 3), dtype=np.uint8))
    entry = sample.to_manifest_entry()
    assert "image" not in entry
    assert entry["image_path"] == str(sample.image_path)
    assert json.loads(json.dumps(entry)) == entry


def test_manifest_round_trip() -> None:
    sample = make_sample()
    restored = Sample.from_manifest_entry(
        json.loads(json.dumps(sample.to_manifest_entry()))
    )
    assert restored == sample
    assert restored.source_dataset is SourceDataset.MPIIFaceGaze
    assert restored.camera_intrinsics == make_sample().camera_intrinsics


def test_from_manifest_entry_rejects_unknown_source_dataset() -> None:
    entry = make_sample().to_manifest_entry()
    with pytest.raises(ValueError):
        Sample.from_manifest_entry({**entry, "source_dataset": "eth-xgaze"})


def test_manifest_round_trip() -> None:
    samples = [
        make_sample(sample_id=f"mpiifacegaze/{x}")
        for x in ["a", "b"]
    ]
    path = Path("/tmp/test_manifest_round_trip.json")

    write_manifest(samples, path)
    restored = read_manifest(path)

    assert restored == samples
    path.unlink()


def test_manifest_is_byte_identical_across_runs() -> None:
    samples = [
        make_sample(sample_id=f"mpiifacegaze/{x}")
        for x in ["a", "b"]
    ]
    path_a = Path("/tmp/test_manifest_a.json")
    path_b = Path("/tmp/test_manifest_b.json")

    write_manifest(samples, path_a)
    write_manifest(samples, path_b)

    assert path_a.read_bytes() == path_b.read_bytes()
    path_a.unlink()
    path_b.unlink()


def test_manifest_entry_is_json_serializable_and_excludes_pixels() -> None:
    sample = make_sample(sample_id="mpiifacegaze/a")
    entry = sample.to_manifest_entry()

    assert "image" not in entry
    assert json.loads(json.dumps(entry)) == entry
