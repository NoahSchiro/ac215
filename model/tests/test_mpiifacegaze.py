"""Tests for the MPIIFaceGaze dataset."""

import shutil
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from src.data.mpiifacegaze import MPII, parse_mpii, parse_mpii_sample
from src.data.schema import DeviceType, PointPx, SizePx, SourceDataset

_CAM_MATRIX = np.array([[1000.0, 0.0, 640.0], [0.0, 1040.0, 360.0], [0.0, 0.0, 1.0]])


def valid_line(image: str = "day01/0005.jpg", gaze: str = "1148 290") -> str:
    """A 28-column annotation line: path, gaze px, 12 landmarks, 6 pose, 3 fc, 3 gt, eye."""
    return " ".join(
        [image, gaze, *["0"] * 12, *["0"] * 6, *["0"] * 3, *["0"] * 3, "left"]
    )


def make_subject(
    root: Path,
    name: str = "p00",
    *,
    lines: list[str] | None = None,
    images: tuple[str, ...] = ("day01/0005.jpg", "day01/0009.jpg"),
    with_calibration: bool = True,
) -> Path:
    subject_dir = root / name
    (subject_dir / "day01").mkdir(parents=True, exist_ok=True)
    for image in images:
        (subject_dir / image).touch()
    if with_calibration:
        calibration = subject_dir / "Calibration"
        calibration.mkdir(exist_ok=True)
        savemat(
            calibration / "Camera.mat",
            {"cameraMatrix": _CAM_MATRIX, "distCoeffs": np.zeros((1, 5))},
        )
        savemat(
            calibration / "screenSize.mat",
            {
                "width_pixel": 1440.0,
                "height_pixel": 900.0,
                "width_mm": 286.5,
                "height_mm": 179.0,
            },
        )
    if lines is not None:
        (subject_dir / f"{name}.txt").write_text("\n".join(lines) + "\n")
    return subject_dir


def test_parse_sample_builds_sample(tmp_path: Path) -> None:
    subject_dir = make_subject(tmp_path, "p00")
    sample = parse_mpii_sample(subject_dir, valid_line())

    assert sample is not None
    assert sample.sample_id == "mpiifacegaze/p00/day01/0005"
    assert sample.subject_id == "p00"
    assert sample.source_dataset is SourceDataset.MPIIFaceGaze
    assert sample.device_type is DeviceType.Laptop
    assert sample.gaze_target_px == PointPx(x=1148.0, y=290.0)
    assert sample.screen_size_px == SizePx(width=1440, height=900)
    assert sample.screen_size_mm.width == pytest.approx(286.5)
    assert sample.screen_size_mm.height == pytest.approx(179.0)
    assert sample.camera_intrinsics.focal_length_px == pytest.approx(1020.0)
    assert sample.camera_intrinsics.principal_point == PointPx(x=640.0, y=360.0)
    assert sample.camera_intrinsics.is_estimate is False
    assert sample.image_path == (subject_dir / "day01" / "0005.jpg").resolve()
    assert sample.image is None


def test_parse_dataset_drops_and_prints_total(tmp_path: Path, capsys) -> None:
    make_subject(tmp_path, "p00", lines=[valid_line(), "bad line"])
    make_subject(tmp_path, "p01", with_calibration=False, lines=[valid_line()])

    samples = parse_mpii(tmp_path)

    assert len(samples) == 1
    assert samples[0].subject_id == "p00"
    out = capsys.readouterr().out
    assert "dropping sample in p00" in out
    assert "dropping sample in p01" in out
    assert "parsed 1 datapoints, dropped 2 of 3" in out


def test_parse_dataset_is_deterministic(tmp_path: Path) -> None:
    make_subject(tmp_path, "p00", lines=[valid_line(), valid_line("day01/0009.jpg")])

    first = parse_mpii(tmp_path)
    second = parse_mpii(tmp_path)

    assert first == second


# p00's real calibration; the fixture face is one of p00's images
_CAM = np.array([[996.4509, 0.0, 624.6634], [0.0, 998.166, 364.0874], [0.0, 0.0, 1.0]])


def make_mini_dataset(root: Path) -> Path:
    """Two subjects (p00, p01) with one detectable image each."""
    face = Path(__file__).parent / "data" / "face.jpg"
    for p in ("p00", "p01"):
        subj = root / p
        (subj / "day01").mkdir(parents=True, exist_ok=True)
        shutil.copy(face, subj / "day01" / "0005.jpg")
        calib = subj / "Calibration"
        calib.mkdir(exist_ok=True)
        savemat(
            calib / "Camera.mat",
            {"cameraMatrix": _CAM, "distCoeffs": np.zeros((1, 5))},
        )
        savemat(
            calib / "screenSize.mat",
            {
                "width_pixel": 1440.0,
                "height_pixel": 900.0,
                "width_mm": 286.5,
                "height_mm": 179.0,
            },
        )
        line = " ".join(
            [
                "day01/0005.jpg",
                "1148 290",
                *["0"] * 12,
                *["0"] * 6,
                *["0"] * 3,
                *["0"] * 3,
                "left",
            ]
        )
        (subj / f"{p}.txt").write_text(line + "\n")
    return root


@pytest.fixture
def mini_dataset(tmp_path: Path) -> Path:
    """Synthetic two-subject dataset; the webdataset cache lands inside it."""
    return make_mini_dataset(tmp_path / "MPIIFaceGaze")


def test_builds_webdataset_on_first_use(mini_dataset: Path) -> None:
    cache = mini_dataset / "MPIIFaceGaze_wd"
    assert not cache.is_dir()

    ds = MPII([0, 1], dataset_root=mini_dataset)

    assert cache.is_dir()  # built and cached for later runs
    assert len(ds) == 2


def test_loads_existing_webdataset_without_raw_data(mini_dataset: Path) -> None:
    MPII([0, 1], dataset_root=mini_dataset)  # first use builds the cache
    for child in mini_dataset.iterdir():  # raw data disappears, cache stays
        if child.name != "MPIIFaceGaze_wd":
            shutil.rmtree(child)

    ds = MPII([0, 1], dataset_root=mini_dataset)  # second use loads cache only

    assert len(ds) == 2


def test_getitem_computes_crops_and_label(mini_dataset: Path) -> None:
    ds = MPII(0, dataset_root=mini_dataset)

    item = ds[0]

    assert item["sample_id"] == "mpiifacegaze/p00/day01/0005"
    assert item["face"].shape == (224, 224, 3)
    assert item["left_eye"].shape == (36, 60, 3)
    assert item["right_eye"].shape == (36, 60, 3)
    # label = raw gaze target / screen size, from the annotation line
    assert item["label"].dtype == np.float32
    assert item["label"] == pytest.approx(
        np.array([1148 / 1440, 290 / 900], dtype=np.float32)
    )
    assert np.allclose(item["R_norm"] @ item["R_norm"].T, np.eye(3), atol=1e-9)


def test_participants_select_subjects(mini_dataset: Path) -> None:
    ds_p00 = MPII(0, dataset_root=mini_dataset)
    ds_p01 = MPII([1], dataset_root=mini_dataset)

    assert len(ds_p00) == 1
    assert len(ds_p01) == 1
    assert {s.subject_id for s in ds_p00.samples} == {"p00"}
    assert {s.subject_id for s in ds_p01.samples} == {"p01"}


def test_rejects_out_of_range_participants(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="p00..p14"):
        MPII(15, dataset_root=tmp_path)
    with pytest.raises(ValueError, match="p00..p14"):
        MPII(-1, dataset_root=tmp_path)
