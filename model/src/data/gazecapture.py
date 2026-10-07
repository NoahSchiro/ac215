"""Parse raw GazeCapture files into Sample records. Contains
the GazeCapture class as well

GazeCapture provides no camera calibration and no physical screen sizes,
so both are estimated:
- camera intrinsics: iTracker's (Krafka et al., CVPR 2016) constant 62.5
  degree field-of-view assumption, applied to the distributed 480x640 /
  640x480 frames; the focal length is averaged across axes like the
  MPIIFaceGaze adapter, and the principal point sits at the frame center.
  All of this is flagged via `CameraIntrinsics.is_estimate`
- screen mm: per-device active display area derived from the panel's
  point resolution and ppi

Frames are stored portrait (480 wide x 640 high) or landscape (640x480)
following the per-frame screen orientation in screen.json
"""

import json
import math
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from torch.utils.data import Dataset

from src.data.schema import (
    CameraIntrinsics,
    DeviceType,
    PointPx,
    Sample,
    SizeMm,
    SizePx,
    SourceDataset,
    make_sample_id,
    read_manifest,
    write_manifest,
)
from src.data.utils import (
    filter_dataset,
    get_landmarker,
    head_pose_est,
    landmark_detection,
    normalize_face,
    transform_label,
)

_OFFICIAL_SPLITS = ("train", "val", "test")
"""GazeCapture's subject-disjoint benchmark splits (look at
info.json for more info)."""


@dataclass(frozen=True, slots=True)
class Recording:
    """One recording's annotations as index-aligned per-frame tuples."""

    device_name: str
    split: str
    frame_files: tuple[str, ...]
    gaze_x_pts: tuple[float, ...]
    gaze_y_pts: tuple[float, ...]
    screen_w_pts: tuple[int, ...]
    screen_h_pts: tuple[int, ...]
    screen_orientation: tuple[int, ...]


def _column(data: dict[str, Any], key: str, name: str, length: int) -> tuple[Any, ...]:
    """Extract one per-frame column."""
    values = data.get(key)
    if not isinstance(values, list) or len(values) != length:
        raise ValueError(f"{name} column {key!r} must be a list of length {length}")
    return tuple(values)


@lru_cache(maxsize=2)
def load_recording(subject_dir: Path) -> Recording:
    """Read one recording's JSON annotations.

    Cached (small LRU) so that parsing a recording's frames in order reads
    each JSON file once instead of once per frame.
    """

    def read_json(name: str) -> Any:
        path = subject_dir / name
        if not path.is_file():
            raise ValueError(f"annotation not found: {name}")
        return json.loads(path.read_text(encoding="utf-8"))

    info = read_json("info.json")
    frame_files = read_json("frames.json")
    dot_info = read_json("dotInfo.json")
    screen = read_json("screen.json")
    if not isinstance(info, dict) or not isinstance(info.get("DeviceName"), str):
        raise TypeError("info.json must hold a DeviceName string")
    split = info.get("Dataset")
    if not isinstance(split, str):
        raise TypeError("info.json must hold a Dataset string")
    if split not in _OFFICIAL_SPLITS:
        raise ValueError(
            f"info.json Dataset must be one of {', '.join(_OFFICIAL_SPLITS)}"
        )
    if not isinstance(frame_files, list) or not frame_files:
        raise ValueError("frames.json must be a non-empty list")
    if not isinstance(dot_info, dict) or not isinstance(screen, dict):
        raise TypeError("dotInfo.json and screen.json must be objects")
    num_frames = len(frame_files)
    return Recording(
        device_name=info["DeviceName"],
        split=split,
        frame_files=tuple(str(name) for name in frame_files),
        gaze_x_pts=_column(dot_info, "XPts", "dotInfo.json", num_frames),
        gaze_y_pts=_column(dot_info, "YPts", "dotInfo.json", num_frames),
        screen_w_pts=_column(screen, "W", "screen.json", num_frames),
        screen_h_pts=_column(screen, "H", "screen.json", num_frames),
        screen_orientation=_column(screen, "Orientation", "screen.json", num_frames),
    )


@lru_cache(maxsize=2)
def _estimated_intrinsics(orientation: int) -> CameraIntrinsics:
    """Estimated intrinsics for a stored frame (iTracker 62.5 deg FOV).

    Portrait screen orientations (1/2) have portrait frames. Landscape
    ones (3/4) have landscape frames.
    """
    frame_long_px, frame_short_px = 640, 480
    # iTracker assumes a constant 62.5 degree field of view for every device
    # and derives the focal length as f = 0.5 * dim / tan(fov / 2)
    itracker_fov_deg = 62.5

    if orientation in (3, 4):
        width, height = frame_long_px, frame_short_px
    else:
        width, height = frame_short_px, frame_long_px
    tan_half_fov = math.tan(math.radians(itracker_fov_deg) / 2)
    focal = 0.5 * (width / 2 + height / 2) / tan_half_fov  # mean of f_x, f_y
    return CameraIntrinsics(
        focal_length_px=focal,
        principal_point=PointPx(x=width / 2, y=height / 2),
        is_estimate=True,
    )


def _device_type(device_name: str) -> DeviceType:
    """Map a GazeCapture DeviceName to a DeviceType."""
    if device_name.startswith("iPhone"):
        return DeviceType.Phone
    if device_name.startswith("iPad"):
        return DeviceType.Tablet
    raise ValueError(f"unknown DeviceName {device_name!r}")


def _screen_size_mm(device_name: str, orientation: int) -> SizeMm:
    """Physical screen size, oriented to match screen.json's per-frame W/H.

    Landscape orientations (3/4) swap width/height
    """
    # Physical screen sizes (portrait, width x height in mm).
    # GazeCapture does not provide them, so these are computed from
    # each panel's point resolution and ppi (e.g. iPhone 6: 750x1334
    # points at 326 ppi -> 58.4 x 103.9 mm). Keyed by the DeviceName
    # strings in info.json.
    screen_size = {
        "iPhone 4S": SizeMm(width=49.9, height=74.8),  # 640x960 @ 326 ppi
        "iPhone 5": SizeMm(width=49.9, height=88.5),  # 640x1136 @ 326 ppi
        "iPhone 5S": SizeMm(width=49.9, height=88.5),
        "iPhone 5C": SizeMm(width=49.9, height=88.5),
        "iPhone 6": SizeMm(width=58.4, height=103.9),  # 750x1334 @ 326 ppi
        "iPhone 6s": SizeMm(width=58.4, height=103.9),
        "iPhone 6 Plus": SizeMm(width=68.4, height=121.7),  # 1080x1920 @ 401 ppi
        "iPhone 6s Plus": SizeMm(width=68.4, height=121.7),
        "iPad 2": SizeMm(width=147.7, height=196.9),  # 768x1024 @ 132 ppi
        "iPad 3": SizeMm(width=147.7, height=196.9),
        "iPad 4": SizeMm(width=147.7, height=196.9),
        "iPad Air": SizeMm(width=147.7, height=196.9),
        "iPad Air 2": SizeMm(width=147.7, height=196.9),
        "iPad Mini": SizeMm(width=119.6, height=159.5),  # 768x1024 @ 163 ppi
        "iPad Pro": SizeMm(width=196.9, height=262.6),  # 2048x2732 @ 264 ppi
    }
    if device_name not in screen_size:
        raise ValueError(f"unknown DeviceName {device_name!r}")
    portrait = screen_size[device_name]
    if orientation in (3, 4):
        return SizeMm(width=portrait.height, height=portrait.width)
    else:
        return portrait


def parse_gazecapture_sample(subject_dir: Path, frame_idx: int) -> Sample | None:
    """Parse one frame of a GazeCapture recording into a Sample.

    `subject_dir` is a recording directory (e.g. ``.../GazeCapture/00002``)
    and `frame_idx` indexes frames.json. Returns None (with a printed
    reason) if anything is invalid.
    """
    try:
        recording = load_recording(subject_dir)
        if not 0 <= frame_idx < len(recording.frame_files):
            raise ValueError(
                f"frame index {frame_idx} out of range 0..{len(recording.frame_files) - 1}"
            )
        frame_file = recording.frame_files[frame_idx]
        image_path = subject_dir / "frames" / frame_file
        if not image_path.is_file():
            raise ValueError(f"image not found: frames/{frame_file}")

        gaze_x = float(recording.gaze_x_pts[frame_idx])
        gaze_y = float(recording.gaze_y_pts[frame_idx])
        if not (math.isfinite(gaze_x) and math.isfinite(gaze_y)):
            raise ValueError(
                f"gaze target is not finite: {recording.gaze_x_pts[frame_idx]},"
                f" {recording.gaze_y_pts[frame_idx]}"
            )
        orientation = int(recording.screen_orientation[frame_idx])
        return Sample(
            sample_id=make_sample_id(
                SourceDataset.GazeCapture,
                subject_dir.name,
                recording.split,
                f"frame-{frame_idx:06d}",
            ),
            source_dataset=SourceDataset.GazeCapture,
            subject_id=subject_dir.name,
            image_path=image_path.resolve(),
            camera_intrinsics=_estimated_intrinsics(orientation),
            screen_size_mm=_screen_size_mm(recording.device_name, orientation),
            screen_size_px=SizePx(
                width=int(recording.screen_w_pts[frame_idx]),
                height=int(recording.screen_h_pts[frame_idx]),
            ),
            gaze_target_px=PointPx(x=gaze_x, y=gaze_y),
            device_type=_device_type(recording.device_name),
        )
    except Exception as exc:  # noqa: BLE001 anything unparseable just gets dropped
        print(f"dropping sample in {subject_dir.name}: {exc}")
        return None


def parse_gazecapture(root: Path) -> list[Sample]:
    """Parse the entire GazeCapture dataset at `root` into Samples."""
    root = Path(root)
    samples: list[Sample] = []
    dropped = 0
    total = 0

    for subject_dir in sorted(root.iterdir()):
        if not (subject_dir.is_dir() and re.fullmatch(r"\d{5}", subject_dir.name)):
            continue
        try:
            num_frames = len(load_recording(subject_dir).frame_files)
        except Exception as exc:  # noqa: BLE001
            print(f"dropping subject {subject_dir.name}: {exc}")
            continue
        for frame_idx in range(num_frames):
            total += 1
            sample = parse_gazecapture_sample(subject_dir, frame_idx)
            if sample is None:
                dropped += 1
            else:
                samples.append(sample)

    print(f"parsed {len(samples)} datapoints, dropped {dropped} of {total}")
    return samples


class GazeCapture(Dataset):
    """Quality-filtered GazeCapture as a torch Dataset.

    Args:
        splits: split(s) to include ("train", "val", "test"). Accepts a
            single string or a list of strings
        dataset_root: raw GazeCapture root, only needed when the filtered
            manifest does not exist yet (it is built and cached on first
            use).
    """

    def __init__(
        self,
        splits: str | list[str],
        dataset_root: os.PathLike[str],
    ) -> None:

        manifest_path = Path(dataset_root) / "gazecapture_filtered.json"

        if isinstance(splits, str):
            splits = [splits]
        for split in splits:
            if split not in _OFFICIAL_SPLITS:
                raise ValueError(
                    f"split {split!r} is unknown: GazeCapture has"
                    f" {', '.join(_OFFICIAL_SPLITS)}"
                )
        self.splits = sorted(set(splits))

        if manifest_path.is_file():
            self.samples = read_manifest(manifest_path)
        else:
            print(f"no manifest at {manifest_path}; building it from the raw dataset")
            self.samples = filter_dataset(parse_gazecapture(Path(dataset_root)))
            write_manifest(self.samples, manifest_path)

        wanted = set(self.splits)
        self.samples = [
            sample
            for sample in self.samples
            if sample.sample_id.split("/")[2] in wanted
        ]

        self.landmarker: Any = None  # created lazily, per worker process

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]

        bgr = cv2.imread(str(sample.image_path))
        # Should never happen
        if bgr is None:
            raise FileNotFoundError(
                f"image missing at training time: {sample.image_path}"
            )
        image = np.asarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).astype(np.uint8)

        if self.landmarker is None:
            self.landmarker = get_landmarker()
        landmarks = landmark_detection(self.landmarker, image)
        # Should never happen
        if landmarks is None:
            raise RuntimeError(f"face detection failed for {sample.sample_id}")

        R, t = head_pose_est(landmarks, sample.camera_intrinsics)
        face, left_eye, right_eye, R_virtual = normalize_face(
            image, landmarks, R, t, sample.camera_intrinsics
        )
        fraction, _ = transform_label(
            sample.gaze_target_px, sample.screen_size_px, R_virtual
        )

        return {
            "sample_id": sample.sample_id,
            "face": face,
            "left_eye": left_eye,
            "right_eye": right_eye,
            "label": np.array(fraction, dtype=np.float32),
            "R_norm": R_virtual,
        }
