"""Parse raw MPIIFaceGaze files into Sample records.

This module is the only code that knows the MPIIFaceGaze on-disk format.

Dataset layout (subjects p00-p14):

    <root>/pXX/pXX.txt                      one line per image, 28 whitespace-separated columns
    <root>/pXX/dayNN/NNNN.jpg               face crop
    <root>/pXX/Calibration/Camera.mat       cameraMatrix (3x3, MATLAB v5)
    <root>/pXX/Calibration/screenSize.mat   screen size in pixels and millimeters

Of the 28 annotation columns we only use 0 (image path, relative to the
subject folder) and 1-2 (gaze target on the screen in pixels). The rest
(landmarks, dataset head pose, 3D face center / gaze target, eval eye)
is ignored: Steps 2-3 recompute it, and nothing dataset-specific may
leak past the schema.
"""

import math
import re
from functools import cache
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from src.data.schema import (
    CameraIntrinsics,
    DeviceType,
    PointPx,
    Sample,
    SizeMm,
    SizePx,
    SourceDataset,
    make_sample_id,
)


@cache
def _load_calibration(
    subject_dir: Path,
) -> tuple[CameraIntrinsics, SizePx, SizeMm]:
    camera = loadmat(subject_dir / "Calibration" / "Camera.mat")
    K = np.asarray(camera["cameraMatrix"], dtype=float)
    screen = loadmat(subject_dir / "Calibration" / "screenSize.mat")
    intrinsics = CameraIntrinsics(
        focal_length_px=(float(K[0, 0]) + float(K[1, 1])) / 2,
        principal_point=PointPx(x=float(K[0, 2]), y=float(K[1, 2])),
    )
    screen_size_px = SizePx(
        width=int(np.asarray(screen["width_pixel"]).item()),
        height=int(np.asarray(screen["height_pixel"]).item()),
    )
    screen_size_mm = SizeMm(
        width=float(np.asarray(screen["width_mm"]).item()),
        height=float(np.asarray(screen["height_mm"]).item()),
    )
    return intrinsics, screen_size_px, screen_size_mm


def parse_mpii_sample(subject_dir: Path, line: str) -> Sample | None:
    """Parse one annotation line into a Sample; None (with a printed reason) if invalid."""
    try:
        fields = line.split()
        if len(fields) != 28:
            raise ValueError(f"expected 28 columns, got {len(fields)}")
        image_path = subject_dir / fields[0]
        if not image_path.is_file():
            raise ValueError(f"image not found: {fields[0]}")
        gaze_x = float(fields[1])
        gaze_y = float(fields[2])
        if not (math.isfinite(gaze_x) and math.isfinite(gaze_y)):
            raise ValueError(f"gaze target is not finite: {fields[1]}, {fields[2]}")

        intrinsics, screen_size_px, screen_size_mm = _load_calibration(subject_dir)
        return Sample(
            sample_id=make_sample_id(
                SourceDataset.MPIIFaceGaze,
                subject_dir.name,
                *Path(fields[0]).with_suffix("").parts,
            ),
            source_dataset=SourceDataset.MPIIFaceGaze,
            subject_id=subject_dir.name,
            image_path=image_path.resolve(),
            camera_intrinsics=intrinsics,
            screen_size_mm=screen_size_mm,
            screen_size_px=screen_size_px,
            gaze_target_px=PointPx(x=gaze_x, y=gaze_y),
            device_type=DeviceType.Laptop,
        )
    except Exception as exc:  # noqa: BLE001 anything unparseable just gets dropped
        print(f"dropping sample in {subject_dir.name}: {exc}")
        return None


def parse_mpii(root: Path) -> list[Sample]:
    """Parse the entire MPIIFaceGaze dataset at `root` into Samples."""
    root = Path(root)
    samples: list[Sample] = []
    dropped = 0
    total = 0

    for subject_dir in sorted(root.iterdir()):
        if not (subject_dir.is_dir() and re.fullmatch(r"p\d{2}", subject_dir.name)):
            continue
        annotation_path = subject_dir / f"{subject_dir.name}.txt"
        if not annotation_path.is_file():
            print(f"dropping subject {subject_dir.name}: no annotation file")
            continue
        for line in annotation_path.read_text(encoding="utf-8").splitlines():
            total += 1
            sample = parse_mpii_sample(subject_dir, line)
            if sample is None:
                dropped += 1
            else:
                samples.append(sample)

    print(f"parsed {len(samples)} datapoints, dropped {dropped} of {total}")
    return samples
