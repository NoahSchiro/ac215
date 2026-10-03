"""Parse raw MPIIFaceGaze files into Sample records. Contains
the MPIIDataset class as well
"""

import math
import os
import re
from functools import cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.io import loadmat
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


class MPII(Dataset):
    """Quality-filtered MPIIFaceGaze as a torch Dataset.

    Args:
        participants: participant number(s) to include (0..14 -> p00..p14).
            Accepts a single int or a list of ints
        root: raw MPIIFaceGaze root, only needed when the filtered manifest
            does not exist yet (it is built and cached on first use).
    """

    def __init__(
        self,
        participants: int | list[int],
        dataset_root: os.PathLike[str],
    ) -> None:

        manifest_path = Path(dataset_root) / "mpii_filtered.json"

        # MPIIFaceGaze has 15 participants: p00..p14
        num_participants = 15

        if isinstance(participants, int):
            participants = [participants]
        for p in participants:
            if not 0 <= p < num_participants:
                raise ValueError(
                    f"participant {p} is out of range: MPIIFaceGaze has"
                    f" p00..p{num_participants - 1}"
                )
        self.participants = sorted({f"p{p:02d}" for p in participants})

        if manifest_path.is_file():
            self.samples = read_manifest(manifest_path)
        else:
            print(f"no manifest at {manifest_path}; building it from the raw dataset")
            self.samples = filter_dataset(parse_mpii(Path(dataset_root)))
            write_manifest(self.samples, manifest_path)

        wanted = set(self.participants)
        self.samples = [s for s in self.samples if s.subject_id in wanted]

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
