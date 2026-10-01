"""Canonical per-sample schema. Every dataset MUST produce these artifacts

- `sample_id` is globally unique across the whole manifest and always
  starts with the source dataset tag, e.g.
  ``"mpiifacegaze/p01/day01/0001"`` or
  ``"gazecapture/123456789/rec-0/frame-000123"``. Use `make_sample_id` to
  build ids. Global uniqueness cannot be checked by this module
- `gaze_target_px` holds the raw label in the source dataset's native
  units (for MPIIFaceGaze: screen pixels; for GazeCapture: screen points).
- `image` / `image_path`: the raw image (full frame or provided crop, per
  dataset). Pixels are an in-memory-only field (uint8, H×W×3, RGB
  channel order).
- `camera_intrinsics.is_estimate` marks intrinsics the adapter had to
  guess rather than read from dataset metadata
- Equality/hash cover the metadata fields only; `image` pixels are
  excluded (use `numpy.array_equal` explicitly if you need pixel
  equality). This keeps samples with loaded pixels safely comparable.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

ImageArray = NDArray[np.uint8]
"""Raw image pixels: uint8, shape (H, W, 3), RGB channel order."""


class SourceDataset(StrEnum):
    """Dataset a sample originated from. Extension point for new adapters."""

    MPIIFaceGaze = "mpiifacegaze"
    # GazeCapture = "gazecapture" #eventually...


class DeviceType(StrEnum):
    """Physical device category used at capture time."""

    Laptop = "laptop"
    Phone = "phone"
    Tablet = "tablet"


@dataclass(frozen=True, slots=True)
class PointPx:
    """A 2D point in pixel units."""

    x: float
    y: float


@dataclass(frozen=True, slots=True)
class SizeMm:
    """Physical extent in millimeters."""

    width: float
    height: float

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError(f"Width must be positive, got {self.width}")
        if self.height <= 0:
            raise ValueError(f"Height must be positive, got {self.height}")


@dataclass(frozen=True, slots=True)
class SizePx:
    """Extent in screen pixels."""

    width: int
    height: int

    def __post_init__(self) -> None:
        if not isinstance(self.width, int):
            raise TypeError(f"width must be an int, got {self.width}")
        if self.width <= 0:
            raise ValueError(f"width must be positive, got {self.width}")
        if not isinstance(self.height, int):
            raise TypeError(f"height must be an int, got {self.height}")
        if self.height <= 0:
            raise ValueError(f"height must be positive, got {self.height}")


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    """Pinhole camera intrinsics, per dataset or estimated."""

    focal_length_px: float
    principal_point: PointPx
    is_estimate: bool = False

    def __post_init__(self) -> None:
        if self.focal_length_px <= 0:
            raise ValueError(f"Width must be positive, got {self.focal_length_px}")


def make_sample_id(source_dataset: SourceDataset | str, *parts: str) -> str:
    """Build a unique sample id: `{source_tag}/{part}/{part}`."""
    tag = SourceDataset(source_dataset).value
    cleaned = tuple(part.strip("/") for part in parts)
    if not cleaned or any(not part for part in cleaned):
        raise ValueError("sample_id parts must be non-empty")
    return "/".join((tag, *cleaned))


class CameraIntrinsicsEntry(TypedDict):
    focal_length_px: float
    principal_point: list[float]
    is_estimate: bool


class SampleManifestEntry(TypedDict):
    """JSON-serializable form of `Sample`"""

    sample_id: str
    source_dataset: str
    subject_id: str
    image_path: str
    camera_intrinsics: CameraIntrinsicsEntry
    screen_size_mm: list[float]
    screen_size_px: list[int]
    gaze_target_px: list[float]
    device_type: str


@dataclass(frozen=True, slots=True, kw_only=True)
class Sample:
    """Canonical per-sample record. See module docstring."""

    sample_id: str
    source_dataset: SourceDataset
    subject_id: str
    image_path: Path
    image: ImageArray | None = field(default=None, compare=False)
    camera_intrinsics: CameraIntrinsics
    screen_size_mm: SizeMm
    screen_size_px: SizePx
    gaze_target_px: PointPx
    device_type: DeviceType

    def __post_init__(self) -> None:
        if not isinstance(self.source_dataset, SourceDataset):
            raise TypeError(
                f"source_dataset must be a SourceDataset, got {self.source_dataset!r}"
            )
        if not isinstance(self.device_type, DeviceType):
            raise TypeError(
                f"device_type must be a DeviceType, got {self.device_type!r}"
            )
        if not self.sample_id:
            raise ValueError("sample_id must be non-empty")
        tag = f"{self.source_dataset.value}/"
        if not self.sample_id.startswith(tag):
            raise ValueError(
                f"sample_id must start with the source dataset tag {tag}, got {self.sample_id}"
            )
        if not self.sample_id[len(tag) :]:
            raise ValueError(
                f"sample_id must have at least one key part after the tag, got {self.sample_id}"
            )
        if not self.subject_id:
            raise ValueError("subject_id must be non-empty")

        if self.image is not None:
            if self.image.ndim != 3 or self.image.shape[2] != 3:
                raise ValueError(
                    f"image must have shape (H, W, 3), got {self.image.shape}"
                )
            if self.image.shape[0] == 0 or self.image.shape[1] == 0:
                raise ValueError("image must have non-zero height and width")
            if self.image.dtype != np.uint8:
                raise ValueError(f"image dtype must be uint8, got {self.image.dtype}")

    def to_manifest_entry(self) -> SampleManifestEntry:
        """JSON-serializable manifest form; image pixels are never included."""
        intrinsics = self.camera_intrinsics
        return SampleManifestEntry(
            sample_id=self.sample_id,
            source_dataset=self.source_dataset.value,
            subject_id=self.subject_id,
            image_path=str(self.image_path),
            camera_intrinsics=CameraIntrinsicsEntry(
                focal_length_px=intrinsics.focal_length_px,
                principal_point=[
                    intrinsics.principal_point.x,
                    intrinsics.principal_point.y,
                ],
                is_estimate=intrinsics.is_estimate,
            ),
            screen_size_mm=[self.screen_size_mm.width, self.screen_size_mm.height],
            screen_size_px=[self.screen_size_px.width, self.screen_size_px.height],
            gaze_target_px=[self.gaze_target_px.x, self.gaze_target_px.y],
            device_type=self.device_type.value,
        )

    @classmethod
    def from_manifest_entry(cls, entry: SampleManifestEntry) -> Sample:
        """Rebuild a metadata-only Sample (``image`` is None) from a manifest entry."""
        intrinsics = entry["camera_intrinsics"]
        return cls(
            sample_id=entry["sample_id"],
            source_dataset=SourceDataset(entry["source_dataset"]),
            subject_id=entry["subject_id"],
            image_path=Path(entry["image_path"]),
            camera_intrinsics=CameraIntrinsics(
                focal_length_px=float(intrinsics["focal_length_px"]),
                principal_point=PointPx(
                    x=float(intrinsics["principal_point"][0]),
                    y=float(intrinsics["principal_point"][1]),
                ),
                is_estimate=bool(intrinsics["is_estimate"]),
            ),
            screen_size_mm=SizeMm(
                width=float(entry["screen_size_mm"][0]),
                height=float(entry["screen_size_mm"][1]),
            ),
            screen_size_px=SizePx(
                width=int(entry["screen_size_px"][0]),
                height=int(entry["screen_size_px"][1]),
            ),
            gaze_target_px=PointPx(
                x=float(entry["gaze_target_px"][0]),
                y=float(entry["gaze_target_px"][1]),
            ),
            device_type=DeviceType(entry["device_type"]),
        )
