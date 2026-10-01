"""Tests for landmark_detection."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.data.schema import ImageArray, PointPx
from src.data.utils import get_landmarker, landmark_detection

# I pulled this image from MPII
FACE_IMAGE_PATH = Path(__file__).parent / "data" / "face.jpg"


def load_face_image() -> ImageArray:
    return np.asarray(Image.open(FACE_IMAGE_PATH).convert("RGB"))


def test_detects_landmarks_on_face() -> None:
    image = load_face_image()
    model = get_landmarker()
    landmarks = landmark_detection(model, image)

    assert landmarks is not None
    # 468 face mesh points + 10 iris points
    assert len(landmarks) == 478
    assert all(isinstance(point, PointPx) for point in landmarks)


def test_landmarks_are_in_pixel_coordinates() -> None:
    image = load_face_image()
    model = get_landmarker()
    landmarks = landmark_detection(model, image)

    assert landmarks is not None
    height, width = image.shape[0], image.shape[1]
    assert all(0 <= point.x <= width for point in landmarks)
    assert all(0 <= point.y <= height for point in landmarks)


def test_eye_landmark_anatomy() -> None:
    """Subject's right eye (33) must sit image-left of their left eye (263)."""
    image = load_face_image()
    model = get_landmarker()
    landmarks = landmark_detection(model, image)

    assert landmarks is not None
    height = int(image.shape[0])
    assert landmarks[33].x < landmarks[263].x
    assert landmarks[33].y == pytest.approx(landmarks[263].y, abs=height * 0.05)


def test_detection_is_deterministic() -> None:
    image = load_face_image()
    model = get_landmarker()

    first = landmark_detection(model, image)
    second = landmark_detection(model, image)

    assert first is not None and second is not None
    assert first == second


def test_returns_none_without_face() -> None:
    noise = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    model = get_landmarker()

    assert landmark_detection(model, noise) is None
