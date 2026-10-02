"""Tests for landmark_detection."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from src.data.schema import CameraIntrinsics, ImageArray, PointPx
from src.data.utils import (
    _GENERIC_FACE_MODEL,
    _MODEL_LANDMARK_IDS,
    get_landmarker,
    head_pose_est,
    landmark_detection,
)

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


def make_landmarks_from_projection(
    rvec: np.ndarray, tvec: np.ndarray, camera_matrix: np.ndarray
) -> list[PointPx]:
    """Apply rotation / translation vectors to generic 3D face."""
    projected, _ = cv2.projectPoints(
        _GENERIC_FACE_MODEL, rvec, tvec, camera_matrix, None
    )
    landmarks = [PointPx(x=0.0, y=0.0)] * 478
    for i, (x, y) in zip(_MODEL_LANDMARK_IDS, projected.reshape(-1, 2)):
        landmarks[i] = PointPx(x=float(x), y=float(y))
    return landmarks


def test_recovers_known_pose_from_synthetic_projection() -> None:
    """solvePnP must invert the projection of a known pose exactly."""
    intrinsics = CameraIntrinsics(
        focal_length_px=1000.0, principal_point=PointPx(x=640.0, y=360.0)
    )
    camera_matrix = np.array([
        [1000.0, 0.0, 640.0],
        [0.0, 1000.0, 360.0],
        [0.0, 0.0, 1.0]]
    )
    rvec_true = np.array([[0.15], [-0.25], [0.05]])  # yaw/pitch/roll
    tvec_true = np.array([[3.0], [-4.0], [60.0]])    # x/y/z

    landmarks = make_landmarks_from_projection(rvec_true, tvec_true, camera_matrix)
    r_mat, t_mat = head_pose_est(landmarks, intrinsics)

    R_true, _ = cv2.Rodrigues(rvec_true)
    assert np.allclose(r_mat, R_true, atol=1e-6)
    assert np.allclose(t_mat, tvec_true.reshape(3), atol=1e-6)


def test_fronto_parallel_face_gives_identity_rotation() -> None:
    intrinsics = CameraIntrinsics(
        focal_length_px=1000.0, principal_point=PointPx(x=0.0, y=0.0)
    )
    # Face with no rotation, 60 cm in front of camera
    landmarks = make_landmarks_from_projection(
        np.zeros((3, 1)), np.array([[0.0], [0.0], [60.0]]), np.eye(3) * 1000.0
    )

    r_mat, t_mat = head_pose_est(landmarks, intrinsics)

    assert np.allclose(r_mat, np.eye(3), atol=1e-6)
    assert np.allclose(t_mat, [0.0, 0.0, 60.0], atol=1e-6)


def test_rejects_landmark_lists_without_the_model_points() -> None:
    intrinsics = CameraIntrinsics(
        focal_length_px=1000.0, principal_point=PointPx(x=0.0, y=0.0)
    )

    with pytest.raises(ValueError, match="Face Mesh"):
        head_pose_est([PointPx(x=0.0, y=0.0)] * 10, intrinsics)
