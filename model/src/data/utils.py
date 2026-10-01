"""
Primary preprocessing tools:

1. Landmarker pulls out key features of face
"""

import os
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import mediapipe as mp
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)

from src.data.schema import ImageArray, PointPx


def get_landmarker() -> Any:
    """Create the FaceLandmarker."""
    # This lands at project_root/model/models/landmarker.task
    # TODO: there might be an argument that we want these weights
    # stored somewhere more generally, since we need it in the
    # browser too... Keep it here for now
    model_dir = Path(__file__).resolve().parents[2] / "models"
    model_path = model_dir / "landmarker.task"

    # If model isn't locally present, fetch and cache
    os.makedirs(model_dir, exist_ok=True)
    if not model_path.is_file():
        print(f"downloading face landmarker model to {model_path}")
        tmp_path = model_path.with_name(model_path.name + ".part")
        urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/latest/face_landmarker.task",
            tmp_path,
        )
        tmp_path.replace(model_path)

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
    )
    return FaceLandmarker.create_from_options(options)


def landmark_detection(landmarker, image: ImageArray) -> list[PointPx] | None:
    """Detect facial landmarks in an RGB uint8 (H, W, 3) image.

    Returns keypoints in pixel coordinates, or None when no face is
    detected.
    """
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None
    (face,) = result.face_landmarks  # num_faces=1 -> exactly zero or one
    height, width = image.shape[0], image.shape[1]
    return [PointPx(x=landmark.x * width, y=landmark.y * height) for landmark in face]
