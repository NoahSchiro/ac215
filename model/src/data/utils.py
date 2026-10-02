"""
Primary preprocessing tools:

1. Landmarker pulls out key features of face
2. head_pose_est estimates head rotation/translation from those landmarks
"""

import os
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)
from numpy.typing import NDArray

from src.data.schema import CameraIntrinsics, ImageArray, PointPx


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

"""
Generic 3D face model for head pose: 

Taken from MediaPipe's canonical face model (google-ai-edge/mediapipe,
canonical_face_model.obj) reference points. Coordinates are [x,y,z]
where y is up, z faces the camera. Units are centimeters

These values are what we would expect from a generic face 
looking straight into the camera 
"""
_GENERIC_FACE_MODEL = np.array([
    [-4.446, -2.664, -3.173],  # right eye, outer corner (MediaPipe landmark 33)
    [-1.856, -2.585, -3.758],  # right eye, inner corner (MediaPipe landmark 133)
    [1.856, -2.585, -3.758],   # left eye, inner corner  (MediaPipe landmark 362)
    [4.446, -2.664, -3.173],   # left eye, outer corner  (MediaPipe landmark 263)
    [-2.456, 4.343, -4.284],   # mouth, right corner     (MediaPipe landmark 61)
    [2.456, 4.343, -4.284],    # mouth, left corner      (MediaPipe landmark 291)
    [0.0, 1.127, -7.476],      # nose tip                (MediaPipe landmark 1)
    [0.0, 9.403, -4.264],      # chin                    (MediaPipe landmark 152)
])

# MediaPipe Face Mesh indices of key points we are interested in
_MODEL_LANDMARK_IDS = (33, 133, 362, 263, 61, 291, 1, 152)


def head_pose_est(
    landmarks: list[PointPx], intrinsics: CameraIntrinsics
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Estimate head pose by computing a rotation + translation matrix from a
    generic 3D face to the observed subject face.

    Args:
        landmarks: pixel-space keypoints from `landmark_detection` (MediaPipe
            Face Mesh, 468+ points).
        intrinsics: camera calibration for the image; estimated ones stay
            flagged via `CameraIntrinsics.is_estimate` in the schema.

    Returns:
        (R, t) mapping the 3D model into the OpenCV camera frame
        (+X image right, +Y image down, +Z forward): R is the 3x3 rotation
        matrix, t the model origin's translation in the model's units
        (~centimetres; expect t[2] ~ 50-60 for MPIIFaceGaze webcam distance).
        A face looking straight at the camera gives R ~ identity.
    """
    # We need to ensure out landmark feature vector has all of the features we want
    if len(landmarks) < max(_MODEL_LANDMARK_IDS) + 1:
        raise ValueError(
            f"head pose needs the full Face Mesh (>= {max(_MODEL_LANDMARK_IDS) + 1} points),"
            f" got {len(landmarks)}"
        )
    
    # Extract only the points we care about
    image_points = np.array(
        [[landmarks[i].x, landmarks[i].y] for i in _MODEL_LANDMARK_IDS],
        dtype=np.float64,
    )
    camera_matrix = np.array(
        [
            [intrinsics.focal_length_px, 0.0, intrinsics.principal_point.x],
            [0.0, intrinsics.focal_length_px, intrinsics.principal_point.y],
            [0.0, 0.0, 1.0],
        ]
    )
    # This func finds the 6 DOF rotation + translation vector (rvec, tvec)
    # between the subjects current pose and the "looking straight-on" pose 
    found, rvec, tvec = cv2.solvePnP(
        _GENERIC_FACE_MODEL,
        image_points,
        camera_matrix,
        None,  # lens distortion is not modeled
        flags=cv2.SOLVEPNP_SQPNP,
    )
    if not found:
        raise ValueError("solvePnP did not converge")

    # Vector -> matrix
    rotation = np.asarray(cv2.Rodrigues(rvec)[0], dtype=np.float64)
    translation = np.asarray(tvec, dtype=np.float64).reshape(3)
    return rotation, translation
