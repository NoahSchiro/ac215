"""
Primary preprocessing tools:

1. Landmarker pulls out key features of face
2. head_pose_est estimates head rotation/translation from those landmarks
3. normalize_face warps the image into a canonical face/eye frame
"""

import io
import json
import math
import os
import tarfile
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np
import webdataset as wds
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)
from numpy.typing import NDArray
from tqdm import tqdm

from src.data.schema import CameraIntrinsics, ImageArray, PointPx, Sample, SizePx


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
_GENERIC_FACE_MODEL = np.array(
    [
        [-4.446, -2.664, -3.173],  # right eye, outer corner (MediaPipe landmark 33)
        [-1.856, -2.585, -3.758],  # right eye, inner corner (MediaPipe landmark 133)
        [1.856, -2.585, -3.758],  # left eye, inner corner  (MediaPipe landmark 362)
        [4.446, -2.664, -3.173],  # left eye, outer corner  (MediaPipe landmark 263)
        [-2.456, 4.343, -4.284],  # mouth, right corner     (MediaPipe landmark 61)
        [2.456, 4.343, -4.284],  # mouth, left corner      (MediaPipe landmark 291)
        [0.0, 1.127, -7.476],  # nose tip                (MediaPipe landmark 1)
        [0.0, 9.403, -4.264],  # chin                    (MediaPipe landmark 152)
    ]
)

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


def normalize_face(
    image: ImageArray,
    landmarks: list[PointPx],
    R: NDArray[np.float64],
    t: NDArray[np.float64],
    intrinsics: CameraIntrinsics,
) -> tuple[ImageArray, ImageArray, ImageArray, NDArray[np.float64]]:
    """Warp the image into a canonical, head-centered frame.

    We place a virtual camera at a fixed position and normalize the face to
    be at a fixed position away from the camera with a fixed rotation

    Args:
        image: input RGB image.
        landmarks: pixel-space Face Mesh points (needs the iris points 468/473).
        R, t: head pose from `head_pose_est` (model -> camera frame).
        intrinsics: camera calibration for this image.

    Returns:
        normalize face crop, normalized eye crops, and the virtual
        camera rotation R_norm.
    """
    # Iris centres from the Face Mesh.
    left_iris_id = 473
    right_iris_id = 468

    # Check we have enough landmarks
    if len(landmarks) < max(left_iris_id, right_iris_id) + 1:
        raise ValueError(
            f"normalization needs the full refined Face Mesh"
            f" (>= {max(left_iris_id, right_iris_id) + 1} points),"
            f" got {len(landmarks)}"
        )

    def normalize(v: NDArray[np.float64]) -> NDArray[np.float64]:
        return v / np.linalg.norm(v)

    # Fixed camera parameters for normalization. These
    # need be identical for every sample from every dataset
    # in both python and typescript
    norm_dist = 60.0  # (cm)
    norm_focal = 960.0  # (cm)
    face_size = 224  # (px)

    # Virtual camera rotation:
    # z-axis = camera to face
    # x-axis = straight up
    z_axis = normalize(R @ np.array([0.0, 0.0, 1.0]))
    head_x = R @ np.array([1.0, 0.0, 0.0])
    # this trick ensures that the x_axis is orthogonal with the z-axis,
    # even if it is not *exactly* the x rotation of the head
    x_axis = normalize(head_x - (head_x @ z_axis) * z_axis)
    y_axis = np.cross(z_axis, x_axis)
    R_virtual = np.column_stack((x_axis, y_axis, z_axis))

    # Virtual camera center
    C_virtual = t - norm_dist * z_axis
    K = np.array(
        [  # camera specs
            [intrinsics.focal_length_px, 0.0, intrinsics.principal_point.x],
            [0.0, intrinsics.focal_length_px, intrinsics.principal_point.y],
            [0.0, 0.0, 1.0],
        ]
    )
    K_norm = np.array(
        [  # virtual camera specs
            [norm_focal, 0.0, face_size / 2],
            [0.0, norm_focal, face_size / 2],
            [0.0, 0.0, 1.0],
        ]
    )

    # H_norm is a translation matrix which answers: for each pixel captured by
    # a real camera, where would that be on on an image taken by our virtual camera
    H_norm = norm_dist * K @ R_virtual @ np.linalg.inv(K_norm) + (K @ C_virtual)[
        :, None
    ] * np.array([[0.0, 0.0, 1.0]])

    # Finally we can apply the rotations / translations to our assets
    face = np.asarray(
        cv2.warpPerspective(
            image,
            H_norm,
            (face_size, face_size),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        ),
        dtype=np.uint8,
    )

    H_src_to_dst = np.linalg.inv(H_norm)

    eye_size = (36, 60)  # (height px, width px)

    def warped(point: PointPx) -> tuple[float, float]:
        out = H_src_to_dst @ np.array([point.x, point.y, 1.0])
        return out[0] / out[2], out[1] / out[2]

    left = np.asarray(
        cv2.getRectSubPix(
            face, (eye_size[1], eye_size[0]), warped(landmarks[left_iris_id])
        ),
        dtype=np.uint8,
    )
    right = np.asarray(
        cv2.getRectSubPix(
            face, (eye_size[1], eye_size[0]), warped(landmarks[right_iris_id])
        ),
        dtype=np.uint8,
    )
    return face, left, right, R_virtual


def transform_label(
    gaze_target_px: PointPx,
    screen_size_px: SizePx,
    R_virtual: NDArray[np.float64],
    gaze_direction_cam: NDArray[np.float64] | None = None,
) -> tuple[tuple[float, float], NDArray[np.float64] | None]:
    """Convert a raw gaze label to match the normalized image.

    We output two formats of labels
        - a fraction of the screen (x/W, y/H)
        - Normalized gaze direction (the angle off of the z-axis)

    Args:
        gaze_target_px: raw on-screen gaze position.
        screen_size_px: resolution of the screen the position refers to.
        R_virtual: virtual camera rotation from `normalize_face`.
        gaze_direction_cam: optional 3D gaze direction in the camera frame
            (e.g. gt - fc when a dataset provides it) to also get the
            angle-based label.

    Returns:
        (screen_fraction, direction_norm)
    """
    fraction = (
        gaze_target_px.x / screen_size_px.width,
        gaze_target_px.y / screen_size_px.height,
    )

    direction: NDArray[np.float64] | None = None
    if gaze_direction_cam is not None:
        direction = R_virtual.T @ np.asarray(gaze_direction_cam, dtype=np.float64)
        length = float(np.linalg.norm(direction))
        if not math.isfinite(length) or length == 0.0:
            raise ValueError("gaze direction must be a finite, non-zero vector")
        direction = direction / length

    return fraction, direction


def check_sample(
    landmarker: Any, sample: Sample, max_reproj: float = 0.1 
) -> dict[str, Any] | str:
    """Run every preprocessing step on one sample.

    Quality checks first (cheapest first):
    - gaze target inside the screen
    - image readable
    - face detection succeeds
    - head pose solvable
    - reprojection error
    - iris points near their eye corners

    Samples passing all checks continue through face normalization and
    the label transform.

    Args:
        landmarker: FaceLandmarker instance (e.g. from `get_landmarker`).
        sample: parsed sample whose image lives at `image_path`.
        max_reproj: mean reprojection error, as a fraction of the
            projected eye-corner distance, above which the sample is
            dropped.

    Returns:
        The per-item dict
        ``{"sample_id", "face", "left_eye", "right_eye", "label", "R_norm"}``
        or a string naming the reason the sample was dropped.
    """
    fx = sample.gaze_target_px.x / sample.screen_size_px.width
    fy = sample.gaze_target_px.y / sample.screen_size_px.height
    if not (math.isfinite(fx) and math.isfinite(fy)) or not (
        0 <= fx <= 1 and 0 <= fy <= 1
    ):
        return "gaze_out_of_bounds"

    image_bgr = cv2.imread(str(sample.image_path))
    if image_bgr is None:
        return "image_unreadable"
    image = np.asarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)).astype(np.uint8)

    landmarks = landmark_detection(landmarker, image)
    if landmarks is None:
        return "face_not_detected"

    try:
        R, t = head_pose_est(landmarks, sample.camera_intrinsics)
    except ValueError:
        return "pose_failed"

    # reprojection error, normalized by the projected eye-corner distance
    camera_focal = sample.camera_intrinsics.focal_length_px
    camera_x = sample.camera_intrinsics.principal_point.x
    camera_y = sample.camera_intrinsics.principal_point.y
    K = np.array([
        [camera_focal, 0.0, camera_x],
        [0.0, camera_focal, camera_y],
        [0.0, 0.0, 1.0],
    ])
    observed = np.array(
        [[landmarks[i].x, landmarks[i].y] for i in _MODEL_LANDMARK_IDS]
    )
    projected, _ = cv2.projectPoints(
        _GENERIC_FACE_MODEL, cv2.Rodrigues(R)[0], t, K, None
    )
    projected = projected.reshape(-1, 2)
    eye_span = float(np.linalg.norm(projected[0] - projected[3]))
    reproj = float(np.linalg.norm(projected - observed, axis=1).mean()) / max(
        eye_span, 1e-9
    )
    if reproj > max_reproj:
        return "reprojection_error"

    # iris centres must sit near their eye corners' bounding box
    for iris_id, corner_a, corner_b in ((468, 33, 133), (473, 263, 362)):
        a, b, iris = landmarks[corner_a], landmarks[corner_b], landmarks[iris_id]
        margin = 0.25 * max(abs(b.x - a.x), abs(b.y - a.y))
        in_box = (
            min(a.x, b.x) - margin <= iris.x <= max(a.x, b.x) + margin
            and min(a.y, b.y) - margin <= iris.y <= max(a.y, b.y) + margin
        )
        if not in_box:
            return "eyes_off_face"

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


def build_webdataset(
    samples: list[Sample], path: str | os.PathLike[str], num_threads: int | None = None
) -> None:
    """Preprocess samples with `check_sample` and cache them as a webdataset.

    Drops invalid samples. Kept samples are written to shard files under
    `path` (a directory), keyed by their sample_id:

    - `{key}.json`      manifest metadata, plus `label` and `R_norm`
    - `{key}.png`       normalized face crop (224x224x3 RGB)
    - `{key}.eyes.npy`  uint8 (2, 36, 60, 3); [0] left, [1] right eye

    Args:
        samples: parsed samples (e.g. from `parse_mpii`).
        path: directory for the shard files; created if missing.
        num_threads: pool size; defaults to the number of CPU cores.
    """
    path = Path(path)
    workers = num_threads or os.cpu_count() or 1
    workers = max(1, min(workers, len(samples)))

    thread_local = threading.local()
    landmarker_lock = threading.Lock()

    def thread_landmarker() -> Any:
        """One FaceLandmarker instance per worker thread."""
        if getattr(thread_local, "landmarker", None) is None:
            with landmarker_lock:
                thread_local.landmarker = get_landmarker()
        return thread_local.landmarker

    drop_counts: Counter[tuple[str, str]] = Counter()
    written = 0

    path.mkdir(parents=True, exist_ok=True)
    for stale in path.glob("*.tar"):
        stale.unlink()  # a rebuild must not mix in shards of a previous run

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = pool.map(
            lambda s: check_sample(thread_landmarker(), s),
            samples,
        )
        with wds.ShardWriter(
            str(path / f"{path.name}-%06d.tar"), maxcount=1000  # ~150 MB per shard
        ) as sink:
            for sample, result in tqdm(zip(samples, results), total=len(samples)):
                if isinstance(result, str):
                    drop_counts[(sample.source_dataset.value, result)] += 1
                    continue
                sink.write({
                    "__key__": result["sample_id"],
                    "json": {
                        **sample.to_manifest_entry(),
                        "label": result["label"].tolist(),
                        "R_norm": result["R_norm"].tolist(),
                    },
                    "png": result["face"],
                    "eyes.npy": np.stack(
                        [result["left_eye"], result["right_eye"]]
                    ),
                })
                written += 1

    print(
        f"wrote {written}/{len(samples)} samples to {path}"
        f" ({len(samples) - written} dropped)"
    )
    for (dataset, reason), count in sorted(drop_counts.items()):
        print(f"  {dataset} / {reason}: {count}")


@cache
def shard_index(shard: str) -> dict[str, tuple[int, int]]:
    """Scan a shard once: member name -> (data offset, size) for every member.

    The index is tiny (a few ints per member), so caching it for every
    shard costs nothing and reads never have to rescan the file or keep
    tarfile handles open.
    """
    with tarfile.open(shard, "r") as tf:
        return {
            info.name: (info.offset_data, info.size)
            for info in tf
            if info.isfile()
        }


def read_shard_member(shard: str, name: str) -> bytes:
    """Read one member's bytes from a shard."""
    index = shard_index(shard)
    if name not in index:
        raise FileNotFoundError(f"member {name} missing in {shard}")
    offset, size = index[name]
    with open(shard, "rb") as f:
        f.seek(offset)
        return f.read(size)


def read_webdataset_sample(shard: str, key: str) -> dict[str, Any]:
    """Read one preprocessed sample from a shard written by `build_webdataset`.

    Args:
        shard: shard file the sample lives in.
        key: the sample's sample_id.

    Returns:
        The per-item dict
        ``{"sample_id", "face", "left_eye", "right_eye", "label", "R_norm"}``
        - exactly what `check_sample` computed when the sample was cached.
    """
    meta = json.loads(read_shard_member(shard, f"{key}.json"))
    if not isinstance(meta, dict):
        raise TypeError(f"metadata for {key} in {shard} is not a JSON object")

    bgr = cv2.imdecode(
        np.frombuffer(read_shard_member(shard, f"{key}.png"), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if bgr is None:
        raise ValueError(f"corrupt face crop for {key} in {shard}")
    face = np.asarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)

    eyes = np.load(io.BytesIO(read_shard_member(shard, f"{key}.eyes.npy")))

    return {
        "sample_id": str(meta["sample_id"]),
        "face": face,
        "left_eye": eyes[0],
        "right_eye": eyes[1],
        "label": np.array(meta["label"], dtype=np.float32),
        "R_norm": np.array(meta["R_norm"], dtype=np.float64),
    }
