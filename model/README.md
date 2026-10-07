# Gaze Tracking model and deployment

## Data preprocessing

The preprocessing pipeline turns the raw [MPIIFaceGaze](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/its-written-all-over-your-face-full-face-appearance-based-gaze-estimation) dataset into a quality-filtered, normalized torch `Dataset`. Everything lives in `src/data/`

### Setup

```bash
uv sync
```

If you have `git cloned` with `git lfs` installed, your data will be placed in the correct location.

```
data/MPIIFaceGaze/
├── p00/
│   ├── p00.txt                  # one line per image, 28 whitespace-separated columns
│   ├── day01/0005.jpg           # face crops
│   └── Calibration/
│       ├── Camera.mat           # camera intrinsics (fx, fy, cx, cy)
│       └── screenSize.mat       # screen size in pixels and millimeters
├── p01/ ... p14/
```

The Face Landmarker model (`models/landmarker.task`) is downloaded automatically on first use. Note: MediaPipe's runtime needs `libEGL`/`libGLESv2` on Linux.

### Pipeline steps

**1. Parsing** — `parse_mpii()` / `parse_mpii_sample()` (`src/data/mpiifacegaze.py`)
The only code that knows the on-disk MPIIFaceGaze format. Of each annotation line's 28 columns we use the image path and the gaze target on screen (pixels). Per-subject calibration comes from `Camera.mat` (focal length = mean of fx/fy, principal point) and `screenSize.mat` (screen size in px and mm). Anything that doesn't parse is dropped with a printed reason and counted.
  * Output: `Sample` records per the schema in `src/data/schema.py`. This is a standard format every later step consumes. Every dataset must output a list of `Sample`

**2. Face/landmark detection** — `landmark_detection(landmarker, image)` (`src/data/utils.py`)
MediaPipe Face Landmarker (478 points with iris refinement) on RGB uint8 images. 
  * Output: pixel-space keypoints, or `None` when no face is found. The detector is the same one intended for the browser inference.

**3. Head pose** — `head_pose_est(landmarks, intrinsics)`
Fits a generic 3D face model to the landmarks with `cv2.solvePnP`. 
  * Output: rotation `R` (3×3) and translation `t` (centimeters) mapping the model into the camera frame; a face looking straight at the camera gives `R = np.eye(3)` (identity matrix).

**4. Normalization** — `normalize_face(image, landmarks, R, t, intrinsics)`
Warps the image so the face is upright, fronto-parallel and at a fixed scale, regardless of original pose/distance. We do this by placing a virtual camera 60 cm along the head's viewing axis (focal 960 px) and the image is resampled by mapping the true camera to this virtual camera. 
  * Output: 224×224 face crop, 36×60 left/right eye crops, and the virtual-camera rotation `R_virtual`.

**5. Label transform** — `transform_label(gaze_target_px, screen_size_px, R_virtual, gaze_direction_cam=None)`
Canonical label = normalized screen fraction `(x/W, y/H)` If a 3D gaze direction is available, it is also rotated by `R_virtual.T` into the normalized frame as a unit vector. Fractions are deliberately not clamped; out-of-bounds labels are filtered in the next step.
  * Output: normalized labels as fraction of screen and as 3D gaze direction

**6. Quality filtering** — `filter_dataset(samples, num_threads=None)`
Iterates through all `Sample` and drops samples if they cannot pass these preproceesing steps. Prints a summary on what was dropped and why.
  * Output: only high quality `Sample`

**7. torch Dataset** — `MPII(participants, dataset_root)` (`src/data/mpiifacegaze.py`)
Loads the filtered manifest, or builds it from scratch and places it at `data/MPIIFaceGaze/mpii_filtered.json`. Selects participants for subject-disjoint train/test splits. `__getitem__` loads the image and computes steps 2–5 **on the fly**, returning:

```python
{
    "sample_id": str,                  # "mpiifacegaze/p00/day01/0005"
    "face":       uint8 (224, 224, 3), # normalized face crop (RGB)
    "left_eye":   uint8 (36, 60, 3),   # subject's anatomical left eye
    "right_eye":  uint8 (36, 60, 3),   # subject's anatomical right eye
    "label":      float32 (2,),        # normalized screen fraction
    "R_norm":     float64 (3, 3),      # virtual-camera rotation
}
```

### Running it

`main.py` builds the filtered manifest or loads it if it is already built

```bash
uv run python main.py
```

### Notes

- Nothing but the manifest is written to disk; crops and labels are recomputed per `__getitem__`. At our current scale, this is okay, but if we start getting up to millions of samples, a significant fraction of our compute time will be re-running preprocessing steps for each epoch. So might be worth it to cache.
- Filtering thresholds live as constants in `filter_dataset`; the virtual-camera parameters are constants in `normalize_face`. Both must stay identical across datasets and match the TypeScript port of steps 2 and 4 planned for browser inference.
- Adding a second dataset should be simple write a new parser to `Sample`, pass those samples through `filter_dataset()` and wrap it up in a torch Dataset.

## Training

### Why this runs in a container

MediaPipe's FaceLandmarker aborts on macOS, inside Metal, with both the CPU and
GPU delegate. The pipeline needs it for manifest building and for every
`__getitem__`, so the model cannot run natively on a Mac at all. A Linux
container has no Metal, so MediaPipe falls back to XNNPACK — the same path CI
uses on Ubuntu. Linux users can skip the container and run `train.py` directly.

The tradeoff: Docker Desktop on macOS does not pass through Apple's GPU, so
training in the container is CPU-only. Once preprocessed crops are cached to
disk (PR #32), training will no longer need MediaPipe and can run natively on
MPS instead.

### Running it

```bash
docker compose run --rm train                      # defaults in docker-compose.yml
docker compose run --rm train --epochs 1 --val 5   # arguments override
```

Or directly, without compose:

```bash
docker build -t gaze-model:dev ./model
docker run --rm --shm-size=2g \
  -v "$PWD/data:/data" -v "$PWD/model/checkpoints:/app/checkpoints" \
  gaze-model:dev --data-root /data/MPIIFaceGaze --val 13 --test 14 --epochs 1
```

`--shm-size=2g` is required. PyTorch's DataLoader passes tensors between worker
processes through `/dev/shm`, which Docker defaults to 64MB; a batch of 224x224
images overruns it and fails with `No space left on device`, which looks like a
full disk but is not. Compose sets this via `shm_size`.

### What the training code does

| Module | Role |
|---|---|
| `src/training/splits.py` | Participant-disjoint splits. `Split` validates pairwise disjointness at construction, so an overlapping split fails loudly rather than reporting an inflated number. |
| `src/training/model.py` | `GazeNet`: an ImageNet-pretrained ResNet-18 over the face crop, plus one eye tower shared across both eyes (the left is flipped into the right's orientation). 11.48M parameters, 11.31M of which is the backbone. |
| `src/training/metrics.py` | Mean/median/p95 plus a per-participant breakdown, with converters to cm and degrees so results compare against published MPIIFaceGaze figures. |
| `src/training/loop.py` | Train/eval loops, per-epoch and best checkpointing, resume. Experiment tracking goes through a `RunLogger` protocol, so wandb stays an optional import. |

The output head is a constructor argument (`screen_fraction` or
`gaze_direction`) because the label frame is still open: `normalize_face` warps
the crop into a head-centered frame, but the emitted label is a raw screen
fraction, which depends on head position relative to the screen. Swapping the
head is a CLI flag, not a rewrite.

### Data versioning

`write_manifest` emits byte-identical JSON for identical input, so
`sha256 data/MPIIFaceGaze/mpii_filtered.json` is a dataset snapshot ID. Record
it alongside any reported result.
