# ---------------------------------------------------------------------------
# Training image for the gaze model.
#
# This exists because MediaPipe's FaceLandmarker aborts on macOS (it fails
# inside Metal, with both the CPU and GPU delegate). The pipeline needs it for
# both manifest building and per-sample preprocessing, so the model cannot run
# on a Mac at all. A Linux container has no Metal, so MediaPipe falls back to
# XNNPACK and works -- the same path CI uses on Ubuntu.
# ---------------------------------------------------------------------------

# uv publishes images with Python preinstalled. pyproject.toml requires
# >=3.14, and pinning the exact base means every teammate and CI build
# resolves the same interpreter rather than whatever their machine has.
FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

# MediaPipe links against OpenGL/EGL at import time even when running on CPU.
# These are present on a desktop Linux box but not in a slim image, so without
# them `import mediapipe` fails. Same three packages CI installs.
#   --no-install-recommends keeps the image small
#   rm -rf /var/lib/apt/lists/* drops the package index, which is ~40MB of
#   cache we never need again
RUN apt-get update && apt-get install -y --no-install-recommends \
        libegl1 libgles2 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies are copied and installed BEFORE the source code, on purpose.
# Docker caches each layer and reuses it when its inputs are unchanged.
# Source changes on every commit; dependencies change rarely. In this order a
# code edit reuses the cached dependency layer, so rebuilds take seconds
# instead of re-resolving and re-downloading torch every time.
COPY pyproject.toml uv.lock ./

# --frozen installs exactly what uv.lock pins and fails if the lockfile is out
# of date, rather than silently resolving something newer. That is what makes
# the image reproducible: same lockfile in, same versions out.
# --no-install-project skips installing our own code, which is not here yet.
RUN uv sync --frozen --no-install-project

# Bake the face landmarker weights into the image.
#
# get_landmarker() otherwise downloads this from Google's CDN on first use.
# That means a container needs network at runtime, and worse, the URL says
# "latest" -- so two builds a month apart could silently get different
# preprocessing. Fetching it at build time pins it to the image.
# ADD fetches a URL without needing curl in the image, which keeps the slim
# base slim and leaves the cached dependency layer above untouched.
ADD https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task \
    /app/models/landmarker.task
RUN chmod 644 /app/models/landmarker.task

# Now the source. Everything below this line rebuilds on a code change; the
# expensive layers above are reused from cache.
COPY src/ ./src/
COPY tests/ ./tests/
COPY main.py train.py ./

# PYTHONUNBUFFERED makes print() appear immediately rather than sitting in a
# buffer, so `docker logs` shows training progress live instead of in bursts.
ENV PYTHONUNBUFFERED=1

# The dataset is ~1.3GB and the GazeCapture tar is 136GB. Neither belongs in
# an image: images get pushed to registries and rebuilt often, and data has a
# different lifecycle from code. Both are mounted at runtime instead.
VOLUME ["/data", "/app/checkpoints"]

# uv run resolves the project's virtualenv, so callers do not have to know
# where it lives. Arguments passed to `docker run` append to this.
ENTRYPOINT ["uv", "run", "python", "train.py"]
CMD ["--help"]
