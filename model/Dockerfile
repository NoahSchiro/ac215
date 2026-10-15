# Initial Dockerfile
# Stage 1: Install Python dependencies
FROM python:3.14 AS builder

WORKDIR /app

RUN pip install uv

# Copy dependency files first for Docker layer caching
COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev

# Stage 2: Create the runtime container
FROM python:3.14-slim AS runtime

WORKDIR /app

# MediaPipe requires these Linux runtime libraries
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libegl1 libgles2 && \
    rm -rf /var/lib/apt/lists/*

# Copy the installed environment from the builder
COPY --from=builder /app/.venv /app/.venv

# Copy the application code
COPY . .

# Use the Python interpreter from the virtual environment
CMD ["/app/.venv/bin/python", "main.py"]