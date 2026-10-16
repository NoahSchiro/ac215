# AC215 Project: Gaze Tracking Maze

This repo tracks the progress of the final project for AC215 @ Harvard for Fall 2026.

## Proposal

`./proposal/`

Full proposal can be found [here](./proposal/README.md). The basic idea is that users will use [gaze tracking](https://en.wikipedia.org/wiki/Eye_tracking) to guide a ball through a maze.

## Training and backend

`./model/`

Training and backend deployment (API wrapper, containerization, `kube.yaml`)

### Data

`./model/data/` (.gitignored)

While I would include the data here through `git lfs`,

- The GazeCapture dataset comes with a license which does not allow for redistribution. It comes as a `tar` full of `tar.gz` so to unpack:
  - Move the `tar` to `./data/`
  - `tar -xf ./data/GazeCapture.tar -C ./data/GazeCapture/`
  - `cd data/GazeCapture/`
  - `for file in *.tar.gz; do tar -xzf "$file" && rm "$file"; done`
- The MPIIFaceGaze dataset has no license restrictions, and as such, can be fetched by `git clone`-ing this repo (**if you have `git lfs` installed!!**). Once you have cloned, just run `tar -xzf data/MPIIFaceGaze.tar.gz -C data`

## Web UI

`./ui/`

The Gaze Maze frontend is built with React, TypeScript, and Vite. The UI is containerized using a multi-stage Docker build: Node.js builds the application, and Nginx serves the compiled files.

### Running the UI with Docker Compose

**Prerequisite:** Docker Desktop must be installed and running.

From the project root, run:

```bash
docker compose up --build -d
```

Open the application in your browser:

**http://localhost:8080**

To verify that the container is running:

```bash
docker compose ps
```

To stop the UI:

```bash
docker compose down
```

### Building the UI Container Directly

Alternatively, build and run the UI without Docker Compose:

```bash
docker build -t ac215-ui ./ui
docker run --rm -p 8080:80 ac215-ui
```

The UI will be available at **http://localhost:8080**.

### Container Configuration

- `ui/Dockerfile` — Multi-stage Node.js and Nginx build.
- `ui/.dockerignore` — Excludes local dependencies, build outputs, and environment files.
- `compose.yaml` — Defines the UI service and exposes port 8080.

### Integration Notes

The current Docker Compose configuration contains the UI service only. The model training container is being developed separately under Issue #40.

When both branches are merged, the Compose configurations should be combined so that the UI and model services can be managed together.

### Verification

The UI container was successfully built and launched using Docker Compose. The Gaze Maze homepage was verified to load at `http://localhost:8080`.

This verifies frontend serving and container startup; full maze gameplay, webcam calibration, and model integration have not yet been tested.
