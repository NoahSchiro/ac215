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

Front end web ui

## Containerized Data Preprocessing

The MPIIFaceGaze preprocessing pipeline can be executed using Docker Compose. This setup follows the containerization workflow covered in AC215 Lectures 2–4.

### Prerequisites

- Docker Desktop with the Docker Engine running
- The MPIIFaceGaze dataset, downloaded separately
- The dataset extracted to `data/MPIIFaceGaze/`

The dataset is not included in this repository. It is licensed under CC BY-NC-SA 4.0. See the dataset's original documentation for attribution and usage conditions.

### Build and Run

From the project root:

```bash
docker compose build preprocess
docker compose up
```

Docker Compose builds the image from `model/Dockerfile` and runs `model/preprocess.py`.

The container reads the dataset from `/data/MPIIFaceGaze` and writes generated artifacts to `/artifacts`. These directories are connected to the host using Docker Compose bind mounts.

The demonstration processes five samples from participant `p00`.

### Generated Artifacts

After a successful run:

```text
artifacts/
├── mpii_filtered.json
├── index.json
└── shards/
    └── shards-w000-000000.tar
```

- `mpii_filtered.json`: Metadata manifest for the selected input samples.
- `index.json`: Manifest SHA256 and output shard hashes.
- `shards/`: Preprocessed WebDataset archives containing normalized image crops and labels.

The generated artifacts are excluded from Git because they may contain participant face images.

### Data Version Tracking

The SHA256 hash of `mpii_filtered.json` serves as the input metadata snapshot identifier. Identical input metadata produced the same manifest SHA256 across repeated runs.

The SHA256 of each output shard is recorded in `index.json` to identify the generated artifact. Shard hashes may differ between runs.

This approach provides lightweight version tracking without requiring DVC.

### Evidence

The `evidence/` directory contains:

- `preprocess-summary.log`: Successful preprocessing execution logs.
- `sample-output.json`: Sample count, manifest SHA256, and output shard SHA256.

The five-sample demonstration successfully processed 5/5 samples with zero dropped samples and exited with code 0.

### Limitations

This demonstration currently requires a locally extracted copy of MPIIFaceGaze. The committed evidence contains metadata and logs, not a self-contained runnable sample dataset.

The repository does not redistribute original participant photographs or processed face crops.
