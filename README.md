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
  - Move the `tar` to `./model/data/`
  - `tar -xf GazeCapture.tar -C GazeCapture/`
  - `cd GazeCapture/`
  - `for file in *.tar.gz; do tar -xzf "$file" && rm "$file"; done`
- The MPIIFaceGaze dataset has no license restrictions, and as such, can be fetched by `git clone`-ing this repo (**if you have `git lfs` installed!!**). Once you have cloned, just run `tar -xzf data/MPIIFaceGaze.tar.gz -C data`

## Web UI

`./ui/`

Front end web ui
