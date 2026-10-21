# Experiment log

## Baseline: MPIIFaceGaze, ResNet-18 + shared eye tower

Raw per-epoch records: [`mpiifacegaze-baseline-run.jsonl`](./mpiifacegaze-baseline-run.jsonl)

**Setup.** 11.48M parameters (11.31M of which is an ImageNet-pretrained
ResNet-18 face tower; one eye tower shared across both crops). Participant-
disjoint splits: train p00-p12, validate p13, test p14. AdamW, lr 1e-4,
weight decay 1e-4, batch 256, screen-fraction output. Apple MPS.

Dataset version: 37,350 of 37,667 MPIIFaceGaze samples survive quality
filtering (243 gaze out of bounds, 72 no face detected, 2 reprojection
error). Snapshot id is the sha256 of `data/MPIIFaceGaze/mpii_filtered.json`.

| epoch | train loss | val loss | val error | p95 | seconds |
|---|---|---|---|---|---|
| 0 | 0.02421 | 0.02122 | 0.17491 | 0.3833 | 705 |
| 1 | 0.00740 | 0.02392 | 0.18764 | 0.4026 | 852 |
| **2** | **0.00510** | **0.01397** | **0.13995** | **0.3149** | 1973 |
| 3 | 0.00385 | 0.01721 | 0.15680 | 0.3495 | 703 |
| 4 | 0.00294 | 0.01990 | 0.17174 | 0.3742 | 826 |

Best model is epoch 2. Error is Euclidean distance in screen-fraction units;
on the 286x179mm displays in this dataset that is **4.72 cm, about 4.5
degrees** at a 60 cm viewing distance. Published results on MPIIFaceGaze sit
around 2.4-3.7 degrees, so a first baseline with no pretraining corpus is in
a reasonable range.

## Finding: we are data-limited, not compute-limited

From epoch 3 onward **training loss keeps falling while validation loss
rises** — the model is memorizing rather than generalizing. With only 13
training identities, an 11.5M-parameter network overfits within about three
passes, so more epochs actively make it worse. This is why the loop
checkpoints on best validation error and supports `--patience`.

Epoch 2's 1973s is an artifact of the host swapping, not of the model: six
DataLoader workers on a 16GB machine exhausted memory. Two workers restored
~700s/epoch.

## Next iterations, in expected order of payoff

1. **GazeCapture pretraining** — ~1,450 identities against 13. Directly
   attacks the limit above. Adapter merged; `--init-from` loads a pretrained
   checkpoint with a fresh optimizer.
2. **Data augmentation** — none in this run. Flip, crop jitter and
   photometric jitter now exist in `src/training/augment.py`.
3. **Freeze more of the backbone** — less capacity to memorize.
4. **Per-user calibration** — ridge regression on the 9-point calibration
   flow. Published work roughly halves error with calibration; this fits a
   small linear correction on the frozen network rather than retraining it.
