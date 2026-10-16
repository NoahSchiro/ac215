"""On-device batch augmentation for training.

Augmentation runs inside `forward_batch` on the collated batch -- after the
DataLoader, on `device` -- so evaluation never sees it, the webdataset cache
stays untouched, both datasets get it for free, and the work is vectorized on
the GPU instead of adding to the CPU-bound loader workers. Each augmentation
takes and returns a collated batch with uint8 HWC crops; label-shaped entries
("label", "direction") are transformed to match the geometry.
"""

import math
from typing import Any

import torch
import torch.nn.functional as F


def random_flip(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Mirror half the samples horizontally, labels included.

    The face flips, the eye crops swap sides (each mirrored, which keeps the
    left/right orientation convention the shared eye tower expects), and the
    labels follow the mirrored world: screen-fraction x becomes 1 - x,
    direction x negates.
    """
    flip_p = 0.5
    out = dict(batch)
    n = out["face"].shape[0]
    mask = torch.rand(n, device=device) < flip_p
    m = mask.view(-1, 1, 1, 1)

    face = out["face"].to(device)
    out["face"] = torch.where(m, face.flip(-1), face)
    left = out["left_eye"].to(device)
    right = out["right_eye"].to(device)
    out["left_eye"] = torch.where(m, right.flip(-1), left)
    out["right_eye"] = torch.where(m, left.flip(-1), right)

    # The mirrored world sees the gaze point mirrored on the screen and the
    # gaze vector mirrored about the head's vertical axis.
    if "label" in out:
        label = out["label"].to(device).float().clone()
        label[mask, 0] = 1.0 - label[mask, 0]
        out["label"] = label
    if "direction" in out:
        direction = out["direction"].to(device).float().clone()
        direction[mask, 0] *= -1.0
        out["direction"] = direction
    return out


def random_crop_jitter(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Apply one random rigid perturbation per sample to all crops at once.

    Rotation +/-5 deg, scale +/-10% and translation +/-4% of the image, with
    reflective padding. The same transform hits the face and both eyes, so it
    reads as alignment noise of the whole normalization rather than a warping
    of the face itself.
    """
    max_rotation = 5.0 * math.pi / 180
    max_scale_delta = 0.1
    max_translation = 0.04
    image_keys = ("face", "left_eye", "right_eye")

    out = dict(batch)
    n = out["face"].shape[0]
    theta = torch.zeros(n, 2, 3, device=device)
    angle = (torch.rand(n, device=device) * 2 - 1) * max_rotation
    scale = 1 + (torch.rand(n, device=device) * 2 - 1) * max_scale_delta
    cos, sin = torch.cos(angle) * scale, torch.sin(angle) * scale
    theta[:, 0, 0] = cos
    theta[:, 0, 1] = -sin
    theta[:, 1, 0] = sin
    theta[:, 1, 1] = cos
    theta[:, 0, 2] = (torch.rand(n, device=device) * 2 - 1) * max_translation
    theta[:, 1, 2] = (torch.rand(n, device=device) * 2 - 1) * max_translation
    for key in image_keys:
        img = out[key].to(device).float().permute(0, 3, 1, 2)
        grid = F.affine_grid(theta, img.shape, align_corners=False)
        img = F.grid_sample(img, grid, padding_mode="reflection", align_corners=False)
        out[key] = img.permute(0, 2, 3, 1).round().clamp_(0, 255).to(torch.uint8)
    return out


def random_photometric_jitter(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Random per-sample brightness and contrast, each +/-20%."""
    max_delta = 0.2
    image_keys = ("face", "left_eye", "right_eye")

    out = dict(batch)
    for key in image_keys:
        img = out[key].to(device).float()
        n = img.shape[0]
        brightness = 1 + (torch.rand(n, 1, 1, 1, device=device) * 2 - 1) * max_delta
        contrast = 1 + (torch.rand(n, 1, 1, 1, device=device) * 2 - 1) * max_delta
        mean = img.mean(dim=(1, 2, 3), keepdim=True)
        img = (img - mean) * contrast + mean * brightness
        out[key] = img.round().clamp_(0, 255).to(torch.uint8)
    return out


def augment_batch(
    batch: dict[str, Any],
    device: torch.device,
    flip: bool = False,
    crop_jitter: bool = False,
    photometric: bool = False,
) -> dict[str, Any]:
    """Chain the enabled augmentations over a collated batch.

    With every switch off, the batch passes through unchanged.
    """
    if flip:
        batch = random_flip(batch, device)
    if crop_jitter:
        batch = random_crop_jitter(batch, device)
    if photometric:
        batch = random_photometric_jitter(batch, device)
    return batch
