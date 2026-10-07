"""Multi-tower gaze network.

One tower over the normalized face crop, one shared tower over both eye
crops. The eye tower is shared deliberately: eyes are near mirror images, so
one set of weights applied to both (with the left crop flipped into the
right's orientation) halves the parameters for no measured loss.

The output head is a constructor argument because the label frame is still
an open question. `screen_fraction` matches what `MPII` currently emits;
`gaze_direction` is the formulation the literature uses with normalized
crops, where you un-normalize with `R_norm` and intersect the screen plane.
Swapping between them should not require touching the backbone.

Size matters here beyond the usual: these weights are CDN-delivered and run
in the browser, so parameter count is a product constraint, not just a
training one. `parameter_count()` is there to keep that visible.
"""

from typing import Literal

import torch
import torchvision
from torch import nn

OutputMode = Literal["screen_fraction", "gaze_direction"]

_OUTPUT_DIMS: dict[str, int] = {"screen_fraction": 2, "gaze_direction": 3}

# ImageNet statistics, required because the face backbone is pretrained.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class EyeTower(nn.Module):
    """Compact conv stack for 36x60 eye crops.

    ResNet is a poor fit at this resolution -- its stem downsamples by 4
    before the first block, which leaves almost nothing. This keeps spatial
    detail for three blocks instead.
    """

    def __init__(self, out_features: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            self._block(3, 32),
            self._block(32, 64),
            self._block(64, 128),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.project = nn.Linear(128, out_features)
        self.out_features = out_features

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, eye: torch.Tensor) -> torch.Tensor:
        return self.project(self.features(eye))


class GazeNet(nn.Module):
    """Face tower + shared eye tower, fused into a configurable head.

    Args:
        output_mode: `screen_fraction` emits (B, 2) in [0, 1] via sigmoid;
            `gaze_direction` emits an L2-normalized (B, 3) unit vector.
        pretrained: ImageNet initialization for the face backbone. With 15
            training identities this matters a great deal -- from scratch is
            not a serious option at this data scale.
        use_eyes: set False to ablate the eye towers and measure what they
            are actually worth.
    """

    def __init__(
        self,
        output_mode: OutputMode = "screen_fraction",
        *,
        pretrained: bool = True,
        use_eyes: bool = True,
        eye_features: int = 128,
        hidden: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if output_mode not in _OUTPUT_DIMS:
            raise ValueError(
                f"output_mode must be one of {sorted(_OUTPUT_DIMS)}, got {output_mode!r}"
            )
        self.output_mode: OutputMode = output_mode
        self.use_eyes = use_eyes

        weights = (
            torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        backbone = torchvision.models.resnet18(weights=weights)
        face_features = int(backbone.fc.in_features)
        backbone.fc = nn.Identity()
        self.face_tower = backbone

        fused = face_features
        if use_eyes:
            self.eye_tower: nn.Module | None = EyeTower(eye_features)
            fused += 2 * eye_features
        else:
            self.eye_tower = None

        self.head = nn.Sequential(
            nn.Linear(fused, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, _OUTPUT_DIMS[output_mode]),
        )

    def forward(
        self,
        face: torch.Tensor,
        left_eye: torch.Tensor | None = None,
        right_eye: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = [self.face_tower(face)]

        if self.eye_tower is not None:
            if left_eye is None or right_eye is None:
                raise ValueError("use_eyes=True requires both eye crops")
            # Flip the left crop so the shared tower always sees one orientation.
            parts.append(self.eye_tower(torch.flip(left_eye, dims=(-1,))))
            parts.append(self.eye_tower(right_eye))

        logits = self.head(torch.cat(parts, dim=-1))

        if self.output_mode == "screen_fraction":
            # Labels are fractions of the screen, so the range is known.
            return torch.sigmoid(logits)
        return logits / torch.linalg.vector_norm(logits, dim=-1, keepdim=True)

    def parameter_count(self) -> int:
        """Trainable parameters. Watch this: the weights ship to a browser."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def prepare_images(batch: torch.Tensor, device: torch.device) -> torch.Tensor:
    """uint8 (B, H, W, 3) from the Dataset -> normalized float (B, 3, H, W).

    The Dataset hands back HWC uint8 because that is what OpenCV produces;
    conversion happens here, on device, so the DataLoader stays cheap.
    """
    if batch.ndim != 4 or batch.shape[-1] != 3:
        raise ValueError(f"expected (B, H, W, 3), got {tuple(batch.shape)}")
    images = batch.to(device, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0
    mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return (images - mean) / std


def select_device(preference: str = "auto") -> torch.device:
    """Pick a device. `auto` prefers CUDA, then Apple MPS, then CPU."""
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
