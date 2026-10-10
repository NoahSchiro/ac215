"""Multi-tower gaze network.

One tower over the normalized face crop, one tower shared across both eye
crops: the left crop is flipped into the right's orientation so a single set
of weights sees one canonical orientation.

`output_mode` selects the head. `screen_fraction` emits (B, 2) in [0, 1] and
is what we train on today; `gaze_direction` emits an L2-normalized (B, 3) and
is wired up for if we move to the normalized-space formulation.
"""

from typing import Literal

import torch
import torchvision
from torch import nn

OutputMode = Literal["screen_fraction", "gaze_direction"]

_OUTPUT_DIMS: dict[str, int] = {"screen_fraction": 2, "gaze_direction": 3}


class EyeTower(nn.Module):
    """Compact conv stack for 36x60 eye crops.

    A ResNet stem downsamples by 4 before its first block, which leaves almost
    nothing at this resolution, so this keeps spatial detail for three blocks.
    """

    def __init__(self, out_features: int = 128) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for in_channels, out_channels in ((3, 32), (32, 64), (64, 128)):
            layers += [
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
        layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten()]
        self.features = nn.Sequential(*layers)
        self.project = nn.Linear(128, out_features)
        self.out_features = out_features

    def forward(self, eye: torch.Tensor) -> torch.Tensor:
        return self.project(self.features(eye))


class GazeNet(nn.Module):
    """Face tower plus a shared eye tower, fused into a configurable head.

    Args:
        output_mode: which head to build; see the module docstring.
        pretrained: ImageNet initialization for the face backbone. With few
            training identities this matters; set False for random init.
        use_eyes: set False to ablate the eye towers and measure their worth.
    """

    def __init__(
        self,
        *,
        output_mode: OutputMode = "screen_fraction",
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
        self.pretrained = pretrained
        self.use_eyes = use_eyes

        backbone = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
            if pretrained
            else None
        )
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
            parts.append(self.eye_tower(torch.flip(left_eye, dims=(-1,))))
            parts.append(self.eye_tower(right_eye))

        logits = self.head(torch.cat(parts, dim=-1))

        if self.output_mode == "screen_fraction":
            return torch.sigmoid(logits)
        return logits / torch.linalg.vector_norm(logits, dim=-1, keepdim=True)

    def parameter_count(self) -> int:
        """Trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
