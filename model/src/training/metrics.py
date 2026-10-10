"""Evaluation metrics.

Mean error is the headline number, but it does not say whether the model is
usable as a controller: low mean error with high frame-to-frame jitter gives a
pointer that vibrates. We report spread alongside the mean, and break results
down per participant so an average cannot hide the subject the model fails on.
"""

from dataclasses import dataclass, field

import numpy as np
import torch
from numpy.typing import NDArray


@dataclass
class ErrorStats:
    """Aggregated error for one evaluation pass."""

    mean: float
    median: float
    p95: float
    count: int
    per_participant: dict[str, ErrorStats] = field(default_factory=dict)

    def worst_participant(self) -> tuple[str, ErrorStats] | None:
        if not self.per_participant:
            return None
        subject = max(self.per_participant, key=lambda k: self.per_participant[k].mean)
        return subject, self.per_participant[subject]


def screen_fraction_error(
    predicted: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Per-sample Euclidean error in screen-fraction units (both (B, 2))."""
    if predicted.shape != target.shape:
        raise ValueError(f"shape mismatch: {predicted.shape} vs {target.shape}")
    return torch.linalg.vector_norm(predicted - target, dim=-1)


def angular_error_deg(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Angle between predicted and target gaze directions in degrees (both (B, 3))."""
    predicted = predicted / torch.linalg.vector_norm(predicted, dim=-1, keepdim=True)
    target = target / torch.linalg.vector_norm(target, dim=-1, keepdim=True)
    cosine = torch.clamp((predicted * target).sum(dim=-1), -1.0, 1.0)
    return torch.rad2deg(torch.arccos(cosine))


def fraction_to_cm(
    fraction_error: NDArray[np.float64] | float,
    screen_width_mm: float,
    screen_height_mm: float,
) -> NDArray[np.float64] | float:
    """Convert screen-fraction error to centimetres on a given display.

    Fraction error is dimensionless, so the physical error depends on the
    screen. The diagonal extent is used as the scale.
    """
    if screen_width_mm <= 0 or screen_height_mm <= 0:
        raise ValueError("screen dimensions must be positive")
    diagonal_cm = float(np.hypot(screen_width_mm, screen_height_mm)) / 10.0
    return fraction_error * diagonal_cm


def cm_to_degrees(
    error_cm: NDArray[np.float64] | float, viewing_distance_cm: float = 60.0
) -> NDArray[np.float64] | float:
    """Convert on-screen error to visual angle, for comparison with the literature."""
    if viewing_distance_cm <= 0:
        raise ValueError("viewing distance must be positive")
    return np.rad2deg(np.arctan2(error_cm, viewing_distance_cm))


def summarize(errors: torch.Tensor, subject_ids: list[str] | None = None) -> ErrorStats:
    """Reduce per-sample errors to an ErrorStats, optionally split by subject."""
    if errors.numel() == 0:
        raise ValueError("cannot summarize an empty error tensor")
    values = errors.detach().float().cpu().numpy()

    per_participant: dict[str, ErrorStats] = {}
    if subject_ids is not None:
        if len(subject_ids) != len(values):
            raise ValueError(
                f"got {len(subject_ids)} subject ids for {len(values)} errors"
            )
        grouped: dict[str, list[float]] = {}
        for subject, value in zip(subject_ids, values, strict=True):
            grouped.setdefault(subject, []).append(float(value))
        per_participant = {
            subject: summarize(torch.tensor(group))
            for subject, group in sorted(grouped.items())
        }

    return ErrorStats(
        mean=float(np.mean(values)),
        median=float(np.median(values)),
        p95=float(np.percentile(values, 95)),
        count=int(values.size),
        per_participant=per_participant,
    )
