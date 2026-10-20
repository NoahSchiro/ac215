"""Tests for on-device batch augmentation."""

from typing import Any

import pytest
import torch

from src.training.augment import augment_batch


def make_batch(n: int = 64) -> dict[str, Any]:
    """A small collated batch with distinct labels per sample."""
    torch.manual_seed(0)
    return {
        "face": torch.randint(0, 256, (n, 224, 224, 3), dtype=torch.uint8),
        "left_eye": torch.randint(0, 256, (n, 36, 60, 3), dtype=torch.uint8),
        "right_eye": torch.randint(0, 256, (n, 36, 60, 3), dtype=torch.uint8),
        "label": torch.rand(n, 2),
        "direction": torch.nn.functional.normalize(torch.rand(n, 3), dim=-1),
        "sample_id": [f"s{i}" for i in range(n)],
    }


def test_no_flags_passes_through() -> None:
    batch = make_batch()
    out = augment_batch(batch, torch.device("cpu"))
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(out[key], value)
        else:
            assert out[key] == value


def test_flip_swaps_eyes_and_transforms_labels() -> None:
    batch = make_batch()
    out = augment_batch(batch, torch.device("cpu"), flip=True)

    changed = (out["face"] != batch["face"]).any(dim=(1, 2, 3))
    n = len(batch["face"])
    # A fair coin over 64 samples: neither side of the coin should be empty.
    assert 16 < changed.sum() < n - 16

    # Flipped samples: mirrored face, swapped mirrored eyes, and the labels
    # follow the mirrored world.
    assert torch.equal(out["face"][changed], batch["face"][changed].flip(-1))
    assert torch.equal(
        out["left_eye"][changed], batch["right_eye"][changed].flip(-1)
    )
    assert torch.equal(
        out["right_eye"][changed], batch["left_eye"][changed].flip(-1)
    )
    assert out["label"][changed, 0] == pytest.approx(1.0 - batch["label"][changed, 0])
    assert out["direction"][changed, 0] == pytest.approx(-batch["direction"][changed, 0])
    assert out["direction"][changed, 1:] == pytest.approx(batch["direction"][changed, 1:])

    # Unflipped samples pass through untouched.
    same = ~changed
    assert torch.equal(out["face"][same], batch["face"][same])
    assert out["label"][same, 0] == pytest.approx(batch["label"][same, 0])


def test_flip_works_without_a_direction_key() -> None:
    batch = make_batch()
    del batch["direction"]
    out = augment_batch(batch, torch.device("cpu"), flip=True)
    assert "label" in out


def test_crop_jitter_preserves_shape_and_range() -> None:
    batch = make_batch()
    out = augment_batch(batch, torch.device("cpu"), crop_jitter=True)

    for key in ("face", "left_eye", "right_eye"):
        assert out[key].shape == batch[key].shape
        assert out[key].dtype == torch.uint8
        assert out[key].min() >= 0 and out[key].max() <= 255
        # Continuous jitter always changes the pixels.
        assert not torch.equal(out[key], batch[key])


def test_photometric_jitter_changes_images() -> None:
    batch = make_batch()
    out = augment_batch(batch, torch.device("cpu"), photometric=True)

    for key in ("face", "left_eye", "right_eye"):
        assert out[key].shape == batch[key].shape
        assert out[key].dtype == torch.uint8
        assert not torch.equal(out[key], batch[key])
