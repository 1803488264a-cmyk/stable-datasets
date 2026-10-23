"""Optional benchmark regressions; requires the benchmark dependencies."""

import pytest
from PIL import Image


torch = pytest.importorskip("torch")
pytest.importorskip("stable_pretraining")

from benchmarks.dataset import get_config  # noqa: E402
from benchmarks.models import ssl_augmentation, val_transform  # noqa: E402


def test_cmu_training_and_validation_image_contract():
    config = get_config("cmufaces")
    sample = {"image": Image.new("L", (32, 30), color=128), "label": 2}
    for transform in (val_transform(config), ssl_augmentation(config, (64, 64), (0.08, 1.0))):
        result = transform(sample.copy())
        assert result["image"].shape == (3, 64, 64)
        assert torch.isfinite(result["image"]).all()
        assert result["label"] == 2


@pytest.mark.parametrize("requested_scale", [(0.08, 1.0), (0.05, 0.4)])
def test_cmu_augmentation_preserves_pose_constraints(requested_scale):
    config = get_config("cmufaces")
    transform = ssl_augmentation(config, (64, 64), requested_scale)
    crop = next(op for op in transform.args if type(op).__name__ == "RandomResizedCrop")
    flip = next(op for op in transform.args if type(op).__name__ == "RandomHorizontalFlip")
    assert crop.scale[0] >= 0.8
    assert crop.scale[0] <= crop.scale[1] <= 1.0
    assert flip.p == 0.0


def test_existing_cifar_augmentation_defaults_are_unchanged():
    config = get_config("cifar10")
    transform = ssl_augmentation(config, (32, 32), (0.08, 1.0))
    crop = next(op for op in transform.args if type(op).__name__ == "RandomResizedCrop")
    flip = next(op for op in transform.args if type(op).__name__ == "RandomHorizontalFlip")
    assert tuple(crop.scale) == (0.08, 1.0)
    assert flip.p == 0.5
