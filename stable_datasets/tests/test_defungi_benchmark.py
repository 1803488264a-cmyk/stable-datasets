"""Optional DeFungi benchmark configuration and transform checks."""

import pytest
from PIL import Image


torch = pytest.importorskip("torch")
pytest.importorskip("stable_pretraining")

from benchmarks.dataset import get_config  # noqa: E402
from benchmarks.models import ssl_augmentation, val_transform  # noqa: E402


@pytest.mark.parametrize("training", [True, False])
def test_actual_image_transform_contract(training):
    config = get_config("defungi")
    image = Image.new("RGB", (500, 500), (120, 140, 130))
    transform = ssl_augmentation(config, config.image_size, (0.08, 1.0)) if training else val_transform(config)
    result = transform({"image": image, "label": 4})
    assert result["image"].shape == (3, 96, 96)
    assert torch.isfinite(result["image"]).all()
    assert result["label"] == 4


def test_defungi_crop_and_flip_policy():
    config = get_config("defungi")
    transform = ssl_augmentation(config, config.image_size, (0.08, 1.0))
    crop = next(op for op in transform.args if type(op).__name__ == "RandomResizedCrop")
    flip = next(op for op in transform.args if type(op).__name__ == "RandomHorizontalFlip")
    assert tuple(crop.scale) == (0.5, 1.0)
    assert flip.p == 0.5


def test_cmu_configuration_is_unchanged():
    config = get_config("cmufaces")
    assert config.image_size == (64, 64)
    assert config.horizontal_flip_p == 0.0
    assert config.min_crop_scale == 0.8
    assert config.mean == [0.27856666]
    assert config.std == [0.24860810]
