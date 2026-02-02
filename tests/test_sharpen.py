import numpy as np
from PIL import Image

from classical.sharpen import sharpen_image
from tests.utils import images_are_different


def test_sharpen_changes_image():
    # проверяем, что sharpen меняет изображение
    
    rng = np.random.default_rng(seed=2024)

    # базовое изображение с шумом
    base = np.full((128, 128, 3), 120, dtype=np.uint8)
    noise = rng.integers(-20, 20, size=base.shape, dtype=np.int16)

    noisy = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(noisy, mode="RGB")

    out = sharpen_image(
        img,
        amount=0.8,
        radius=1,
        threshold=0,
    )

    assert images_are_different(img, out)
