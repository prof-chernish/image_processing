import numpy as np
from PIL import Image

from classical.denoise import denoise_image
from tests.utils import images_are_different


def test_denoise_changes_image():
    # проверяем, что denoise меняет изображение
    
    # создаём тестовое изображение с шумом
    rng = np.random.default_rng(seed=42)

    base = np.full((128, 128, 3), 128, dtype=np.uint8)
    noise = rng.integers(-10, 10, size=base.shape, dtype=np.int16)

    noisy = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(noisy, mode="RGB")

    # применяем denoise
    out = denoise_image(img, h=3)

    # проверяем, что изображение изменилось
    assert images_are_different(img, out)
