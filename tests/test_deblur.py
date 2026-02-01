from PIL import Image
import numpy as np

from models.deblur.deblur import deblur_image
from tests.utils import images_are_different

def test_deblur_changes_image():
    rng = np.random.default_rng(seed=2025)

    # базовое изображение
    base = np.full((128, 128, 3), 120, dtype=np.uint8)

    # добавляем шум
    noise = rng.integers(-25, 25, size=base.shape, dtype=np.int16)
    noisy = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    img = Image.fromarray(noisy, mode="RGB")

    out = deblur_image(img)

    # 1. размер должен совпадать
    assert out.size == img.size

    # 2. изображение должно отличаться
    assert images_are_different(img, out)
