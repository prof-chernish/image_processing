import numpy as np
from PIL import Image

from classical.post_denoise import post_denoise_image
from tests.utils import images_are_different


def test_post_denoise_changes_image():
    rng = np.random.default_rng(seed=123)

    base = np.full((128, 128, 3), 128, dtype=np.uint8)
    noise = rng.integers(-15, 15, size=base.shape, dtype=np.int16)

    noisy = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(noisy, mode="RGB")

    out = post_denoise_image(img, sigma=0.6)

    assert images_are_different(img, out)
