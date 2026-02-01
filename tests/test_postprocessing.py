from PIL import Image
import numpy as np

from postprocessing.preview import apply_postprocessing
from tests.utils import images_are_different

def test_postprocessing_changes_image():
    rng = np.random.default_rng(seed=777)

    base = np.full((128, 128, 3), 130, dtype=np.uint8)
    noise = rng.integers(-20, 20, size=base.shape, dtype=np.int16)

    noisy = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(noisy, mode="RGB")

    out = apply_postprocessing(
        img,
        blur_strength=1.0,
        sharpen_strength=30,
    )

    assert images_are_different(img, out)


def test_postprocessing_noop_when_disabled():
    img = Image.new("RGB", (128, 128), color="gray")

    out = apply_postprocessing(
        img,
        blur_strength=0.0,
        sharpen_strength=0,
    )

    # должно быть строго то же самое
    assert not images_are_different(img, out)
