from PIL import Image
import numpy as np

from models.colorize.colorize import colorize_image
from tests.utils import images_are_different

def test_colorize_changes_grayscale_image():
    # создаём grayscale изображение с градиентом
    h, w = 128, 128
    gradient = np.tile(
        np.linspace(0, 255, w, dtype=np.uint8),
        (h, 1),
    )

    img_gray = Image.fromarray(gradient, mode="L")

    out = colorize_image(img_gray, size=128)

    # 1. выход должен быть RGB
    assert out.mode == "RGB"

    # 2. изображение должно отличаться от исходного
    assert images_are_different(img_gray.convert("RGB"), out)


