import numpy as np
from PIL import Image

from models.upscale.modes import mode_improve_details, mode_upscale_image
from tests.utils import images_are_different


def test_upscale_resize_x2_changes_size():
    # проверяем, что upscale увеличивает изображение ровно в 2 раза, если выбран масштаб 2

    img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))

    out = mode_upscale_image(img, scale=2)

    assert out.size == (128, 128)


def test_upscale_resize_x4_changes_size_more():
    # проверяем, что upscale увеличивает изображение ровно в 4 раза, если выбран масштаб 4

    img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))

    out = mode_upscale_image(img, scale=4)

    assert out.size == (256, 256)


def test_enhance_keeps_size_but_changes_image():
    # проверяем режим улучшения детализации без изменения размера
    img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))

    out = mode_improve_details(img)

    # размер не меняется
    assert out.size == img.size

    # содержимое меняется
    assert images_are_different(img, out)
