import numpy as np
import pytest
from PIL import Image

from pipeline import ImageTooLargeError, process_image
from tests.utils import images_are_different


def make_image(w, h, random=False):
    # вспомогательная функция для создания изображения
    if random:
        arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    else:
        arr = np.zeros((h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def test_pipeline_all_steps_enabled():
    # проверяем, что пайплайн работает при всех включенных шагах
    # итоговое изображение должно меняться
    
    img = make_image(64, 64, random=True)

    out = process_image(
        img,
        do_denoise=True,
        do_deblur=True,
        do_colorize=True,
        upscale_mode="enhance",
        do_post_denoise=True,
        do_sharpen=True,
    )

    assert isinstance(out, Image.Image)
    assert out.size == img.size
    assert images_are_different(img, out)


def test_pipeline_noop_when_all_disabled():
    # проверяем, что пайплайн работает при всех выключенных шагах
    # итоговое изображение не должно меняться
    
    img = make_image(64, 64, random=True)

    out = process_image(
        img,
        do_denoise=False,
        do_deblur=False,
        do_colorize=False,
        upscale_mode=None,
        do_post_denoise=False,
        do_sharpen=False,
    )

    assert isinstance(out, Image.Image)
    assert out.size == img.size
    assert not images_are_different(img, out)


def test_pipeline_some_steps_enabled():
    # проверяем, что пайплайн работает при некоторых включенных шагах
    # итоговое изображение должно меняться

    img = make_image(64, 64, random=True)

    out = process_image(
        img,
        do_denoise=True,
        do_deblur=False,
        do_colorize=False,
        upscale_mode=None,
        do_post_denoise=True,
        do_sharpen=True,
    )

    assert isinstance(out, Image.Image)
    assert out.size == img.size
    assert images_are_different(img, out)


def test_pipeline_rejects_too_large_image():
    # программа не должна работать с изображениями, размером больше 2_000_000 пикселей
    img = make_image(2000, 1100)

    with pytest.raises(ImageTooLargeError):
        process_image(
            img,
            do_denoise=False,
            do_deblur=False,
            do_colorize=False,
            upscale_mode=None,
            do_post_denoise=False,
            do_sharpen=False,
        )


def test_pipeline_accepts_boundary_size_noop():
    # программа должна работать с изображениями, размером равным 2_000_000 пикселей
    img = make_image(2000, 1000, random=True)

    out = process_image(
        img,
        do_denoise=False,
        do_deblur=False,
        do_colorize=False,
        upscale_mode=None,
        do_post_denoise=False,
        do_sharpen=False,
    )

    assert isinstance(out, Image.Image)
    assert out.size == img.size
    assert not images_are_different(img, out)
