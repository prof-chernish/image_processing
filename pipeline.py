from PIL import Image

from classical.denoise import denoise_image
from classical.post_denoise import post_denoise_image
from classical.sharpen import sharpen_image
from models.colorize.colorize import colorize_image
from models.deblur.deblur import deblur_image
from models.upscale.modes import mode_improve_details, mode_upscale_image


class ImageTooLargeError(Exception):
    """Изображение превышает допустимый размер"""

    pass


def check_image_size_allowed(
    image: Image.Image,
    *,
    max_pixels: int = 2_000_000,
):
    w, h = image.size
    pixels = w * h

    if pixels > max_pixels:
        raise ImageTooLargeError(
            f"Изображение слишком большое для обработки.\n\n"
            f"Размер: {w}×{h} px\n"
            f"Максимально допустимо: ~1000×2000 px\n\n"
            f"Пожалуйста, уменьшите изображение и попробуйте снова."
        )


def process_image(
    pil_image: Image.Image,
    *,
    do_denoise: bool = True,
    do_deblur: bool = True,
    do_colorize: bool = True,
    upscale_mode: str | None = None,  # None | "resize" | "enhance"
    upscale_scale: int = 2,  # используется только для resize
    do_post_denoise: bool = True,
    do_sharpen: bool = True,
) -> Image.Image:
    """
    Основной пайплайн обработки изображения.
    Вход / выход: PIL.Image
    """

    img = pil_image
    # проверка размера
    check_image_size_allowed(img)

    # denoise
    if do_denoise:
        img = denoise_image(img, h=1)

    # deblur
    if do_deblur:
        img = deblur_image(img)

    # colorize
    if do_colorize:
        img = colorize_image(img)

    # upscale
    if upscale_mode == "resize":
        img = mode_upscale_image(img, scale=upscale_scale)

    elif upscale_mode == "enhance":
        img = mode_improve_details(img)

    # postprocessing (стабилизация краёв)
    if do_post_denoise:
        img = post_denoise_image(img)

    # postprocessing (повышение резкостиЫ)
    if do_sharpen:
        img = sharpen_image(img)

    return img
