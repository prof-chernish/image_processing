from PIL import Image

from classical.denoise import denoise_image
from models.deblur.deblur import deblur_image
from models.colorize.colorize import colorize_image
from models.upscale.upscale import upscale_image
from classical.post_denoise import post_denoise_image
from classical.sharpen import sharpen_image

class ImageTooLargeError(Exception):
    """Изображение превышает допустимый размер"""
    pass



def check_image_size_allowed(
    image: Image.Image,
    *,
    max_pixels: int = 12_000_000,
):
    w, h = image.size
    pixels = w * h

    if pixels > max_pixels:
        raise ImageTooLargeError(
            f"Изображение слишком большое для обработки.\n\n"
            f"Размер: {w}×{h} px\n"
            f"Максимально допустимо: ~3000×4000 px\n\n"
            f"Пожалуйста, уменьшите изображение и попробуйте снова."
        )




def process_image(
    pil_image: Image.Image,
    *,
    do_denoise: bool = True,
    do_deblur: bool = True,
    do_colorize: bool = True,
    do_upscale: bool = True,
    do_post_denoise: bool = True,
    do_sharpen: bool = True,
) -> Image.Image:
    """
    Основной пайплайн обработки изображения.
    Вход / выход: PIL.Image
    """

    img = pil_image
    # 🔒 ЕДИНАЯ РАННЯЯ ПРОВЕРКА
    check_image_size_allowed(img)

    # 1. Ultra-light denoise (по умолчанию OFF)
    if do_denoise:
        img = denoise_image(img, h=1)

    # 2. Deblur
    if do_deblur:
        img = deblur_image(img)

    # 3. Colorize
    if do_colorize:
        img = colorize_image(img)

    # 4. Upscale
    if do_upscale:
        img = upscale_image(img)


    # 5. Post-upscale denoise (стабилизация краёв)
    if do_post_denoise:
        img = post_denoise_image(img)

    # 6. Sharpen (мягкий, OFF по умолчанию)
    if do_sharpen:
        img = sharpen_image(img)

    return img
