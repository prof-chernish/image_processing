from PIL import Image

from models.upscale.upscale import upscale_image


def mode_upscale_image(pil_image: Image.Image, scale: int = 2) -> Image.Image:
    """
    Режим 1: Увеличение изображения.
    Размер в пикселях увеличивается.
    """
    return upscale_image(pil_image, scale=scale)


def mode_improve_details(
    pil_image: Image.Image,
    scale: int = 2,
) -> Image.Image:
    """
    Режим 2: Улучшение детализации без изменения размера.
    Алгоритм:
    upscale -> resize обратно к исходному размеру
    """
    # исходный размер
    original_size = pil_image.size  # (width, height)

    # апскейл (получаем больше деталей)
    upscaled = upscale_image(pil_image, scale=scale)

    # уменьшаем обратно
    improved = upscaled.resize(original_size, resample=Image.LANCZOS)

    return improved
