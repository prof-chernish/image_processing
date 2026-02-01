from PIL import Image, ImageFilter


def apply_postprocessing(
    base_img: Image.Image,
    blur_strength: float,
    sharpen_strength: int,
) -> Image.Image:
    """
    Быстрая интерактивная постобработка изображения.

    base_img         — результат основного пайплайна (НЕ меняется)
    blur_strength    — 0.0 .. 3.0 (радиус GaussianBlur)
    sharpen_strength — 0 .. 100 (сила UnsharpMask)
    """

    img = base_img

    if blur_strength > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=float(blur_strength)))

    if sharpen_strength > 0:
        img = img.filter(
            ImageFilter.UnsharpMask(
                radius=2, percent=int(sharpen_strength), threshold=3
            )
        )

    return img
