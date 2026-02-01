import numpy as np
from PIL import Image


def denoise_image(
    pil_image: Image.Image,
    h: int = 1,
    template_window_size: int = 7,
    search_window_size: int = 21,
) -> Image.Image:
    """
    Мягкий денойз

    """
    import cv2

    img = np.array(pil_image.convert("RGB"))

    denoised = cv2.fastNlMeansDenoisingColored(
        img,
        None,
        h,  # сила цветового денойза
        h,  # сила яркостного денойза
        template_window_size,
        search_window_size,
    )

    return Image.fromarray(denoised)
