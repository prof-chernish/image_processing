import cv2
import numpy as np
from PIL import Image


def post_denoise_image(
    pil_image: Image.Image,
    sigma: float = 0.6,
) -> Image.Image:
    """
    Лёгкое сглаживание после апскейла.
    Стабилизация краёв.
    """

    img = np.array(pil_image.convert("RGB"))

    smoothed = cv2.GaussianBlur(
        img,
        ksize=(0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
    )

    return Image.fromarray(smoothed)
