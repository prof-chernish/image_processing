import numpy as np
from PIL import Image


def sharpen_image(
    pil_image: Image.Image,
    amount: float = 0.3,
    radius: int = 1,
    threshold: int = 0,
) -> Image.Image:
    """
    Нерезкое маскирование (Unsharp Mask).

    amount    — сила шарпа
    radius    — радиус размытия
    threshold — порог
    """
    import cv2

    img = np.array(pil_image.convert("RGB"))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # размытие
    blurred = cv2.GaussianBlur(
        img_bgr,
        ksize=(0, 0),
        sigmaX=radius,
        sigmaY=radius,
    )

    # unsharp mask
    sharpened = cv2.addWeighted(
        img_bgr,
        1.0 + amount,
        blurred,
        -amount,
        0,
    )

    # опциональный threshold
    if threshold > 0:
        low_contrast_mask = np.abs(img_bgr - blurred) < threshold
        sharpened[low_contrast_mask] = img_bgr[low_contrast_mask]

    sharpened_rgb = cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)
    return Image.fromarray(sharpened_rgb)
