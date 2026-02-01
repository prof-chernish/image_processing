import numpy as np
from PIL import Image


def images_are_different(img1: Image.Image, img2: Image.Image) -> bool:
    """
    Возвращает True, если изображения различаются.
    Используется для проверки факта применения преобразования.
    """
    arr1 = np.array(img1)
    arr2 = np.array(img2)

    if arr1.shape != arr2.shape:
        return True

    return not np.array_equal(arr1, arr2)
