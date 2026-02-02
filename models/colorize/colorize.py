import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage import color as skcolor

from .model_loader import get_model


def _load_img_rgb(pil_image: Image.Image) -> np.ndarray:
    """
    PIL.Image -> RGB numpy array (HWC, uint8)
    """
    img = pil_image.convert("RGB")
    arr = np.asarray(img)
    if arr.ndim == 2:  # grayscale -> 3ch
        arr = np.tile(arr[:, :, None], (1, 1, 3))
    elif arr.ndim == 3 and arr.shape[2] == 4:  # RGBA -> RGB
        arr = arr[:, :, :3]
    return arr


def _resize_rgb(img_rgb: np.ndarray, hw=(256, 256)) -> np.ndarray:
    return np.asarray(
        Image.fromarray(img_rgb).resize((hw[1], hw[0]), resample=Image.BICUBIC)
    )


def _preprocess_l(img_rgb_orig: np.ndarray, hw=(256, 256)):
    """
    RGB -> L channel (orig + resized) as torch tensors
    """
    img_rgb_rs = _resize_rgb(img_rgb_orig, hw=hw)

    lab_orig = skcolor.rgb2lab(img_rgb_orig)
    lab_rs = skcolor.rgb2lab(img_rgb_rs)

    l_orig = lab_orig[:, :, 0]
    l_rs = lab_rs[:, :, 0]

    tens_l_orig = torch.tensor(l_orig, dtype=torch.float32)[None, None, :, :]
    tens_l_rs = torch.tensor(l_rs, dtype=torch.float32)[None, None, :, :]
    return tens_l_orig, tens_l_rs


def _postprocess_to_rgb(tens_l_orig: torch.Tensor, out_ab: torch.Tensor) -> Image.Image:
    """
    L + ab -> RGB PIL.Image
    """
    hw_orig = tens_l_orig.shape[2:]
    hw = out_ab.shape[2:]

    if hw_orig != hw:
        out_ab = F.interpolate(
            out_ab, size=hw_orig, mode="bilinear", align_corners=False
        )

    lab = torch.cat([tens_l_orig, out_ab], dim=1)[0].permute(1, 2, 0).cpu().numpy()

    rgb = skcolor.lab2rgb(lab)
    rgb_u8 = (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(rgb_u8)



def colorize_image(
    pil_image: Image.Image,
    size: int = 256,
) -> Image.Image:
    """
    ECCV16 colorization.
    Input : PIL.Image (RGB or grayscale)
    Output: PIL.Image (RGB)
    """

    # PIL -> numpy RGB
    img_rgb = _load_img_rgb(pil_image)

    # Preprocess (L channel)
    tens_l_orig, tens_l_rs = _preprocess_l(img_rgb, hw=(size, size))

    # Model
    model = get_model()

    # Inference
    with torch.inference_mode():
        out_ab = model(tens_l_rs)

    # Postprocess -> RGB PIL
    return _postprocess_to_rgb(tens_l_orig, out_ab)
