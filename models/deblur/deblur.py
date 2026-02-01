import numpy as np
import torch
from PIL import Image

from models.deblur.model_loader import get_model

# ---------- конфиг ----------
DEVICE = torch.device("cpu")
# ---------------------------


# ---------- инициализация модели (один раз) ----------
_model = None


def _get_model():
    global _model
    if _model is None:
        _model = get_model(device=DEVICE)
    return _model


# ----------------------------------------------------


def deblur_image(pil_image: Image.Image) -> Image.Image:
    """
    Деблюр изображения.
    Вход / выход: PIL.Image (RGB)
    """

    model = _get_model()

    # --- preprocess ---
    img = np.array(pil_image.convert("RGB")).astype(np.float32)
    h, w, _ = img.shape

    # паддинг до кратности 32
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32

    img_padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

    x = img_padded / 127.5 - 1.0
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
    x = x.to(DEVICE)

    # --- inference ---
    with torch.no_grad():
        y = model(x)

    # --- postprocess ---
    y = y.squeeze(0).permute(1, 2, 0).cpu().numpy()
    y = (y + 1.0) * 127.5
    y = np.clip(y, 0, 255).astype(np.uint8)

    # убираем паддинг
    y = y[:h, :w]

    return Image.fromarray(y)
