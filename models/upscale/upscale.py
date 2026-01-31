from PIL import Image
import numpy as np
import torch
from pathlib import Path

from realesrgan import RealESRGANer
from basicsr.archs.srvgg_arch import SRVGGNetCompact


# ---------- Конфиг ----------
ROOT = Path(__file__).resolve().parents[2]   # корень проекта
WEIGHTS_PATH = ROOT / "weights" / "realesr-general-x4v3.pth"
DEVICE = torch.device("cpu")

# безопасные значения для CPU
TILE = 256
TILE_PAD = 10
MAX_SIDE = 1200  # честный лимит для CPU
DEFAULT_SCALE = 2
# ----------------------------


# ---------- Инициализация модели (ОДИН РАЗ) ----------
_model = SRVGGNetCompact(
    num_in_ch=3,
    num_out_ch=3,
    num_feat=64,
    num_conv=32,
    upscale=4,
    act_type="prelu",
)

_upsampler = RealESRGANer(
    scale=4,
    model_path=str(WEIGHTS_PATH),
    model=_model,
    tile=TILE,
    tile_pad=TILE_PAD,
    pre_pad=0,
    half=False,
    device=DEVICE,
)
# ----------------------------------------------------


def _check_size(pil_image: Image.Image):
    w, h = pil_image.size
    if max(w, h) > MAX_SIDE:
        raise ValueError(
            f"Изображение слишком большое для CPU-апскейла. "
            f"Максимальная сторона: {MAX_SIDE}px, сейчас: {max(w, h)}px."
        )


def upscale_image(pil_image: Image.Image, scale: int = DEFAULT_SCALE) -> Image.Image:
    """
    Апскейл изображения (×2 по умолчанию).
    Вход/выход: PIL.Image
    """
    if scale not in (2, 4):
        raise ValueError("scale должен быть 2 или 4")

    _check_size(pil_image)

    img = np.array(pil_image.convert("RGB"))

    with torch.no_grad():
        out, _ = _upsampler.enhance(img, outscale=scale)

    return Image.fromarray(out)
