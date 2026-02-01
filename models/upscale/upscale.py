from pathlib import Path

import numpy as np
import torch
from basicsr.archs.srvgg_arch import SRVGGNetCompact
from PIL import Image
from realesrgan import RealESRGANer

# ---------- Конфиг ----------
ROOT = Path(__file__).resolve().parents[2]  # корень проекта
WEIGHTS_PATH = ROOT / "weights" / "realesr-general-x4v3.pth"
DEVICE = torch.device("cpu")

# безопасные значения для CPU
TILE = 256
TILE_PAD = 10
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


def upscale_image(pil_image: Image.Image, scale: int = DEFAULT_SCALE) -> Image.Image:
    """
    Апскейл изображения (×2 по умолчанию).
    Вход/выход: PIL.Image
    """
    if scale not in (2, 4):
        raise ValueError("scale должен быть 2 или 4")

    img = np.array(pil_image.convert("RGB"))

    with torch.no_grad():
        out, _ = _upsampler.enhance(img, outscale=scale)

    return Image.fromarray(out)
