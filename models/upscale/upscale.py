from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]  # корень проекта
WEIGHTS_PATH = ROOT / "weights" / "realesr-general-x4v3.pth"
DEVICE = torch.device("cpu")

TILE = 256
TILE_PAD = 10
DEFAULT_SCALE = 2


_upsampler = None


def get_upsampler():
    
    global _upsampler

    if _upsampler is not None:
        return _upsampler

    from basicsr.archs.srvgg_arch import SRVGGNetCompact
    from realesrgan import RealESRGANer

    model = SRVGGNetCompact(
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
        model=model,
        tile=TILE,
        tile_pad=TILE_PAD,
        pre_pad=0,
        half=False,
        device=DEVICE,
    )

    return _upsampler



def upscale_image(pil_image: Image.Image, scale: int = DEFAULT_SCALE) -> Image.Image:
    if scale not in (2, 4):
        raise ValueError("scale должен быть 2 или 4")

    img = np.array(pil_image.convert("RGB"))

    upsampler = get_upsampler()

    with torch.no_grad():
        out, _ = upsampler.enhance(img, outscale=scale)

    return Image.fromarray(out)

