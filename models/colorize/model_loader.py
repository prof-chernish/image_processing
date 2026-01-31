from pathlib import Path
import torch
import torch.nn as nn

from models.colorize.eccv16 import ECCVGenerator

ROOT = Path(__file__).resolve().parents[2]   # корень проекта
WEIGHTS_PATH = ROOT / "weights" / "colorization_release_v2-9b330a0b.pth"
DEVICE = torch.device("cpu")

def load_state_dict_flexible(weights_path: str) -> dict:
    obj = torch.load(weights_path, map_location="cpu")
    if isinstance(obj, dict):
        # common patterns: {"state_dict": ...} or direct state_dict
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if all(isinstance(k, str) for k in obj.keys()):
            return obj
    raise ValueError(f"Unsupported weights format in: {weights_path}")

_model = None

def get_model(device=DEVICE):
    global _model
    if _model is None:
        model = ECCVGenerator().eval()
        sd = load_state_dict_flexible(WEIGHTS_PATH)
        model.load_state_dict(sd, strict=True)
        model.to(device)
        _model = model
    return _model
