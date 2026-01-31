from pathlib import Path
import torch
import torch.nn as nn

from models.deblur.fpn_mobilenet import FPNMobileNet

ROOT = Path(__file__).resolve().parents[2]   # корень проекта
WEIGHTS_PATH = ROOT / "weights" / "fpn_mobilenet.h5"
DEVICE = torch.device("cpu")


def load_weights(model, weights_path):
    ckpt = torch.load(weights_path, map_location="cpu")
    state_dict = ckpt["model"]

    clean = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        clean[k] = v

    model.load_state_dict(clean, strict=False)
    return model


def get_model(device=DEVICE):
    model = FPNMobileNet(
        norm_layer=nn.BatchNorm2d,
        pretrained=False
    )

    load_weights(model, WEIGHTS_PATH)
    model.to(device)

    # используем train(), т.к. модель обучалась и инференсилась в этом режиме
    model.train()

    return model
