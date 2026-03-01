import torch
import random
import numpy as np
import os
from mylib.engine.Config import Config
import models


def generate_model(cfg: Config, ema: bool = False):
    model = getattr(models, cfg.get('model.name'))(num_classes=
            cfg.get('model.num_channels_output'))
    if ema:
        for param in model.parameters():
            param.detach_()
    return model

def get_best_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device


def get_proper_device(device_str=None):
    """Return torch.device from config string (e.g. 'cuda', 'cuda:0', 'cpu')."""
    if device_str is None or device_str == '':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_str)
    print(f"Using device: {device}")
    return device

def set_seed(inc, base_seed=2023):

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

    seed = base_seed + inc
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    torch.cuda.manual_seed_all(seed + 3)
    os.environ['PYTHONHASHSEED'] = str(seed + 4)

    