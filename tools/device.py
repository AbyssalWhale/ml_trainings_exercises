import logging
import os

import torch
from torch._C.cpp import nn

from tools.helper_system import get_model_saving_dir


def get_device() -> torch.device:
    """
    Returns the best available device (CUDA if available, else CPU).
    Returns:
        torch.device: The selected device.
    """
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    logging.info("Getting device. CUDA available: %s. Selected device: %s", cuda_available, device)
    return device

def get_model(name: str, device) -> nn.Module:
    path = os.path.join(get_model_saving_dir(), name)
    logging.info("trying to load model from path: %s", path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found in path: {path}")
    return torch.load(path, map_location=device, weights_only=False)