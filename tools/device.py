import logging
import torch


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
