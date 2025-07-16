import logging

from sympy.printing.pytorch import torch


def get_device():
    logging.info(f"getting device. Is CUDA available? - {torch.cuda.is_available()}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
