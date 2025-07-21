import logging
import os
from zoneinfo import reset_tzpath

import pandas as pd

import torchvision
from tools.helper_system import get_project_dir


def download_and_get_mnist_data_sets() -> tuple:
    """
    Loads and returns the MNIST training and validation datasets.

    Returns:
        tuple: (train_set, valid_set)
    """
    data_path = os.path.join(get_project_dir(), "data")
    logging.info("Getting MNIST datasets. Data path: %s", data_path)
    try:
        train_set = torchvision.datasets.MNIST(data_path, train=True, download=True)
        valid_set = torchvision.datasets.MNIST(data_path, train=False, download=True)
        logging.info(
            "Loaded train set size: %d, valid set size: %d",
            len(train_set),
            len(valid_set))
        return train_set, valid_set
    except Exception as e:
        logging.error("Error loading MNIST datasets: %s", e)
        raise

def get_asl_data_set() -> tuple:
    try:
        data_path = os.path.join(get_project_dir(), "data", "MNIST", "asl")
        train_df = pd.read_csv(os.path.join(data_path, "sign_mnist_train.csv"))
        valid_df = pd.read_csv(os.path.join(data_path, "sign_mnist_valid.csv"))
        return train_df, valid_df
    except FileNotFoundError as e:
        logging.error("ASL dataset files not found: %s", e)
        raise
