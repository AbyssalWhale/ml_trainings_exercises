import logging
import os
import torchvision
from tools.helper_system import get_project_dir


def get_mnist_data_sets() -> tuple:
    """
    Loads and returns the MNIST training and validation datasets.

    Returns:
        tuple: (train_set, valid_set)
    """
    data_path = os.path.join(get_project_dir(), "data")
    logging.info(f"Getting MNIST datasets. Data path: {data_path}")
    try:
        train_set = torchvision.datasets.MNIST(data_path, train=True, download=True)
        valid_set = torchvision.datasets.MNIST(data_path, train=False, download=True)
        logging.info(f"Loaded train set size: {len(train_set)}, valid set size: {len(valid_set)}")
        return train_set, valid_set
    except Exception as e:
        logging.error(f"Error loading MNIST datasets: {e}")
        raise
