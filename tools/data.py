import logging
import os
import torchvision
from tools.helper_system import get_project_dir


def get_mnist_data_sets():
    data_path = os.path.join(os.path.join(get_project_dir(), "data"))
    logging.info(f"getting training and validation MNIST data sets and storing them: {data_path}")
    train_set = torchvision.datasets.MNIST(data_path, train=True, download=True)
    valid_set = torchvision.datasets.MNIST(data_path, train=False, download=True)
    return train_set, valid_set