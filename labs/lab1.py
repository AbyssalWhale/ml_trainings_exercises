import logging
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from tools.data import download_and_get_mnist_data_sets
from tools.device import get_device
from tools.training import compile_model, train_and_validate_model


def lab1() -> None:
    """
    Main entry point for Lab 1. Loads data, prepares model,
    trains, validates, and predicts on MNIST.
    """
    logging.info("Starting Lab 1")
    try:
        device = get_device()
        train_set, valid_set = download_and_get_mnist_data_sets()

        logging.info("DATA PREPARATION")
        image_0, image_0_label = train_set[0]
        trans, image_0_tensor = __convert_image_to_tensor__(image_0)

        logging.info("Assign transforms and split data to batches with size 32")
        train_set.transform = trans
        valid_set.transform = trans
        batch_size = 32
        loaders = {
            'train': DataLoader(train_set, batch_size=batch_size, shuffle=True),
            'valid': DataLoader(valid_set, batch_size=batch_size)
        }

        logging.info("PREPARE TRAIN AND VALIDATE MODEL")
        layers = __get_model_layers__()
        model = compile_model(device, layers)
        loss_function = nn.CrossEntropyLoss()
        train_and_validate_model(
            epochs=5,
            model=model,
            train_loader=loaders['train'],
            valid_loader=loaders['valid'],
            device=device,
            loss_function=loss_function,
            is_lab1=True
        )

        logging.info("PREDICTION")
        prediction = model(image_0_tensor.unsqueeze(0).to(device))
        logging.info("Prediction result as tensor: %s", prediction)
        logging.info(
            "Expected class: %s Predicted class: %s",
            image_0_label,
            prediction.argmax().item()
        )
    except (RuntimeError, ValueError, TypeError) as e:
        logging.error("Error in lab1: %s", e)
        raise


def __convert_image_to_tensor__(image) -> tuple:
    """
    Converts a PIL image to a PyTorch tensor using a transformation pipeline.
    Returns the transform and the tensor.
    """
    logging.info("Creating transformation pipeline and converting image to tensor")
    trans = transforms.Compose([transforms.ToTensor()])
    tensor = trans(image)
    logging.info(
        "Image tensor\nPIL Images have a potential integer range of [0, 255], "
        "but the ToTensor class converts it to a float range of [0.0, 1.0].\n"
        "Min: %s\nMax: %s\nSize: %s\nDevice: %s",
        tensor.min(), tensor.max(), tensor.size(), tensor.device
    )
    return trans, tensor

def __get_model_layers__() -> list:
    """
    Prepares and returns the layers for the neural network model.
    """
    logging.info("Preparing layers for the model")
    input_size = 1 * 28 * 28
    n_classes = 10
    layers = [
        nn.Flatten(),
        nn.Linear(input_size, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, n_classes)
    ]
    logging.info(
        "Input size: %d (1*28*28), Classes: %d, Model layers: %s",
        input_size,
        n_classes, layers
    )
    return layers
