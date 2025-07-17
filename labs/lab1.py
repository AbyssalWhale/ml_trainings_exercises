import logging
import torch
from torch import nn
from torchvision import transforms
from torch.optim import Adam
from torch.utils.data import DataLoader

from tools.data import get_mnist_data_sets
from tools.device import get_device


def lab1() -> None:
    """
    Main entry point for Lab 1. Loads data, prepares model, trains, validates, and predicts on MNIST.
    """
    logging.info("Starting Lab 1")
    try:
        device = get_device()
        train_set, valid_set = get_mnist_data_sets()

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
        model = __compile_model__(device, layers)
        loss_function = nn.CrossEntropyLoss()
        __train_and_validate_model__(
            epochs=5,
            model=model,
            loaders=loaders,
            device=device,
            loss_function=loss_function
        )

        logging.info("PREDICTION")
        prediction = model(image_0_tensor.unsqueeze(0).to(device))
        logging.info("Prediction result as tensor: %s", prediction)
        logging.info("Expected class: %s Predicted class: %s", image_0_label, prediction.argmax().item())
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
    logging.info("Input size: %d (1*28*28), Classes: %d, Model layers: %s", input_size, n_classes, layers)
    return layers


def __compile_model__(device, layers: list) -> nn.Module:
    """
    Compiles and returns a PyTorch sequential model on the specified device.
    """
    model = nn.Sequential(*layers)
    model.to(device)
    try:
        model = torch.compile(model)
        logging.info("Model compiled with torch.compile.")
    except Exception as e:
        logging.warning(
            "torch.compile failed: %s. Using uncompiled model.",
            e
        )
    logging.info(
        "Model device: %s",
        next(model.parameters()).device
    )
    return model


def __train_and_validate_model__(epochs: int, model: nn.Module, loaders: dict, device, loss_function) -> None:
    """
    Trains and validates the model for the specified number of epochs.
    Args:
        epochs (int): Number of epochs.
        model (nn.Module): The model to train.
        loaders (dict): Dictionary with 'train' and 'valid' DataLoader.
        device: Device to use.
        loss_function: Loss function.
    """
    train_loader = loaders['train']
    valid_loader = loaders['valid']
    for epoch in range(epochs):
        logging.info("Epoch %d/%d - Training", epoch + 1, epochs)
        __train_model__(model=model, loader=train_loader, device=device, loss_function=loss_function)
        logging.info("Epoch %d/%d - Validating", epoch + 1, epochs)
        __validate_model__(model=model, loader=valid_loader, device=device, loss_function=loss_function)


def __train_model__(model: nn.Module, loader: DataLoader, device, loss_function) -> None:
    """
    Trains the model for one epoch using the training data loader.
    """
    optimizer = Adam(model.parameters())
    train_n = len(loader.dataset)
    loss = 0.0
    accuracy = 0.0
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        optimizer.zero_grad()
        batch_loss = loss_function(output, y)
        batch_loss.backward()
        optimizer.step()
        loss += batch_loss.item()
        accuracy += get_batch_accuracy(output, y, train_n)
    logging.info('Train - Loss: %.4f Accuracy: %.4f', loss, accuracy)


def __validate_model__(model: nn.Module, loader: DataLoader, device, loss_function) -> None:
    """
    Validates the model using the validation data loader.
    """
    loss = 0.0
    accuracy = 0.0
    valid_N = len(loader.dataset)
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss += loss_function(output, y).item()
            accuracy += get_batch_accuracy(output, y, valid_N)
    logging.info('Valid - Loss: %.4f Accuracy: %.4f', loss, accuracy)


def get_batch_accuracy(output: torch.Tensor, y: torch.Tensor, N: int) -> float:
    """
    Calculates the accuracy for a batch.
    Returns the fraction of correct predictions in the batch.
    """
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(y.view_as(pred)).sum().item()
    return correct / N
