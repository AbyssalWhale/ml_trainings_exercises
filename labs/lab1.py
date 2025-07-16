import logging
import torch
import torch.nn as nn
from tools.data import get_mnist_data_sets
from tools.device import get_device
from torchvision import transforms
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader


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
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_set, batch_size=batch_size)

        logging.info("PREPARE TRAIN AND VALIDATE MODEL")
        layers = __get_model_layers__()
        model = __compile_model__(device, layers)
        loss_function = nn.CrossEntropyLoss()
        __train_and_validate_model__(
            epochs=5,
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            device=device,
            loss_function=loss_function
        )

        logging.info("PREDICTION")
        prediction = model(image_0_tensor.unsqueeze(0).to(device))
        logging.info(f"Prediction result as tensor: {prediction}")
        logging.info(f"Expected class: {image_0_label} Predicted class: {prediction.argmax().item()}")
    except Exception as e:
        logging.error(f"Error in lab1: {e}")
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
        f"Image tensor\nPIL Images have a potential integer range of [0, 255], "
        f"but the ToTensor class converts it to a float range of [0.0, 1.0].\n"
        f"Min: {tensor.min()}\nMax: {tensor.max()}\nSize: {tensor.size()}\nDevice: {tensor.device}"
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
    logging.info(f"Input size: {input_size} (1*28*28), Classes: {n_classes}, Model layers: {layers}")
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
        logging.warning(f"torch.compile failed: {e}. Using uncompiled model.")
    logging.info(f"Model device: {next(model.parameters()).device}")
    return model

def __train_and_validate_model__(epochs: int, model: nn.Module, train_loader: DataLoader, valid_loader: DataLoader, device, loss_function) -> None:
    """
    Trains and validates the model for the specified number of epochs.
    """
    for epoch in range(epochs):
        logging.info(f"Epoch {epoch+1}/{epochs} - Training")
        __train_model__(
            model=model,
            train_loader=train_loader,
            device=device,
            loss_function=loss_function
        )
        logging.info(f"Epoch {epoch+1}/{epochs} - Validating")
        __validate_model__(
            model=model,
            valid_loader=valid_loader,
            device=device,
            loss_function=loss_function
        )

def __train_model__(model: nn.Module, train_loader: DataLoader, device, loss_function) -> None:
    """
    Trains the model for one epoch using the training data loader.
    """
    optimizer = Adam(model.parameters())
    train_n = len(train_loader.dataset)
    loss = 0.0
    accuracy = 0.0
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        optimizer.zero_grad()
        batch_loss = loss_function(output, y)
        batch_loss.backward()
        optimizer.step()
        loss += batch_loss.item()
        accuracy += get_batch_accuracy(output, y, train_n)
    logging.info(f'Train - Loss: {loss:.4f} Accuracy: {accuracy:.4f}')

def __validate_model__(model: nn.Module, valid_loader: DataLoader, device, loss_function) -> None:
    """
    Validates the model using the validation data loader.
    """
    loss = 0.0
    accuracy = 0.0
    valid_N = len(valid_loader.dataset)
    model.eval()
    with torch.no_grad():
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss += loss_function(output, y).item()
            accuracy += get_batch_accuracy(output, y, valid_N)
    logging.info(f'Valid - Loss: {loss:.4f} Accuracy: {accuracy:.4f}')

def get_batch_accuracy(output: torch.Tensor, y: torch.Tensor, N: int) -> float:
    """
    Calculates the accuracy for a batch.
    Returns the fraction of correct predictions in the batch.
    """
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(y.view_as(pred)).sum().item()
    return correct / N