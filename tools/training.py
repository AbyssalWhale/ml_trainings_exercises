import logging

import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Adam
import torchvision.transforms.v2 as transforms

def train_and_validate_model(
        epochs,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        device,
        loss_function,
        is_lab1: bool = False,
        random_transforms: transforms.Compose = None
):
    """
    Trains and validates the model for the specified number of epochs.
    Args:
        random_transforms:
        is_lab1:
        valid_loader:
        train_loader:
        epochs (int): Number of epochs.
        model (nn.Module): The model to train.
        loaders (dict): Dictionary with 'train' and 'valid' DataLoader.
        device: Device to use.
        loss_function: Loss function.
    """
    logging.info("train and validate model")
    for epoch in range(epochs):
        logging.info("Epoch %d/%d - Training", epoch + 1, epochs)
        train_model(
            model=model,
            loader=train_loader,
            device=device,
            loss_function=loss_function,
            random_transforms=random_transforms
        )
        logging.info("Epoch %d/%d - Validating", epoch + 1, epochs)
        validate_model(
            model=model,
            loader=valid_loader,
            device=device,
            loss_function=loss_function,
            is_lab1=is_lab1
        )


def compile_model(device, layers: list) -> nn.Module:
    """
    Compiles and returns a PyTorch sequential model on the specified device.
    """
    model = nn.Sequential(*layers)
    model.to(device)
    try:
        model = torch.compile(model)
        logging.info("Model compiled with torch.compile. Model layers: %s", layers)
    except (RuntimeError, TypeError) as e:
        logging.warning("torch.compile failed: %s. Using uncompiled model.", e)
    logging.info(
        "Model device: %s",
        next(model.parameters()).device
    )
    return model


def train_model(
        model: nn.Module,
        loader: DataLoader,
        device,
        loss_function,
        random_transforms: transforms.Compose
) -> None:
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
        output = model(x) if random_transforms is None else model(random_transforms(x))
        optimizer.zero_grad()
        batch_loss = loss_function(output, y)
        batch_loss.backward()
        optimizer.step()
        loss += batch_loss.item()
        accuracy += get_batch_accuracy(output, y, train_n)
    logging.info('Train - Loss: %.4f Accuracy: %.4f', loss, accuracy)


def validate_model(
        model: nn.Module,
        loader: DataLoader,
        device,
        loss_function,
        is_lab1: bool = False
) -> None:
    """
    Validates the model using the validation data loader.
    """
    loss = 0.0
    accuracy = 0.0
    valid_N = len(loader.dataset)
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = (x.to(device), y.to(device)) if is_lab1 else (x, y)
            output = model(x)
            loss += loss_function(output, y).item()
            accuracy += get_batch_accuracy(output, y, valid_N)
    logging.info('Valid - Loss: %.4f Accuracy: %.4f', loss, accuracy)


def get_batch_accuracy(output, y, N):
    """
    Calculates the accuracy for a batch.
    Returns the fraction of correct predictions in the batch.
    """
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(y.view_as(pred)).sum().item()
    return correct / N

