import logging

import torch

from models.data.data_set import DataSet
from tools.data import get_asl_data_set
from tools.device import get_device
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import nn


def lab2():
    logging.info("Starting Lab 2")
    try:
        device = get_device()
        train_df, valid_df = get_asl_data_set()

        y_train = train_df.pop('label')
        y_valid = valid_df.pop('label')
        x_train = train_df.values
        x_valid = valid_df.values
        logging.info("extracted labels and images from the dataframes. ")
        logging.info(f"train images labels: {y_train.shape} and total images with pixel pear each: {x_train.shape}")
        logging.info(f"validation images labels: {y_valid.shape} and total images with pixel pear each: {x_valid.shape}")

        logging.info("reshaping images from 1D to 2D (28x28 pixels) first 20 images"
                     " and putting them in single plot (40 x 40 inches)")
        plt.figure(figsize=(40, 40))
        num_images = 20
        for i in range(num_images):
            row = x_train[i]
            label = y_train[i]

            image = row.reshape(28, 28)
            # putting our image as a subplot: taking 1 row, 20 columns, and the i-th image
            plt.subplot(1, num_images, i + 1)
            plt.title(label, fontdict={'fontsize': 30})
            plt.axis('off')
            plt.imshow(image, cmap='gray')

        logging.info("normalizing the data by dividing pixel values (train and validation data sets) by 255")
        x_train = train_df.values / 255
        x_valid = valid_df.values / 255

        logging.info(f"converting data in PyTorch tensors and creating DataLoader with batch sizes 32 for train and validation data sets")
        train_data, train_loader = __get_train_data_and_loader(
            x_df=x_train,
            y_df=y_train,
            device=device,
            batch_size=32,
            shuffle=True
        )
        valid_data, valid_loader = __get_train_data_and_loader(
            x_df=x_valid,
            y_df=y_valid,
            device=device,
            batch_size=32,
            shuffle=False
        )
        logging.info(f"image as tensors: {next(iter(train_loader))}")

        logging.info("CREATE LAYERS AND COMPILE MODEL")
        layers = __get_model_layers__()
        model = __compile_model__(device, layers)
        loss_function = nn.CrossEntropyLoss()

        logging.info("TRAIN AND VALIDATE MODEL")
        epochs = 20
        for epoch in range(epochs):
            logging.info("Epoch %d/%d - Training", epoch + 1, epochs)
            __train_model__(model, train_loader, device, loss_function)
            __validate_model__(model, valid_loader, device, loss_function)



    except (RuntimeError, ValueError, TypeError) as e:
        logging.error("Error in lab2: %s", e)
        raise

def __get_model_layers__() -> list:
    """
    Prepares and returns the layers for the neural network model.
    """
    logging.info("Preparing layers for the model")
    input_size = 1 * 28 * 28
    n_classes = 24
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

def __compile_model__(device, layers: list) -> nn.Module:
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
            output = model(x)
            loss += loss_function(output, y).item()
            accuracy += get_batch_accuracy(output, y, valid_N)
    logging.info('Valid - Loss: %.4f Accuracy: %.4f', loss, accuracy)

def get_batch_accuracy(output, y, N):
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(y.view_as(pred)).sum().item()
    return correct / N

def __get_train_data_and_loader(x_df, y_df, device, batch_size, shuffle=True):
    train_data = DataSet(x_df, y_df, device)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
    return train_data, train_loader