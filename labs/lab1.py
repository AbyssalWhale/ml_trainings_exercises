import logging
from wsgiref.util import request_uri

import torch
import torch.nn as nn
from tools.data import get_mnist_data_sets
from tools.device import get_device
from torchvision import transforms
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader


def lab1():
    logging.info(f"starting lab 1")
    device = get_device()
    train_set, valid_set = get_mnist_data_sets()

    logging.info("DATA PREPARATION")
    image_0, image_0_label = train_set[0]
    trans, image_0_tensor = __convert_image_to_tensor__(image_0)

    logging.info(f"assign transforms and splitting data to batches with size 32")
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
    prediction = model(image_0_tensor)
    logging.info(f"prediction result as tensor: {prediction}")
    logging.info(f"expected class: {image_0_label} predicted class: {prediction.argmax().item()}")


def __convert_image_to_tensor__(image):
    logging.info(f"creating transformation pipeline and converting image to tensors")
    trans = transforms.Compose([transforms.ToTensor()])
    tensor = trans(image)
    logging.info(f"Image tensor"
          f"\nPIL Images have a potential integer range of [0, 255], but the ToTensor class converts it to a float range of [0.0, 1.0]."
          f"\nMin: {tensor.min()}"
          f"\nMax: {tensor.max()}"
          f"\nSize: {tensor.size()}"
          f"\nDevice: {tensor.device}"
          f"\nEntire tensor: {tensor}")
    return trans, tensor

def __get_model_layers__():
    logging.info(f"preparing layers for the model")
    input_size = 1 * 28 * 28
    n_classes = 10
    layers = [
        nn.Flatten(),
        nn.Linear(input_size, 512),  # Input
        nn.ReLU(),  # Activation for input
        nn.Linear(512, 512),  # Hidden
        nn.ReLU(),  # Activation for hidden
        nn.Linear(512, n_classes)  # Output
    ]
    logging.info(f"input size is {input_size} - 1 * 28 * 28 - where 1 is channel (Grayscale), 28 is height, and 28 is width of the image")
    logging.info(f"{n_classes} - number of categories (or labels) our model is trying to predict")
    logging.info(f"model layers: {layers}")
    return layers

def __compile_model__(device, layers):
    model = nn.Sequential(*layers)
    model.to(device)
    model = torch.compile(model)
    return model

def __train_and_validate_model__(epochs, model, train_loader, valid_loader, device, loss_function):
    for epoch in range(epochs):
        logging.info(f"training cycle epoch: {epoch}")
        logging.info(f"training")
        __train_model__(
            model=model,
            train_loader=train_loader,
            device=device,
            loss_function=loss_function
        )
        logging.info(f"validating")
        __validate_model__(
            model=model,
            valid_loader=valid_loader,
            device=device,
            loss_function=loss_function
        )

def __train_model__(model, train_loader, device, loss_function):
    optimizer = Adam(model.parameters())
    train_n = len(train_loader.dataset)
    loss = 0
    accuracy = 0

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
    logging.info('Train - Loss Results: {:.4f} Accuracy: {:.4f}'.format(loss, accuracy))

def __validate_model__(model, valid_loader, device, loss_function):
    loss = 0
    accuracy = 0
    valid_N = len(valid_loader.dataset)
    model.eval()
    with torch.no_grad():
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss += loss_function(output, y).item()
            accuracy += get_batch_accuracy(output, y, valid_N)
    logging.info('Valid - Loss: {:.4f} Accuracy: {:.4f}'.format(loss, accuracy))

def get_batch_accuracy(output, y, N):
    """a function to calculate the accuracy for each batch.
    The result is a fraction of the total accuracy, so we can add the accuracy of each batch together to get the total."""
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(y.view_as(pred)).sum().item()
    return correct / N