import logging

import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Adam
from models.data.data_set import DataSetL3
from models.nn.custom_cnn_layers_l4 import CustomCNNLayersL4
from tools.data import get_asl_data_set
from tools.device import get_device
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms.v2 as transforms

from tools.training import get_batch_accuracy

IMG_HEIGHT = 28
IMG_WIDTH = 28
IMG_CHS = 1
N_CLASSES = 24
BATCH_SIZE = 32

def lab4():
    try:
        """Lab 4: Model is still lagging with validation accuracy. 
        We are increasing the size and variance of the data set to make model more robust by applying technique name - data augmentation.
        """
        logging.info("LAB4. PREPARATION")
        device = get_device()
        train_df, valid_df = get_asl_data_set()

        logging.info("creating our data loaders")
        train_data = DataSetL3(
            base_df=train_df,
            device=device,
            img_chs=IMG_CHS,
            img_height=IMG_HEIGHT,
            img_width=IMG_WIDTH
        )
        train_loader = DataLoader(
            dataset=train_data,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        valid_data = DataSetL3(
            base_df=valid_df,
            device = device,
            img_chs = IMG_CHS,
            img_height = IMG_HEIGHT,
            img_width = IMG_WIDTH
        )
        valid_loader = DataLoader(
            dataset=valid_data,
            batch_size=BATCH_SIZE
        )

        logging.info("design model")
        base_model = nn.Sequential(*_get_layers())
        loss_function = nn.CrossEntropyLoss()
        optimizer = Adam(base_model.parameters())
        # model = torch.compile(base_model.to(device))
        logging.info("Model compiled with torch.compile. Model layers: %s", base_model.to(device))

        logging.info("performing data argumentation")
        row_0 = train_df.head(1)
        y_0 = row_0.pop('label')
        x_0 = row_0.values / 255
        x_0 = x_0.reshape(IMG_CHS, IMG_WIDTH, IMG_HEIGHT)
        x_0 = torch.tensor(x_0)
        logging.info("reshaped image: %s", x_0.shape)
        image = F.to_pil_image(x_0)
        plt.figure()
        plt.imshow(image, cmap='gray')
        plt.title("reshaped image")
        # plt.show()
        print("")

        logging.info("performing example of random resize crop")
        trans = transforms.Compose([
            transforms.RandomResizedCrop((IMG_WIDTH, IMG_HEIGHT), scale=(.7, 1), ratio=(1, 1)),
        ])
        new_x_0 = trans(x_0)
        image = F.to_pil_image(new_x_0)
        plt.figure()
        plt.imshow(image, cmap='gray')
        plt.title("resized image")
        # plt.show()

        logging.info("performing example of random horizontal flip")
        trans = transforms.Compose([
            transforms.RandomHorizontalFlip()
        ])
        new_x_0 = trans(x_0)
        image = F.to_pil_image(new_x_0)
        plt.imshow(image, cmap='gray')
        plt.title("random horizontal flip image")
        # plt.show()

        logging.info("performing example of random rotation")
        trans = transforms.Compose([
            transforms.RandomRotation(10)
        ])
        new_x_0 = trans(x_0)
        image = F.to_pil_image(new_x_0)
        plt.imshow(image, cmap='gray')
        plt.title("random rotation")
        # plt.show()

        logging.info("performing example of colors manipulation")
        brightness = .2  # Change to be from 0 to 1
        contrast = .5  # Change to be from 0 to 1

        trans = transforms.Compose([
            transforms.ColorJitter(brightness=brightness, contrast=contrast)
        ])
        new_x_0 = trans(x_0)
        image = F.to_pil_image(new_x_0)
        plt.imshow(image, cmap='gray')
        plt.title("colors manipulation")
        # plt.show()

        logging.info("apply all techniques to the data set")
        random_transforms = transforms.Compose([
            transforms.RandomRotation(5),
            transforms.RandomResizedCrop((IMG_WIDTH, IMG_HEIGHT), scale=(.9, 1), ratio=(1, 1)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=.2, contrast=.5)
        ])
        new_x_0 = random_transforms(x_0)
        image = F.to_pil_image(new_x_0)
        plt.imshow(image, cmap='gray')
        plt.title("all applied techniques")
        # plt.show()

        logging.info("train and validate model")
        epochs = 20

        for epoch in range(epochs):
            print('Epoch: {}'.format(epoch))
            _train(
                device=device,
                model=base_model.to(device),
                train_loader=train_loader,
                optimizer=optimizer,
                loss_function=loss_function,
                train_N=len(train_data),
                random_transforms=random_transforms
            )
            _validate(
                model=base_model.to(device),
                valid_loader=valid_loader,
                loss_function=loss_function,
                valid_N=len(valid_data)
            )

    except (RuntimeError, ValueError, TypeError) as e:
        logging.error("Error in lab2: %s", e)
        raise

def _get_layers() -> list[nn.Module]:
    flattened_img_size = 75 * 3 * 3
    return [
        CustomCNNLayersL4(IMG_CHS, 25, 0),  # 25 x 14 x 14
        CustomCNNLayersL4(25, 50, 0.2),  # 50 x 7 x 7
        CustomCNNLayersL4(50, 75, 0),  # 75 x 3 x 3
        # Flatten to Dense Layers
        nn.Flatten(),
        nn.Linear(flattened_img_size, 512),
        nn.Dropout(.3),
        nn.ReLU(),
        nn.Linear(512, N_CLASSES)
    ]

def _train(
        device,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: Adam,
        loss_function: nn.Module,
        train_N: int,
        random_transforms: transforms.Compose
):
    loss = 0
    accuracy = 0

    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        output = model(random_transforms(x))  # Updated
        optimizer.zero_grad()
        batch_loss = loss_function(output, y)
        batch_loss.backward()
        optimizer.step()

        loss += batch_loss.item()
        accuracy += get_batch_accuracy(output, y, train_N)
    logging.info('Train - Loss: {:.4f} Accuracy: {:.4f}'.format(loss, accuracy))

def _validate(
        model: nn.Module,
        valid_loader: DataLoader,
        loss_function: nn.Module,
        valid_N: int
):
    loss = 0
    accuracy = 0

    model.eval()
    with torch.no_grad():
        for x, y in valid_loader:
            output = model(x)

            loss += loss_function(output, y).item()
            accuracy += get_batch_accuracy(output, y, valid_N)
    logging.info('Valid - Loss: {:.4f} Accuracy: {:.4f}'.format(loss, accuracy))