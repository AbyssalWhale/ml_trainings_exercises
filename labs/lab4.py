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
        model = torch.compile(base_model.to(device))
        logging.info("Model compiled with torch.compile. Model layers: %s", model)

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

        logging.info("performing random resize crop")
        trans = transforms.Compose([
            transforms.RandomResizedCrop((IMG_WIDTH, IMG_HEIGHT), scale=(.7, 1), ratio=(1, 1)),
        ])
        new_x_0 = trans(x_0)
        image = F.to_pil_image(new_x_0)
        plt.figure()
        plt.imshow(image, cmap='gray')
        plt.title("resized image")
        # plt.show()
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