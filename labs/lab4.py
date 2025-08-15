import logging

import torch
from torch.utils.data import DataLoader
from torch import nn
import torchvision.transforms.functional as F
import torchvision.transforms.v2 as transforms
import matplotlib.pyplot as plt
from models.data.data_set import DataSetL3
from models.nn.custom_cnn_layers_l4 import CustomCNNLayersL4
from tools.data import get_asl_data_set
from tools.device import get_device
from tools.training import train_and_validate_model

IMG_HEIGHT = 28
IMG_WIDTH = 28
IMG_CHS = 1
N_CLASSES = 24
BATCH_SIZE = 32


def lab4():
    """Lab 4: Model is still lagging with validation accuracy.
    We are increasing the size and variance of the data set
    to make model more robust by applying technique name - data argumentation.
    """
    try:
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
            device=device,
            img_chs=IMG_CHS,
            img_height=IMG_HEIGHT,
            img_width=IMG_WIDTH
        )
        valid_loader = DataLoader(
            dataset=valid_data,
            batch_size=BATCH_SIZE
        )

        logging.info("design model")
        base_model = nn.Sequential(*_get_layers())
        loss_function = nn.CrossEntropyLoss()
        # model = torch.compile(base_model.to(device))
        logging.info("Model compiled with torch.compile. Model layers: %s", base_model.to(device))

        logging.info("performing data argumentation")
        row_0 = train_df.head(1)
        row_0.pop('label')
        x_0 = row_0.values / 255
        x_0 = x_0.reshape(IMG_CHS, IMG_WIDTH, IMG_HEIGHT)
        x_0 = torch.tensor(x_0)
        random_transforms = _data_argumentation_exp(x_0)

        train_and_validate_model(
            epochs=20,
            model=base_model.to(device),
            train_loader=train_loader,
            valid_loader=valid_loader,
            device=device,
            loss_function=loss_function,
            random_transforms=random_transforms
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


def _data_argumentation_exp(x_0: torch.Tensor):
    """This function is used to experiment with data argumentation techniques.
    It is not used in the lab4 function, but it can be used to test different data argumentation techniques.
    """

    _, axes = plt.subplots(1, 5, figsize=(20, 4))
    # Original image
    axes[0].imshow(F.to_pil_image(x_0), cmap='gray')
    axes[0].set_title("Original")
    axes[0].axis('off')

    # Random resize crop
    trans_resize = transforms.Compose([
        transforms.RandomResizedCrop((IMG_WIDTH, IMG_HEIGHT), scale=(.7, 1), ratio=(1, 1)),
    ])
    new_x_0_resize = trans_resize(x_0)
    axes[1].imshow(F.to_pil_image(new_x_0_resize), cmap='gray')
    axes[1].set_title("random resize crop")
    axes[1].axis('off')

    # Random horizontal flip
    trans_flip = transforms.Compose([
        transforms.RandomHorizontalFlip()
    ])
    new_x_0_flip = trans_flip(x_0)
    axes[2].imshow(F.to_pil_image(new_x_0_flip), cmap='gray')
    axes[2].set_title("Horizontal Flip")
    axes[2].axis('off')
    # Random rotation
    trans_rot = transforms.Compose([
        transforms.RandomRotation(10)
    ])
    new_x_0_rot = trans_rot(x_0)
    axes[3].imshow(F.to_pil_image(new_x_0_rot), cmap='gray')
    axes[3].set_title("Rotation")
    axes[3].axis('off')
    # Color manipulation + all techniques
    random_transforms = transforms.Compose([
        transforms.RandomRotation(5),
        transforms.RandomResizedCrop((IMG_WIDTH, IMG_HEIGHT), scale=(.9, 1), ratio=(1, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=.2, contrast=.5)
    ])
    new_x_0_all = random_transforms(x_0)
    axes[4].imshow(F.to_pil_image(new_x_0_all), cmap='gray')
    axes[4].set_title("All Techniques")
    axes[4].axis('off')
    plt.suptitle("Image Augmentation Examples", fontsize=16)
    plt.tight_layout()
    plt.show()
    return random_transforms
