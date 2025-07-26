import logging
import os.path

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn
from models.data.data_set import DataSet
from tools.data import get_asl_data_set
from tools.device import get_device
from tools.helper_system import get_labs_data_saving_dir
from tools.training import compile_model, train_and_validate_model

# Constants
IMG_SIZE = 28
NUM_CLASSES = 24
HIDDEN_SIZE = 512
BATCH_SIZE = 32
NUM_IMAGES_TO_PLOT = 20
NORMALIZATION_DIVISOR = 255.0
EPOCHS = 20


def lab2():
    """
    Main function for Lab 2:
    Loads ASL dataset, visualizes, normalizes, prepares loaders, builds and trains model.
    """
    logging.info("Starting Lab 2")
    try:
        # Get the device (GPU if available, else CPU)
        device = get_device()
        logging.info("Device selected for computation: %s", device)

        # Load the ASL training and validation datasets as pandas DataFrames
        train_df, valid_df = get_asl_data_set()
        logging.info("Loaded ASL training and validation datasets.")

        # Separate labels from image data
        y_train = train_df.pop('label')
        y_valid = valid_df.pop('label')
        x_train = train_df.values
        x_valid = valid_df.values
        logging.info("Extracted labels and images from the dataframes.")
        logging.info(
            "Train labels shape: %s, Train images shape: %s",
            y_train.shape, x_train.shape
        )
        logging.info(
            "Validation labels shape: %s, Validation images shape: %s",
            y_valid.shape, x_valid.shape
        )

        # Visualize the first NUM_IMAGES_TO_PLOT images from the training set
        logging.info(
            "Reshaping images from 1D to 2D (%dx%d pixels) for the first %d images and plotting them.",
            IMG_SIZE, IMG_SIZE, NUM_IMAGES_TO_PLOT
        )
        plt.figure(figsize=(40, 40))
        for i in range(NUM_IMAGES_TO_PLOT):
            row = x_train[i]
            label = y_train[i]
            image = row.reshape(IMG_SIZE, IMG_SIZE)
            plt.subplot(1, NUM_IMAGES_TO_PLOT, i + 1)
            plt.title(label, fontdict={'fontsize': 30})
            plt.axis('off')
            plt.imshow(image, cmap='gray')
        # Save the plot to a file for later review
        plt.savefig(os.path.join(get_labs_data_saving_dir("lab2"), "sample_images.png"))
        logging.info("Saved a plot of the first %d training images to sample_images.png.", NUM_IMAGES_TO_PLOT)
        # plt.show()  # Uncomment if running interactively

        # Normalize pixel
        # values to the range [0, 1] for better neural network performance
        logging.info("Normalizing the data by dividing pixel values by %d.", NORMALIZATION_DIVISOR)
        x_train = train_df.values.astype(float) / NORMALIZATION_DIVISOR
        x_valid = valid_df.values.astype(float) / NORMALIZATION_DIVISOR
        logging.info("Normalization complete. Example pixel value: %.4f", x_train[0][0])

        # Convert data to PyTorch tensors and create DataLoaders for batching
        logging.info("Converting data to PyTorch tensors and creating DataLoaders with batch size %d.", BATCH_SIZE)
        _, train_loader = _get_train_data_and_loader(
            x_df=x_train,
            y_df=y_train,
            device=device,
            batch_size=BATCH_SIZE,
            shuffle=True
        )
        _, valid_loader = _get_train_data_and_loader(
            x_df=x_valid,
            y_df=y_valid,
            device=device,
            batch_size=BATCH_SIZE,
            shuffle=False
        )
        logging.info("Example batch from train_loader: %s", next(iter(train_loader)))

        # Build the neural network model
        logging.info("Creating model layers and compiling the model.")
        layers = _get_model_layers()
        model = compile_model(device, layers)
        loss_function = nn.CrossEntropyLoss()
        logging.info("Model and loss function ready.")

        # Train and validate the model for a number of epochs
        train_and_validate_model(
            epochs=EPOCHS,
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            device=device,
            loss_function=loss_function,
            is_lab1=True
        )

    except (RuntimeError, ValueError, TypeError) as e:
        logging.error("Error in lab2: %s", e)
        raise


def _get_model_layers() -> list:
    """
    Prepares and returns the layers for the neural network model.
    """
    logging.info("Preparing layers for the model")
    input_size = 1 * IMG_SIZE * IMG_SIZE
    layers = [
        nn.Flatten(),
        nn.Linear(input_size, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZE, NUM_CLASSES)
    ]
    logging.info(
        "Input size: %d (1*%d*%d), Classes: %d, Model layers: %s",
        input_size,
        IMG_SIZE, IMG_SIZE, NUM_CLASSES, layers
    )
    return layers


def _get_train_data_and_loader(x_df, y_df, device, batch_size, shuffle=True):
    train_data = DataSet(x_df, y_df, device)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
    return train_data, train_loader
