import logging

from models.data.data_set import DataSet
from tools.data import get_asl_data_set
from tools.device import get_device
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


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

        BATCH_SIZE = 32
        logging.info("converting data in PyTorch tensors and creating DataLoader for train and validation data sets")
        train_data = DataSet(x_train, y_train, device)
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        train_N = len(train_loader.dataset)

        valid_data = DataSet(x_valid, y_valid, device)
        valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE)
        valid_N = len(valid_loader.dataset)
        batch = next(iter(train_loader))


    except (RuntimeError, ValueError, TypeError) as e:
        logging.error("Error in lab2: %s", e)
        raise