import logging

from torch.utils.data import DataLoader
from models.data.data_set import DataSetL3
from tools.data import get_asl_data_set
from tools.device import get_device

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
        logging.info("PREPARATION")
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

        valid_data = DataSetL3(valid_df)
        valid_loader = DataLoader(
            dataset=valid_data,
            batch_size=BATCH_SIZE
        )
    except (RuntimeError, ValueError, TypeError) as e:
        logging.error("Error in lab2: %s", e)
        raise