import logging

from models.data.data_set import DataSetL3
from tools.data import get_asl_data_set
from tools.device import get_device

IMG_HEIGHT = 28
IMG_WIDTH = 28
IMG_CHS = 1
BATCH_SIZE = 32

def lab3():
    """
    CNN introduction - using data from lab2
    """
    try:
        logging.info("PREPARATION")
        device = get_device()
        logging.info("Device selected for computation: %s", device)

        logging.info("loading ASL training and validation datasets. Ots already flattened")
        train_df, valid_df = get_asl_data_set()

        logging.info("getting the first 5 rows of the training dataset...")
        sample_df = train_df.head().copy()
        sample_df.pop('label')
        sample_x = sample_df.values
        logging.info(
            "example below is a sample of the training dataset. "
            "It does not have which pixels are near each other. Convolutions - can not be applied on it."
            "Convolutions is mathematical operation used primarily in CNN with primary goal output a new matrix, "
            "feature map")
        logging.info(sample_x)
        logging.info(
            "we need to convert the current shape (5, 784) to (5, 1, 28, 28)"
            "With NumPy arrays, we can pass a -1 for any dimension we wish to remain the same.")
        sample_x = sample_x.reshape(-1, IMG_CHS, IMG_HEIGHT, IMG_WIDTH)
        logging.info("convertion result: %s", sample_x.shape)
        logging.info("creating our data loaders")
        train_data = DataSetL3(
            base_df=train_df,
            device=device
        )
    except (RuntimeError, ValueError, TypeError) as e:
        logging.error("Error in lab2: %s", e)
        raise