import logging

from torch.utils.data import DataLoader
from torch import nn
from models.data.data_set import DataSetL3
from tools.data import get_asl_data_set
from tools.device import get_device
from tools.training import train_and_validate_model

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
        train_data = DataSetL3(base_df=train_df, device=device)
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

        valid_data = DataSetL3(base_df=valid_df, device=device)
        valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE)

        layers = _get_layers()
        model = nn.Sequential(*layers)
        # Uncomment if you want to use torch.compile for optimization. Failing on mac m
        # model = torch.compile(model)
        loss_function = nn.CrossEntropyLoss()

        train_and_validate_model(
            epochs=20,
            model=model.to(device),
            train_loader=train_loader,
            valid_loader=valid_loader,
            device=device,
            loss_function=loss_function,
            is_lab1=False
        )

        logging.info("Example batch from train_loader: %s", next(iter(train_loader)))
    except (RuntimeError, ValueError, TypeError) as e:
        logging.error("Error in lab2: %s", e)
        raise


def _get_layers() -> list[nn.Module]:
    n_classes = 24
    kernel_size = 3
    flattened_img_size = 75 * 3 * 3
    layers = [
        # First convolution
        nn.Conv2d(IMG_CHS, 25, kernel_size, stride=1, padding=1),  # 25 x 28 x 28
        nn.BatchNorm2d(25),
        nn.ReLU(),
        nn.MaxPool2d(2, stride=2),  # 25 x 14 x 14
        # Second convolution
        nn.Conv2d(25, 50, kernel_size, stride=1, padding=1),  # 50 x 14 x 14
        nn.BatchNorm2d(50),
        nn.ReLU(),
        nn.Dropout(.2),
        nn.MaxPool2d(2, stride=2),  # 50 x 7 x 7
        # Third convolution
        nn.Conv2d(50, 75, kernel_size, stride=1, padding=1),  # 75 x 7 x 7
        nn.BatchNorm2d(75),
        nn.ReLU(),
        nn.MaxPool2d(2, stride=2),  # 75 x 3 x 3
        # Flatten to Dense
        nn.Flatten(),
        nn.Linear(flattened_img_size, 512),
        nn.Dropout(.3),
        nn.ReLU(),
        nn.Linear(512, n_classes)
    ]
    return layers
