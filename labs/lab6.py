import logging

from PIL import Image
from tools.device import get_device
from torchvision.models import vgg16
from torchvision.models import VGG16_Weights


from tools.helper_system import get_lab_data_file_path, show_image


def lab6():
    """Lab 6: Using existing models."""
    try:
        logging.info("LAB6. PREPARATION")

        logging.info("checking and loading model")
        device = get_device()
        weights = VGG16_Weights.DEFAULT
        model = vgg16(weights=weights)
        model.to(device)

        # getting model transforms
        pre_trans = weights.transforms()

        # loading image
        show_image(
            image_path=get_lab_data_file_path(lab_name="lab6", item_name="happy_dog.jpg"),
            title="Happy Dog"
        )

        pass
    except (RuntimeError, ValueError, TypeError) as e:
        logging.error("Error in lab6: %s", e)
        raise


