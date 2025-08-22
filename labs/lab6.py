import logging

from tools.device import get_device
from torchvision.models import vgg16
from torchvision.models import VGG16_Weights



def lab6():
    """Lab 6: Using existing models."""
    try:
        logging.info("LAB6. PREPARATION")
        logging.info("checking and loading model")
        device = get_device()
        weights = VGG16_Weights.DEFAULT
        model = vgg16(weights=weights)
        model.to(device)
        pass
    except (RuntimeError, ValueError, TypeError) as e:
        logging.error("Error in lab6: %s", e)
        raise