import logging

from tools.device import get_device, get_model


def lab5():
    """Lab 5: Make predictions by using model trained in lab4 but new images that model has not seen yet.
    It's called: inference. Make sure lab4 is executed before lab5.
    """
    try:
        logging.info("LAB4. PREPARATION")
        device = get_device()
        model = get_model(name="lab4_model.pth", device=device)
        pass
    except (RuntimeError, ValueError, TypeError) as e:
        logging.error("Error in lab5: %s", e)
        raise