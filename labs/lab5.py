import logging
import os.path

from tools.device import get_device, get_model
from tools.helper_system import get_lab_data_path
from PIL import Image
import matplotlib.pyplot as plt


def lab5():
    """Lab 5: Make predictions by using model trained in lab4 but new images that model has not seen yet.
    It's called: inference. Make sure lab4 is executed before lab5.
    """
    try:
        logging.info("LAB4. PREPARATION")
        device = get_device()
        model = get_model(name="lab4_model.pth", device=device)

        logging.info("preparing images for inference")
        path_image_a, path_image_b = get_and_show_lab_images()

        pass
    except (RuntimeError, ValueError, TypeError) as e:
        logging.error("Error in lab5: %s", e)
        raise

def get_and_show_lab_images() -> tuple:
    path_image_a = get_lab_data_path(lab_name="lab5", item_name="a.png")
    path_image_b = get_lab_data_path(lab_name="lab5", item_name="b.png")

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(Image.open(path_image_a), cmap='gray')
    axes[0].set_title("Image means letter: A")
    axes[0].axis('off')

    axes[1].imshow(Image.open(path_image_b), cmap='gray')
    axes[1].set_title("Image means letter: B")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()
    return path_image_a, path_image_b