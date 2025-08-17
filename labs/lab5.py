import logging
import os.path

import torch
from tools.device import get_device, get_model
from tools.helper_system import get_lab_data_path
from PIL import Image
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import torchvision.io as tv_io
import torchvision.transforms.v2 as transforms

IMG_WIDTH = 28
IMG_HEIGHT = 28
ALPHABET = "abcdefghiklmnopqrstuvwxy"

def lab5():
    """Lab 5: Make predictions by using model trained in lab4 but new images that model has not seen yet.
    It's called: inference. Make sure lab4 is executed before lab5.
    """
    try:
        logging.info("LAB5. PREPARATION")
        logging.info("checking and loading model")
        device = get_device()
        model = get_model(name="lab4_model.pth", device=device)

        logging.info("preparing images")
        path_image_a, path_image_b = get_and_show_lab_images()
        processed_image_a, processed_image_b = scale_and_show_image(path_image_a, path_image_b)

        logging.info("making predictions")
        # the squeeze removes dimensions of 1, unsqueeze adds a dimension of 1 at the index we specify.
        # The first dimension is usually the batch dimension, so we can say .unsqueeze(0).
        batched_image_a = processed_image_a.unsqueeze(0)
        batched_image_b = processed_image_b.unsqueeze(0)

        # putting on the same device
        batched_image_model_a = batched_image_a.to(device)
        batched_image_model_b = batched_image_b.to(device)

        output_a = model(batched_image_model_a)
        output_b = model(batched_image_model_b)

        logging.info("understand predictions results by using alphabet %s", ALPHABET)
        prediction_a = output_a.argmax(dim=1).item()
        prediction_b = output_b.argmax(dim=1).item()

        logging.info("prediction for image A: %s", ALPHABET[prediction_a])
        logging.info("prediction for image B: %s", ALPHABET[prediction_b])

        pass
    except (RuntimeError, ValueError, TypeError) as e:
        logging.error("Error in lab5: %s", e)
        raise

def get_and_show_lab_images() -> tuple:
    logging.info("preparing images for inference")
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

def scale_and_show_image(path_image_a: str, path_image_b: str) -> tuple:
    logging.info("scaling the images to 28x28 pixels and converting to grayscale")
    image_a = tv_io.read_image(path_image_a, tv_io.ImageReadMode.GRAY)
    image_b = tv_io.read_image(path_image_b, tv_io.ImageReadMode.GRAY)

    preprocess_trans = transforms.Compose([
        transforms.ToDtype(torch.float32, scale=True),  # Converts [0, 255] to [0, 1]
        transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),
        transforms.Grayscale()  # From Color to Gray
    ])

    processed_image_a = preprocess_trans(image_a)
    processed_image_b = preprocess_trans(image_b)

    _, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(F.to_pil_image(processed_image_a), cmap='gray')
    axes[0].set_title("scaled image A")
    axes[0].axis('off')

    axes[1].imshow(F.to_pil_image(processed_image_b), cmap='gray')
    axes[1].set_title("scaled image B")
    axes[1].axis('off')

    plt.suptitle("scaled images", fontsize=16)
    plt.tight_layout()
    plt.show()

    return  processed_image_a, processed_image_b
