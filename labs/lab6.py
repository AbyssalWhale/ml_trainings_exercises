import json
import logging

import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torchvision.io as tv_io
import torchvision.transforms.functional as F
from torchvision.models import vgg16, VGG16_Weights

from tools.device import get_device
from tools.helper_system import get_lab_data_file_path, show_image, get_item_from_data_dir


def lab6():
    """Lab 6: Using existing models."""
    try:
        logging.info("LAB6. PREPARATION")

        logging.info("checking and loading model")
        device = get_device()
        weights = VGG16_Weights.DEFAULT
        model = vgg16(weights=weights)
        model.to(device)

        logging.info("transforming image to model input format")
        pre_trans = weights.transforms()
        happy_dog_path = get_lab_data_file_path(lab_name="lab6", item_name="happy_dog.jpg")

        # loading a list of classes (for all images) to interpret model output
        brown_bear_path = get_lab_data_file_path(lab_name="lab6", item_name="brown_bear.jpg")
        sleepy_cat_path = get_lab_data_file_path(lab_name="lab6", item_name="sleepy_cat.jpg")
        # vgg_classes = json.load(open(get_item_from_data_dir(name="imagenet_class_index.json")))
        with open(get_item_from_data_dir(name="imagenet_class_index.json"), encoding="utf-8") as f:
            vgg_classes = json.load(f)
            readable_prediction(
                device=device,
                pre_trans=pre_trans,
                model=model,
                image_path=happy_dog_path,
                vgg_classes=vgg_classes
            )
            readable_prediction(
                device=device,
                pre_trans=pre_trans,
                model=model,
                image_path=brown_bear_path,
                vgg_classes=vgg_classes
            )
            readable_prediction(
                device=device,
                pre_trans=pre_trans,
                model=model,
                image_path=sleepy_cat_path,
                vgg_classes=vgg_classes
            )

        doggy_door(
            device=device,
            model=model,
            pre_trans=pre_trans,
            image_path=brown_bear_path
        )
        doggy_door(
            device=device,
            model=model,
            pre_trans=pre_trans,
            image_path=happy_dog_path
        )
        doggy_door(
            device=device,
            model=model,
            pre_trans=pre_trans,
            image_path=sleepy_cat_path
        )
    except (RuntimeError, ValueError, TypeError) as e:
        logging.error("Error in lab6: %s", e)
        raise


def load_and_process_image(
        device,
        pre_trans,
        file_path
):
    # Print image's original shape, for reference
    logging.info('original image shape %s', mpimg.imread(file_path).shape)
    show_image(
        image_path=file_path,
        title="original image before processing"
    )

    image = tv_io.read_image(file_path).to(device)
    image = pre_trans(image)  # weights.transforms()
    image = image.unsqueeze(0)  # Turn into a batch
    logging.info('processed image %s', image.shape)

    plot_image = F.to_pil_image(torch.squeeze(image))
    plt.title("processed image")
    plt.imshow(plot_image, cmap='gray')
    plt.axis('off')
    plt.show()

    return image


def readable_prediction(
        device,
        pre_trans,
        model,
        image_path,
        vgg_classes
):
    logging.info("Making predictions for image: %s", image_path)
    # Show image
    show_image(
        image_path=image_path,
        title="image for prediction"
    )
    # Load and pre-process image
    image = load_and_process_image(
        device=device,
        pre_trans=pre_trans,
        file_path=image_path
    )
    # Make predictions
    output = model(image)[0]  # Unbatch
    predictions = torch.topk(output, 3)
    indices = predictions.indices.tolist()
    # Print predictions in readable form
    out_str = "Top results: "
    pred_classes = [vgg_classes[str(idx)][1] for idx in indices]
    out_str += ", ".join(pred_classes)
    logging.info("predictions results: %s", out_str)

    return predictions


def doggy_door(device, model, pre_trans, image_path):
    logging.info("prediction if image is dog or not: %s", image_path)
    show_image(image_path=image_path, title="image to check if dog or not")
    image = load_and_process_image(
        device=device,
        pre_trans=pre_trans,
        file_path=image_path
    )
    idx = model(image).argmax(dim=1).item()
    logging.info("Predicted index: %s", idx)
    if 151 <= idx <= 268:
        logging.info("Doggy come on in!")
    elif 281 <= idx <= 285:
        logging.info("Kitty stay inside!")
    else:
        logging.info("You're not a dog! Stay outside!")
