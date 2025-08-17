import logging
import os
from IPython import get_ipython


def get_project_dir() -> str:
    """
    Finds and returns the project root directory by searching for main.py.
    Returns:
        str: Absolute path to the project root directory.
    Raises:
        Exception: If the project root cannot be found.
    """
    logging.info("Trying to find project root directory...")
    current_dir = os.path.abspath(os.path.dirname(__file__))
    while not os.path.isfile(os.path.join(current_dir, 'main.py')):
        parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
        if parent_dir == current_dir:
            logging.error("Could not find project root directory.")
            raise FileNotFoundError("Could not find project root directory.")
        current_dir = parent_dir
    logging.info("Project root directory found: %s", current_dir)
    return current_dir


def get_labs_data_saving_dir(lab_name: str) -> str:
    """
    Constructs the path to the directory where lab data should be saved.

    Args:
        lab_name (str): The name of the lab.

    Returns:
        str: The absolute path to the lab's data saving directory.
    """
    project_dir = get_project_dir()
    labs_data_dir = os.path.join(project_dir, "data", "labs_saving", lab_name)
    lab_data_saving_dir = os.path.join(labs_data_dir, lab_name)

    if not os.path.exists(lab_data_saving_dir):
        os.makedirs(lab_data_saving_dir)
        logging.info("created directory for %s data: %s", lab_name, lab_data_saving_dir)

    return lab_data_saving_dir

def get_lab_data_path(lab_name: str, item_name: str) -> str:
    """
    return path to the directory where data is ready to use for lab or intended to be used for lab.
    Args:
        item_name:
        lab_name:
    Returns:

    """
    project_dir = get_project_dir()
    labs_data_dir = os.path.join(project_dir, "data", "labs_rts_data", lab_name, item_name)
    logging.info("getting path for lab data item %s", labs_data_dir)
    if not os.path.exists(labs_data_dir):
        raise FileNotFoundError(
            f"Directory for lab {lab_name} does not exist: {labs_data_dir}. "
            "Make sure to run the lab preparation first."
        )
    return labs_data_dir

def get_model_saving_dir() -> str:
    """
    Constructs the path to the directory where model data should be saved.

    Args:
        lab_name (str): The name of the lab.

    Returns:
        str: The absolute path to the lab's model saving directory.
    """
    project_dir = get_project_dir()
    model_saving_dir = os.path.join(project_dir, "models", "saved")

    if not os.path.exists(model_saving_dir):
        os.makedirs(model_saving_dir)
        logging.info("created directory for %s models: %s", model_saving_dir)

    return model_saving_dir


def shut_down() -> None:
    """
    Shuts down the IPython kernel if running in an IPython environment.
    """
    logging.info("Shutting down...")
    ipython = get_ipython()
    if ipython is not None:
        ipython.kernel.do_shutdown(True)
        logging.info("IPython kernel shut down.")
    else:
        logging.info("Not running in IPython environment. No shutdown performed.")
