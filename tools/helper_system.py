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
    labs_data_dir = os.path.join(project_dir, "data", "labs", lab_name)
    lab_data_saving_dir = os.path.join(labs_data_dir, lab_name)

    if not os.path.exists(lab_data_saving_dir):
        os.makedirs(lab_data_saving_dir)
        logging.info(f"Created directory for %s data: %s", lab_name, lab_data_saving_dir)

    return lab_data_saving_dir


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
