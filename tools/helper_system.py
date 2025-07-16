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
            raise Exception("Could not find project root directory.")
        current_dir = parent_dir
    logging.info(f"Project root directory found: {current_dir}")
    return current_dir


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
