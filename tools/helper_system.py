import logging
import os
from IPython import get_ipython


def get_project_dir():
    logging.info("trying to find project root directory...")
    current_dir = os.path.abspath(os.path.dirname(__file__))
    while not os.path.isfile(os.path.join(current_dir, 'main.py')):
        parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
        if parent_dir == current_dir:
            raise Exception("could not find project root directory.")
        current_dir = parent_dir
    return current_dir

def shut_down():
    logging.info("shutting down")
    ipython = get_ipython()
    if ipython is not None:
        ipython.kernel.do_shutdown(True)