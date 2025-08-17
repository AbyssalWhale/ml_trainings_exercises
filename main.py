import logging

from labs.lab1 import lab1
from labs.lab2 import lab2
from labs.lab3 import lab3
from labs.lab4 import lab4
from labs.lab5 import lab5
from tools.helper_system import shut_down

logging.basicConfig(
    filename='lab_executions_logs.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)


def main():
    try:
        # lab1()
        # shut_down()
        # lab2()
        # shut_down()
        # lab3()
        # shut_down()
        # lab4()
        # shut_down()
        lab5()
    finally:
        shut_down()


if __name__ == "__main__":
    main()
