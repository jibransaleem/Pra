import logging
import os
import yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "Logs")
YML_PATH = os.path.join(BASE_DIR, "params.yaml")

os.makedirs(LOG_DIR, exist_ok=True)


class Files:
    def __init__(self):
        pass
    def load_params(self):
        with open(YML_PATH, "r") as file:
            data = yaml.safe_load(file)
        return data


class log:
    def __init__(self, Logger_name, file_name):
        self.file_name = file_name
        self.logger_name = Logger_name
        self.formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s \n")

    def cli_(self):
        cli_handler = logging.StreamHandler()
        cli_handler.setFormatter(self.formatter)
        return cli_handler

    def file_(self):
        file_name = f"{self.file_name}.log"
        file_path = os.path.join(LOG_DIR, file_name)
        file_handler = logging.FileHandler(filename=file_path)
        file_handler.setFormatter(self.formatter)
        return file_handler

    def get_logger(self):
        my_logger = logging.getLogger(self.logger_name)
        my_logger.setLevel("INFO")
        my_logger.addHandler(self.cli_())
        my_logger.addHandler(self.file_())
        return my_logger
obj   = Files()
print(obj.load_params())