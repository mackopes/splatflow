import os

from splatflow.config import Config


def initialise_directories(config: Config):
    required_directories = ["datasets", "models"]

    for dir in required_directories:
        full_dir_path = os.path.join(config.splatflow_data_root, dir)
        os.makedirs(full_dir_path, exist_ok=True)


def initialise(config: Config):
    initialise_directories(config)
