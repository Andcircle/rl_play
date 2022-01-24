import os
import yaml
import sys
import logging
import os

def init_logging(logging_dir):
    logging_path = os.path.join(logging_dir, 'log')
    file_writer = logging.FileHandler(logging_path)
    formatter = logging.Formatter('%(asctime)s %(name)-3s %(levelname)-3s %(message)s')
    file_writer.setLevel(logging.INFO)
    file_writer.setFormatter(formatter)

    console_writer = logging.StreamHandler(sys.stdout)
    console_writer.setLevel(logging.INFO)

    root = logging.getLogger()
    root.addHandler(file_writer)
    root.addHandler(console_writer)
    root.setLevel(logging.INFO)


def load_config(config_path):
    with open(config_path, 'r') as rf:
        config = yaml.load(rf)

    return config


def save_config(config, save_dir):
    save_path = save_dir + '/config.yml'
    with open(save_path, 'w') as wf:
        yaml.dump(config, wf)



