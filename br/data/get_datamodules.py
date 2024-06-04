from hydra.utils import instantiate
import yaml

import yaml
import os

CONFIG_PATH = CYTODL_CONFIG_PATH + "/results/"

configs = os.listdir(CONFIG_PATH)
DATA_LIST = {}
for config in configs:
    data = config.split(".")[0]
    with open(CONFIG_PATH + config) as stream:
        a = yaml.safe_load(stream)
        DATA_LIST[data] = a


def get_data(dataset_name, batch_size, debug=False):
    config_list = DATA_LIST[dataset_name]
    data = []
    for config_path in config_list:
        with open(config_path, "r") as stream:
            config = yaml.safe_load(stream)
            if batch_size:
                config["batch_size"] = batch_size
                if "shuffle" in config.keys():
                    config["shuffle"] = False
                if debug:
                    config["subsample"] = {}
                    config["subsample"]["train"] = 1
                    config["subsample"]["valid"] = 1
                    config["subsample"]["test"] = 1
            data.append(instantiate(config))
    return data
