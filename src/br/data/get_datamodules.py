import os

import yaml
from hydra.utils import instantiate

from br.models.utils import get_all_configs_per_dataset


def get_data(dataset_name, batch_size, results_path, debug=False):
    DATA_LIST = get_all_configs_per_dataset(results_path)
    config_list = DATA_LIST[dataset_name]

    # Get config path from CYTODL_CONFIG_PATH
    cytodl_config_path = os.environ.get("CYTODL_CONFIG_PATH")

    data = []
    for config_path in config_list["data_paths"]:
        config_path = cytodl_config_path + config_path
        with open(config_path) as stream:
            config = yaml.safe_load(stream)
            if batch_size:
                config["batch_size"] = batch_size
                if "shuffle" in config.keys():
                    config["shuffle"] = False
                if debug:
                    if config["_target_"] == "cyto_dl.datamodules.DataframeDatamodule":
                        config["subsample"] = {}
                        config["subsample"]["train"] = 4
                        config["subsample"]["valid"] = 4
                        config["subsample"]["test"] = 4
            data.append(instantiate(config))
    return data
