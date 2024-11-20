import pandas as pd
import yaml
from cyto_dl.models.utils.mlflow import get_config, load_model_from_checkpoint
from hydra._internal.utils import _locate
from hydra.utils import instantiate

from br.data.get_datamodules import get_data
from br.models.utils import get_all_configs_per_dataset


def load_model_from_path(dataset, results_path, strict=False, split="val", device="cuda:0"):
    MODEL_INFO = get_all_configs_per_dataset(results_path)
    models = MODEL_INFO[dataset]
    model_manifest = pd.read_csv(models["orig_df"])
    model_sizes = []
    all_models = []
    for j, ckpt_path in enumerate(models["model_checkpoints"]):
        if "model_paths" in models.keys():
            config_path = models["model_paths"][j]
        else:
            config_path = ckpt_path.split("ckpt")[0] + "yaml"
        with open(config_path) as stream:
            config = yaml.safe_load(stream)
        model_conf = config["model"]
        model_class = model_conf.pop("_target_")
        model_conf = instantiate(model_conf)
        model_class = _locate(model_class)
        all_models.append(
            model_class.load_from_checkpoint(
                ckpt_path, **model_conf, strict=strict, map_location=device
            ).eval()
        )
        model_sizes.append(config["model/params/total"])

    return all_models, models["names"], model_sizes, model_manifest


def load_model_from_mlflow(dataset, results_path, split="val"):
    TRACKING_URI = "https://mlflow.a100.int.allencell.org"
    MODEL_INFO = get_all_configs_per_dataset(results_path)
    models = MODEL_INFO[dataset]
    model_manifest = pd.read_csv(models["orig_df"])
    model_sizes = []
    all_models = []
    for i in models["run_ids"]:
        all_models.append(
            load_model_from_checkpoint(
                TRACKING_URI,
                i,
                path=f"checkpoints/{split}/loss/best.ckpt",
                strict=False,
            )
        )
        config = get_config(TRACKING_URI, i, "./tmp")
        model_sizes.append(config["model/params/total"])

    return all_models, models["names"], model_sizes, model_manifest


def get_data_and_models(dataset_name, batch_size, results_path, debug=False):
    data_list = get_data(dataset_name, batch_size, results_path, debug)
    all_models, run_names, model_sizes, model_manifest = load_model_from_path(
        dataset_name, results_path
    )  # default list of models in load_models.py
    return data_list, all_models, run_names, model_sizes, model_manifest
